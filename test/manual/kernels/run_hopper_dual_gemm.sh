#!/usr/bin/env bash
# Run from any directory inside this checkout. All logs survive a failed stage.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash test/manual/kernels/run_hopper_dual_gemm.sh STAGE [OUTPUT_DIR]
Stages: check, compile, debug, unit, bench, sanitizer, all
all = check + compile + unit + bench; sanitizer is explicitly requested.
Environment: CUDA_VISIBLE_DEVICES (default 0), PYTHON (default python),
BENCH_RUNS (default 1), BENCH_DTYPES (default bf16), REPETITIONS (default 50).
bench tests TP1/TP4 local widths, rotation 1/4, tokens 1/2/4/8/9/16, auto dispatch and all three explicit tactics.
EOF
}

stage="${1:-all}"
case "$stage" in
    -h|--help) usage; exit 0 ;;
    check|compile|debug|unit|bench|sanitizer|all) ;;
    *) usage; exit 2 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
out="${2:-/tmp/hopper-dual-gemm-results}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"

run_logged() {
    local name="$1"
    shift
    echo "Running $name; log: $out/$name.log"
    "$@" 2>&1 | tee "$out/$name.log"
}

check() {
    git rev-parse HEAD > "$out/revision.txt"
    nvidia-smi > "$out/nvidia-smi.txt"
    run_logged environment "$py" - <<'PY'
import json
import os
import shutil
import subprocess
import sys

import cutlass
import torch
import sglang
from cutlass.cute.nvgpu import warpgroup
from cutlass.pipeline import PipelineTmaAsync

assert torch.cuda.is_available(), "CUDA PyTorch / visible GPU unavailable"
cap = torch.cuda.get_device_capability()
assert cap == (9, 0), f"Requires SM90; got {cap}"
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else shutil.which("nvcc")
if not nvcc and os.path.isfile("/usr/local/cuda/bin/nvcc"):
    nvcc = "/usr/local/cuda/bin/nvcc"
if nvcc:
    print(subprocess.check_output([nvcc, "--version"], text=True))
assert hasattr(warpgroup, "MmaF16BF16Op"), "CuTe DSL lacks Hopper WGMMA support"
props = torch.cuda.get_device_properties(0)
target = "sm_90a"  # Explicit cute.compile target in the kernel wrapper
print(json.dumps({
    "gpu": props.name, "capability": cap, "sm_count": props.multi_processor_count,
    "memory_bytes": props.total_memory, "torch": torch.__version__,
    "torch_cuda": torch.version.cuda, "sglang_source": sglang.__file__,
    "python_executable": sys.executable,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "jit_target": target,
    "cutlass_dsl": cutlass.__version__, "nvcc": nvcc,
}, indent=2))
PY
}

compile() {
    run_logged compile "$py" - <<'PY'
import torch
from sglang.kernels.ops.gemm import hopper_dual_gemm_swiglu
from sglang.kernels.ops.gemm.cutedsl_hopper_dual_gemm import HOPPER_DUAL_GEMM_TACTICS

for dtype in (torch.bfloat16, torch.float16):
    for m in (1, 9):
        x = torch.randn(m, 128, device="cuda", dtype=dtype) * 0.25
        w = torch.randn(130, 128, device="cuda", dtype=dtype) * 0.1
        for tactic in range(len(HOPPER_DUAL_GEMM_TACTICS)):
            print(f"Compiling/running dtype={dtype}, M={m}, tactic={tactic}", flush=True)
            y = hopper_dual_gemm_swiglu(x, w, tactic)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all().item()
print("All twelve tactic/dtype/token-tile specializations compiled and launched.", flush=True)
PY
}

find_sanitizer() {
    local candidate
    if command -v compute-sanitizer >/dev/null 2>&1; then
        command -v compute-sanitizer
        return 0
    fi
    for candidate in "${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" \
        "${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer" \
        /usr/local/cuda/compute-sanitizer/compute-sanitizer; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

debug() {
    local variant case_dir failed=0
    local sanitizer_path
    local -a flags launcher
    launcher=()
    if sanitizer_path="$(find_sanitizer)"; then
        echo "Using compute-sanitizer: $sanitizer_path"
        launcher=("$sanitizer_path" --tool memcheck --error-exitcode 1)
        echo "$sanitizer_path" > "$out/debug-mode.txt"
    else
        echo "compute-sanitizer not found; running four direct diagnostic launches."
        echo "These runs do not perform sanitizer checks; install the CUDA compute-sanitizer tool for instruction localization."
        echo "direct launches; no sanitizer checks" > "$out/debug-mode.txt"
    fi
    for variant in original static-registers aligned aligned-static-registers; do
        flags=()
        case "$variant" in
            static-registers) flags=(--static-registers) ;;
            aligned) flags=(--aligned) ;;
            aligned-static-registers) flags=(--aligned --static-registers) ;;
        esac
        case_dir="$out/debug-$variant"
        mkdir -p "$case_dir"
        # A poisoned CUDA context must never be reused by the next case.
        if run_logged "debug-$variant" env \
            CUDA_LAUNCH_BLOCKING=1 CUTE_DSL_DISABLE_FILE_CACHING=True \
            CUTE_DSL_DUMP_DIR="$case_dir" \
            ${launcher[@]+"${launcher[@]}"} \
            "$py" test/manual/kernels/debug_hopper_dual_gemm.py ${flags[@]+"${flags[@]}"}; then
            echo PASS > "$case_dir/status.txt"
        else
            echo FAIL > "$case_dir/status.txt"
            failed=1
        fi
    done
    return "$failed"
}

unit() {
    run_logged unit "$py" test/registered/kernels/ops/gemm/test_hopper_dual_gemm.py -v
}

bench() {
    local dtype width rotation run name
    for dtype in ${BENCH_DTYPES:-bf16}; do
        for width in 14336 3584; do
            for rotation in 1 4; do
                for ((run=1; run<=${BENCH_RUNS:-1}; run++)); do
                    name="bench-${dtype}-i${width}-rotation${rotation}-run${run}"
                    run_logged "$name" "$py" test/manual/kernels/bench_hopper_dual_gemm.py \
                        --hidden 4096 --intermediate "$width" --dtype "$dtype" \
                        --rotation "$rotation" --repetitions "${REPETITIONS:-50}" \
                        --output "$out/$name.jsonl"
                done
            done
        done
    done
}

sanitizer() {
    local sanitizer_path
    if ! sanitizer_path="$(find_sanitizer)"; then
        echo "ERROR: compute-sanitizer not found in PATH or the CUDA Toolkit; sanitizer checks did not run." >&2
        return 1
    fi
    local tool
    for tool in memcheck racecheck synccheck; do
        run_logged "$tool" "$sanitizer_path" --tool "$tool" --error-exitcode 1 \
            "$py" test/registered/kernels/ops/gemm/test_hopper_dual_gemm.py -v
    done
}

check
case "$stage" in
    all) compile; unit; bench ;;
    check) ;;
    *) "$stage" ;;
esac
echo "Finished $stage; results: $out"
