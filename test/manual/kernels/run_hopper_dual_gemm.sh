#!/usr/bin/env bash
# Run from any directory inside this checkout. All logs survive a failed stage.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash test/manual/kernels/run_hopper_dual_gemm.sh STAGE [OUTPUT_DIR]
Stages: check, compile, unit, bench, sanitizer, all
all = check + compile + unit + bench; sanitizer is explicitly requested.
Environment: CUDA_VISIBLE_DEVICES (default 0), PYTHON (default python),
BENCH_RUNS (default 1), BENCH_DTYPES (default bf16), REPETITIONS (default 50).
bench tests TP1/TP4 local widths, rotation 1/4, and tokens 1/2/4/8/9/16, and all three tactics.
EOF
}

stage="${1:-all}"
case "$stage" in
    -h|--help) usage; exit 0 ;;
    check|compile|unit|bench|sanitizer|all) ;;
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
    command -v compute-sanitizer >/dev/null
    local tool
    for tool in memcheck racecheck synccheck; do
        run_logged "$tool" compute-sanitizer --tool "$tool" --error-exitcode 1 \
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
