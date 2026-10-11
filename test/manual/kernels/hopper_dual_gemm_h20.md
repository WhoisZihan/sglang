# H20: Hopper CuTe DSL DualGEMM validation

This version replaces the earlier C++ cp.async implementation. The kernel is
CuTe DSL, with TMA loads, a producer warpgroup, a WGMMA consumer warpgroup,
multistage shared storage, and a register SwiGLU epilogue. It is a candidate for
upstream validation; GPU compilation, numerical validation, sanitizer checks,
and performance tuning have not run on the macOS development machine.
Automatic model routing is not enabled until measured full-MLP gains are known.

## Apply the code

The full patch is based on SGLang `de487f8039e06853b5f376fd5f068ea9d7c400bb`.
Use `full.patch` for a checkout without the old prototype, or `upgrade.patch`
for a checkout with the earlier three commits / original handoff patch.
Do not apply both. From the server checkout:

```bash
git apply --check /path/to/hopper-dsl-handoff/full.patch
git apply /path/to/hopper-dsl-handoff/full.patch
```

Use the CUDA Python environment corresponding to this checkout, including its
pinned `nvidia-cutlass-dsl` package (currently 4.8.0). The operator uses the DSL
compiler, not the C++ `load_jit`/nvcc path. You do not need to rebuild the AOT
`sglang-kernel` wheel. `compute-sanitizer` and `ncu`, when used, come from the
CUDA Toolkit. A working driver and CuTe DSL compiler/runtime are required.

```bash
cd /path/to/sglang
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
# Fresh processes invalidate the process-local compiled-function cache.
bash test/manual/kernels/run_hopper_dual_gemm.sh check /tmp/h20-dualgemm-dsl
```

The check records GPU, compute capability, SM count, PyTorch/CUDA/CuTe versions,
source location, Python executable, and nvcc version when available. The kernel
compilation target is explicitly `sm_90a`. Source fingerprints from the old
`SGLANG_JIT_CACHE_DIR` do not control this DSL operator.

## Compile and test

```bash
bash test/manual/kernels/run_hopper_dual_gemm.sh compile /tmp/h20-dualgemm-dsl
bash test/manual/kernels/run_hopper_dual_gemm.sh unit /tmp/h20-dualgemm-dsl
```

Compilation covers BF16/FP16, both token tiles (8 and 16), and all three tactics.
The compile smoke checks launch and finite output. Unit tests verify independent
FP64 references with low-precision projection rounding, all tactics, short
pipelines, feature/token/K padding, ring-buffer wraparound, graph replay with
changed inputs and weights on a nondefault stream, invalid storage, empty batch,
decode limits, and the torch.compile custom-op/fake contract.

The supported scope is contiguous BF16/FP16 X `[M,K]` and W `[2*I,K]`, M=0..16,
positive K divisible by 64, positive I, 16-byte-aligned input pointers, and no
autograd, bias, LoRA, or quantization. Out-of-bounds TMA loads zero-fill padding;
output stores are predicated. Keep numerical failures as failures, and send
`compile.log` immediately if compilation fails.

## Diagnose a runtime illegal instruction

If compilation finishes but the first launch reports CUDA error 715, run:

```bash
bash test/manual/kernels/run_hopper_dual_gemm.sh debug /tmp/h20-dualgemm-debug
```

This runs four separate processes under compute-sanitizer memcheck, retaining
line information, PTX and CUBIN. The first reproduces BF16 M=1, I=65, K=128,
tactic 0. The second removes dynamic register redistribution only. The last
two use M=8, I=64 to remove feature/token padding, with and without register
redistribution. TMA, WGMMA, warp specialization and the pipeline remain active
in all cases. These are diagnostic variants, not performance measurements.
The public operator still uses its original register policy.

The runner searches PATH and the CUDA Toolkit for compute-sanitizer. If the
tool is unavailable, it prints a notice and still runs all four cases directly
with blocking launches. Those direct runs are not sanitizer checks. The selected
mode is recorded in `debug-mode.txt`; the explicit `sanitizer` stage requires
the tool and fails with an explanatory error if it cannot be found.

Failed cases do not prevent the remaining cases from running; the stage exits
nonzero if any case fails. Send `environment.log`, `debug-*.log`, and the four
`debug-*` artifact directories. A passing static-register case helps isolate
the fault but does not establish the exact failing instruction without the
sanitizer report and generated code. No GPU validation is available locally.

The first H100 PCIe report reproduced error 715 for M=1, I=65 with both
register policies. Both M=8, I=64 variants completed their numerical comparison.
The failing PTX used a 1D TMA X load at the reported source line; the aligned
variants used 2D loads. TMA inputs now keep their row extent dynamic to prevent
singleton rows from disappearing from the tensor map; K remains static.
Rerun `compile`, `unit`, and `debug` after this fix. The original reports also
contained 41 cuGetProcAddress_v2 API errors in every case, including the two
numerically successful cases; these are distinct from the kernel error 715.
These first reports preceded the singleton-row fix.

The subsequent H100 PCIe unit run passed the numerical/padding/pipeline,
CUDA Graph replay, storage, and tactic contract tests. The fullgraph test
failed while Dynamo traced the selector's lazy import, before entering the
kernel. The Hopper public entry now resolves its fixed backend through an
`assume_constant_result` helper, leaving the registered torch custom op in
the graph. The fullgraph regression clears the selector cache before tracing.
An isolated CPU check on PyTorch 2.14.1 reproduced the old lazy-import failure
and passed fullgraph AOT/fake execution with both cold and warm selector caches
after the fix, using a substitute custom op rather than the CUDA kernel.
This integration fix needs a repeat of the GPU unit suite; performance and
clean sanitizer results remain pending.

## Measure all tactics

```bash
bash test/manual/kernels/run_hopper_dual_gemm.sh bench /tmp/h20-dualgemm-dsl
BENCH_RUNS=3 BENCH_DTYPES="bf16 fp16" \
  bash test/manual/kernels/run_hopper_dual_gemm.sh bench /tmp/h20-dualgemm-dsl-repeat
```

The sweep covers M=1/2/4/8/9/16, hidden=4096, intermediate=14336/3584, rotation
1/4, and tactics 0/1/2. Tactics are `(features, reduction, stages)`:

- 0: `(64, 128, 3)`
- 1: `(128, 128, 3)`
- 2: `(64, 64, 5)`

The initial default is tactic 0; this is not a measured tuning conclusion.
The comparison is one merged gate/up linear plus SGLang SwiGLU, and reports
front-half and complete local MLP (including the same down projection) separately.
The two widths simulate TP1/TP4 local dimensions, without communication.
CUDA Graph replay and events measure five groups of samples; allocation,
compilation, and correctness checks are outside timing. All rotating tensor
sets are checked numerically. JSONL includes configuration, timing, speedup,
front maximum absolute error, relative L2 error, and environment information.
`speedup > 1` means DualGEMM is faster. Rotation=1 repeats the same weights;
rotation=4 rotates weights to assess cache sensitivity, not serving throughput.

There is also a CI benchmark using the repository marker's L2-aware rotation:

```bash
python test/registered/kernels/benchmark/gemm/bench_hopper_dual_gemm.py
```

## Sanitizers, profiling and return files

```bash
bash test/manual/kernels/run_hopper_dual_gemm.sh sanitizer /tmp/h20-dualgemm-dsl
# Profile one configuration and inspect registers, spills, memory and stalls.
ncu --set full --target-processes all \
  python test/manual/kernels/bench_hopper_dual_gemm.py \
  --tokens 8 --tactics 0 --intermediate 14336 --rotation 4 --repetitions 1

tar -czf /tmp/h20-dualgemm-dsl-results.tar.gz -C /tmp h20-dualgemm-dsl
```

Sanitizer runs memcheck/racecheck/synccheck with nonzero error exit status.
Send `environment.log`, `compile.log`, `unit.log`, `bench-*.jsonl` and sanitizer
logs. If a stage fails, include its `.log` and environment/revision metadata. Repeat
measurements on an idle GPU under consistent power/clock settings. H20 results
apply to H20; H100/H200 need their own calibration. Choose default tactics and
model eligibility using stable complete-MLP results, including regressions.

One command for the initial round:

```bash
bash test/manual/kernels/run_hopper_dual_gemm.sh all /tmp/h20-dualgemm-dsl
```
