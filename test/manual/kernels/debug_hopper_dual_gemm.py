"""One launch per process, with line information and optional register ablation."""

import argparse
import json
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.gemm.cutedsl_hopper_dual_gemm import (
    HopperDualGemmKernel,
    _tma_tensor,
    _validate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-registers", action="store_true")
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    m, n, k = (8, 64, 128) if args.aligned else (1, 65, 128)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    weight = torch.randn(2 * n, k, device="cuda", dtype=x.dtype) * 0.1
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _validate(x, weight)
    # Compute the independent reference and finish input creation before the
    # suspect launch, so an asynchronous failure has only one possible source.
    gate, up = (x.double() @ weight.double().T).chunk(2, dim=-1)
    reference = (
        torch.nn.functional.silu(gate.to(x.dtype).float()) * up.to(x.dtype).float()
    ).to(x.dtype)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cutlass": cutlass.__version__,
                "shape": [m, n, k],
                "dtype": str(x.dtype),
                "tactic": 1,
                "redistribute_registers": not args.static_registers,
                "dump_dir": os.environ.get("CUTE_DSL_DUMP_DIR"),
            }
        ),
        flush=True,
    )
    arguments = (
        *(_tma_tensor(t) for t in (x, *weight.chunk(2, dim=0))),
        from_dlpack(out.detach(), assumed_align=16),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cute.compile(
        HopperDualGemmKernel(m, 1, redistribute_registers=not args.static_registers),
        *arguments,
        options="--gpu-arch=sm_90a --generate-line-info --keep-ptx --keep-cubin",
    )
    print("Compilation complete; launching exactly once", flush=True)
    compiled(*arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, atol=0.006, rtol=0.03)
    print("Launch and numerical comparison passed", flush=True)


if __name__ == "__main__":
    main()
