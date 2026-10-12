"""Compare merged linear + SwiGLU with Hopper TMA/WGMMA, including down GEMM."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import hopper_dual_gemm_swiglu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("tokens", [1, 2, 4, 8, 9, 16], [8])
@marker.parametrize("intermediate", [14336, 3584], [3584])
@marker.parametrize("dtype", [torch.bfloat16, torch.float16], [torch.bfloat16])
@marker.parametrize("tactic", [-1, 0, 1, 4], [-1])
@marker.parametrize("full_mlp", [False, True])
@marker.benchmark("implementation", ["merged", "dual"])
def benchmark(tokens, intermediate, dtype, tactic, full_mlp, implementation):
    if torch.cuda.get_device_capability() != (9, 0):
        return marker.skip("requires SM90")
    x = torch.randn(tokens, 4096, device="cuda", dtype=dtype) * 0.25
    weight = torch.randn(2 * intermediate, 4096, device="cuda", dtype=dtype) * 0.05
    down = torch.randn(4096, intermediate, device="cuda", dtype=dtype) * 0.05

    def run(x, weight, down):
        y = (
            silu_and_mul(torch.nn.functional.linear(x, weight))
            if implementation == "merged"
            else hopper_dual_gemm_swiglu(x, weight, tactic)
        )
        return torch.nn.functional.linear(y, down) if full_mlp else y

    return marker.do_bench(
        run,
        input_args=(x, weight, down),
        # Account for and rotate exactly the tensors read by the measured path.
        memory_args=(x, weight, down) if full_mlp else (x, weight),
        graph_clone_args=(0, 1, 2) if full_mlp else (0, 1),
    )


if __name__ == "__main__":
    benchmark.run()
