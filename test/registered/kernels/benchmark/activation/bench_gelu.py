import itertools

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.activation.gelu import new_gelu, quick_gelu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@torch.compile
def torch_new_gelu(x):
    return F.gelu(x, approximate="tanh")


@torch.compile
def torch_quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


OPS = {"new": (new_gelu, torch_new_gelu), "quick": (quick_gelu, torch_quick_gelu)}
DTYPES = [torch.float16]
SIZES = [
    1,
    128,
    4096,
    65536,
    1048576,
]


def benchmark(op, dtype, size):
    x = create_random(size, dtype=dtype)
    # Both implementations use the same cold-L2 measurement and quantile.
    jit, torch_impl = OPS[op]
    jit_s = marker.do_bench(jit, input_args=(x,)).times[0]
    torch_s = marker.do_bench(torch_impl, input_args=(x,)).times[0]
    return jit_s * 1e6, torch_s * 1e6, torch_s / jit_s


def main():
    print(
        f"{'op':>6} {'dtype':>10} {'size':>10} | {'JIT(us)':>10} {'torch(us)':>10} {'speedup':>9}"
    )
    for op, dtype, size in itertools.product(OPS, DTYPES, SIZES):
        jit_us, torch_us, speedup = benchmark(op, dtype, size)
        print(
            f"{op:>6} {str(dtype).removeprefix('torch.'):>10} {size:>10} | "
            f"{jit_us:>10.4f} {torch_us:>10.4f} {speedup:>8.2f}x"
        )


if __name__ == "__main__":
    main()
