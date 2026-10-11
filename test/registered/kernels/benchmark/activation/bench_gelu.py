import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.activation.gelu import _gelu_module, new_gelu, quick_gelu
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


def jit_vectorized(x, quick):
    # Preserve the original vector-width selection as an A/B baseline.
    out = torch.empty_like(x)
    _gelu_module(x.dtype, quick, scalar_small=False).run(x, out)
    return out


@marker.parametrize("op", ["new", "quick"])
@marker.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], [torch.bfloat16]
)
@marker.parametrize(
    "size",
    [
        1,
        7,
        8,
        9,
        31,
        32,
        33,
        64,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1024,
        4096,
        65536,
        1048576,
    ],
    [32, 127, 256, 257, 512, 4096, 65536],
)
@marker.benchmark("impl", ["jit", "vectorized", "torch"])
def benchmark(op, dtype, size, impl):
    x = create_random(size, dtype=dtype)
    if impl == "vectorized":
        return marker.do_bench(jit_vectorized, input_args=(x, op == "quick"))
    fn = OPS[op][0 if impl == "jit" else 1]
    # marker rotates inputs and outputs to measure with cold L2.
    return marker.do_bench(fn, input_args=(x,))


if __name__ == "__main__":
    benchmark.run()
