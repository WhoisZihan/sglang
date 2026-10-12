"""Compare the Hopper TMA/WGMMA backend with merged gate/up GEMM plus SGLang SwiGLU.

Reports both the front half and the full local MLP, including the same down
projection. Graph replay uses a rotating input/weight set exceeding L2 when
requested. Compilation, allocation, and numerical checks are outside timing.
"""

import argparse
import json
from pathlib import Path

import cutlass
import torch

from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import hopper_dual_gemm_swiglu
from sglang.kernels.ops.gemm.cutedsl_hopper_dual_gemm import (
    HOPPER_DUAL_GEMM_TACTICS,
    _pick_hopper_tactic,
)


def _time(fn, repetitions):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repetitions)
    return sorted(samples)[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 9, 16])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--rotation", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, help="Write clean JSONL separately from JIT logs"
    )
    parser.add_argument(
        "--tactics", nargs="+", type=int, default=[-1, *HOPPER_DUAL_GEMM_TACTICS]
    )
    args = parser.parse_args()
    if max(args.tokens) > 16:
        parser.error("this backend targets decode with at most 16 tokens")
    if any(t != -1 and t not in HOPPER_DUAL_GEMM_TACTICS for t in args.tactics):
        parser.error("invalid tactic")
    if (
        min(
            args.tokens
            + [args.hidden, args.intermediate, args.repetitions, args.rotation]
        )
        <= 0
    ):
        parser.error("shapes, repetitions, and rotation must be positive")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("")

    def emit(record):
        line = json.dumps(record)
        print(line, flush=True)
        if args.output:
            with args.output.open("a") as stream:
                stream.write(line + "\n")

    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass": cutlass.__version__,
        "backend": "cute_dsl_sm90a_tma_warp_specialized",
        "tactic_schema": "hopper_one_based_v1",
        "tactics": HOPPER_DUAL_GEMM_TACTICS,
        "dtype": args.dtype,
        "rotation": args.rotation,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "multiprocessor_count": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        "weight_bytes_per_rotation": 6 * args.hidden * args.intermediate,
        "front_weight_bytes_total": 4 * args.hidden * args.intermediate * args.rotation,
        "mlp_weight_bytes_total": 6 * args.hidden * args.intermediate * args.rotation,
    }
    emit({"metadata": metadata})
    for m in args.tokens:
        inputs = [
            torch.randn(m, args.hidden, device="cuda", dtype=dtype) * 0.25
            for _ in range(args.rotation)
        ]
        weights = [
            torch.randn(2 * args.intermediate, args.hidden, device="cuda", dtype=dtype)
            * 0.05
            for _ in inputs
        ]
        downs = [
            torch.randn(args.hidden, args.intermediate, device="cuda", dtype=dtype)
            * 0.05
            for _ in inputs
        ]

        def baseline(x, w):
            return silu_and_mul(torch.nn.functional.linear(x, w))

        for tactic in args.tactics:
            selected_tactic = (
                _pick_hopper_tactic(
                    m,
                    args.hidden,
                    args.intermediate,
                    metadata["multiprocessor_count"],
                    dtype,
                )
                if tactic == -1
                else tactic
            )
            max_abs, relative_l2 = 0.0, 0.0
            for x, w, down in zip(inputs, weights, downs, strict=True):
                actual = hopper_dual_gemm_swiglu(x, w, tactic)
                expected = baseline(x, w)
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
                torch.testing.assert_close(
                    torch.nn.functional.linear(actual, down),
                    torch.nn.functional.linear(expected, down),
                    rtol=0.05,
                    atol=0.03,
                )
                error = actual.float() - expected.float()
                max_abs = max(max_abs, error.abs().max().item())
                relative_l2 = max(
                    relative_l2,
                    (error.norm() / expected.float().norm().clamp_min(1e-6)).item(),
                )
            row = {
                "tokens": m,
                "tactic": tactic,
                "selected_tactic": selected_tactic,
                "configuration": HOPPER_DUAL_GEMM_TACTICS[selected_tactic],
                "front_max_abs_error": max_abs,
                "front_relative_l2_error": relative_l2,
            }
            for full_mlp in (False, True):
                timings = {}
                for name in ("baseline", "dual"):

                    def run():
                        for xi, wi, di in zip(inputs, weights, downs, strict=True):
                            y = (
                                baseline(xi, wi)
                                if name == "baseline"
                                else hopper_dual_gemm_swiglu(xi, wi, tactic)
                            )
                            if full_mlp:
                                torch.nn.functional.linear(y, di)

                    timings[name] = _time(run, args.repetitions) / args.rotation
                row["mlp" if full_mlp else "front"] = {
                    "baseline_us": timings["baseline"],
                    "dual_us": timings["dual"],
                    "speedup": timings["baseline"] / timings["dual"],
                }
            emit(row)


if __name__ == "__main__":
    main()
