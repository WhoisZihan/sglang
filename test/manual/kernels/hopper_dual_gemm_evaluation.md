# Hopper DualGEMM and runtime follow-up assessment

Baseline: SGLang `de487f8039`, runtime assessment inspected 2026-10-10;
Hopper design updated 2026-10-11. GPU compilation and measurements are pending.

## 1. Hopper DualGEMM + SwiGLU

The implementation now lives in `ops/gemm/cutedsl_hopper_dual_gemm.py`.
It preserves the Blackwell `[gate; up]` weight layout and explicit projection
rounding before fused SwiGLU, with one shared X tile per stage. Architecture
specifics are SM90 TMA, WGMMA and register accumulators; no TMEM or tcgen05.

Each CTA has a 128-thread producer warpgroup and a 128-thread consumer
warpgroup. Warp 0 issues all three TMA transfers; their full combined byte
count is attached to a single stage barrier. Four consumer warps return stage
ownership only after both gate and up MMAs have completed. One committed MMA
group remains in flight to overlap stages. The final wait completes before
reading register accumulators. Output stores predicate token/feature tails;
TMA zero-fills out-of-bounds input coordinates, including a partial K=128 tile.

The operator supports FP16/BF16 decode M=0..16, positive K divisible by 64,
positive intermediate size, contiguous/aligned input storage, and no autograd,
bias, quantization, or LoRA. Three explicit tactics compare feature width,
K tile and pipeline depth. Tactic 0 is the initial policy, not a measured
optimum. Public registration uses `KernelBackend.CUTE_DSL`; Torch custom-op
and fake registration preserve the torch.compile boundary. No model fast path
is enabled until GPU evidence determines profitable shapes and tactics.

See `hopper_dual_gemm_h20.md` for the updated compile, unit, sanitizer and
benchmark commands. Old C++/nvcc and JIT-cache instructions no longer apply.
The numerical tests cover all tactics and pipeline boundary cases; manual
benchmark compares one merged gate/up GEMM plus existing SGLang activation,
and includes the same down GEMM. The registered benchmark uses the project
marker's L2-aware rotation. Local static checks cannot establish device
correctness, pipeline safety, occupancy, or speedup. Use H20 results to tune
H20; validate H100/H200 separately before generalizing the model policy.

## 2. FA multi-step draft metadata: narrower scope than the TODO suggests

Relevant code: `FlashAttentionMultiStepBackend.init_forward_metadata_out_graph`
in `python/sglang/srt/layers/attention/flashattention_backend.py` calls every
step's backend. Its TODO requests incremental metadata updates.

Current graph-replay topk=1 already uses `_set_decode_page_metadata`, which calls
the fused `normal_decode_set_metadata` kernel. It fills lengths, prefix sums
and page tables; translated unified-pool paths additionally copy from the
KV location plan. Consequently the proposal is cross-step reuse, not fusion
of an entirely unfused sequence. FA scheduler metadata is normal-decode-only
at this call site; it is not an additional repeated draft optimization target.

For step s in the topk=1 ordinary path, cache length is base length + s + 1.
Within a single iteration the request set and existing prefix pages can be
stable, while the tail/page boundary changes. Prefix sums can be derived from
the base prefix sum plus `(s+1)*arange(bs+1)`. This does NOT make the whole
metadata interchangeable: graph-owned buffers, sequence lengths, tail pages and
per-step write locations remain distinct.

Recommended first scope: topk=1, ordinary non-translating MHA pool, no SWA or
encoder/local attention. Build the prefix page table once for the iteration,
maintain a stable graph-visible buffer and update lengths/tail pages per step.
Keep fallback for translated pools and other geometries. Add broader support
only after measuring this version.

Topk>1 has a second opportunity: its normal prefix table is gathered/copied on
every step even though the common prefix is stable. The expansion table is
different per step; page_size>1 includes the last partial page, deduplication
and `last_page_lens`. Treat this as a separate follow-up rather than sharing
all buffers indiscriminately. For SWA/unified pools, allocator translations and
physical row lifetimes must be honored before any cross-step aliasing.

Measure metadata preparation, full draft iteration and full decode iteration
for batch sizes 1/8/32, contexts 1K/8K/32K/128K, steps 2/4/8 and page sizes
1/16/64. Profile the existing path before coding. A launch-count reduction
alone is not evidence of improved serving performance. The recent
[PR #42803](https://github.com/sgl-project/sglang/pull/42803) optimizes other
speculative KV-index builders; keep comparison and scope separate.

Assessment: plausible long-context performance PR, moderate/high correctness
risk, uncertain small-context benefit. Lower priority than the Hopper experiment
until a profile establishes a material repeated-prefix cost.

## 3. Mixed chunked prefill with logprobs: concrete functional gap

`Scheduler.get_new_batch_prefill` currently disables mixed batches when either
the prefill or running batch requests logprobs. Removing that guard alone is
incorrect:

1. `ScheduleBatch.mix_with_running` appends 1-token decode rows with a logprob
   start of zero. They would be interpreted as having one input-logprob row.
   Decode rows should contribute zero input rows while still retaining their
   sampling/output-logprob row.
2. `merge_batch` combines output top-k/token-ID preferences but does not create
   a valid extended input-logprob token-ID tensor when only the decode side
   requested logprobs. The forward batch and logits processor consume this
   tensor independently of the merged preferences.
3. `LogitsMetadata.from_forward_batch` and `_get_pruned_states` derive input
   and sampling indices from extend lengths and logprob starts. Their existing
   zero-input-row convention (`start == extend_len`) retains a last-token sample
   row, which should be reused for decode rows rather than introducing a
   second logits computation.
4. `BatchResultProcessor.process_batch_result_prefill` knows about decoding
   requests for cache handling but still routes logprobs through
   `_apply_prefill_logprobs`. Decode-tail output values must be appended without
   treating the processed decode token as a new prompt-logprob entry. Its input
   cursor must advance by exactly the number of prefill/scoring rows consumed.

Suggested first PR: text-only, ordinary non-speculative serving, eager forward;
cover both non-overlap and overlap before allowing the feature on overlap.
Keep unsupported graph/backend/spec/encoder/beam combinations gated explicitly.
Full prefill graphs already have eager logits-tail capability checks; mixed
support must pass their existing admission rules, not bypass them.

Required validation matrix:

- Only prefill requests logprobs, only decode requests logprobs, or both.
- A batch also contains requests without logprobs.
- Output-only logprobs, top logprobs and selected-token logprobs.
- Chunked prompt logprobs with start offsets and cache-hit prefixes.
- Finishing/in-flight decode rows under overlap, streaming and cursor accounting.
- Zero input rows, where output logprobs and sampling still remain present.

Compare against mixed chunk disabled, controlling prompts and generated lengths.
Check per-token logprobs, ordering, counts and top/token-ID values independently
of generated text. Report TTFT/TPOT tails and throughput; mixed chunking changes
their tradeoff, so a throughput gain alone is not sufficient.

Assessment: stronger functional justification than FA metadata optimization;
medium implementation scope with substantial bookkeeping/overlap testing.
It does not require a novel kernel, but it does require GPU end-to-end validation.

## Recommended order

1. Compile and validate the Hopper kernel; benchmark full local MLP shapes.
2. If promising, tune only measured shapes and integrate one ordinary dense model.
3. Implement mixed logprob support as a separate runtime PR.
4. Profile FA metadata; pursue cross-step prefix reuse if the cost is material.

No speedup is claimed, and no PR has been opened by this experiment.
