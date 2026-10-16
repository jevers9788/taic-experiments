# Report

## Status (2026-10-16)

- **V100 (sm_70): complete.** 9 baselines, identity and hand-written PTX checks,
  and TAIC end to end with a stub model (no API calls). Table:
  `results/baselines.md`.
- **L4 (control) and T4: not run.** No on-demand capacity in any US zone on
  2026-10-15/16 (logs in `logs/`).

## Question

Old GPUs lose performance on kernels with no library implementation because
Triton stopped optimizing for them (no tensor cores below sm_80, PORTING.md 7).
Can an LLM writing PTX directly fill that gap? Old GPUs also stand in for any
hardware a compiler serves poorly, the paper's own pitch.

Hypotheses:
1. For kernels without a library path, much of the lost performance is software.
2. An LLM can recover some of it, without an expert.
3. Low-precision storage decoded in registers (FP8) can beat the FP16 ceiling
   on bandwidth-bound work.

Out of scope: economics (cost per token) and scale-out claims.

## V100 results (p50; measured ceilings 825 GB/s, 97 TFLOP/s cuBLAS FP16)

| Kernel | Triton | Best stock |
|---|---|---|
| FP16 GEMM | 13.4 ms (10.5% of ceiling) | cuBLAS 1.92 ms (73.5%) |
| BitDelta | 12.7 ms | eager 3.0 ms |
| FlashAttention | 3.65 ms | SDPA 0.76 ms |
| FP16 GEMV | 0.130 ms (31% BW) | cuBLAS 0.054 ms (75% BW) |
| RMSNorm | 0.013 ms (57% BW) | eager 0.035 ms |
| Softmax | 0.0123 ms | eager 0.0123 ms (below timer resolution) |
| SageAttention | fails to compile | eager 2.25 ms |
| FP8 GEMM | fails (no fp8e4nv on sm_70) | FP16 upcast + cuBLAS 2.06 ms |
| FP8 GEMV | fails (same) | FP16 upcast 0.178 ms (3.3x slower than FP16 GEMV) |

Hardware confirmed the upstream FlashAttention reference mismatch (88% of
outputs; PORTING.md 3).

## Interpretation

- Primary metric is speedup vs autotuned Triton, as in the paper. cuBLAS/SDPA
  and the measured ceilings are context: they show what is attainable.
- Not expected: beating cuBLAS on plain GEMM. Realistic: closing Triton's gap
  toward it, and beating stock paths that are generic workarounds (BitDelta,
  FP8 GEMV, SageAttention).
- No absolute Triton latencies exist to reproduce: the paper reports speedups
  only and the repos ship no archived runs.

## Next

1. L4 control and T4 sessions when capacity allows.
2. Add Mamba-2 (2), FlashSinkhorn, Forgetting Attention, BitDelta batched
   (upstream references, labelled as such). Mamba-2 is the most adopted
   (Granite 4.0, Nemotron-H, Falcon-H1).
3. Triton-free launcher, so TAIC can target kernels Triton can't compile
   (SageAttention, FP8); PORTING.md 2.
4. Real TAIC runs (user): GEMM, BitDelta, FlashAttention, FP8 GEMV first.
5. Optional: parallel pre-compilation for slow pre-Ampere autotunes.

## Publishing

- Ours: harness, findings, results, write-up. Add a LICENSE; scrub usernames
  and project name from logs.
- Not ours: TAIC/PTX Gym code (no license; link pinned commits).
- Ask the authors first: TAIC-generated PTX (derived from unlicensed sources).
- Report upstream: two TAIC startup bugs, wheel missing `kernels/papers`,
  FlashAttention reference mismatch.
