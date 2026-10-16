# Inspection findings, not GPU results

Sources: TAIC 6a00d1261c803fecdfa74c1c349b339a026b06ad and its PTX Gym
submodule aa634dbc537454230111fafac3ee1f7769dbfa6c. No upstream source was edited.

1. PTX Gym bundles patched Triton 3.6.0 and CUDA 12.8.93 ptxas. Use a separate
   Python environment; stock Triton cannot replace its PTX-injection changes.
2. TAIC README and implementation disagree about the initial candidate.
   `agent.py:run_agent_loop` defaults `start_triton_generated_ptx=False`, but
   always creates a baseline evaluator and dumps Triton PTX before opening a
   provider session. Source-first generation still requires successful Triton
   compilation. Compiler-independent evaluation is NOT implemented yet.
   Worse, PTX injection in the patched Triton only swaps the module at the cubin
   stage (`triton_src/python/triton/compiler/compiler.py:324`); every earlier
   stage (TTIR, TTGIR, LLIR, PTX) still runs. So where Triton fails to lower a
   kernel for a target, PTX Gym cannot evaluate ANY candidate for it, and
   `scripts/generate.py` cannot run it either. Planned fix: a Triton-free ctypes
   CUDA-driver launcher using the contract captured where Triton works.
   `scripts/baselines.py` now saves each kernel's Triton PTX (entry ABI) and
   compile metadata to `results/<gpu>/ptx/`; the grid is not yet captured.
3. FlashAttention forward launches with IS_CAUSAL=False, while forward_torch
   applies a triangular causal mask. The local baseline draft uses a noncausal
   independent reference and reports the upstream eager reference as a failure
   if it disagrees. No source computation or tolerance has been silently changed.
   SDPA output-only timing excludes the LSE output; it is not a complete drop-in
   replacement for the upstream output-plus-LSE contract.
4. RMSNorm's reference squares FP16 values, while the Triton kernel promotes to
   FP32. The independent reference uses FP32 intermediates. Its tuning space
   includes block sizes that change the reduction domain; correctness must gate
   acceptance, not just successful compilation.
5. Upstream autotuning compares candidates against an initial Triton output,
   not an independent reference. Draft baselines apply an independent check to
   the selected result. They do not yet retune all candidates against that
   independent reference if selection fails. Such results must remain failed.
6. SageAttention's current independent-reference coverage is incomplete: the
   draft uses its upstream PyTorch reference. Additional validation is required.

7. No tensor cores in Triton below sm_80. `AccelerateMatmul.cpp:373` in the pinned
   Triton rejects MMA for compute capability < 80 ("has been deprecated. It falls
   back to the FMA path"), so `tl.dot` on T4 (sm_75) and V100 (sm_70) uses scalar
   FMA even though both have tensor cores (`mma.sync` m16n8k8 / m8n8k4).
   Upstream: triton-lang/triton PR #5066 (merged 2024-11-05) added the fallback and
   changed supported hardware from CC 7.0+ to 8.0+; the PR gives no rationale.
   Maintainers later (2025-03-03, same PR thread): pre-Ampere support is "best
   effort and won't be tested on our side", and they are "unlikely to accept
   complex performance related patches". Consequence: the primary metric stays
   speedup vs autotuned Triton on the same contract, as in the paper. The paper
   picks Triton as the boundary to exclude PyTorch's fusion and library dispatch
   (§3), and its weak-baseline warning (§3.2) concerns the Triton *source*, which
   is unchanged here; a weak *lowering* for this target is what AI lowering
   replaces. cuBLAS / SDPA and the measured ceilings are reported as context
   (attainability, and to avoid overclaiming), not as the comparison baseline.

8. Released TAIC cannot start. `agent.py:545` and `utils/setup.py:199` pass an operator object to
   `ptx_gym.dump_kernel_ptx`, which (in the pinned PTX Gym, `ptx_gym/api.py`)
   takes a kernel *name*, so every run raises `Unknown kernel <...object>` before
   any model call (observed on V100, logs/v100-taic-fixture.log). Workaround
   without editing upstream: `scripts/taic.py` rebinds it to the operator-based
   `ptx_gym.helpers.triton.dump_kernel_ptx`, as TAIC's run_identity_compare.py uses.

9. Prebuilt-wheel install differs from upstream's editable install in two ways
   (found on V100): PTX Gym looks for ptxas inside its source tree
   (`helpers/environment.py`), and `ptx_gym/kernels/papers/` has no `__init__.py`,
   so the wheel omits the paper kernels (NCU subprocess failed with
   `No module named 'ptx_gym.kernels.papers'`). setup-vm.sh symlinks the wheel's
   ptxas into the source tree and loads ptx_gym from the source tree via a .pth.

10. TAIC prompt caveats (`.upstream/taic/prompts/`). (a) The FP16-GEMM and
    FlashAttention expert rule blocks never fire: `prompts/initial.py:81-92` gates
    them on `ptx_gym.kernels.level2_float16` / `FlashAttentionFloat16Kernel`, which
    don't exist in the pinned PTX Gym; only the convolution block fires. The paper's
    runs may have included them. (The FA block also describes a causal 8x16x256x64
    kernel, not ours.) (b) Prompts differ by provider: OpenAI "Fastest PTX", Anthropic
    "Basic PTX ... prioritize correctness", plus extra GEMM guidance; cross-provider
    comparisons also compare instructions. (c) The performance rule pushes ldmatrix,
    tensor cores and async copies "when useful for the target"; sm_70 lacks ldmatrix
    and cp.async, so expect failed attempts on V100.

No sm_75 portability claim can be made without running on a T4.
