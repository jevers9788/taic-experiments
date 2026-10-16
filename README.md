# TAIC older-GPU baseline experiment

Scope: L4 control and T4 (`sm_75`), $20 incremental cloud budget, no model calls.
Upstream sources are private local checkouts under `.upstream/`, excluded from
version control; no upstream code is redistributed here.

## Pinned sources

- TAIC: https://github.com/francois141/triton_ai_compiler at
  `6a00d1261c803fecdfa74c1c349b339a026b06ad`
- PTX Gym (TAIC submodule): `aa634dbc537454230111fafac3ee1f7769dbfa6c`
- Patched Triton identifies itself as 3.6.0.

The released README claims a compiler-PTX starting default, but pinned agent.py
defaults `start_triton_generated_ptx=False`. It still compiles the Triton reference
and dumps its PTX before any model call, even in direct-generation mode. Its default model
is `gpt-6-astra`, reasoning effort `max`. Do not run `agent.py` during setup.

## Cloud

Project `<gcp-project>`, existing L4 `l4-20260917-140617-17234`, zone `us-west1-a`.
A two-hour STOP scheduling limit was installed before attempting startup.
Initial startup failed with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`; see logs.
No calibration numbers are available yet.

Pricing sources (planning only, not measured charges):
https://cloud.google.com/products/compute/pricing/accelerator-optimized
https://cloud.google.com/products/compute/gpus-pricing

Stop VMs when idle. Disk charges persist while stopped. Use `/tmp` staging and
absolute paths for report downloads. NCU must run through `sudo -E` and report
ownership must be restored afterward.

## Resume (draft; GPU execution blocked)

1. Check `logs/final-instances.log` and confirm GCP capacity. Before starting
   the existing L4, set a two-hour STOP limit with `gcloud compute instances
   set-scheduling ... --max-run-duration=7200s --instance-termination-action=STOP`.
   `scripts/provision.sh l4|t4 ZONE...` tries zones in order and creates at most one VM with the same limit.
   These are per-start limits, not a cumulative $20 billing guard. Track all
   sessions and disk charges; do not repeatedly restart without checking budget.
2. Copy this project's scripts to an absolute directory on the VM. Run
   `bash scripts/setup-vm.sh` there, saving stdout/stderr to a log. Driver setup
   can reboot; reconnect and rerun. Do not run this against active shared jobs.
   Building patched Triton on a 4-vCPU GPU VM exceeded the 2h limit, so build it
   once on a CPU VM instead: `bash scripts/provision.sh cpu ZONE...`, then
   `scripts/build-wheel.sh` there (Proton and C++ unit tests off). Built
   2026-10-15 on n2-standard-32 in 928 s (`logs/cpu-build-wheel.log`):
   `.cache/wheels/ptx_gym-3.6.0-cp310-cp310-linux_x86_64.whl`, sha256
   `165d41bc…6fcb`. Copy it to the GPU VM and run setup with
   `TRITON_WHEEL=/abs/path/to/that.whl`. Delete the CPU VM afterwards.
3. From the experiment root, `source scripts/env.sh`. Capture `nvidia-smi`,
   `nvcc --version`, `ptxas --version`, `ncu --version`, package freeze, and exact
   commands. Setup installs TAIC's uv.lock exactly (torch 2.11.0, CUDA 13 runtime;
   kernels use patched Triton's ptxas 12.8.93 = PTX ISA 8.7, as in the paper).
   CUDA 13 dropped sm_60/sm_70: V100/P100 will need a CUDA 12 torch (a deviation).
4. Start with `.venv/bin/python scripts/baselines.py SoftmaxFloat16Kernel
   --output results/l4-softmax.json` (one shell line), followed by GEMM.
   `bash scripts/run-baselines.sh` runs the seven cases (six upstream + FP8 GEMM) serially.
5. `.venv/bin/python scripts/check_candidate.py --identity` checks identity PTX;
   omit `--identity` for the handwritten ReLU fixture. Neither command calls
   an LLM. Both require a GPU and the patched Triton installation.
6. Generation uses TAIC itself (the paper's agent), with its defaults, which match
   the paper: direct generation from the Triton source (NOT seeded with Triton's
   PTX; TAIC's README wrongly says otherwise), plan -> up to three candidate
   patches -> keep only safe + correct + faster, NCU and PTX-ISA skills, budget
   gate checked before each request.
   - Verify (no network): `.venv/bin/python scripts/taic_fixture.py --trace-path
     results/<gpu>/taic-fixture` runs TAIC's real loop with a stub model that
     returns the hand-written ReLU PTX. Part of `run-session.sh`.
   - Real run (paid API calls; you run it), from the experiment root after
     `source scripts/env.sh`:
     `OPENAI_API_KEY=... .venv/bin/python scripts/taic.py KERNEL --max-budget USD --trace-path runs`
     (`scripts/taic.py` = TAIC's CLI with an upstream startup crash worked
     around; plain `python -m agent` crashes, see PORTING.md item 8)
     (defaults: `--provider openai --model gpt-6-astra --reasoning-effort max`,
     as in the paper; the paper used $15 per kernel for its harder kernels).
     `--provider anthropic|openrouter` and `--model` switch models. Cost is
     only tracked, and `--max-budget` only enforced, for models in TAIC's
     pricing table (`.upstream/taic/utils/cost.py`); others log "cost
     unavailable". Per-run cost: `runs/<run>/prices.log`.
   - TAIC needs Triton to compile the kernel first (PORTING.md item 2).

See REPORT.md for remaining work and PORTING.md for correctness concerns.

The temporary L4 scheduling change was removed after capacity failures, restoring
its prior policy. The instance remains stopped. No model service was modified.
