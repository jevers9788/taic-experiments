# TAIC older-GPU baseline experiment

Does LLM-written PTX ("AI lowering", arXiv 2609.36800) help on GPUs that Triton no
longer optimizes for? Harness, baselines and results on a V100 (sm_70); see
REPORT.md for results and PORTING.md for findings. Upstream TAIC and PTX Gym are
checked out under `.upstream/` by setup and are not redistributed here.

## Pinned sources

- TAIC: https://github.com/francois141/triton_ai_compiler at
  `6a00d1261c803fecdfa74c1c349b339a026b06ad`
- PTX Gym (TAIC submodule): `aa634dbc537454230111fafac3ee1f7769dbfa6c`
- Patched Triton identifies itself as 3.6.0.

The released README claims a compiler-PTX starting default, but pinned agent.py
defaults `start_triton_generated_ptx=False`. It still compiles the Triton reference
and dumps its PTX before any model call, even in direct-generation mode. Its default model
is `gpt-6-astra`, reasoning effort `max`. Do not run `agent.py` during setup.

## Cloud (GCP)

Copy `config.example.sh` to `config.local.sh` (gitignored) and set `GCP_PROJECT`
(a project with GPU quota) and, optionally, VM naming and the remote directory.
Scripts read it via `scripts/config.sh`. VMs are on-demand with a run-time STOP
limit set at creation; GPU capacity is often exhausted, so `provision.sh` tries
zones in order (`logs/*-create-*.log` show past attempts).

Pricing sources (planning only, not measured charges):
https://cloud.google.com/products/compute/pricing/accelerator-optimized
https://cloud.google.com/products/compute/gpus-pricing

Stop VMs when idle. Disk charges persist while stopped. Use `/tmp` staging and
absolute paths for report downloads. NCU must run through `sudo -E` and report
ownership must be restored afterward.

## Running it

1. `bash scripts/provision.sh l4|t4|v100|cpu ZONE...` creates at most one VM
   (first zone with capacity) with a run-time STOP limit. Limits are per start,
   not a billing cap: track sessions and disk charges.
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
   - Kernels Triton can't compile on the target (e.g. SageAttention, FP8 on sm_70):
     prefix the command with `PTX_LAUNCHER_KERNELS=<Kernel>` (PORTING.md 14);
     speedups are then vs the stock PyTorch path, not Triton.
   - TAIC re-autotunes Triton at startup (~1 h for FP16 GEMM on V100). Add
     `--tuning-from results/<gpu>/KERNEL.json` (ours) to reuse the config our
     baseline saved; generation stays direct and Triton is still timed live.
     Do not use TAIC's `--start-json` for this: it requires a PTX candidate and
     seeds generation from it (a different method from the paper's).
     The fixture takes the same flag and `--kernel` (non-ReLU kernels get
     Triton's own PTX back, so ~1.0x).
   - Large kernels on pre-Ampere GPUs: add `--skip-racecheck` (racecheck takes
     ~18 min per FP16 GEMM candidate on V100). Before reporting a result, run
     `scripts/racecheck.py KERNEL runs/<run>/final_candidate.ptx --threads-x N
     --tuning-from results/<gpu>/KERNEL.json` on it (PORTING.md 11).

See REPORT.md for remaining work and PORTING.md for correctness concerns.

The temporary L4 scheduling change was removed after capacity failures, restoring
its prior policy. The instance remains stopped. No model service was modified.
