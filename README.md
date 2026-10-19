# Triton AI Compiler - Legacy Hardware Experiments

Does LLM-written PTX ("AI lowering", arXiv 2609.36800) help on GPUs that Triton no
longer optimizes for? Harness, baselines and results on a V100 (sm_70).

See [REPORT.md](REPORT.md) for measured performance, validation status, and remaining
experiments. [PORTING.md](PORTING.md) documents compatibility problems, upstream
bugs, and the workarounds used to run the experiments.

## Pinned sources

- TAIC: https://github.com/francois141/triton_ai_compiler at
  `6a00d1261c803fecdfa74c1c349b339a026b06ad`
- PTX Gym (TAIC submodule): `aa634dbc537454230111fafac3ee1f7769dbfa6c`
- PTX Gym includes a modified Triton 3.6.0 that can execute candidate PTX in place
  of compiler-generated PTX. Stock Triton cannot replace this build.

## Cloud (GCP)

This experiment is only supported on GCP for now.
Copy `config.example.sh` to `config.local.sh` and set `GCP_PROJECT`. VM naming and the remote directory are optional.

## Running it

1. Run `bash scripts/provision.sh v100 ZONE...` (or choose `l4`, `t4`, or `cpu`).
   It creates one VM in the first zone with capacity, with a two-hour STOP limit.
2. Build the PTX Gym wheel once on a CPU VM with `bash scripts/build-wheel.sh`
   and reuse it on GPU VMs. Copy this project's scripts and the wheel to the GPU
   VM, then run `TRITON_WHEEL=/abs/path/to/wheel.whl bash scripts/setup-vm.sh`.
   For V100, also set `TORCH_CUDA12=1` to install a compatible PyTorch build.
3. Run commands from the experiment root after `source scripts/env.sh`.
   Setup pins dependencies and saves their versions; `bash scripts/run-session.sh v100`
   records GPU/tool versions and runs the offline checks and all baselines.
4. Start with `.venv/bin/python scripts/baselines.py SoftmaxFloat16Kernel
   --output results/v100/SoftmaxFloat16Kernel.json` for a single baseline.
   `bash scripts/run-baselines.sh` runs all nine cases serially.

5. `.venv/bin/python scripts/check_candidate.py --identity` checks that Triton's
   own PTX passes the evaluation pipeline. Omit `--identity` to check handwritten
   ReLU PTX. Neither calls a model.

6. To generate PTX (paid API calls), set `OPENAI_API_KEY` in your environment and run:

   ```bash
   .venv/bin/python scripts/taic.py KERNEL --max-budget USD --trace-path runs
   ```

## Notes

- **Generation:** TAIC defaults to `gpt-6-astra` with `max` reasoning effort and
  generates directly from Triton source. It uses profiler feedback to propose
  improvements, keeping faster candidates that pass the configured checks.
  The budget gate runs before each request, so a request can finish over budget.
- **Offline check:** `.venv/bin/python scripts/taic_fixture.py --trace-path results/<gpu>/taic-fixture`
  exercises TAIC's loop with a stub model and handwritten ReLU PTX. It makes no
  API calls and is included in `run-session.sh`.
- **Kernels Triton cannot compile:** prefix the generation command with
  `PTX_LAUNCHER_KERNELS=<Kernel>` for cases such as SageAttention or FP8 on V100.
  This bypasses Triton compilation. In this mode, output fields named `triton`
  or `speedup_vs_triton` refer to the stock PyTorch baseline (PORTING.md, item 14).
- **Reuse autotuning:** add `--tuning-from results/<gpu>/KERNEL.json` to reuse a
  saved baseline configuration. Generation stays direct and the baseline is
  timed live. `--start-json` instead resumes from a PTX candidate. The offline
  fixture also accepts `--tuning-from` and `--kernel`.
- **Racecheck:** keep it enabled unless its measured cost is prohibitive
  (~18 minutes per FP16 GEMM candidate on V100). If you use `--skip-racecheck`,
  run the following check before reporting a candidate; a numerical pass alone
  is insufficient (PORTING.md, item 11):

  ```bash
  .venv/bin/python scripts/racecheck.py KERNEL runs/<run>/final_candidate.ptx \
    --threads-x N --tuning-from results/<gpu>/KERNEL.json
  ```
