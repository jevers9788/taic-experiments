# Status: GPU capacity blocked; no measurements

On 2026-10-15, GCP rejected startup of the existing L4 twice and T4 creation in
six zones: us-west1-a, us-west1-b, us-central1-a, us-central1-c, us-east1-c,
us-east4-a. Each reported ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS. Raw error
logs are under logs/. Both n1-standard-8 and n1-standard-4 T4 configurations
were attempted. These are capacity failures, not evidence about CUDA or Triton.

The final inventory contains only the original stopped L4, and no taic disks.
No GPU benchmark, CUDA assembly check, sanitizer run, profiling run, or model
API request completed. There are no latency, throughput, speedup, or cost
measurements to report. The CSV marks all pending kernel/GPU pairs explicitly.

## Prepared

- Pinned TAIC and PTX Gym checkouts excluded from redistribution.
- Draft Ubuntu/CUDA 12.8 setup and serial baseline runner.
- Draft no-model handwritten ReLU evaluation smoke test using PTX Gym's ABI,
  assembly, sanitizers, differential verifier, timing, and NCU evaluator.
- Source-inspection findings in PORTING.md.

## Not complete

- L4 calibration and all T4 baselines, BF16-input case, throughput/peak context.
- GPU validation of all scripts and meaningful hardware correctness tests.
- Porting fixes established by observed sm_75 errors.
- Compiler-independent evaluation where Triton itself fails.
- Generation uses TAIC itself (paper defaults). Its offline end-to-end check
  (`scripts/taic_fixture.py`, stub model) has not yet run on a GPU.
- No absolute Triton latencies exist to reproduce: the paper reports only
  speedups, and the released repos contain no archived baseline runs.
- A measured ranking. None is defensible yet.

## Open questions

- When will an authorized GCP zone have capacity for these GPUs?
- Which pinned kernels actually compile on sm_75?
- Which upstream reference/tuning defects reproduce on hardware?
- Which library path wins under each exact kernel contract?

No change to experiment scope is implied by this blocker. Resume with the L4
calibration and T4 setup when capacity is available. Do not interpret draft
scripts as a verified ready-to-run environment.
