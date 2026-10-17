# Draft issue for francois141/ptx_gym (not yet posted)

**Title:** Sanitizer/NCU timeout leaves the kernel running on the GPU and skews later timings

**Pinned commit:** aa634dbc537454230111fafac3ee1f7769dbfa6c

## Summary

`_run_sanitizer_tool` (`ptx_gym/evaluation/sanitizer.py:150-173`) and the NCU runner
(`ptx_gym/evaluation/ncu.py:715-741`) call `subprocess.run(..., timeout=...)`. On
timeout, Python kills only the direct child (`compute-sanitizer` / `ncu`). The
processes it spawned, `TreeLauncherSubreaper` and the
`python -m ptx_gym.evaluation.sanitizer --run-request ...` target, keep running on
the GPU. Every benchmark in the same session then shares the GPU with the orphan,
so both the candidate and the Triton baseline are timed too slow.

## Reproduction (V100-SXM2-16GB, sm_70)

`MatrixMultiplicationFloat16`, config block_m=32, block_n=64, block_k=32, num_warps=4:

1. `triton.testing.do_bench` on `forward_triton`: **13.53 ms** p50.
2. `diagnose_ptx(..., sanitizer_tool="racecheck", timeout_seconds=60)` on the
   operator's own Triton PTX returns "Compute Sanitizer timed out after 60 seconds."
3. `ps` / `nvidia-smi --query-compute-apps` still show `TreeLauncherSubreaper` and the
   target Python process on the GPU.
4. Same benchmark again: **31.65 ms** p50.

racecheck alone needs ~1105 s for this GEMM on V100 (memcheck 10 s, synccheck and
initcheck 5 s each), so the default 120 s timeout fires on every large-kernel
candidate, and every timing after it in the session is affected.

## Suggested fix

Start the tool in its own session and kill the whole process group on timeout:

```python
with subprocess.Popen(cmd, start_new_session=True, stdout=PIPE, stderr=PIPE, text=True) as proc:
    try:
        out, err = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
```

When NCU runs through a `sudo` wrapper (needed for performance counters), the
process tree is root-owned and can't be signalled by the caller, so the wrapper
itself needs a bound, e.g. `timeout -k 5 <secs> ncu ...`.

Happy to open a PR with this change.
