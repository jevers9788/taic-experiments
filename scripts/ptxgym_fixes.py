"""Runtime fixes for the pinned PTX Gym, applied on import (no upstream edits).

1. Timeouts kill the whole process tree (PORTING.md 12). PTX Gym runs
   compute-sanitizer and ncu via `subprocess.run(timeout=...)`, which kills only the
   direct child; the kernel under test kept running on the GPU and slowed every later
   timing (13.5 -> 31.6 ms for FP16 GEMM on V100). ncu runs under sudo, so its
   root-owned tree is bounded inside scripts/ncu-root.sh instead.
2. skip_racecheck(): optional; run racecheck later with scripts/racecheck.py.
"""
import os
import signal
import subprocess

from ptx_gym.evaluation import ncu, sanitizer


class _GroupKillingSubprocess:
    """Stand-in for the `subprocess` module whose run() kills the process group on timeout."""

    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def run(args, *, timeout=None, capture_output=False, check=False, **kwargs):
        if capture_output:
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with subprocess.Popen(args, start_new_session=True, **kwargs) as proc:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise
        result = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
        if check:
            result.check_returncode()
        return result


sanitizer.subprocess = ncu.subprocess = _GroupKillingSubprocess()


def skip_racecheck():
    """Run memcheck, synccheck and initcheck only (racecheck: ~18 min on V100 GEMM)."""
    sanitizer.SANITIZER_TOOLS = tuple(t for t in sanitizer.SANITIZER_TOOLS if t != 'racecheck')


if __name__ == '__main__':
    # Self-check: a timed-out process group must leave no survivors.
    try:
        _GroupKillingSubprocess.run(['bash', '-c', 'sleep 60 & sleep 60'], timeout=1)
    except subprocess.TimeoutExpired:
        pass
    survivors = subprocess.run(['pgrep', '-f', '^sleep 60$'], capture_output=True, text=True).stdout
    assert not survivors.strip(), f'survivors: {survivors}'
    skip_racecheck()
    assert 'racecheck' not in sanitizer.SANITIZER_TOOLS
    print('ok')
