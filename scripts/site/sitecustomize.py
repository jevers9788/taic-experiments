"""Apply launcher mode at interpreter start when $PTX_LAUNCHER_KERNELS is set.

On PYTHONPATH via scripts/env.sh, so PTX Gym's sanitizer and ncu subprocesses
(which run their own Python) patch the same kernels as the parent process.
"""
import os

if os.environ.get('PTX_LAUNCHER_KERNELS'):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import launcher_mode  # noqa: F401  (applies itself)
    except ImportError:  # e.g. a Python without torch/ptx_gym
        pass
