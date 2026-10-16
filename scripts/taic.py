"""Run TAIC's CLI unchanged, with one upstream bug worked around (see PORTING.md 8).

Released agent.py and utils/setup.py call `dump_kernel_ptx(<operator object>)` but
import the name-based `ptx_gym.api.dump_kernel_ptx`, so every run crashes before any
model call. Point both at the operator-based helper instead (as TAIC's own
run_identity_compare.py does). Usage: python scripts/taic.py KERNEL [agent flags]

Optional `--tuning-from results/<gpu>/<Kernel>.json` (ours, not TAIC's) reuses the
Triton config our baseline autotune saved instead of re-autotuning at startup.
Generation stays direct (no starting candidate); see use_saved_tuning().
"""
import argparse
import json
from pathlib import Path
import sys

import agent  # TAIC, from .upstream/taic on PYTHONPATH
from ptx_gym.helpers.triton import dump_kernel_ptx
import utils.response
import utils.setup

# Both call sites pass an operator object: agent.py:545 and utils/setup.py:199.
agent.dump_kernel_ptx = dump_kernel_ptx
utils.setup.dump_kernel_ptx = dump_kernel_ptx


def use_saved_tuning(path):
    """Build every TAIC evaluator from a saved tuning config, without a starting candidate.

    Upstream only accepts `autotune_metrics` via --start-json, which also requires a
    PTX candidate (seeded generation). verifier_for_kernel(kernel, autotune_metrics)
    is the evaluator path that --start-json uses (fixed config, no autotune); we pass
    our config to it for the agent's evaluator and the provider tools' verifiers.
    The Triton baseline is still timed live on every evaluation (evaluate.py).
    """
    record = json.loads(Path(path).read_text())
    metrics = {'selected_config': record['autotune']['parameters'],
               'tuning_result': record['autotune']}
    real = utils.response.verifier_for_kernel

    def verifier(kernel_name, autotune_metrics=None, **kwargs):
        assert kernel_name == record['kernel'], f'{path} is for {record["kernel"]}'
        return real(kernel_name, autotune_metrics or metrics, **kwargs)

    agent.verifier_for_kernel = utils.response.verifier_for_kernel = verifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--tuning-from', type=Path)
    args, sys.argv[1:] = parser.parse_known_args()
    if args.tuning_from:
        use_saved_tuning(args.tuning_from)
    agent.main()
