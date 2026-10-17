"""Run compute-sanitizer racecheck on a saved PTX candidate, with a long timeout.

For candidates from TAIC runs made with --skip-racecheck that we consider promising.
Usage: python scripts/racecheck.py KERNEL PTX_FILE --threads-x N
         --tuning-from results/<gpu>/KERNEL.json [--timeout 3600]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import ptxgym_fixes  # noqa: F401  (process-group kill on timeout)
from ptx_gym.evaluation.sanitizer import diagnose_ptx
from ptx_gym.evaluation.types import Payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel')
    parser.add_argument('ptx', type=Path)
    parser.add_argument('--threads-x', type=int, required=True)
    parser.add_argument('--tuning-from', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    tuning = json.loads(args.tuning_from.read_text())['autotune']['parameters']
    payload = Payload(ptx=args.ptx.read_text(), threads_x=args.threads_x)
    result = diagnose_ptx(args.kernel, payload.to_launch_dict(), sanitizer_tool='racecheck',
                          timeout_seconds=args.timeout, tuning_config=tuning)
    print(json.dumps({k: result.get(k) for k in ('clean', 'error', 'diagnostics')}, indent=2))
    sys.exit(0 if result.get('clean') is True else 1)


if __name__ == '__main__':
    main()
