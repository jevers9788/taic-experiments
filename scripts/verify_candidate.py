"""Independently check and time a saved PTX candidate (e.g. from a TAIC trace).

Correctness uses baselines.py's independent references and edge cases, not PTX Gym's
Triton-vs-candidate check. Timing puts the candidate, autotuned Triton and the stock
PyTorch path in one process, interleaved over several rounds (rotating order), so
they are directly comparable. Racecheck is separate: scripts/racecheck.py.
Usage: python scripts/verify_candidate.py KERNEL PTX_FILE --threads-x N
         --tuning-from results/<gpu>/KERNEL.json --output results/<gpu>/NAME.json
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent))
import ptxgym_fixes  # noqa: F401  (process-group kill on timeouts)
import torch
from baselines import compare, edge_input, flops, measure_peaks, reference, roofline, traffic_bytes
from ptx_gym.evaluation.performance import benchmark
from ptx_gym.kernels import resolve_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel')
    parser.add_argument('ptx', type=Path)
    parser.add_argument('--threads-x', type=int, required=True)
    parser.add_argument('--tuning-from', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tuning = json.loads(args.tuning_from.read_text())['autotune']['parameters']
    op = resolve_kernel(args.kernel)(ptx={'tuning_config': tuning})
    op.set_ptx({'ptx': args.ptx.read_text(), 'num_threads_x': args.threads_x})
    functions = {'candidate': lambda x: op.forward_triton(x, ptx=True)[0],
                 'triton': lambda x: op.forward_triton(x)[0],
                 'pytorch_eager': lambda x: op.forward_torch(x)}
    tolerance = getattr(op, 'verification_tolerance', 1e-3)
    record = {'command': sys.argv, 'kernel': args.kernel, 'ptx_file': str(args.ptx),
              'gpu': torch.cuda.get_device_name(), 'tuning': tuning, 'rtol': tolerance,
              'atol': tolerance, 'peaks': measure_peaks(), 'correct': {}, 'p50_rounds_ms': {}}
    for label, fn in functions.items():
        try:
            for case in ['random'] * 5 + ['zeros', 'small']:
                x = edge_input(args.kernel, op.get_random_input(), case)
                compare(fn(x), reference(args.kernel, op, x), tolerance)
            record['correct'][label] = True
        except Exception:
            record['correct'][label] = False
            record.setdefault('errors', {})[label] = traceback.format_exc()
    timed = [k for k, ok in record['correct'].items() if ok]
    x = op.get_random_input()
    for r in range(args.rounds):
        for label in timed[r % len(timed):] + timed[:r % len(timed)]:
            p50 = benchmark(lambda: functions[label](x)).p50
            record['p50_rounds_ms'].setdefault(label, []).append(p50)
    record['p50_ms'] = {k: statistics.median(v) for k, v in record['p50_rounds_ms'].items()}
    record['roofline'] = {k: roofline(v, traffic_bytes(x, functions[k](x)), flops(args.kernel, x),
                                      record['peaks']) for k, v in record['p50_ms'].items()}
    if 'candidate' in record['p50_ms']:
        record['speedup_vs'] = {k: v / record['p50_ms']['candidate']
                                for k, v in record['p50_ms'].items() if k != 'candidate'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record.get(k) for k in ('correct', 'p50_ms', 'speedup_vs')}, indent=2))


if __name__ == '__main__':
    main()
