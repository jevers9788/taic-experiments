"""Offline end-to-end check of TAIC (paper defaults) with no model calls.

Runs TAIC's own agent loop, prompts, PTX Gym evaluation, planning round and trace
output unchanged; only the provider session is replaced, by a stub that answers every
request with the hand-written ReLU PTX from check_candidate.py, or, for any other
--kernel, with Triton's own PTX (identity candidate; expect ~1.0x).
Usage (after `source scripts/env.sh`): python scripts/taic_fixture.py --trace-path DIR
  [--kernel NAME] [--tuning-from results/<gpu>/<Kernel>.json]  (see scripts/taic.py)
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import taic  # TAIC with the upstream dump_kernel_ptx fix applied
from check_candidate import handwritten_relu
from ptx_gym.helpers.triton import dump_kernel_ptx
from utils.response_format import PtxKernel

agent = taic.agent

captured = {}


class Reply:
    """Just enough of a provider response for TAIC's response_json_text()."""

    def __init__(self, data):
        self.output_text = json.dumps(data)

    def model_dump(self, mode=None):
        return {'fixture': True, 'output_text': self.output_text}


class FixtureSession:
    """Replaces TAIC's ProviderSession; never touches the network. Cost is $0."""

    def __init__(self, kernel_name):
        self.kernel_name = kernel_name

    def request_json(self, *, pipeline, **_):
        op = captured['evaluator'].operator  # autotuned; its config fixes the contract
        threads = op.num_warps * 32
        ptx = dump_kernel_ptx(op)
        if self.kernel_name == 'ReLUFloat16Kernel':
            ptx = handwritten_relu(ptx, op.block_size, threads)
        print(f'=== Fixture stub answering {pipeline} at {time.strftime("%FT%TZ", time.gmtime())} ===',
              flush=True)
        kernel = {'ptx': ptx,
                  'num_threads_x': threads, 'num_threads_y': 1, 'num_threads_z': 1,
                  'difficulties': []}
        if pipeline == 'initial_candidate':
            return Reply(kernel), 0.0, None
        if pipeline == 'improvement_plan':
            return Reply({'improvements': [{'name': 'fixture', 'rationale': 'offline check',
                                            'instruction': 'resubmit unchanged'}]}), 0.0, None
        if pipeline.endswith('failure_analysis'):
            return Reply({'root_cause': 'fixture', 'repair_instruction': 'none'}), 0.0, None
        # candidate / repairs: metadata reply plus the "patched" PTX, as tool calls would give.
        metadata = {k: v for k, v in kernel.items() if k != 'ptx'}
        return Reply(metadata), 0.0, PtxKernel(**kernel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-path', required=True, type=Path)
    parser.add_argument('--kernel', default='ReLUFloat16Kernel')
    parser.add_argument('--tuning-from', type=Path)
    parser.add_argument('--skip-racecheck', action='store_true')
    args = parser.parse_args()
    if args.skip_racecheck:
        taic.ptxgym_fixes.skip_racecheck()
    if args.tuning_from:
        taic.use_saved_tuning(args.tuning_from)
    real_verifier_for_kernel = agent.verifier_for_kernel

    def capturing_verifier(*a, **k):
        captured['evaluator'] = real_verifier_for_kernel(*a, **k)
        return captured['evaluator']

    agent.verifier_for_kernel = capturing_verifier
    agent.create_provider_session = lambda *a, **k: FixtureSession(args.kernel)
    # provider='openai' selects the paper's default prompt path; the session is the stub.
    final = json.loads(agent.run_agent_loop(
        args.kernel, model='fixture', provider='openai', max_tool_rounds=1,
        max_repair_attempts=0, reasoning_effort=None, trace_path=args.trace_path))
    print(json.dumps({k: final.get(k) for k in ('speedup', 'p50', 'run_cost_usd_so_far',
                                                'gpu_type')}, indent=2))
    print('Evaluator tuning config:', captured['evaluator'].operator.best_config)
    assert final['p50'] and final['speedup'], 'TAIC did not produce a verified candidate'


if __name__ == '__main__':
    main()
