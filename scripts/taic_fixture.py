"""Offline end-to-end check of TAIC (paper defaults) with no model calls.

Runs TAIC's own agent loop, prompts, PTX Gym evaluation, planning round and trace
output unchanged; only the provider session is replaced, by a stub that answers every
request with the hand-written ReLU PTX from check_candidate.py.
Usage (after `source scripts/env.sh`): python scripts/taic_fixture.py --trace-path DIR
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import taic  # TAIC with the upstream dump_kernel_ptx fix applied
from check_candidate import handwritten_relu
from ptx_gym.helpers.triton import dump_kernel_ptx
from utils.response_format import PtxKernel

agent = taic.agent

captured = {}
real_verifier_for_kernel = agent.verifier_for_kernel


def capturing_verifier(*args, **kwargs):
    captured['evaluator'] = real_verifier_for_kernel(*args, **kwargs)
    return captured['evaluator']


class Reply:
    """Just enough of a provider response for TAIC's response_json_text()."""

    def __init__(self, data):
        self.output_text = json.dumps(data)

    def model_dump(self, mode=None):
        return {'fixture': True, 'output_text': self.output_text}


class FixtureSession:
    """Replaces TAIC's ProviderSession; never touches the network. Cost is $0."""

    def request_json(self, *, pipeline, **_):
        op = captured['evaluator'].operator  # autotuned; its config fixes the contract
        threads = op.num_warps * 32
        kernel = {'ptx': handwritten_relu(dump_kernel_ptx(op), op.block_size, threads),
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
    args = parser.parse_args()
    agent.verifier_for_kernel = capturing_verifier
    agent.create_provider_session = lambda *a, **k: FixtureSession()
    # provider='openai' selects the paper's default prompt path; the session is the stub.
    final = json.loads(agent.run_agent_loop(
        'ReLUFloat16Kernel', model='fixture', provider='openai', max_tool_rounds=1,
        max_repair_attempts=0, reasoning_effort=None, trace_path=args.trace_path))
    print(json.dumps({k: final.get(k) for k in ('speedup', 'p50', 'run_cost_usd_so_far',
                                                'gpu_type')}, indent=2))
    assert final['p50'] and final['speedup'], 'TAIC did not produce a verified candidate'


if __name__ == '__main__':
    main()
