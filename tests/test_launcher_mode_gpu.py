"""GPU checks for launcher mode (TAIC/PTX Gym without compiling Triton for the target).

Run on a GPU VM from the experiment root, after `source scripts/env.sh`:
  PTX_LAUNCHER_KERNELS=SageAttentionICLR2025,MatrixVectorMultiplicationFloat16Kernel \
    .venv/bin/python tests/test_launcher_mode_gpu.py results/<gpu>
"""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import launcher_mode  # noqa: F401  (also applied at startup by scripts/site/sitecustomize.py)
from ptx_gym.evaluation.evaluate import TritonPTXCandidateEvaluator
from ptx_gym.evaluation.types import Payload
from ptx_gym.helpers.kernels import extract_specification_from_operator
from ptx_gym.helpers.triton import dump_kernel_ptx
from ptx_gym.kernels import resolve_kernel
import torch

results = Path(sys.argv[1])
assert launcher_mode.ACTIVE == {'SageAttentionICLR2025', 'MatrixVectorMultiplicationFloat16Kernel'}

# 1. The generated entry template equals Triton's real entry where Triton compiles
#    (Softmax and BitDelta are not in launcher mode here).
for name in ['SoftmaxFloat16Kernel', 'BitDeltaNeurIPS2024Matmul']:
    op = resolve_kernel(name)(ptx={'tuning_config': {}})
    real = dump_kernel_ptx(op)
    template = launcher_mode.template_for(op)
    entry = lambda p: p[p.index('.visible .entry'):p.index(')', p.index('.visible .entry')) + 1]
    assert entry(template) == entry(real), f'{name}:\n{entry(template)}\n!=\n{entry(real)}'
    print(f'{name}: template entry == Triton entry', flush=True)

# 2. GEMV in launcher mode: Triton's own PTX passes PTX Gym's full evaluation
#    (sanitizer subprocess, verifier, timing, ncu), timed against the stock path.
op = resolve_kernel('MatrixVectorMultiplicationFloat16Kernel')(ptx={'tuning_config': json.loads(
    (results / 'MatrixVectorMultiplicationFloat16Kernel.json').read_text())['autotune']['parameters']})
ptx = (results / 'ptx' / 'MatrixVectorMultiplicationFloat16Kernel.ptx').read_text()
result = TritonPTXCandidateEvaluator(type(op), operator=op).evaluate(Payload(ptx=ptx, threads_x=op.num_warps * 32))
assert result.passed, result.message
assert result.triton_p50 < 0.09, f'baseline should be stock torch.mv (~0.053 ms), got {result.triton_p50}'
print(f'GEMV: passed; candidate {result.p50:.4f} ms vs stock {result.triton_p50:.4f} ms; '
      f'ncu available={result.ncu_report.get("available")}', flush=True)

# 3. SageAttention (Triton can't compile it on sm_70): constexprs survive, the template
#    comes from the recorded launch, and a do-nothing candidate runs through the whole
#    pipeline, failing only the correctness check.
op = resolve_kernel('SageAttentionICLR2025')(ptx={'tuning_config': {}})
spec = extract_specification_from_operator(op)
assert spec.constexpr_values.get('stride_qn') == 64, spec.constexpr_values
template = dump_kernel_ptx(op)
assert template.count('.param') == 10 and '.target sm_70' in template, template
result = TritonPTXCandidateEvaluator(type(op), operator=op).evaluate(Payload(ptx=template, threads_x=128))
assert result.compiles and result.sanitizer_report.get('clean') is True, result.message
assert not result.correct and result.message == 'Correctness check failed', result.message
print('Sage: template + sanitizer subprocess + verifier ran without Triton; correctness failed as expected', flush=True)

# 4. TAIC end to end on SageAttention with the stub model (answers with the template).
trace = results / 'taic-fixture-sage-launcher'
run = subprocess.run([sys.executable, 'scripts/taic_fixture.py', '--kernel', 'SageAttentionICLR2025',
                      '--tuning-from', str(results / 'SageAttentionICLR2025.json'), '--skip-racecheck',
                      '--trace-path', str(trace)], capture_output=True, text=True)
assert run.returncode != 0 and 'Initial candidate must compile and pass verification' in run.stderr, run.stderr[-2000:]
prompt = next(trace.glob('*/iteration_000_try_00_initial_candidate_prompt.txt')).read_text()
assert '.visible .entry kernel(' in prompt and 'stride_qn' in prompt, prompt[:2000]
print('TAIC on Sage: prompt built without Triton; stub candidate rejected by correctness', flush=True)
print('ok')
