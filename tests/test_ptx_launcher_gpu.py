"""GPU checks for the Triton-free PTX launcher.

Run on a GPU VM from the experiment root, after `source scripts/env.sh`:
  .venv/bin/python tests/test_ptx_launcher_gpu.py results/<gpu>
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import torch
from ptx_launcher import launch_ptx, record_launch
from ptx_gym.kernels import resolve_kernel

results = Path(sys.argv[1])

# 1. Triton's own PTX, launched by us, must reproduce Triton's own launch bit for bit.
#    Covers a 1-D grid, a callable grid (GEMM), extra scalars, and packed int inputs.
for name in ['MatrixVectorMultiplicationFloat16Kernel', 'SoftmaxFloat16Kernel',
             'MatrixMultiplicationFloat16', 'BitDeltaNeurIPS2024Matmul']:
    tuning = json.loads((results / f'{name}.json').read_text())['autotune']['parameters']
    op = resolve_kernel(name)(ptx={'tuning_config': tuning})
    torch.manual_seed(0)
    x = op.get_random_input()
    expected = op.forward_triton(x)[0].clone()
    _, compiled = op.forward_triton(x)
    launch, out = record_launch(op, x)
    out.zero_()
    launch_ptx(compiled.asm['ptx'], launch, shared_bytes=compiled.metadata.shared)
    torch.cuda.synchronize()
    assert torch.equal(out, expected), f'{name}: launcher output differs from Triton'
    print(f'{name}: identical to Triton ({launch.grid} x {launch.block})', flush=True)

# 2. SageAttention: recording the launch must not compile (Triton fails on sm_70).
op = resolve_kernel('SageAttentionICLR2025')(ptx={'tuning_config': {}})
x = op.get_random_input()
launch, out = record_launch(op, x)
assert launch.grid == (8, 16, 16), launch.grid
assert launch.block == (128, 1, 1), launch.block
assert len(launch.args) == 8 and all(isinstance(a, torch.Tensor) for a in launch.args), launch.args
assert out.shape == (16, 16, 1024, 64), out.shape
print('SageAttentionICLR2025: launch recorded without compiling', flush=True)

# 3. A PTX entry that doesn't match the recorded arguments is rejected before launch.
bad = '.version 8.7\n.target sm_70\n.address_size 64\n.visible .entry kernel(\n\t.param .u32 n\n)\n{\n\tret;\n}\n'
try:
    launch_ptx(bad, launch)
    raise AssertionError('expected ValueError for a mismatched entry')
except ValueError as error:
    print(f'mismatched entry rejected: {error}', flush=True)
print('ok')
