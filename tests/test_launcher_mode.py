"""GPU-free checks for launcher mode. Run: python3 tests/test_launcher_mode.py"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from launcher_mode import arg_kind, entry_template, kernels_from_env
from ptx_launcher import parse_entry


class FakeTensor:
    def data_ptr(self):
        return 0x7f0000000000


assert [arg_kind(v) for v in (FakeTensor(), 7, 0.5)] == ['ptr', 'i32', 'fp32']
for bad in (True, 2**40, 'x'):
    try:
        arg_kind(bad)
        raise AssertionError(f'expected ValueError for {bad!r}')
    except ValueError:
        pass

# Must match Triton 3.6's own entry exactly (here: SoftmaxFloat16Kernel on V100).
TRITON_SOFTMAX_ENTRY = """.visible .entry kernel(
	.param .u64 .ptr .global .align 1 kernel_param_0,
	.param .u64 .ptr .global .align 1 kernel_param_1,
	.param .u32 kernel_param_2,
	.param .u32 kernel_param_3,
	.param .u32 kernel_param_4,
	.param .u32 kernel_param_5,
	.param .u64 .ptr .global .align 1 kernel_param_6,
	.param .u64 .ptr .global .align 1 kernel_param_7
)"""
ptx = entry_template(['ptr', 'ptr', 'i32', 'i32', 'i32', 'i32'], ('8.7', 'sm_70', 64))
assert TRITON_SOFTMAX_ENTRY in ptx, ptx
assert ptx.startswith('.version 8.7\n.target sm_70\n.address_size 64\n'), ptx
name, params = parse_entry(ptx)
assert name == 'kernel' and [t for t, _ in params] == ['u64', 'u64'] + ['u32'] * 4 + ['u64', 'u64']

assert kernels_from_env({}) == set()
assert kernels_from_env({'PTX_LAUNCHER_KERNELS': 'SageAttentionICLR2025, MatrixMultiplicationFloat8'}) == {
    'SageAttentionICLR2025', 'MatrixMultiplicationFloat8'}
print('ok')
