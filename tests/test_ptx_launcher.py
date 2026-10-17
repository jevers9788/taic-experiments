"""GPU-free checks for the Triton-free PTX launcher. Run: python3 tests/test_ptx_launcher.py"""
import ctypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from ptx_launcher import pack_args, parse_entry, resolve_grid, split_args

# Entry declaration as Triton 3.6 writes it: pointers carry qualifiers, scalars don't,
# and two scratch pointers are appended.
PTX = """
.version 8.7
.target sm_70
.address_size 64
.visible .entry kernel(
	.param .u64 .ptr .global .align 1 kernel_param_0,
	.param .u64 .ptr .global .align 1 kernel_param_1,
	.param .u32 kernel_param_2,
	.param .u64 .ptr .global .align 1 kernel_param_3,
	.param .u64 .ptr .global .align 1 kernel_param_4
)
.reqntid 128, 1, 1
{
	ret;
}
"""

name, params = parse_entry(PTX)
assert name == 'kernel'
assert params == [('u64', 'kernel_param_0'), ('u64', 'kernel_param_1'), ('u32', 'kernel_param_2'),
                  ('u64', 'kernel_param_3'), ('u64', 'kernel_param_4')], params
for bad in ['.version 8.7\n.target sm_70\n', '.visible .func helper(.param .u32 x) { ret; }']:
    try:
        parse_entry(bad)
        raise AssertionError('expected ValueError for PTX without an entry')
    except ValueError:
        pass

# Runtime args are packed by the PTX param types; missing trailing pointers (Triton's
# scratch buffers) are passed as null.
array, keep = pack_args([t for t, _ in params], [0x7f0000001000, 0x7f0000002000, 7])
assert len(array) == 5
assert ctypes.cast(array[0], ctypes.POINTER(ctypes.c_uint64))[0] == 0x7f0000001000
assert ctypes.cast(array[2], ctypes.POINTER(ctypes.c_uint32))[0] == 7
assert ctypes.cast(array[4], ctypes.POINTER(ctypes.c_uint64))[0] == 0
for values in ([1], [1, 2, 3, 4, 5, 6]):  # too few non-pointer params / too many values
    try:
        pack_args([t for t, _ in params], values)
        raise AssertionError(f'expected ValueError for {values}')
    except ValueError:
        pass
try:
    pack_args(['f16'], [1])
    raise AssertionError('expected ValueError for unsupported param type')
except ValueError:
    pass

# Triton calls kernel[grid](*args, **kwargs); constexprs go in the meta dict.
names = ['x_ptr', 'out_ptr', 'n', 'BLOCK_SIZE']
runtime, meta = split_args(names, {'BLOCK_SIZE'}, (10, 20, 30), {'BLOCK_SIZE': 64, 'num_warps': 4})
assert runtime == [10, 20, 30] and meta == {'BLOCK_SIZE': 64}, (runtime, meta)
runtime, meta = split_args(names, {'n', 'BLOCK_SIZE'}, (10, 20, 30, 64), {})
assert runtime == [10, 20] and meta == {'n': 30, 'BLOCK_SIZE': 64}, (runtime, meta)
# Triton specializes None arguments away, so they are not PTX params either.
runtime, meta = split_args(names, {'BLOCK_SIZE'}, (10, None, 30), {'BLOCK_SIZE': 64})
assert runtime == [10, 30] and meta == {'BLOCK_SIZE': 64, 'out_ptr': None}, (runtime, meta)

assert resolve_grid((8, 16, 16), {}) == (8, 16, 16)
assert resolve_grid((5,), {}) == (5, 1, 1)
assert resolve_grid(lambda m: (4096 // m['BLOCK_M'], 4096 // m['BLOCK_N']),
                    {'BLOCK_M': 32, 'BLOCK_N': 64}) == (128, 64, 1)
for bad in [(0,), (1, 2, 3, 4)]:
    try:
        resolve_grid(bad, {})
        raise AssertionError(f'expected ValueError for grid {bad}')
    except ValueError:
        pass
print('ok')
