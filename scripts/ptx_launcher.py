"""Launch a PTX kernel without compiling its Triton source (PORTING.md 2 and 13).

PTX Gym swaps candidate PTX in at Triton's cubin stage, so every earlier Triton stage
must still succeed; where Triton can't lower a kernel for the target (e.g.
SageAttention's int8 dot on sm_70), no candidate can run. Here Triton's source stays
the spec: we record the operator's own `kernel[grid](...)` call without compiling it,
assemble the candidate with PTX Gym's bundled ptxas (the paper's toolchain), and launch
it through the CUDA driver API (ctypes, no new dependencies) on PyTorch's stream.
"""
import ctypes
from dataclasses import dataclass
import functools
import re
import subprocess
import tempfile

PARAM_CTYPES = {'u64': ctypes.c_uint64, 'b64': ctypes.c_uint64, 's64': ctypes.c_int64,
                'u32': ctypes.c_uint32, 'b32': ctypes.c_uint32, 's32': ctypes.c_int32,
                'f32': ctypes.c_float, 'f64': ctypes.c_double}
ENTRY = re.compile(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)', re.S)
PARAM = re.compile(r'\.param\s+\.(\w+)\b[^,]*?(\w+)\s*(?:,|$)')
TRITON_SCRATCH_POINTERS = 2  # Triton 3.6 appends global and profile scratch pointers


@dataclass
class Launch:
    grid: tuple
    block: tuple
    args: list  # runtime arguments in PTX parameter order (tensors or scalars)


def parse_entry(ptx):
    """Return (entry name, [(type, name), ...]) of the PTX kernel entry."""
    match = ENTRY.search(ptx)
    if match is None:
        raise ValueError('No .visible .entry declaration in PTX')
    return match[1], PARAM.findall(match[2])


def pack_args(types, values):
    """Pack values for cuLaunchKernel; trailing scratch pointers not given are null."""
    if len(values) > len(types):
        raise ValueError(f'{len(values)} arguments for {len(types)} PTX params')
    missing = types[len(values):]
    if len(missing) > TRITON_SCRATCH_POINTERS or any(t not in ('u64', 'b64') for t in missing):
        raise ValueError(f'{len(values)} arguments for PTX params {types}')
    keep = []
    for kind, value in zip(types, list(values) + [0] * len(missing)):
        if kind not in PARAM_CTYPES:
            raise ValueError(f'Unsupported PTX param type .{kind}')
        keep.append(PARAM_CTYPES[kind](value))
    array = (ctypes.c_void_p * len(keep))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in keep])
    return array, keep


def split_args(names, constexpr_names, args, kwargs):
    """Bind kernel[grid](*args, **kwargs) to the kernel's params.

    Returns (runtime values in order, meta), where meta holds the constexprs and the
    None arguments, which Triton specializes away rather than passing to PTX.
    """
    bound = dict(zip(names, args))
    bound.update({k: v for k, v in kwargs.items() if k in names})
    runtime = [bound[n] for n in names if n in bound and n not in constexpr_names and bound[n] is not None]
    meta = {n: bound[n] for n in names if n in bound and (n in constexpr_names or bound[n] is None)}
    return runtime, meta


def resolve_grid(grid, meta):
    grid = tuple(grid(meta) if callable(grid) else grid)
    if not 1 <= len(grid) <= 3 or any(int(g) < 1 for g in grid):
        raise ValueError(f'Invalid grid {grid}')
    return tuple(int(g) for g in grid) + (1,) * (3 - len(grid))


class _Recorder:
    """Stands in for a JITFunction: records kernel[grid](...) without compiling."""

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def record_launch(op, inputs):
    """Run op.forward_triton with its Triton kernel replaced by a recorder.

    Returns (Launch, output tensor the kernel writes). Nothing is compiled.
    """
    fn, recorder = op.compiled_kernel, _Recorder()
    op.compiled_kernel = recorder
    try:
        result = op.forward_triton(inputs)
    finally:
        op.compiled_kernel = fn
    names = fn.arg_names
    runtime, meta = split_args(names, {names[i] for i in fn.constexprs}, recorder.args, recorder.kwargs)
    bound = dict(zip(names, recorder.args)) | recorder.kwargs
    warps = recorder.kwargs.get('num_warps', getattr(op, 'num_warps', 4))
    launch = Launch(resolve_grid(recorder.grid, bound), (warps * 32, 1, 1), runtime)
    return launch, result[0] if isinstance(result, tuple) else result


@functools.cache
def _cuda():
    lib = ctypes.CDLL('libcuda.so.1')
    def check(code):
        if code:
            msg = ctypes.c_char_p()
            lib.cuGetErrorString(code, ctypes.byref(msg))
            raise RuntimeError(f'CUDA driver error {code}: {msg.value and msg.value.decode()}')
    return lib, check


@functools.cache
def _load(ptx, arch):
    """Assemble with PTX Gym's bundled ptxas and load; cached so timing loops don't recompile."""
    from ptx_gym.helpers.environment import get_ptxas_path
    name, _ = parse_entry(ptx)
    with tempfile.TemporaryDirectory(prefix='ptx-launcher-') as tmp:
        src, cubin = f'{tmp}/kernel.ptx', f'{tmp}/kernel.cubin'
        with open(src, 'w') as f:
            f.write(ptx)
        done = subprocess.run([str(get_ptxas_path(arch)), f'-arch={arch}', '-o', cubin, src],
                              capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f'ptxas failed:\n{done.stderr}')
        image = open(cubin, 'rb').read()
    lib, check = _cuda()
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    check(lib.cuModuleLoadData(ctypes.byref(module), image))
    check(lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode()))
    return module, function  # module kept alive by the cache


def launch_ptx(ptx, launch, shared_bytes=0, threads_x=None):
    """Assemble and launch `ptx` with a recorded Launch on PyTorch's current stream."""
    import torch
    _, params = parse_entry(ptx)
    values = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in launch.args]
    array, keep = pack_args([kind for kind, _ in params], values)  # validates before any GPU work
    major, minor = torch.cuda.get_device_capability()
    torch.cuda.current_stream()  # ensure PyTorch's CUDA context is current
    _, function = _load(ptx, f'sm_{major}{minor}')
    lib, check = _cuda()
    if shared_bytes > 48 * 1024:
        check(lib.cuFuncSetAttribute(function, 8, shared_bytes))  # MAX_DYNAMIC_SHARED_SIZE_BYTES
    block = (threads_x, 1, 1) if threads_x else launch.block
    check(lib.cuLaunchKernel(function, *launch.grid, *block, shared_bytes,
                             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), array, None))
