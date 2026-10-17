"""Launcher mode: run TAIC / PTX Gym on kernels Triton can't compile for this GPU.

For the kernel classes named in $PTX_LAUNCHER_KERNELS (comma-separated):
- forward_triton(ptx=True) runs the candidate through scripts/ptx_launcher.py;
- forward_triton() returns forward_torch: Triton has no output here, so the
  reference and the timing baseline are the stock PyTorch path. For these kernels
  PTX Gym's/TAIC's "triton_p50" and "speedup_vs_triton" mean stock PyTorch.
- dump_kernel_ptx() returns an entry template in Triton's exact ABI, built from the
  recorded launch; get_kernel_shared_memory_bytes() returns None (no dynamic budget).
Applied on import, and in PTX Gym's sanitizer/ncu subprocesses via
scripts/site/sitecustomize.py (they inherit the env var). See PORTING.md 14.
"""
import os
import sys

ENV = 'PTX_LAUNCHER_KERNELS'
PARAM_DECL = {'ptr': '.u64 .ptr .global .align 1', 'i32': '.u32', 'fp32': '.f32'}
STATIC_SHARED_LIMIT = 48 * 1024  # dynamic shared memory given to candidates that declare it
ACTIVE = set()


def kernels_from_env(environ=os.environ):
    return {name.strip() for name in environ.get(ENV, '').split(',') if name.strip()}


def arg_kind(value):
    """Triton's PTX param kind for a runtime argument."""
    if hasattr(value, 'data_ptr'):
        return 'ptr'
    if isinstance(value, bool):
        raise ValueError('bool runtime arguments are not supported')
    if isinstance(value, int):
        if -2**31 <= value < 2**31:
            return 'i32'
        raise ValueError(f'{value} needs a 64-bit param, which is not supported')
    if isinstance(value, float):
        return 'fp32'
    raise ValueError(f'Unsupported runtime argument {value!r}')


def entry_template(kinds, system):
    """PTX module with Triton's entry ABI (args + two scratch pointers) and an empty body."""
    version, target, address_size = system
    params = ',\n'.join(f'\t.param {PARAM_DECL[k]} kernel_param_{i}'
                        for i, k in enumerate(list(kinds) + ['ptr', 'ptr']))
    return (f'.version {version}\n.target {target}\n.address_size {address_size}\n\n'
            f'.visible .entry kernel(\n{params}\n)\n{{\n\tret;\n}}\n')


def template_for(op):
    from ptx_gym.helpers.environment import get_ptx_system_config
    from ptx_launcher import record_launch
    launch, _ = record_launch(op, op.get_random_input())
    return entry_template([arg_kind(a) for a in launch.args], get_ptx_system_config())


def _patch_operator(cls):
    from triton.runtime.jit import JITFunction
    from ptx_launcher import launch_ptx, record_launch
    original = cls.forward_triton

    def forward_triton(self, inputs, ptx=False):
        if ptx:
            launch, out = record_launch(self, inputs)
            code = self.ptx['ptx']
            launch_ptx(code, launch, threads_x=self.ptx.get('num_threads_x'),
                       shared_bytes=STATIC_SHARED_LIMIT if '.extern .shared' in code else 0)
            return out, None
        if isinstance(self.compiled_kernel, JITFunction):
            return self.forward_torch(inputs), None
        return original(self, inputs)  # a launch recorder is installed: records, never compiles

    cls.forward_triton = forward_triton


def apply():
    """Patch the kernels named in the environment; idempotent."""
    names = kernels_from_env() - ACTIVE
    if not names:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ptx_gym.helpers import triton as helpers
    from ptx_gym.kernels import resolve_kernel
    for name in names:
        _patch_operator(resolve_kernel(name))
    ACTIVE.update(names)
    if getattr(helpers.dump_kernel_ptx, 'launcher_mode', False):
        return
    dump, shared = helpers.dump_kernel_ptx, helpers.get_kernel_shared_memory_bytes

    def dump_kernel_ptx(kernel, inputs=None):
        return template_for(kernel) if type(kernel).__name__ in ACTIVE else dump(kernel, inputs)

    def get_kernel_shared_memory_bytes(kernel, inputs=None):
        return None if type(kernel).__name__ in ACTIVE else shared(kernel, inputs)

    dump_kernel_ptx.launcher_mode = True
    for module in list(sys.modules.values()):  # rebind names already imported elsewhere
        for attr, old, new in (('dump_kernel_ptx', dump, dump_kernel_ptx),
                               ('get_kernel_shared_memory_bytes', shared, get_kernel_shared_memory_bytes)):
            if getattr(module, attr, None) is old:
                setattr(module, attr, new)


if __name__ != '__main__' and kernels_from_env():
    apply()
