"""Offline PTX Gym smoke test. This module never imports a model client."""
import argparse
import json
from pathlib import Path
import re
import sys

import torch
from ptx_gym.evaluation.compilation import compile_ptx
from ptx_gym.evaluation.evaluate import TritonPTXCandidateEvaluator
from ptx_gym.evaluation.types import Payload
from ptx_gym.helpers.triton import dump_kernel_ptx
from ptx_gym.kernels import resolve_kernel


def handwritten_relu(template, block_size, threads):
    """Use only the compiler's ABI declaration, with an independently written body."""
    match = re.search(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)', template, re.S)
    if match is None:
        raise ValueError('No PTX entry declaration found')
    # Triton 3.6 writes e.g. `.param .u64 .ptr .global .align 1 kernel_param_0`.
    params = re.findall(r'\.param\s+\.u64\b[^,]*?(\w+)\s*(?:,|$)', match[2])
    if len(params) < 2:
        raise ValueError('Expected input and output pointer arguments')
    version = re.search(r'\.version\s+([\d.]+)', template)[1]
    target = re.search(r'\.target\s+(\w+)', template)[1]
    # Each CTA covers the same fixed tile as the autotuned reference.
    return f'''.version {version}
.target {target}
.address_size 64
.visible .entry {match[1]}({match[2]})
.reqntid {threads}, 1, 1
{{
.reg .pred p;
.reg .b16 h;
.reg .b32 t, b, i, off;
.reg .b64 x, y, addr, delta;
.reg .f32 value;
ld.param.u64 x, [{params[0]}];
ld.param.u64 y, [{params[1]}];
mov.u32 t, %tid.x;
mov.u32 b, %ctaid.x;
mul.lo.u32 off, b, {block_size};
LOOP:
setp.ge.u32 p, t, {block_size};
@p bra DONE;
add.u32 i, off, t;
mul.wide.u32 delta, i, 2;
add.u64 addr, x, delta;
ld.global.b16 h, [addr];
cvt.f32.f16 value, h;
max.f32 value, value, 0f00000000;
cvt.rn.f16.f32 h, value;
add.u64 addr, y, delta;
st.global.b16 [addr], h;
add.u32 t, t, {threads};
bra LOOP;
DONE:
ret;
}}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/handwritten-relu.json')
    parser.add_argument('--identity', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    cls = resolve_kernel('ReLUFloat16Kernel')
    op = cls()
    template = dump_kernel_ptx(op)
    source = template if args.identity else handwritten_relu(template, op.block_size, op.num_warps * 32)
    payload = Payload(ptx=source, threads_x=op.num_warps * 32)
    compiled = compile_ptx(payload)
    if not compiled.compiles:
        raise RuntimeError(compiled.error)
    op.set_ptx(payload.to_launch_dict())
    for kind in ['random', 'zeros', 'small', 'large']:
        x = torch.randn((op.batch_size, op.size), device='cuda', dtype=torch.float16)
        if kind == 'zeros':
            x.zero_()
        elif kind == 'small':
            x *= 1e-4
        elif kind == 'large':
            x *= 1000
        torch.testing.assert_close(op.forward_triton(x, ptx=True)[0], torch.relu(x), rtol=0, atol=0)
    evaluator = TritonPTXCandidateEvaluator(cls, operator=op)
    result = evaluator.evaluate(payload).to_dict()
    result['command'] = sys.argv
    result['fixture'] = 'triton_identity' if args.identity else 'handwritten_relu'
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    # PTX contains a compiler ABI header; keep it in ignored local artifacts.
    artifact = Path('.cache/fixtures')
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / 'relu.ptx').write_text(source)
    if not result['passed']:
        raise RuntimeError(result['message'])


if __name__ == '__main__':
    main()
