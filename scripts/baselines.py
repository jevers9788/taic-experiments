"""Measure unchanged upstream kernels; failed correctness never earns a timing."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
from ptx_gym.evaluation.performance import benchmark
from ptx_gym.helpers.environment import get_ptx_system_config
from ptx_gym.kernels import resolve_kernel

KERNELS = [
    'SoftmaxFloat16Kernel', 'RMSNormFloat16Kernel',
    'MatrixMultiplicationFloat16', 'FlashAttentionNeurIPS2022Forward',
    'SageAttentionICLR2025', 'BitDeltaNeurIPS2024Matmul',
    # FP8 (e4m3) inputs: native on sm_89, no FP8 hardware on sm_75 (dtype emulation test).
    'MatrixMultiplicationFloat8',
    # GEMV is LLM decoding's bottleneck and purely bandwidth-bound: FP8 storage moves
    # half the bytes of FP16, so it can beat the FP16 ceiling on the same card.
    'MatrixVectorMultiplicationFloat16Kernel', 'MatrixVectorMultiplicationFloat8Kernel',
]


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    return []


def traffic_bytes(x, out):
    """Compulsory DRAM traffic: every input and output byte moved once."""
    # ponytail: lower bound; ignores re-reads (e.g. GEMM tiles), so %-of-peak is optimistic.
    return sum({t.data_ptr(): t.nbytes for t in tensors(x) + tensors(out)}.values())


def flops(name, x):
    """Useful FLOPs for matmul-like kernels; None for memory-bound ones."""
    if name.startswith(('MatrixMultiplication', 'BitDelta')):
        m, k = x[0].shape
        return 2 * m * k * x[2].shape[1]
    if name.startswith(('FlashAttention', 'Sage')):
        b, h, t, d = x[0].shape  # noncausal: QK^T and PV are each 2*T*S*D per head
        return 4 * b * h * t * x[1].shape[2] * d
    return None


def measure_peaks():
    """Achievable ceilings measured on this card, so no spec-sheet numbers are needed."""
    src = torch.empty(2**29, dtype=torch.uint8, device='cuda')
    dst = torch.empty_like(src)
    copy_ms = benchmark(lambda: dst.copy_(src)).p50
    a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    b = torch.randn_like(a)
    mm_ms = benchmark(lambda: a @ b).p50
    return {'copy_GBps': 2 * src.nbytes / copy_ms / 1e6,
            'cublas_fp16_mm_TFLOPs': 2 * 8192**3 / mm_ms / 1e9,
            'method': '512 MiB device copy (read+write) and 8192^3 fp16 torch.mm, p50'}


def roofline(timing_ms, nbytes, nflops, peaks):
    gbps = nbytes / timing_ms / 1e6
    result = {'traffic_bytes': nbytes, 'achieved_GBps': gbps,
              'pct_of_measured_copy_bw': 100 * gbps / peaks['copy_GBps']}
    if nflops:
        tflops = nflops / timing_ms / 1e9
        result.update(flops=nflops, achieved_TFLOPs=tflops,
                      pct_of_measured_fp16_mm=100 * tflops / peaks['cublas_fp16_mm_TFLOPs'])
    return result


def column_major(t):
    return t.t().contiguous().t()


def reference(name, op, x):
    if name.startswith('Softmax'):
        return torch.softmax(x.float(), -1).to(x.dtype)
    if name.startswith('RMSNorm'):
        a, w = x
        a = a.float()
        return (a * torch.rsqrt(a.square().mean(-1, keepdim=True) + op.eps)
                * w.float()).half()
    if name.startswith(('MatrixMultiplicationFloat', 'MatrixVector')):  # FP16/FP8 in, FP16 out
        return (x[0].float() @ x[1].float()).half()
    if name.startswith('FlashAttention'):
        q, k, v = x
        # The pinned launcher specifies IS_CAUSAL=False.
        scores = q.float() @ k.float().transpose(-1, -2) * op.scale
        return (torch.softmax(scores, -1).half() @ v,
                torch.logsumexp(scores, -1))
    if name.startswith('BitDelta'):
        a, packed, _ = x
        bits = torch.arange(32, device=packed.device)
        weights = ((packed[..., None] >> bits) & 1).permute(0, 2, 1)
        weights = weights.reshape(a.shape[1], -1).float() * 2 - 1
        return (a.float() @ weights).half()
    # SageAttention's reference includes its quantization scales and exp2 semantics.
    return op.forward_torch(x)


def compare(actual, expected, tolerance):
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError('Output structure mismatch')
        for a, e in zip(actual, expected):
            compare(a, e, tolerance)
    else:
        torch.testing.assert_close(actual, expected, rtol=tolerance,
                                   atol=tolerance, equal_nan=True)


def edge_input(name, x, case):
    def transform(t):
        if not t.is_floating_point():
            return torch.zeros_like(t) if case == 'zeros' else t
        if case == 'zeros':
            return torch.zeros(t.shape, device=t.device).to(t.dtype)
        if case == 'small':
            # 1e-4 underflows to 0 in float8_e4m3fn; use its smallest normal there.
            value = max(1e-4, torch.finfo(t.dtype).tiny)
            return torch.full(t.shape, value, device=t.device).to(t.dtype)
        return t
    if isinstance(x, torch.Tensor):
        return transform(x)
    # Output buffers are rewritten by the operator.
    return tuple(transform(t) for t in x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel', choices=KERNELS)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Default True lets cuBLAS reduce FP16 GEMMs in FP16; the reference and Triton use FP32.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    record = {'command': sys.argv, 'kernel': args.kernel,
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'ptx_system': get_ptx_system_config(),
              'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
              'implementations': {}, 'seed': 42,
              'torch_flags': {'allow_tf32': False, 'allow_fp16_reduced_precision_reduction': False}}
    cls = resolve_kernel(args.kernel)
    # Construct a fixed-contract object even if upstream autotuning cannot compile.
    op = cls(ptx={'tuning_config': {}})
    record['shape'] = op.get_shape_information()
    tolerance = getattr(op, 'verification_tolerance', 1e-3)
    record['rtol'] = record['atol'] = tolerance
    if not args.check_only:
        record['peaks'] = measure_peaks()
    try:
        tuning_result = op.autotune()
        record['autotune'] = dataclasses.asdict(tuning_result)
    except Exception:
        record['autotune_error'] = traceback.format_exc()
    if 'autotune_error' not in record:
        # Keep the contract (PTX entry ABI + compile metadata) where Triton works, so a
        # Triton-free launcher can reuse it on targets where Triton fails (PORTING.md 2).
        try:
            _, compiled = op.forward_triton(op.get_random_input())
            ptx_dir = Path(args.output).parent / 'ptx'
            ptx_dir.mkdir(parents=True, exist_ok=True)
            (ptx_dir / f'{args.kernel}.ptx').write_text(compiled.asm['ptx'])
            meta = compiled.metadata
            record['triton_metadata'] = {k: str(v) for k, v in (
                meta._asdict() if hasattr(meta, '_asdict') else vars(meta)).items()}
        except Exception:
            record['contract_capture_error'] = traceback.format_exc()
    functions = {'pytorch_eager': lambda x: op.forward_torch(x)}
    if 'autotune_error' not in record:
        functions['triton'] = lambda x: op.forward_triton(x)[0]
    if args.kernel.startswith('FlashAttention'):
        # SDPA returns output only; LSE is not part of this comparison.
        functions['torch_sdpa_output_only'] = lambda x: torch.nn.functional.scaled_dot_product_attention(*x, is_causal=False, scale=op.scale)
    if args.kernel == 'MatrixMultiplicationFloat16':
        functions['torch_mm_cublas'] = lambda x: torch.mm(x[0], x[1], out=x[2])
    # One-time input preparation per implementation; not timed.
    prepare = {}
    if args.kernel == 'MatrixMultiplicationFloat8':
        # The usual fallback where FP8 matmul is unsupported: upcast, then cuBLAS FP16.
        functions['torch_upcast_fp16_mm'] = lambda x: torch.mm(x[0].half(), x[1].half())
        # Native FP8 cuBLAS (sm_89+). Needs column-major B, prepared once like stored weights.
        one = torch.ones((), device='cuda')
        functions['torch_scaled_mm_fp8'] = lambda x: torch._scaled_mm(
            x[0], x[1], scale_a=one, scale_b=one, out_dtype=torch.float16)
        prepare['torch_scaled_mm_fp8'] = lambda x: (x[0], column_major(x[1]), x[2])
    if args.kernel == 'MatrixVectorMultiplicationFloat8Kernel':
        functions['torch_upcast_fp16_mv'] = lambda x: torch.mv(x[0].half(), x[1].half())
    for label, fn in functions.items():
        result = {}
        try:
            for case in ['random'] * 5 + ['zeros', 'small']:
                x = edge_input(args.kernel, op.get_random_input(), case)
                expected = reference(args.kernel, op, x)
                x = prepare.get(label, lambda v: v)(x)
                if label == 'torch_sdpa_output_only':
                    expected = expected[0]
                compare(fn(x), expected, tolerance)
            result['correct'] = True
            result['cases'] = ['random:5', 'zeros', 'small']
            if not args.check_only:
                x = prepare.get(label, lambda v: v)(op.get_random_input())
                timing = benchmark(lambda: fn(x))
                result['timing_ms'] = dataclasses.asdict(timing)
                result['roofline'] = roofline(timing.p50, traffic_bytes(x, fn(x)),
                                              flops(args.kernel, x), record['peaks'])
        except Exception:
            result['correct'] = False
            result['error'] = traceback.format_exc()
        record['implementations'][label] = result
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
