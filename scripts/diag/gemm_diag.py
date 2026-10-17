"""Find where Triton FP16 GEMM goes from ~13 ms (baselines.py) to ~31 ms (TAIC)."""
import sys
sys.path.insert(0, 'scripts')
import torch
from ptx_gym.evaluation.performance import benchmark, evaluate_ptx_performance
from ptx_gym.helpers.triton import clear_triton_cache, dump_kernel_ptx
from ptx_gym.kernels import resolve_kernel

cfg = {'block_m': 32, 'block_n': 64, 'block_k': 32, 'num_warps': 4}
cls = resolve_kernel('MatrixMultiplicationFloat16')
torch.manual_seed(42)

op = cls(ptx={'tuning_config': cfg})            # how TAIC's verifier builds it
x = op.get_random_input()
print('A  fixed-config op, do_bench:', round(benchmark(lambda: op.forward_triton(x)).p50, 3), flush=True)
print('A2 same, fixed=True inputs  :', round(benchmark(lambda: op.forward_triton(op.get_random_input(fixed=True))).p50, 3), flush=True)
clear_triton_cache()
print('B  after clear_triton_cache :', round(benchmark(lambda: op.forward_triton(x)).p50, 3), flush=True)
import taic                                     # TAIC + our launcher patches
print('C  after importing TAIC     :', round(benchmark(lambda: op.forward_triton(x)).p50, 3), flush=True)
op.set_ptx({'ptx': dump_kernel_ptx(op), 'num_threads_x': op.num_warps * 32})
m = evaluate_ptx_performance(op, x)             # PTX Gym's evaluation timing (50 alternating rounds)
print('D  evaluate_ptx_performance : triton', round(m['triton'].p50, 3), 'ptx', round(m['ptx'].p50, 3), flush=True)
print('E  plain do_bench again     :', round(benchmark(lambda: op.forward_triton(x)).p50, 3), flush=True)
