"""Does a timed-out racecheck leave a GPU process running, and does it slow GEMM timing?"""
import subprocess, time
import torch
from ptx_gym.evaluation.performance import benchmark
from ptx_gym.evaluation.sanitizer import diagnose_ptx
from ptx_gym.evaluation.types import Payload
from ptx_gym.helpers.triton import dump_kernel_ptx
from ptx_gym.kernels import resolve_kernel

cfg = {'block_m': 32, 'block_n': 64, 'block_k': 32, 'num_warps': 4}
op = resolve_kernel('MatrixMultiplicationFloat16')(ptx={'tuning_config': cfg})
x = op.get_random_input()
print('before: GEMM p50', round(benchmark(lambda: op.forward_triton(x)).p50, 3), flush=True)
payload = Payload(ptx=dump_kernel_ptx(op), threads_x=op.num_warps * 32)
t = time.time()
r = diagnose_ptx('MatrixMultiplicationFloat16', payload.to_launch_dict(), sanitizer_tool='racecheck',
                 tuning_config=op.best_config, timeout_seconds=60)
print('racecheck returned after', round(time.time() - t), 's; error:', r.get('error'), flush=True)
time.sleep(5)
ps = subprocess.run(['bash', '-c', 'ps -eo pid,etime,cmd | grep -E "[s]anitizer|[T]reeLauncher|ptx_gym.evaluation" | cut -c1-140'],
                    capture_output=True, text=True).stdout
print('surviving processes:\n' + (ps or '(none)'), flush=True)
print('nvidia-smi apps:', subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                                         capture_output=True, text=True).stdout, flush=True)
print('after: GEMM p50', round(benchmark(lambda: op.forward_triton(x)).p50, 3), flush=True)
