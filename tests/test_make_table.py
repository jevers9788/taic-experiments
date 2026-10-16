"""GPU-free check for the baseline table. Run: python3 tests/test_make_table.py"""
import csv
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from make_table import main

ok = lambda ms: {'correct': True, 'timing_ms': {'p50': ms}}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'l4').mkdir()
    (root / 'l4' / 'MatrixMultiplicationFloat16.json').write_text(json.dumps({'implementations': {
        'triton': {**ok(2.0), 'roofline': {'pct_of_measured_copy_bw': 5.0, 'pct_of_measured_fp16_mm': 80.0}},
        'pytorch_eager': ok(1.9), 'torch_mm_cublas': ok(1.5)}}))
    (root / 't4').mkdir()
    (root / 't4' / 'MatrixMultiplicationFloat8.json').write_text(json.dumps({
        'autotune_error': 'Traceback...\nRuntimeError: fp8 dot unsupported',
        'implementations': {'pytorch_eager': ok(9.0), 'torch_scaled_mm_fp8': {'correct': False}}}))
    main(root)
    rows = {(r['gpu'], r['kernel']): r for r in csv.DictReader(open(root / 'baselines.csv'))}
    assert len(rows) == 27
    gemm = rows['L4', 'MatrixMultiplicationFloat16']
    assert gemm['status'] == 'measured' and gemm['triton_p50_ms'] == '2'
    assert gemm['best_stock'] == 'torch_mm_cublas' and gemm['triton_pct_fp16_mm'] == '80'
    fp8 = rows['T4', 'MatrixMultiplicationFloat8']
    assert fp8['status'] == 'triton_failed' and fp8['triton_compiles'] == 'False'
    assert 'fp8 dot unsupported' in fp8['notes'] and 'torch_scaled_mm_fp8 failed' in fp8['notes']
    assert rows['L4', 'SoftmaxFloat16Kernel']['status'] == 'not_run'
print('ok')
