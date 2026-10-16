#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs results
run_id=$(date -u +%Y%m%dT%H%M%SZ)
# All workloads run sequentially; a failed case keeps its error log.
for kernel in SoftmaxFloat16Kernel RMSNormFloat16Kernel MatrixMultiplicationFloat16 FlashAttentionNeurIPS2022Forward SageAttentionICLR2025 BitDeltaNeurIPS2024Matmul MatrixMultiplicationFloat8; do
  log="logs/$run_id-$kernel.log"
  printf 'Command: .venv/bin/python scripts/baselines.py %s --output results/%s-%s.json\n' "$kernel" "$run_id" "$kernel" > "$log"
  .venv/bin/python scripts/baselines.py "$kernel" --output "results/$run_id-$kernel.json" >> "$log" 2>&1 || {
    echo "Failed $kernel; see $log" >&2
  }
done
