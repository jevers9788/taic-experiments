#!/usr/bin/env bash
# One GPU session after setup-vm.sh: versions, loop fixture check, baselines.
# Each step logs its exact command; a failed step does not stop the session.
cd "$(dirname "$0")/.."
source scripts/env.sh
gpu=${1:?usage: run-session.sh l4|t4}
out=results/$gpu; mkdir -p "$out" logs
step() {
  local name=$1; shift
  { echo "Command: $*"; echo "Start: $(date -u +%FT%TZ)"; } > "logs/$gpu-$name.log"
  "$@" >> "logs/$gpu-$name.log" 2>&1; echo "Exit: $? $(date -u +%FT%TZ)" >> "logs/$gpu-$name.log"
}
step versions bash -c 'nvidia-smi; nvcc --version; ptxas --version; ncu --version;
  find .venv -name ptxas -type f -exec {} --version \;; .venv/bin/python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)";
  .venv/bin/uv pip freeze --python .venv/bin/python'
step identity .venv/bin/python scripts/check_candidate.py --identity --output "$out/identity-relu.json"
step handwritten .venv/bin/python scripts/check_candidate.py --output "$out/handwritten-relu.json"
# TAIC end to end with a stub model (no API calls); see scripts/taic_fixture.py.
step taic-fixture .venv/bin/python scripts/taic_fixture.py --trace-path "$out/taic-fixture"
for k in SoftmaxFloat16Kernel MatrixMultiplicationFloat16 RMSNormFloat16Kernel FlashAttentionNeurIPS2022Forward SageAttentionICLR2025 BitDeltaNeurIPS2024Matmul MatrixMultiplicationFloat8 MatrixVectorMultiplicationFloat16Kernel MatrixVectorMultiplicationFloat8Kernel; do
  step "baseline-$k" .venv/bin/python scripts/baselines.py "$k" --output "$out/$k.json"
done
echo SESSION_DONE > "logs/$gpu-session.done"
