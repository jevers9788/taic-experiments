#!/usr/bin/env bash
set -euo pipefail
ncu_bin=/usr/local/cuda-12.8/bin/ncu
# Keep the profiler's privileged execution local; never run the model client as root.
status=0
# No root-owned .pyc files in the venv/source tree.
# Root gets its own Triton cache, so user runs never hit root-owned cache dirs.
sudo -E PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}-root" "$ncu_bin" "$@" || status=$?
for ((i=1; i<=$#; i++)); do
  if [[ ${!i} == -o || ${!i} == --export ]]; then
    j=$((i+1)); report=${!j}
    for path in "$report" "$report.ncu-rep"; do
      if [[ -f $path ]]; then sudo chown "$(id -u):$(id -g)" "$path"; fi
    done
  fi
done
exit "$status"
