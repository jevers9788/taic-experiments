#!/usr/bin/env bash
set -euo pipefail
# Usage: provision.sh l4|t4|v100|cpu ZONE... -- tries zones in order, creates at most one VM.
gpu=$1; shift
case $gpu in
  l4) machine=g2-standard-4; accel=nvidia-l4 ;;
  t4) machine=n1-standard-4; accel=nvidia-tesla-t4 ;;
  v100) machine=n1-standard-8; accel=nvidia-tesla-v100 ;;  # V100 requires N1
  # Wheel build only; 32 vCPUs is the project's CPUS_ALL_REGIONS quota.
  cpu) machine=${MACHINE:-n2-standard-32}; accel= ;;
  *) echo "unknown gpu $gpu" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
source scripts/config.sh
vm=$(vm_name "$gpu")
project=$GCP_PROJECT
mkdir -p logs
if [[ -n $(gcloud compute instances list --project="$project" --filter="name=$vm" --format='value(name)') ]]; then
  echo "$vm already exists. Inspect and reuse it." >&2
  exit 1
fi
for zone in "$@"; do
  log="logs/$gpu-create-$zone-$(date -u +%Y%m%dT%H%M%SZ).log"
  # GPUS_ALL_REGIONS quota is 1, so a success here is the only GPU VM.
  if gcloud compute instances create "$vm" --project="$project" --zone="$zone" \
      --machine-type="$machine" ${accel:+--accelerator="type=$accel,count=1"} \
      --maintenance-policy=TERMINATE --no-restart-on-failure \
      --max-run-duration=7200s --instance-termination-action=STOP \
      --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
      --boot-disk-size=100GB --boot-disk-type=pd-standard \
      --no-service-account --no-scopes --labels=purpose=taic-baselines --quiet >"$log" 2>&1; then
    echo "created $vm in $zone"; exit 0
  fi
  echo "failed $zone: $(grep -m1 -oE '^code: [A-Z_]+' "$log" || tail -1 "$log")" >&2
done
exit 1
