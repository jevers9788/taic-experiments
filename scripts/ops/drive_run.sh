#!/usr/bin/env bash
# Local: wait for a user-started TAIC run on the VM to start and finish, run
# post_run.sh, copy results back, and ALWAYS stop the VM at the end.
# Usage: [GPU=v100] bash scripts/ops/drive_run.sh KERNEL TAG [LAUNCHER_KERNELS]
set -u
KERNEL=$1; TAG=$2; LAUNCH=${3:-}
cd "$(dirname "$0")/../.."
source scripts/config.sh
GPU=${GPU:-v100}; VM=$(vm_name "$GPU"); P=$GCP_PROJECT; Z=$(vm_zone "$VM")
[[ -n $Z ]] || { echo "VM $VM not found in $P" >&2; exit 1; }
ssh_vm() { gcloud compute ssh $VM --zone=$Z --project=$P --ssh-flag=-oConnectTimeout=20 --ssh-flag=-oServerAliveInterval=15 --command="$1" 2>/dev/null; }
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_vm() { gcloud compute instances stop $VM --zone=$Z --project=$P 2>&1 | tail -1; log "VM status: $(gcloud compute instances describe $VM --zone=$Z --project=$P --format='value(status)')"; }
trap stop_vm EXIT
# Deadlines are wall-clock (a sleeping laptop or failed ssh must not extend them).
start=$(date +%s); last_ok=$start; seen=0; idle=0
while true; do
  now=$(date +%s)
  (( !seen && now - start > 3600 )) && { log "no run started within 1 h"; exit 1; }
  (( now - last_ok > 1800 )) && { log "VM unreachable for 30 min"; exit 1; }
  out=$(ssh_vm "pgrep -f '^.venv/bin/python scripts/taic.py $KERNEL' >/dev/null && echo RUNNING; echo OK")
  [[ $out == *OK* ]] || { sleep 60; continue; }
  last_ok=$now
  if [[ $out == *RUNNING* ]]; then seen=1; idle=0; else idle=$((idle+1)); fi
  (( seen && idle >= 3 )) && break
  sleep 60
done
log "TAIC run ended; post-processing"
tar czf .cache/stage/ops.tgz scripts && gcloud compute scp .cache/stage/ops.tgz $VM:/tmp/ --zone=$Z --project=$P 2>/dev/null
ssh_vm "cd ~/$REMOTE_DIR && tar xzf /tmp/ops.tgz 2>/dev/null; rm -f /tmp/out-$TAG.done; GPU=$GPU KERNEL=$KERNEL TAG=$TAG PTX_LAUNCHER_KERNELS=$LAUNCH setsid -f bash scripts/ops/post_run.sh < /dev/null > /tmp/post-$TAG.out 2>&1; echo launched"
for i in $(seq 1 75); do [[ -n $(ssh_vm "ls /tmp/out-$TAG.done 2>/dev/null") ]] && break; sleep 60; done  # racecheck <= 60 min
log "copying results"
rm -rf .cache/stage/out-$TAG && gcloud compute scp $VM:/tmp/out-$TAG.tgz .cache/stage/ --zone=$Z --project=$P 2>/dev/null \
  && tar xzf .cache/stage/out-$TAG.tgz -C .cache/stage && mkdir -p runs && cp -R .cache/stage/out-$TAG/runs/. runs/ \
  && cp .cache/stage/out-$TAG/logs/* logs/ && { mkdir -p results/$GPU; cp .cache/stage/out-$TAG/results/* results/$GPU/ 2>/dev/null; true; } \
  && log "copied" || log "COPY FAILED (results remain on the VM disk)"
