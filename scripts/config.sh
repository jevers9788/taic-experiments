# Experiment settings. Put yours in config.local.sh at the repo root (gitignored;
# template: config.example.sh) or export them. Sourced by the provisioning/ops scripts.
_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f $_root/config.local.sh ]] && source "$_root/config.local.sh"
: "${GCP_PROJECT:?Set GCP_PROJECT in config.local.sh (see config.example.sh)}"
VM_PREFIX=${VM_PREFIX:-taic}      # VMs are named $VM_PREFIX-<gpu>$VM_SUFFIX
VM_SUFFIX=${VM_SUFFIX:-}
REMOTE_DIR=${REMOTE_DIR:-taic-exp}  # experiment dir on the VM, relative to $HOME
vm_name() { echo "$VM_PREFIX-$1$VM_SUFFIX"; }
vm_zone() { gcloud compute instances list --project="$GCP_PROJECT" --filter="name=$1" --format='value(zone.basename())'; }
