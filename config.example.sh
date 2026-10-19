# Copy to config.local.sh (gitignored) and fill in.
GCP_PROJECT=my-gcp-project   # project with GPU quota (GPUS_ALL_REGIONS >= 1)
VM_PREFIX=taic               # VM names: taic-l4, taic-t4, taic-v100, taic-cpu
VM_SUFFIX=                   # optional, e.g. -2026 to keep names unique
REMOTE_DIR=taic-exp          # experiment dir on the VM, under $HOME
