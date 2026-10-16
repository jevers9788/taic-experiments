#!/usr/bin/env bash
set -euo pipefail
# Run on a dedicated Ubuntu 22.04 experiment VM as its normal sudo user.
cd "$(dirname "$0")/.."
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 22.04 ]]
[[ $(uname -m) == x86_64 ]]
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential gcc-12 g++-12 git curl python3-dev python3-venv "linux-headers-$(uname -r)"
if ! nvidia-smi >/dev/null 2>&1; then
  curl -fSL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o /tmp/taic-cuda-installer.pyz
  sudo env CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python3 /tmp/taic-cuda-installer.pyz install_driver --installation-mode=binary --force-version=580.159.04
  nvidia-smi
fi
curl -fSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/taic-cuda-keyring.deb
sudo dpkg -i /tmp/taic-cuda-keyring.deb
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8
# ncu runs Python as root (ncu-root.sh); reclaim anything it left behind.
[[ -e .venv || -e .upstream || -e .cache ]] && sudo chown -R "$(id -u):$(id -g)" .venv .upstream .cache 2>/dev/null || true
python3 -m venv --clear .venv  # no leftovers from earlier installs
.venv/bin/pip install 'uv==0.9.5'
mkdir -p .upstream results
if [[ ! -d .upstream/taic ]]; then
  git clone https://github.com/francois141/triton_ai_compiler.git .upstream/taic
fi
git -C .upstream/taic checkout 6a00d1261c803fecdfa74c1c349b339a026b06ad
git -C .upstream/taic submodule update --init --depth 1
# Match the paper's environment exactly: install TAIC's uv.lock as-is (torch 2.11.0 on the
# CUDA 13 runtime). Its never-true marker keeps stock Triton out; PTX Gym ships patched Triton.
.venv/bin/uv export --frozen --no-hashes --no-emit-package ptx_gym --project .upstream/taic -o results/taic-locked.txt
.venv/bin/uv pip install --python .venv/bin/python --no-deps -r results/taic-locked.txt
# TORCH_CUDA12=1: documented deviation from the paper's CUDA 13 lock, for GPUs CUDA 13 dropped (sm_60/sm_70).
if [[ ${TORCH_CUDA12:-} == 1 ]]; then
  .venv/bin/uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 'torch==2.11.0+cu126'
  .venv/bin/uv pip uninstall --python .venv/bin/python triton  # PTX Gym's wheel provides patched Triton below
  .venv/bin/python -c "import torch; a = torch.cuda.get_arch_list(); print(torch.__version__, torch.version.cuda, a); assert 'sm_70' in a, 'sm_70 missing'"
fi
# Prefer a prebuilt wheel from build-wheel.sh: TRITON_WHEEL=/abs/path/triton-*.whl
if [[ -n ${TRITON_WHEEL:-} ]]; then
  .venv/bin/uv pip install --python .venv/bin/python --no-deps "$TRITON_WHEEL"
  # PTX Gym runs ptxas from its source tree (editable layout); point that at the wheel's copy.
  mkdir -p .upstream/taic/ptx_gym/triton_src/third_party/nvidia/backend
  ln -sfn "$PWD/.venv/lib/python3.10/site-packages/triton/backends/nvidia/bin" .upstream/taic/ptx_gym/triton_src/third_party/nvidia/backend/bin
  # The wheel's ptx_gym copy lacks kernels/papers (no __init__.py upstream, so the build
  # skipped it). Use the pinned source tree, as an editable install would; a .pth file
  # also works under `sudo -E` (ncu), where PYTHONPATH may be dropped.
  rm -rf .venv/lib/python3.10/site-packages/ptx_gym
  echo "$PWD/.upstream/taic/ptx_gym" > .venv/lib/python3.10/site-packages/ptx_gym_source.pth
else
  TRITON_BUILD_PROTON=OFF TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF \
    MAX_JOBS=${MAX_JOBS:-2} .venv/bin/uv pip install --python .venv/bin/python --no-deps -e .upstream/taic/ptx_gym
fi
.venv/bin/uv pip freeze --python .venv/bin/python > results/requirements-resolved.txt
