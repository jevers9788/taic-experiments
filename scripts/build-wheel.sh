#!/usr/bin/env bash
set -euo pipefail
# Build PTX Gym's patched Triton wheel on a CPU-only Ubuntu 22.04 VM (same OS/Python as GPU VMs).
# No GPU or CUDA needed: the build downloads prebuilt LLVM and its pinned NVIDIA tools.
cd "$(dirname "$0")/.."
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 22.04 && $(uname -m) == x86_64 ]]
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git python3-dev python3-venv
if [[ ! -d .upstream/taic ]]; then
  git clone https://github.com/francois141/triton_ai_compiler.git .upstream/taic
fi
git -C .upstream/taic checkout 6a00d1261c803fecdfa74c1c349b339a026b06ad
git -C .upstream/taic submodule update --init --depth 1
python3 -m venv .venv-build
.venv-build/bin/pip install -q --upgrade pip
# Unused by PTX Gym/TAIC: Proton profiler and C++ unit tests. AMD backend is hard-coded upstream.
export TRITON_BUILD_PROTON=OFF TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
start=$(date +%s)
.venv-build/bin/pip wheel --no-deps -v -w dist .upstream/taic/ptx_gym
echo "Wheel build seconds: $(( $(date +%s) - start ))"
python3 --version
ls -l dist
