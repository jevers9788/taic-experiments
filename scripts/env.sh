# Source from the experiment root.
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/.upstream/taic:$PWD/.upstream/taic/ptx_gym${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export PTX_MEMORY_SANITIZER=/usr/local/cuda-12.8/bin/compute-sanitizer
export NCU_PATH="$PWD/scripts/ncu-root.sh"
