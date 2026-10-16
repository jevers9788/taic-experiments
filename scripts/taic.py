"""Run TAIC's CLI unchanged, with one upstream bug worked around (see PORTING.md 8).

Released agent.py and utils/setup.py call `dump_kernel_ptx(<operator object>)` but
import the name-based `ptx_gym.api.dump_kernel_ptx`, so every run crashes before any
model call. Point both at the operator-based helper instead (as TAIC's own
run_identity_compare.py does). Usage: python scripts/taic.py KERNEL [agent flags]
"""
import agent  # TAIC, from .upstream/taic on PYTHONPATH
from ptx_gym.helpers.triton import dump_kernel_ptx
import utils.setup

# Both call sites pass an operator object: agent.py:545 and utils/setup.py:199.
agent.dump_kernel_ptx = dump_kernel_ptx
utils.setup.dump_kernel_ptx = dump_kernel_ptx

if __name__ == '__main__':
    agent.main()
