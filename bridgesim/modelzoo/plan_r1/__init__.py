"""
Plan-R1: Token-based autoregressive trajectory planner (vendored from /hugsim-storage/Plan-R1).

Plan-R1 uses bare module imports (e.g. ``from utils import ...``, ``from layers import ...``).
Inserting this package's directory into sys.path makes those bare imports resolve correctly
without modifying any of the original Plan-R1 source files.
"""
import sys
import os

_PLAN_R1_VENDOR_DIR = os.path.dirname(__file__)
if _PLAN_R1_VENDOR_DIR not in sys.path:
    sys.path.insert(0, _PLAN_R1_VENDOR_DIR)
