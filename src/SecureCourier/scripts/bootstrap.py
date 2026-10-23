"""Resolve all runtime imports within this release, wherever it is unpacked."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
sys.path[:0] = [str(ROOT/'legacy/sc_d/src'), str(ROOT/'legacy/parent/src')]
import scd_common
scd_common.PARENT = ROOT/'legacy/parent'
# No active code in this release reads the historical LEGACY location.
scd_common.LEGACY = ROOT/'legacy'
