"""
Example: mapping baroclinic rossby radius for the global ocean.
"""

# ======================================================================
# MAIN FILE recalling the functions implemented for computing the
# BAROCLINIC ROSSBY RADIUS in a defined region.
# ======================================================================
import sys, os
import time
import logging
import warnings
import numpy as np
import scipy as sp
from functools import partial
from plumbum import cli, colors

try:
    from ..qgbaroclinic.read.ncread import ncRead
    from ..qgbaroclinic.tool.eos import Eos
    from ..qgbaroclinic.read.config import Config
    from ..qgbaroclinic.solve.verticalstructureequation import VerticalStructureEquation
    from ..qgbaroclinic.tool.interpolation import Interpolation
    from ..qgbaroclinic.tool.bvfreq import BVfreq
    from ..qgbaroclinic.write.ncwrite import ncWrite
    from ..qgbaroclinic.tool.filter import Filter
    from ..qgbaroclinic.tool.utils import Utils
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qgbaroclinic.read.ncread import ncRead
    from qgbaroclinic.tool.eos import Eos
    from qgbaroclinic.read.config import Config
    from qgbaroclinic.solve.verticalstructureequation import VerticalStructureEquation
    from qgbaroclinic.tool.interpolation import Interpolation
    from qgbaroclinic.tool.bvfreq import BVfreq
    from qgbaroclinic.write.ncwrite import ncWrite
    from qgbaroclinic.tool.filter import Filter
    from qgbaroclinic.tool.utils import Utils

    