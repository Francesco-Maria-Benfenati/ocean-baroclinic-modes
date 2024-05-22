"""
Example: computing baroclinic modes of motion for a region.
"""

import os, sys
import numpy as np
try:
    from ..qgbaroclinic.model.baroclinicmodes import BaroclinicModes
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qgbaroclinic.model.baroclinicmodes import BaroclinicModes

b = BaroclinicModes("my_testcase_azores")

b.region( latitude = [34.5, 50.2], longitude = [-34, -30])
temp, sal = b.read("./data/test_case/dataset_azores/azores_Jan2021.nc", "thetao", "so")
print(temp, sal)
bathy = b.read("./data/bathymetry/GEBCO_2023.nc", "elevation", lat = [34.5, 50.2], lon=[-34, -30])
mean_depth = np.nanmean(bathy)
b.set_bottomdepth(mean_depth)

print(mean_depth)