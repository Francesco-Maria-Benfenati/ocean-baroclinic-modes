"""
Example: computing baroclinic modes of motion for a region.
"""

import os, sys
import numpy as np

try:
    import qgbaroclinic as qgb
except (ImportError, ModuleNotFoundError):
    sys.path.append(os.path.dirname((os.path.abspath(__file__))))
    import qgbaroclinic

# Define Baroclinic Modes object
obm = qgb.QGBaroclinic(latitude=[34.5, 50.2], longitude=[-34, -30])

# Extract OCEAN variables form NetCDF file
temp, sal, depth, lat = obm.read(
    "./data/reanalysis/", "thetao", "so", "depth", "latitude"
)
print(f"Temperature ha shape: {temp.shape}")
mean_lat = np.mean(lat)
temp = temp.transpose("time", "longitude", "latitude", "depth")
sal =sal.transpose("time", "longitude", "latitude", "depth")
print(f"After Transposing, Temperature has shape: {temp.shape}")
# Extract Bathymetry dataset and compute mean reagion depth
elevation = obm.read(
    "./data/bathymetry/GLO-MFC_001_030_mask_bathy.nc",
    "deptho",
)
mean_depth = np.round(np.abs(np.nanmean(elevation)))

# Compute potential density and average over the region
pot_density = obm.potential_density(temp, sal)
mean_region_pot_density = pot_density.mean(
    dim=["time", "longitude", "latitude"], skipna=True
)
print(mean_region_pot_density.values)
# Run model
obm(mean_region_pot_density, depth.values, mean_lat, mean_depth)

# The output result is stored as attributes
rossby_rad = obm.rossby_rad
vert_structfunc = obm.vert_structfunc
print(f"Rossby radii [km] {rossby_rad/1000}")
