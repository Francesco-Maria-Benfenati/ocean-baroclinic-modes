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
from argparse import ArgumentParser

from model.ncread import ncRead
from model.eos import Eos
from model.config import Config
from model.baroclinicmodes import BaroclinicModes
from model.interpolation import Interpolation
from model.bvfreq import BVfreq
from model.ncwrite import ncWrite


if __name__ == "__main__":
    # Get starting time
    start_time = time.time()

    # READ CONFIG FILE
    # arg parser to get config file path
    # usage example: python main.py -c config.toml
    print("Reading config file ...")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file", required=True)
    args, unknown = parser.parse_known_args(sys.argv)
    # load config
    config = Config(args.config)

    # STORE VARIABLES FROM NetCDF INPUT FILE
    print("Reading data ...")
    indata_path = config.input.indata_path
    read_oce = ncRead(indata_path)
    # Store Potential Temperature, Salinity and mean latitude from NetCDF input file.
    oce_dims = config.input.oce.dims
    # check dimension order in config file
    try:
        assert tuple(oce_dims.keys()) == ("time", "lon", "lat", "depth")
    except AssertionError:
        warnings.warn("Dimension in config file are not in order time, lon, lat, depth")
    oce_vars = config.input.oce.vars
    (
        pot_temperature,
        salinity,
        depth,
    ) = read_oce.variables(*oce_vars.values())
    # Transpose dims in the same order as in config file (time, lon, lat, depth)
    pot_temperature, salinity = read_oce.transpose(
        pot_temperature, salinity, **oce_dims
    )

    # STORE SEA FLOOR DEPTH FROM NetCDF BATHYMETRY FILE
    print("Reading bathymetry ...")
    inbathy_path = config.input.bathy.inbathy_path
    read_bathy = ncRead(inbathy_path)
    bathy_dims = config.input.bathy.dims
    bathy_vars = config.input.bathy.vars
    (seafloor_depth,) = read_bathy.variables(*bathy_vars.values())

    # COMPUTE DENSITY
    print("Computing density ...")
    # Compute Density from Pot. Temperature & Salinity.
    eos = Eos(pot_temperature.values, salinity.values, depth.values)
    mean_region_density = np.nanmean(eos.density, axis=(0, 1, 2))

    # VERTICAL INTERPOLATION (1m grid step)
    print("Vertically interpolating mean density ...")
    grid_step = 1  # [m]
    mean_region_depth = seafloor_depth.mean(dim=bathy_dims.values()).values
    interpolation = Interpolation(depth.values, mean_region_density)
    (interp_dens,) = interpolation.apply_interpolation(
        -grid_step / 2, mean_region_depth + grid_step, grid_step
    )
    interp_depth = np.arange(-grid_step / 2, mean_region_depth + grid_step, grid_step)

    # COMPUTE BRUNT-VAISALA FREQUENCY
    bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(interp_depth, interp_dens)
    bv_freq_sqrd = BVfreq.post_processing(bv_freq_sqrd)
    bv_freq = np.sqrt(bv_freq_sqrd)

    # BAROCLINIC MODES & ROSSBY RADIUS
    print("Computing baroclinic modes and Rossby radii ...")
    # N° of modes of motion considered (including the barotropic one).
    N_motion_modes = config.experiment.n_modes
    mean_lat = np.mean(np.array(config.experiment.domain.lat))
    # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
    baroclinicmodes = BaroclinicModes(bv_freq, mean_lat, grid_step)
    rossby_rad = baroclinicmodes.rossbyrad / 1000  # Rossby radius in [km]
    # Set Rossby radius of mode 0 (barotropic) as NaN
    if rossby_rad[0] > 1e04:
        rossby_rad[0] = np.nan
    structure_func = baroclinicmodes.structfunc
    print(f"Rossby radii [km]: {rossby_rad}")

    # WRITE RESULTS ON OUTPUT FILE
    print("Writing output file ...")
    ncwrite = ncWrite(config.experiment.outpath, filename=config.experiment.name)
    dims = ["mode"]
    modes_of_motion = np.arange(0, config.experiment.n_modes)
    # Rossby radii dataset
    radii_dataset = ncwrite.create_dataset(
        dims="mode", coords={"mode": modes_of_motion}, rossbyrad=rossby_rad
    )
    # Vertical profiles dataset
    # Interpolate Brunt-Vaisala frequency over original depth levels
    interface_depths = np.arange(0, mean_region_depth, grid_step)
    interpolate = sp.interpolate.interp1d(
        interface_depths,
        bv_freq,
        fill_value="extrapolate",
        kind="linear",
    )
    interp_bvfreq = interpolate(depth)
    vertical_profiles_dataset = ncwrite.create_dataset(
        dims="depth",
        coords={"depth": depth.values},
        density=mean_region_density,
        bvfreq=interp_bvfreq,
    )
    # Vertical structure function dataset
    # Interpolate vertical structure function over original depth levels
    interpolate = sp.interpolate.interp1d(
        -interp_depth[1:-1],
        structure_func,
        fill_value="extrapolate",
        kind="linear",
        axis=0,
    )
    interp_structfunc = interpolate(depth)
    vert_struct_func_dataset = ncwrite.create_dataset(
        dims=["depth", "mode"],
        coords={"mode": modes_of_motion, "depth": depth.values},
        structfunc=interp_structfunc,
    )
    ncwrite.save(radii_dataset, vert_struct_func_dataset, vertical_profiles_dataset)

    # Get ending time
    end_time = time.time()
    # Print elapsed time
    elapsed_time = np.round(end_time - start_time, decimals=2)
    print(f"Computing Ocean Baroclinic Modes COMPLETED in {elapsed_time} seconds.")
