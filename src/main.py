# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:10:28 2022

@author: Francesco M. Benfenati
"""
# ======================================================================
# MAIN FILE recalling the functions implemented for computing the
# BAROCLINIC ROSSBY RADIUS in a defined region.
# ======================================================================
import sys
import numpy as np

from argparse import ArgumentParser
import warnings

from model.ncread import ncRead
from model.eos import Eos
from model.config import Config
from model.baroclinicmodes import BaroclinicModes
from model.interpolation import Interpolation
from model.bvfreq import BVfreq

import concurrent.futures
import time
from tqdm import tqdm


def timed_future_progress_bar(future, expected_time=60, increments=100):
    """
    ** BETA VERSION **
    Display progress bar for expected_time seconds.
    Complete early if future completes.
    Wait for future if it doesn't complete in expected_time.
    """
    interval = expected_time / increments
    with tqdm(total=increments) as pbar:
        for i in range(increments - 1):
            if future.done():
                # finish the progress bar
                pbar.update(increments - i)
                return
            else:
                time.sleep(interval)
                pbar.update()
        # if the future still hasn't completed, wait for it.
        future.result()
        pbar.update()


def main() -> None:
    """
    Main code for running the model.
    """

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

    # COMPUTE POTENTIAL DENSITY
    print("Computing potential density ...")
    # Compute Potential Density from Pot. Temperature & Salinity.
    eos = Eos(pot_temperature.values, salinity.values, depth.values)
    # Compute Brunt-Vaisala frequency squared from Potential Density.
    # Store vertical mean potential density as used within the function
    # while computing Brunt-Vaisala frequency squared.
    mean_region_density = np.nanmean(eos.density, axis=(0, 1, 2))
    print("Density mean profile: ", mean_region_density)

    # VERTICAL INTERPOLATION (1m grid step)
    print("Vertically interpolating mean potential density ...")
    grid_step = 1  # [m]
    mean_region_depth = seafloor_depth.mean(dim=bathy_dims.values()).values
    interpolation = Interpolation(depth.values, mean_region_density)
    (interp_dens,) = interpolation.apply_interpolation(
        -grid_step / 2, mean_region_depth + grid_step, grid_step
    )
    interp_depth = -np.arange(-grid_step / 2, mean_region_depth + grid_step, grid_step)
    print("Region mean depth: ", mean_region_depth)
    print("New interpolated depth grid: ", interp_depth)

    # COMPUTE BRUNT-VAISALA FREQUENCY
    bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(interp_depth, interp_dens)
    bv_freq_sqrd = BVfreq.post_processing(bv_freq_sqrd)
    bv_freq = np.sqrt(bv_freq_sqrd)
    print("Brunt-Vaisala frequency profile: ", bv_freq)
    print("Max BV freq value: ", np.max(bv_freq))

    # BAROCLINIC MODES & ROSSBY RADIUS
    print("Computing baroclinic modes and Rossby radii ...")
    # N° of modes of motion considered (including the barotropic one).
    N_motion_modes = config.experiment.n_modes
    mean_lat = np.mean(np.array(config.experiment.domain.lat))
    # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
    baroclinicmodes = BaroclinicModes(bv_freq, mean_lat, grid_step)
    R = 1 / (np.sqrt(baroclinicmodes.eigenvals) * 1000)  # Rossby radius in [km]
    print(f"Rossby radii [km]: {R}")

    # WRITE RESULTS ON OUTPUT FILE
    print("Writing output file ...")


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(main)
        timed_future_progress_bar(future)
        print("Computing QG Baroclinic Modes COMPLETED.")
