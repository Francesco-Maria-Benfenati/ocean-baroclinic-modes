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
from functools import partial
from xarray import Dataset

from model.ncread import ncRead
from model.eos import Eos
from model.config import Config
from model.baroclinicmodes import BaroclinicModes
from model.interpolation import Interpolation
from model.bvfreq import BVfreq
from model.ncwrite import ncWrite
from model.filter import Filter


# Preprocessing for Climatologies with different number of observations.
def drop_dims_from_dataset(dataset: Dataset, drop_dims: list[str]) -> Dataset:
    """
    Drop dimensions from datasets which would give troubles due to incompatible sizes.
    """
    return dataset.drop_dims(drop_dims=drop_dims)


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

    # STORE VARIABLES FROM NetCDF OCE FILE
    print("Reading OCE data ...")
    oce_path = config.input.oce.path
    read_oce = ncRead(oce_path)
    # OCE dimensions, variables and coordinates
    oce_dims = config.input.oce.dims
    oce_vars = config.input.oce.vars
    oce_coords = config.input.oce.coords
    # check dimension order in config file
    sorted_dims = ("time", "lon", "lat", "depth")
    try:
        assert tuple(oce_dims.keys()) == sorted_dims
    except AssertionError:
        warnings.warn(
            "Dimension in config file are expected to be [time, lon, lat, depth]."
        )
    oce_domain = {k: v for k, v in zip(oce_dims.values(), config.domain.values())}
    # Convert datetime to numpy friendly
    oce_domain[oce_dims["time"]] = [
        np.datetime64(t) for t in oce_domain[oce_dims["time"]]
    ]
    drop_dims = config.input.oce.drop_dims
    # Read oce dataset
    oce_dataset = read_oce.dataset(
        dims=oce_dims,
        vars=oce_vars,
        coords=oce_coords,
        preprocess=partial(drop_dims_from_dataset, drop_dims=drop_dims),
        decode_vars=config.input.oce.decode_vars_with_xarray,
        **oce_domain,
    )
    # STORE POT. TEMPERATURE AND SALINITY VARIABLES.
    temperature = oce_dataset[oce_vars["temperature"]]
    salinity = oce_dataset[oce_vars["salinity"]]
    depth = oce_dataset[oce_coords["depth"]]
    latitude = oce_dataset[oce_coords["lat"]]
    # close datasets
    oce_dataset.close()
    print(f"Input Data have shape: {salinity.shape}")

    # STORE SEA FLOOR DEPTH FROM NetCDF BATHYMETRY FILE
    if config.input.bathy.set_bottomdepth is False:
        print("Reading bathymetry ...")
        inbathy_path = config.input.bathy.path
        read_bathy = ncRead(inbathy_path)
        bathy_dims = config.input.bathy.dims
        bathy_vars = config.input.bathy.vars
        bathy_coords = config.input.bathy.coords
        bathy_domain = {
            bathy_dims["lon"]: oce_domain[oce_dims["lon"]],
            bathy_dims["lat"]: oce_domain[oce_dims["lat"]],
        }
        bathy_dataset = read_bathy.dataset(
            dims=bathy_dims,
            vars=bathy_vars,
            coords=bathy_coords,
            **bathy_domain,
            decode_vars=True,
        )
        seafloor_depth = bathy_dataset[bathy_vars["bottomdepth"]]
        mean_region_depth = np.abs(seafloor_depth.mean(dim=bathy_dims.values()).values)
        bathy_dataset.close()
    else:
        mean_region_depth = np.abs(config.input.bathy.set_bottomdepth)
    mean_region_depth = np.round(mean_region_depth, decimals=0)
    print(f"Mean region depth: {mean_region_depth} m")

    # COMPUTE DENSITY
    print("Computing density ...")
    # Compute Density from Pot. Temperature & Salinity.
    if config.input.oce.insitu_temperature is True:
        pot_temperature = Eos.potential_temperature(
            salinity.values, temperature.values, depth.values
        )
    else:
        pot_temperature = temperature
    try:
        eos = Eos(salinity.values, pot_temperature.values, depth.values)
    except AttributeError:
        eos = Eos(salinity.values, pot_temperature, depth.values)
    mean_region_density = np.nanmean(eos.density, axis=(0, 1, 2))

    # VERTICAL INTERPOLATION (1m grid step)
    print("Vertically interpolating mean density ...")
    grid_step = 1  # [m]

    interpolation = Interpolation(depth.values, mean_region_density)
    (interp_dens,) = interpolation.apply_interpolation(
        -grid_step / 2, mean_region_depth + grid_step, grid_step
    )
    # Depth levels (1m grid step)
    interp_depth = np.arange(-grid_step / 2, mean_region_depth + grid_step, grid_step)
    # Interface levels (1m grid step, shifted of 0.5 m respect to depth levels)
    interface_depth = np.arange(len(interp_depth[1:-1]) + 1)

    # COMPUTE BRUNT-VAISALA FREQUENCY
    bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(interp_depth, interp_dens)
    bv_freq = np.sqrt(bv_freq_sqrd)

    # FILTERING BRUNT-VAISALA FREQUENCY PROFILE WITH A LOW-PASS FILTER.
    filter = config.bvfilter.filter
    ####################################################################
    # if filter:
    #     print("Applying low-pass filter to Brunt-Vaisala frequency ...")
    #     filter_order = config.bvfilter.order
    #     cutoff_wavelength = config.bvfilter.cutoff_wavelength
    #     bv_freq_filtered = Filter.lowpass(
    #         bv_freq, grid_step, cutoff_wavelength, filter_order
    #     )
    # else:
    #     bv_freq_filtered = bv_freq
    ####################################################################

    # * Filtering differently above/below 100 m *
    bv_freq_above_pycnocline = bv_freq[:100]
    bv_freq_below_pycnocline = bv_freq[100:]
    if filter:
        print("Applying low-pass filter to Brunt-Vaisala frequency ...")
        filter_order = config.bvfilter.order
        cutoff_wavelength = config.bvfilter.cutoff_wavelength
        bv_freq_filtered_above_pycnocline = Filter.lowpass(
            bv_freq_above_pycnocline,
            grid_step,
            cutoff_wavelength=10,
            order=filter_order,
        )
        bv_freq_filtered_below_pycnocline = Filter.lowpass(
            bv_freq_below_pycnocline,
            grid_step,
            cutoff_wavelength=cutoff_wavelength,
            order=filter_order,
        )
        bv_freq_filtered = np.concatenate(
            (bv_freq_filtered_above_pycnocline, bv_freq_filtered_below_pycnocline)
        )
    else:
        bv_freq_filtered = bv_freq

    ####################################################################
    # Beta plot of BV frequency.
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot(bv_freq, -interface_depth)
    # plt.plot(bv_freq_filtered, -interface_depth)
    # plt.show()
    # plt.close()
    ####################################################################

    # BAROCLINIC MODES & ROSSBY RADIUS
    print("Computing baroclinic modes and Rossby radii ...")
    # N° of modes of motion considered (including the barotropic one).
    N_motion_modes = config.output.n_modes
    mean_lat = np.mean(latitude.values)

    # Warning if the region is too near the equator.
    equator_threshold = 2.0
    if -equator_threshold < np.any(latitude.values) < equator_threshold:
        warnings.warn(
            "The domain area is close to the equator: ! Rossby radii computation might be inaccurate !"
        )

    # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
    baroclinicmodes = BaroclinicModes(
        bv_freq_filtered,
        mean_lat=mean_lat,
        grid_step=grid_step,
        n_modes=N_motion_modes,
    )
    rossby_rad = baroclinicmodes.rossbyrad / 1000  # Rossby radius in [km]
    print(f"Rossby radii [km]: {rossby_rad}")

    # WRITE RESULTS ON OUTPUT FILE
    print("Writing output file ...")
    ncwrite = ncWrite(config.output.path, filename=config.output.filename)
    dims = ["mode"]
    modes_of_motion = np.arange(0, config.output.n_modes)
    # Rossby radii dataset
    radii_dataset = ncwrite.create_dataset(
        dims="mode", coords={"mode": modes_of_motion}, rossbyrad=rossby_rad
    )
    # Vertical profiles dataset
    # Interpolate Brunt-Vaisala frequency (computed at interfaces) over depth levels
    interface_depths = np.arange(0, mean_region_depth + grid_step, grid_step)
    interpolate = sp.interpolate.interp1d(
        interface_depths,
        bv_freq,
        fill_value="extrapolate",
        kind="linear",
    )
    interp_bvfreq = interpolate(interp_depth[1:-1])
    vertical_profiles_dataset = ncwrite.create_dataset(
        dims="depth",
        coords={"depth": interp_depth[1:-1]},
        density=interp_dens[1:-1],
        bvfreq=interp_bvfreq,
    )
    # Vertical structure function dataset
    vert_struct_func_dataset = ncwrite.create_dataset(
        dims=["depth", "mode"],
        coords={"mode": modes_of_motion, "depth": interp_depth[1:-1]},
        structfunc=baroclinicmodes.structfunc,
    )
    ncwrite.save(radii_dataset, vert_struct_func_dataset, vertical_profiles_dataset)

    # Get ending time
    end_time = time.time()
    # Print elapsed time
    elapsed_time = np.round(end_time - start_time, decimals=2)
    print(f"Computing Ocean Baroclinic Modes COMPLETED in {elapsed_time} seconds.")
