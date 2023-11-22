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

from model.ncread import ncRead
from model.eos import Eos
from model.config import Config
from model.baroclinicmodes import BaroclinicModes
from model.interpolation import Interpolation
from model.bvfreq import BVfreq
from model.ncwrite import ncWrite
from model.filter import Filter
from model.utils import Utils


def main():
    """
    Software main function.
    """

    # READ CONFIG FILE
    # arg parser to get config file path
    # usage example: python main.py -c config.toml
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file", required=True)
    args, unknown = parser.parse_known_args(sys.argv)
    # load config
    config = Config(args.config)
    
    # SET LOGGING
    log_path = os.path.join(config.output.path, "qgbaroclinic.log")
    removed = False
    if os.path.exists(log_path):
        os.remove(log_path)
        removed = True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_level = logging.INFO
    logging.basicConfig(
        filename=log_path,
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    if removed:
        logging.info(f"Removed old log file at {log_path}")

    # STORE VARIABLES FROM NetCDF OCE FILE
    logging.info("Reading OCE data ...")
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
        logging.warning(
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
        preprocess=partial(Utils.drop_dims_from_dataset, drop_dims=drop_dims),
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
    logging.info(f"Input Data have shape: {salinity.shape}")

    # STORE SEA FLOOR DEPTH FROM NetCDF BATHYMETRY FILE
    if config.input.bathy.set_bottomdepth is False:
        logging.info("Reading bathymetry ...")
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
    logging.info(f"Mean region depth: {mean_region_depth} m")

    # COMPUTE DENSITY
    logging.info("Computing density ...")
    # Compute Density from Pot. Temperature & Salinity.
    ref_pressure = 0  # reference pressure [dbar]
    if config.input.oce.insitu_temperature is True:
        pot_temperature = Eos.potential_temperature(
            salinity.values, temperature.values, depth.values, ref_press=ref_pressure
        )
    else:
        pot_temperature = temperature
    try:
        eos = Eos(salinity.values, pot_temperature.values, ref_pressure)
    except AttributeError:
        eos = Eos(salinity.values, pot_temperature, ref_pressure)
    mean_region_potdensity = np.nanmean(eos.density, axis=(0, 1, 2))

    # VERTICAL INTERPOLATION (1m grid step)
    logging.info("Vertically interpolating mean density ...")
    grid_step = 1  # [m]
    interpolation = Interpolation(depth.values, mean_region_potdensity)
    (interp_dens, interp_depth) = interpolation.apply_interpolation(
        -grid_step / 2, mean_region_depth + grid_step, grid_step
    )
    # Interface levels (1m grid step, shifted of 0.5 m respect to depth levels)
    interface_depth = interp_depth[:-1] + grid_step / 2
    # New equispatial depth levels.
    depth_levels = interp_depth[1:-1]

    # COMPUTE BRUNT-VAISALA FREQUENCY SQUARED
    bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(interp_depth, interp_dens)

    # RE-INTERPOLATING BRUNT-VAISALA FREQUENCY SQUARED FOR REMOVING NaNs and < 0 values.
    bv_freq_sqrd[np.where(bv_freq_sqrd < 0)] = np.nan
    if np.isnan(bv_freq_sqrd[0]):
        bv_freq_sqrd[0] = 0.0
    interpolate_bvfreqsqrd = Interpolation(interface_depth, bv_freq_sqrd)
    bv_freq_sqrd, interfaces = interpolate_bvfreqsqrd.apply_interpolation(
        0, mean_region_depth + grid_step, grid_step
    )
    assert np.array_equal(interfaces, interface_depth)
    bv_freq = np.sqrt(bv_freq_sqrd)

    # FILTERING BRUNT-VAISALA FREQUENCY PROFILE WITH A LOW-PASS FILTER.
    filter = config.bvfilter.filter
    # If filtering should be depth-dependent.
    depth_dependent_filter = True
    # Filtering the whole profile in the same way.
    if not depth_dependent_filter:
        if filter:
            logging.info("Applying low-pass filter to Brunt-Vaisala frequency ...")
            filter_order = config.bvfilter.order
            cutoff_wavelength = config.bvfilter.cutoff_wavelength
            bv_freq_filtered = Filter.lowpass(
                bv_freq, grid_step, cutoff_wavelength, filter_order
            )
        else:
            bv_freq_filtered = bv_freq
    # Filtering differently above/below 100 m .
    elif depth_dependent_filter:
        bv_freq_above_pycnocline = bv_freq[:100]
        bv_freq_below_pycnocline = bv_freq[100:]
        if filter:
            logging.info("Applying low-pass filter to Brunt-Vaisala frequency ...")
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

    # BAROCLINIC MODES & ROSSBY RADIUS
    logging.info("Computing baroclinic modes and Rossby radii ...")
    # N° of modes of motion considered (including the barotropic one).
    N_motion_modes = config.output.n_modes
    mean_lat = np.mean(latitude.values)
    # Warning if the region is too near the equator.
    equator_threshold = 2.0
    lower_condition = -equator_threshold < np.min(latitude.values) < equator_threshold
    upper_condition = -equator_threshold < np.max(latitude.values) < equator_threshold
    if Utils.andor(lower_condition, upper_condition):
        warnings.warn(
            "The domain area is close to the equator: ! Rossby radii computation might be inaccurate !"
        )

    # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
    # From config file, specify if structure of vertical velocity should be computed instead of Phi.
    baroclinicmodes = BaroclinicModes(
        bv_freq_filtered,
        mean_lat=mean_lat,
        grid_step=grid_step,
        n_modes=N_motion_modes,
        vertvel_method=config.output.vertvel_method,
    )
    rossby_rad = baroclinicmodes.rossbyrad  # / 1000  # Rossby radius in [km]
    logging.info(f"Rossby radii [km]: {rossby_rad/1000}")

    # WRITE RESULTS ON OUTPUT FILE
    logging.info("Writing output file ...")
    ncwrite = ncWrite(config.output.path, filename=config.output.filename)
    dims = ["mode"]
    modes_of_motion = np.arange(0, config.output.n_modes)
    # Rossby radii dataset
    radii_dataset = ncwrite.create_dataset(
        dims="mode", coords={"mode": modes_of_motion}, rossbyrad=rossby_rad
    )
    # Density dataset
    density_dataset = ncwrite.create_dataset(
        dims="depth",
        coords={"depth": depth_levels},
        density=interp_dens[1:-1],
    )
    # Brunt-Vaisala freq. dataset
    bvfreq_dataset = ncwrite.create_dataset(
        dims="depth_interface",
        coords={"depth_interface": interface_depth},
        bvfreq=bv_freq_filtered,
    )
    # Vertical structure function dataset
    if config.output.vertvel_method:
        vert_struct_func_dataset = ncwrite.create_dataset(
            dims=["depth_interface", "mode"],
            coords={"mode": modes_of_motion, "depth_interface": interface_depth},
            structfunc=baroclinicmodes.structfunc,
        )
    else:
        vert_struct_func_dataset = ncwrite.create_dataset(
            dims=["depth", "mode"],
            coords={"mode": modes_of_motion, "depth": depth_levels},
            structfunc=baroclinicmodes.structfunc,
        )
    ncwrite.save(
        radii_dataset, density_dataset, bvfreq_dataset, vert_struct_func_dataset
    )

    # Results with WKB approximation (see Chelton, 1997)
    m = np.arange(N_motion_modes)
    coriolis_param = np.abs(baroclinicmodes.coriolis_param(mean_lat))
    lambda_m = (coriolis_param * m * np.pi) ** (-1) * sp.integrate.trapezoid(
        bv_freq_filtered, interface_depth
    )
    logging.info(
        f"According to WKB approximation (see chelton, 1997), Rossby radii [km] are:{lambda_m / 1000}"
    )


if __name__ == "__main__":
    # Get starting time
    start_time = time.time()

    # SET LOGGING
    logging.captureWarnings(True)

    # RUN MAIN()
    try:
        main()
    except Exception as e:
        # Log any unhandled exceptions
        logging.error(f"An exception occurred: {e}")
    
    # Get ending time
    end_time = time.time()
    # Print elapsed time
    elapsed_time = np.round(end_time - start_time, decimals=2)
    logging.info(
        f"Computing Ocean Baroclinic Modes COMPLETED in {elapsed_time} seconds."
    )
