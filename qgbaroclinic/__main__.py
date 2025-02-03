# ======================================================================
# MAIN FILE recalling the functions implemented for computing the
# BAROCLINIC ROSSBY RADIUS in a defined region.
# ======================================================================
import os
import time
import logging
import warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)


from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial

try:
    from .model import OceBaroclinicModes
    from .read import Config, extract_oce_from_config, extract_bathy_from_config
    from .write import ncWrite
except ImportError:
    from model import OceBaroclinicModes
    from read import Config, extract_oce_from_config, extract_bathy_from_config
    from write import ncWrite


def average_values(
    obm: OceBaroclinicModes,
    potential_density: xr.Variable,
    seafloor_depth: xr.Variable,
    ocean_dict: dict,
    n_modes: int,
) -> None:
    """
    Compute average values in a region.
    """

    logging.info("Computing Baroclinic Modes for 'AVERAGE' output")
    # Mean vaues.
    logging.info("Averaging variables in space and time")
    mean_potential_density = np.nanmean(potential_density, axis=(0, 1, 2))
    mean_latitude = np.nanmean(oce_dict["latitude"], axis=-1)
    mean_seafloor_depth = np.nanmean(np.abs(seafloor_depth), axis=(0, 1))
    # Run MODEL.
    logging.info("Running Model")
    rossby_rad, vert_structfunc = obm(
        mean_potential_density,
        ocean_dict["depth"].values,
        mean_latitude,
        mean_seafloor_depth,
        n_modes=n_modes,
    )
    # Write results on output file.
    logging.info("Saving dataset to netcdf output file")
    modes_of_motion = np.arange(0, n_modes)
    rossrad_dataset = ncwrite.create_dataset(
        dims="mode",
        coords={"mode": modes_of_motion},
        attrs={"name": "baroclinic rossby radius", "units": "meters"},
        rossrad=rossby_rad,
    )
    # Vertical structure function dataset
    vert_struct_func_dataset = ncwrite.create_dataset(
        dims=["depth", "mode"],
        coords={"mode": modes_of_motion, "depth": obm.depth},
        attrs={"name": "normalized vertical structure function", "units": "1"},
        vertstructfunc=vert_structfunc,
    )
    ncwrite.save(rossrad_dataset, vert_struct_func_dataset)
    # Plotting results.
    fig_name = config.output.fig_name
    if fig_name:
        logging.info("Plotting results")
        fig_out_path = os.path.join(config.output.folder_path, fig_name + ".png")
        obm.plot(fig_out_path)


if __name__ == "__main__":
    # Config file as argument.
    parser = ArgumentParser(description="Ocean QG Baroclinic Modes of Motion.")
    parser.add_argument(
        "-c",
        "--config",
        help="(optional) Path to configuration file.",
        required=False,
        type=str,
        default="./config.toml",
    )
    parser.add_argument(
        "-p",
        "--processors",
        help="(optional) Number of processors for parallel computing.",
        required=False,
        type=int,
        default=None,
    )
    args = parser.parse_args()
    # Path to config file.
    path_to_config_file = args.config
    config = Config(os.path.normpath(path_to_config_file))
    # Number of processors for parallel computing
    n_cpus = args.processors
    if n_cpus is None:
        if config.output.type == "map":
            n_cpus = os.cpu_count() - 1
        else:
            n_cpus = 1
    # Get starting time
    start_time = time.time()
    # MAIN TASKS.
    try:
        # SET OUTPUT FILE (& LOGGING)
        ncwrite = ncWrite(
            config.output.folder_path, filename=config.output.filename, logfile=True
        )
        logging.info(f"Using config file {config.config_file}")
        logging.info(f"Using {n_cpus} CPUs for computing")
        # Extract OCEAN variables.
        oce_dict = extract_oce_from_config(config)
        # Extract SEAFLOOR DEPTH from BATHYMETRY.
        seafloor_depth = extract_bathy_from_config(
            config, oce_dict["longitude"], oce_dict["latitude"]
        )
        # Instance of OBM class.
        obm = OceBaroclinicModes()
        # Compute Potential Density from Pot./in-situ Temperature.
        logging.info("Computing Potential Density")
        potential_density = obm.potential_density(
            oce_dict["temperature"],
            oce_dict["salinity"],
            oce_dict["depth"],
            insitu_temperature=config.input.oce.insitu_temperature,
        )
        match config.output.type:
            # Compute AVERAGE values in a region.
            case "average":
                average_values(
                    obm,
                    potential_density,
                    seafloor_depth,
                    oce_dict,
                    config.output.n_modes,
                )
            # Compute 2D MAP in a region.
            case "map":
                pass
            case _:
                raise ValueError(
                    f"Output type '{config.output.type}' is not supported."
                )
    except Exception as e:
        # Log any unhandled exceptions
        logging.exception(f"An exception occurred: {e}")
    # Get ending time
    end_time = time.time()
    # Print elapsed time
    elapsed_time = np.round(end_time - start_time, decimals=2)
    logging.info(
        f"Computing Ocean Baroclinic Modes COMPLETED in {elapsed_time} seconds."
    )
