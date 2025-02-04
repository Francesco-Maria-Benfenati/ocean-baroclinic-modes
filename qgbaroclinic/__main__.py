# ======================================================================
# MAIN FILE recalling the functions implemented for computing the
# BAROCLINIC ROSSBY RADIUS in a defined region.
# ======================================================================
import os
import time
import logging
import numpy as np

from argparse import ArgumentParser

from plot.plotmap import *

try:
    from .model import OceBaroclinicModes
    from .read import Config
    from .write import ncWrite
    from .core import (
        extract_oce_from_config,
        extract_bathy_from_config,
        region_average,
        region_map,
    )
except ImportError:
    from model import OceBaroclinicModes
    from read import Config
    from write import ncWrite
    from core import (
        extract_oce_from_config,
        extract_bathy_from_config,
        region_average,
        region_map,
    )

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
        ocean_dict = extract_oce_from_config(config)
        # Extract SEAFLOOR DEPTH from BATHYMETRY.
        seafloor_depth = extract_bathy_from_config(
            config, ocean_dict["longitude"], ocean_dict["latitude"]
        )
        # Instance of OBM class.
        obm = OceBaroclinicModes()
        # Compute Potential Density from Pot./in-situ Temperature.
        logging.info("Computing Potential Density")
        potential_density = obm.potential_density(
            ocean_dict["temperature"],
            ocean_dict["salinity"],
            ocean_dict["depth"],
            insitu_temperature=config.input.oce.insitu_temperature,
        )
        match config.output.type:
            # Compute AVERAGE values in a region.
            case "average":
                region_average(
                    obm,
                    ncwrite,
                    potential_density,
                    seafloor_depth,
                    ocean_dict,
                    config.output.n_modes,
                )
            # Compute 2D MAP in a region.
            case "map":
                pass
                region_map(
                    obm,
                    ncwrite,
                    potential_density,
                    seafloor_depth,
                    ocean_dict,
                    config.output.n_modes,
                    n_cpus,
                )
            case _:
                raise ValueError(
                    f"Output type '{config.output.type}' is not supported."
                )
        # PLOTTING RESULTS
        fig_name = config.output.fig_name
        if fig_name:
            logging.info("Plotting results")
            match config.output.type:
                case "average":
                    fig_out_path = os.path.join(ncwrite.outfolder, fig_name + ".png")
                    obm.plot(fig_out_path)
                case "map":
                    pm = PlotMap(
                        config.domain.lon, config.domain.lat, ncwrite.outfolder
                    )
                    rossbyrad_out, lon_out, lat_out = pm.rossrad_from_netcdf_output(
                        ncwrite.path
                    )
                    rossbyrad_out = rossbyrad_out.transpose("lat", "lon", "mode")
                    pm.make_plot(
                        rossbyrad_out[:, :, 1] / 1000.0,  # Conversion to [km]
                        lon_out,
                        lat_out,
                        config.output.fig_name,
                        offset=1,
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
