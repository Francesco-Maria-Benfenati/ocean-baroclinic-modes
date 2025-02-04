import logging
import warnings
import numpy as np
import xarray as xr
from typing import Union
from functools import partial
from scipy.interpolate import RegularGridInterpolator
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

try:
    from .model import OceBaroclinicModes
    from .read import Config, ncRead
    from .tool import Utils
    from .write import ncWrite
except ImportError:
    from model import OceBaroclinicModes
    from read import Config, ncRead
    from tool import Utils
    from write import ncWrite

warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_oce_from_config(config: Config) -> dict[Union[NDArray, xr.Variable]]:
    """
    STORE VARIABLES FROM NETCDF OCE FILE, based on config file.

    Returns:
        dict[Union[NDArray, xr.Variable]: dictionary of variables
    """

    logging.info("Reading OCE data")
    oce_path = config.input.oce.path
    read_oce = ncRead(oce_path)
    # OCE dimensions, variables and coordinates.
    oce_dims = config.input.oce.dims
    oce_vars = config.input.oce.vars
    oce_coords = config.input.oce.coords
    # Check dimension order in config file.
    sorted_dims = ("time", "lon", "lat", "depth")
    assert (
        tuple(oce_dims.keys()) == sorted_dims
    ), f"Dimensions in config file are expected to be {sorted_dims}."
    # OCE domain. Convert datetime to numpy friendly. Drop dimensions.
    oce_domain = {k: v for k, v in zip(oce_dims.values(), config.domain.values())}
    oce_domain[oce_dims["time"]] = [
        np.datetime64(t) for t in oce_domain[oce_dims["time"]]
    ]
    drop_dims = config.input.oce.drop_dims
    # Read oce dataset.
    oce_dataset = read_oce.dataset(
        dims=oce_dims,
        vars=oce_vars,
        coords=oce_coords,
        preprocess=partial(Utils.drop_dims_from_dataset, drop_dims=drop_dims),
        decode_vars=config.input.oce.decode_vars_with_xarray,
        **oce_domain,
    )
    temperature = oce_dataset[oce_vars["temperature"]]
    salinity = oce_dataset[oce_vars["salinity"]]
    # Check array size matching.
    assert (
        temperature.shape == salinity.shape
    ), "Temperature and saliniy do not have the same shape."
    output_dict = {
        "temperature": temperature,
        "salinity": salinity,
        "depth": oce_dataset[oce_coords["depth"]],
        "latitude": oce_dataset[oce_coords["lat"]],
        "longitude": oce_dataset[oce_coords["lon"]],
    }
    # Close datasets
    oce_dataset.close()
    return output_dict


def extract_bathy_from_config(
    config: Config, oce_longitude: NDArray, oce_latitude: NDArray
) -> Union[NDArray, xr.Variable]:
    """
    STORE SEA FLOOR DEPTH FROM NETCDF BATHYMETRY FILE, based on config file.

    Args:
        oce_longitude (NDArray): longitude array of ocean grid.
        oce_latitude (NDArray): latitude array of ocean grid.

    Returns:
        Union[NDArray, xr.Variable]: sea floor depth array.
    """

    # Domain defined
    oce_dims = config.input.oce.dims
    oce_domain = {k: v for k, v in zip(oce_dims.values(), config.domain.values())}
    # Read bathy from netcdf
    if config.input.bathy.set_bottomdepth is False:
        logging.info("Reading bathymetry dataset")
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
        seafloor_depth = bathy_dataset[bathy_vars["seafloor_depth"]]
        # INTERPOLATION OF GEBCO BATHYMETRY ON OCEAN GRID
        x = bathy_dataset[bathy_coords["lon"]]
        y = bathy_dataset[bathy_coords["lat"]]
        interp = RegularGridInterpolator(
            (x, y),
            seafloor_depth.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        X, Y = np.meshgrid(oce_longitude, oce_latitude, indexing="ij")
        seafloor_depth = interp((X, Y))
        bathy_dataset.close()
    # Read seafloor depth set by the user
    else:
        logging.info("Sea Floor Depth set as in confi file")
        seafloor_depth = np.ones(
            [oce_longitude.shape[0], oce_latitude.shape[0]], dtype=np.float64
        )
        seafloor_depth *= config.input.bathy.set_bottomdepth
    return seafloor_depth


def region_average(
    obm: OceBaroclinicModes,
    ncwrite: ncWrite,
    potential_density: xr.Variable,
    seafloor_depth: xr.Variable,
    ocean_dict: dict,
    n_modes: int,
) -> None:
    """
    Compute average values in a region.

    Args:
        obm (OceBaroclinicModes): obm object
        potential_density (xr.Variable):
            potential density 4D array (time, lon, lat, depth)
        seafloor_depth (xr.Variable):
            sea floor depth 2D array (lon, lat)
        ocean_dict (dict):
            dictionary containing ocean variables and coordinates
        n_modes (int): number of modes to be computed.
    """

    logging.info("Computing Baroclinic Modes for 'AVERAGE' output")
    # Mean vaues.
    logging.info("Averaging variables in space and time")
    mean_potential_density = np.nanmean(potential_density, axis=(0, 1, 2))
    mean_latitude = np.nanmean(ocean_dict["latitude"], axis=-1)
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


def region_map(
    obm: OceBaroclinicModes,
    ncwrite: ncWrite,
    potential_density: xr.Variable,
    seafloor_depth: xr.Variable,
    ocean_dict: dict,
    n_modes: int,
    n_cpus: int,
) -> None:
    """
    Compute baroclinic modes of motion for a 2D map.

    Args:
        obm (OceBaroclinicModes): obm object
        potential_density (xr.Variable): Potential density 4D array (time, lon, lat, depth)
        seafloor_depth (xr.Variable): Sea floor depth 2D array (lon, lat)
        ocean_dict (dict): Dictionary containing ocean variables and coordinates
        n_modes (int): Number of modes to be computed.
    """

    logging.info("Computing Baroclinic Modes for 'MAP' output")
    # Apply only time averaging to potential density
    mean_potential_density = np.nanmean(
        potential_density, axis=0
    )  # Shape: (lon, lat, depth)
    seafloor_depth = np.abs(seafloor_depth)
    # Get dimensions
    lon_size, lat_size, _ = mean_potential_density.shape
    depth_size = int(np.max(seafloor_depth)) + 1
    depth_array = np.arange(0.5, depth_size, 1.0)  # create depth array
    # Initialize output arrays
    rossby_rad = np.full(
        (lon_size, lat_size, n_modes), np.nan
    )  # Shape: (lon, lat, mode)
    vert_structfunc = np.full(
        (lon_size, lat_size, depth_size, n_modes), np.nan
    )  # Shape: (lon, lat, depth, mode)
    # Parallel computation using ProcessPoolExecutor
    logging.info("Processing each (lon, lat) grid point in parallel")
    tasks = [
        (i, j, mean_potential_density, seafloor_depth, ocean_dict, obm, n_modes)
        for i in range(lon_size)
        for j in range(lat_size)
    ]
    total_tasks = len(tasks)  # Total number of (i, j) points

    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        # Wrap results in tqdm for progress tracking
        with tqdm(total=total_tasks, desc="Computing MAP") as pbar:
            for result in executor.map(compute_modes_at_point, *zip(*tasks)):
                # Store results in arrays
                i, j, rossby_vals, vert_struct_vals = result  # Unpack result
                if rossby_vals is not None and vert_struct_vals is not None:
                    logging.info(f"Processed point (lon id, lat id) : ({i}, {j})")
                    rossby_rad[i, j, :] = rossby_vals
                    vert_structfunc[i, j, : len(vert_struct_vals), :] = vert_struct_vals
                else:
                    logging.info(
                        f"Pot. Density profile is not sufficient to compute values at point (lon id, lat id) : ({i}, {j}))"
                    )
                pbar.update(1)  # Update tqdm progress bar

    # Write results to NetCDF file.
    logging.info("Saving dataset to netcdf output file")
    modes_of_motion = np.arange(0, n_modes)
    rossrad_dataset = ncwrite.create_dataset(
        dims=["lon", "lat", "mode"],
        coords={
            "mode": modes_of_motion,
            "lon": ocean_dict["longitude"],
            "lat": ocean_dict["latitude"],
        },
        attrs={"name": "baroclinic rossby radius", "units": "meters"},
        rossrad=rossby_rad,
    )
    vert_struct_func_dataset = ncwrite.create_dataset(
        dims=["lon", "lat", "depth", "mode"],
        coords={
            "mode": modes_of_motion,
            "lon": ocean_dict["longitude"],
            "lat": ocean_dict["latitude"],
            "depth": depth_array,
        },
        attrs={"name": "normalized vertical structure function", "units": "1"},
        vertstructfunc=vert_structfunc,
    )
    # Save output file.
    ncwrite.save(rossrad_dataset, vert_struct_func_dataset)


def compute_modes_at_point(
    i, j, mean_potential_density, seafloor_depth, ocean_dict, obm, n_modes
):
    """
    Computes baroclinic modes for a given (lon, lat) point.

    Args:
        i, j (int): Indices of the lon-lat grid.
        mean_potential_density (np.array): Time-averaged potential density (lon, lat, depth).
        seafloor_depth (np.array): Seafloor depth (lon, lat).
        ocean_dict (dict): Dictionary with ocean variables.
        obm (OceBaroclinicModes): Baroclinic modes solver.
        n_modes (int): Number of modes.

    Returns:
        tuple: (i, j, rossby_rad[i, j, :], vert_structfunc[i, j, :, :]) or None if skipped.
    """

    potdens_profile = mean_potential_density[i, j, :]  # Extract depth profile
    seafloor = seafloor_depth[i, j]  # Seafloor depth at (lon, lat)
    latitude = ocean_dict["latitude"][j]  # Latitude at index j
    # Compute baroclinic modes
    try:
        rossby_rad, vert_structfunc = obm(
            potdens_profile,
            ocean_dict["depth"].values,
            latitude.values,
            seafloor,
            n_modes=n_modes,
        )
    # Skip invalid land or missing points
    except ValueError:
        return i, j, None, None
    return i, j, rossby_rad, vert_structfunc
