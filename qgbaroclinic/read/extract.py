import sys, os
import logging
import numpy as np
import xarray as xr
from typing import Union
from functools import partial
from scipy.interpolate import RegularGridInterpolator
from numpy.typing import NDArray

try:
    from . import ncRead
    from ..tool import Utils
    from . import Config
except (ImportError, ModuleNotFoundError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tool import Utils
    from read import ncRead
    from read import Config


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


if __name__ == "__main__":
    pass
