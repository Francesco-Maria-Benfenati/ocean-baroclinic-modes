import os
import xarray as xr
from xarray import DataArray, Dataset, Variable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy.typing import NDArray


class ncRead:
    """
    This Class is for reading data from NetCDF input files.
    """

    def __init__(self, inpath: str) -> None:
        """
        Build class object given the input data path.
        """
        # check for file existence
        assert os.path.exists(inpath), FileNotFoundError(f"File {inpath} not found")
        # if the path is a directory, add *.nc to the path for xarray compatibility
        if not os.path.isfile(inpath):
            inpath = os.path.join(inpath, "*.nc")
        self.path = inpath

    def variables(self, *names: tuple[str]) -> tuple[Variable]:
        """
        Extract variable(s) from NetCDF file, given the name(s).
        """
        dataset = xr.open_mfdataset(self.path)
        variables = ()
        for name in names:
            var = dataset.variables[name]
            variables += (var,)
        dataset.close()
        return variables

    def transpose(self, *vars: tuple[Variable], **kdims: tuple[str]) -> tuple[Variable]:
        """
        Transpose variables according to the given dimensions order.
        """
        transposed_vars = ()
        dims = kdims.values()
        for var in vars:
            transposed_vars += (var.transpose(*dims, ..., missing_dims="raise"),)
        return transposed_vars


if __name__ == "__main__":
    read = ncRead("./data/test_case/dataset_azores/azores_Jan2021.nc")
    temp, sal = read.variables("thetao", "so")
    dims = {"time": "time", "lon": "longitude", "lat": "latitude", "depth": "depth"}
    print(temp)
    # Transpose many fields using **kwargs dims
    temp, sal = read.transpose(temp, sal, lat="latitude", depht="depth")
    print(temp.dims)
    # Transpose many fields using **dict dims
    temp, sal = read.transpose(temp, sal, **dims)
    print(temp.dims)
