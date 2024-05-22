import sys, os
import logging
import xarray as xr
from xarray import Dataset, Variable
import numpy as np
import pandas as pd

try:
    from ..src.tools.utils import Utils
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tool.utils import Utils


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

    def dataset(
        self,
        dims: dict[str],
        vars: dict[str] = None,
        coords: dict[str] = None,
        preprocess: callable = None,
        decode_vars: bool = False,
        **domain: dict[list],
    ) -> Dataset:
        """
        Extract Dataset out of a NetCDF file, given dimensions and variables.
        """
        try:
            engine = "h5netcdf"
            dataset = xr.open_mfdataset(
                self.path,
                concat_dim=None,
                combine="by_coords",
                parallel=True,
                preprocess=preprocess,
                engine="h5netcdf",
                cache=True,
                lock=False,
                decode_times=True,
                decode_cf=decode_vars,
                # mask_and_scale = False,
            )
        except (ValueError, OSError):
            engine = "scipy"  # "netcdf4"
            dataset = xr.open_mfdataset(
                self.path,
                concat_dim=None,
                combine="by_coords",
                parallel=True,
                engine=engine,
                cache=True,
                lock=False,
                decode_cf=decode_vars,
                decode_times=True,
                # mask_and_scale = False,
            )
        logging.info(f"Open NetCDF file(s) with {engine} engine.")
        # Keep only variables we are interested in.
        dataset = dataset[list(vars.values())]
        # check that the dimensions are in the dataset
        for key in dims.values():
            assert key in dataset.dims, f"Dimension {key} not found in dataset"
        # Check if domain has coorect labels (i.e. dimension names)
        for key in domain.keys():
            assert key in dataset.dims, "Domain labels do not correspond to dimensions"
        # check that the coords are in the dataset
        if coords is not None:
            for key in coords.values():
                assert key in dataset.coords, f"Coordinate {key} not found in dataset"
                # Check if labels are the same for dims and coords dictionaries
                assert (
                    coords.keys() == dims.keys()
                ), "Labels are not the same in dims and coords dictionaries"
            if coords != dims:
                # Rename coordinates as dimensions
                name_dict = {k: v for k, v in zip(coords.values(), dims.values())}
                dataset = dataset.rename(name_dict)
        # Transpose dataset coherently with the dims argument.
        dataset = dataset.transpose(*dims.values(), ..., missing_dims="raise")
        # Crop dataset
        dataset = self.crop_dataset(dataset, **domain)
        # Decode variables manually
        if decode_vars is False:
            dataset = self.decode_vars(dataset)
        return dataset

    def variables(self, *names: tuple[str], **domain: dict[list]) -> tuple[Variable]:
        """
        Extract variable(s) from NetCDF file, given the name(s).
        """
        try:
            dataset = xr.open_mfdataset(
                self.path,
                concat_dim=None,
                combine="by_coords",
                parallel=True,
                engine="netcdf4", #"h5netcdf",
                cache=True,
                lock=False,
                decode_times=True,
                decode_cf=True,
            )
        except Exception:
            dataset = xr.open_mfdataset(
                    self.path,
                    concat_dim=None,
                    combine="by_coords",
                    parallel=True,
                    engine="scipy",
                    cache=True,
                    lock=False,
                    decode_times=True,
                    decode_cf=True,
                )
        dataset = self.crop_dataset(dataset, **domain)
        variables = ()
        for name in names:
            var = dataset.variables[name]
            variables += (var,)
        dataset.close()
        return variables

    def var_asattr(self, *names: tuple[str]) -> None:
        """
        Set variable(s) as attributes.
        """
        variables = self.variables(*names)
        for i in range(len(names)):
            setattr(self, names[i], variables[i])

    def transpose(self, *vars: tuple[Variable], **kdims: dict[str]) -> tuple[Variable]:
        """
        Transpose variables according to the given dimensions order.
        """
        transposed_vars = ()
        dims = kdims.values()
        print(*dims)
        for var in vars:
            transposed_vars += (var.transpose(*dims, ..., missing_dims="ignore"),)
        return transposed_vars

    def crop_dataset(self, dataset: Dataset, **domain: dict[list]) -> tuple[Variable]:
        """
        Crop the input Variable(s). Kyeword args 'domain' are the dims/coordinates.
        NOTE: dims and coords should be named in the same way.
        """
        coords = dataset.coords
        dims = dataset.dims
        new_domain = dict()
        # check that the dimensions are in the dataset
        for key in domain.keys():
            assert key in dims, f"Kwarg {key} is not among dataset dims: {dims}"
            assert key in coords, f"Kwarg {key} is not among dataset coords: {coords}"
            if domain[key] == []:
                continue
            try:
                [id_min, id_max] = Utils.find_nearvals(coords[key].values, *domain[key])
            except ValueError:
                raise ValueError(
                    "Please, provide both or none domain extremants. Only one is not accepted."
                )
            [id_min, id_max] = np.sort([id_min, id_max])
            new_domain[key] = np.arange(id_min, id_max + 1)
        cropped_dataset = dataset.isel(new_domain, missing_dims="warn")
        return cropped_dataset

    def decode_vars(self, dataset: Dataset) -> Dataset:
        """
        Decode variables "by hand" if having troubles with xarray decoding.
        """
        for var in dataset.var():
            if dataset[var].dtype == "int16":
                scale_factor = dataset[var].scale_factor
                add_offset = dataset[var].add_offset
                decode_values = dataset[var].values * np.float64(
                    scale_factor
                ) + np.float64(add_offset)
                dataset[var].values = decode_values
        return dataset


if __name__ == "__main__":
    read = ncRead("./data/test_case/dataset_azores/azores_Jan2021.nc")
    temp, sal = read.variables("thetao", "so")
    vars = read.variables("thetao", "so")
    print(temp.shape)
    print(type(temp))
    temp, sal = read.variables("thetao", "so", depth=[0, 62.5])
    dims = {"time": "time", "lon": "longitude", "lat": "latitude", "depth": "depth"}
    print(temp.shape)
    # Transpose many fields using **kwargs dims
    temp, sal = read.transpose(temp, sal, lat="latitude", depth="deptho")
    print(temp.dims)
    # Transpose many fields using **dict dims
    temp, sal = read.transpose(temp, sal, **dims)
    print(temp.dims)
    # Test loading coordinates as attributes.
    read.var_asattr("latitude", "longitude")
    print(read.longitude)
    print(read.latitude.values)
    print(temp[0, 0:3, 0, 0].values)
    temp = temp.isel({"longitude": [0, 1, 2]})
    print(temp[0, :, 0, 0].values)
    near_ids = Utils.find_nearvals(np.arange(1, 11), 2.3, 4.5, 6.9)
    assert near_ids == [1, 3, 6]
    dataset = xr.open_dataset("./data/test_case/dataset_azores/azores_Jan2021.nc")
    print("Whole dataset: ", dataset.dims)
    cropped_dataset = read.crop_dataset(
        dataset,
        longitude=[23, 45],
        latitude=[34, 68],
        time=[
            np.datetime64("2021-01-21T12"),
            np.datetime64("2021-01-26T12"),
        ],
    )
    print("Cropped dataset: ", cropped_dataset.dims)
    date0 = pd.Timestamp("2021-01-21T12")
    date1 = pd.Timestamp("2021-01-26T12")
    cropped_dataset = read.crop_dataset(
        dataset,
        longitude=[23, 45],
        latitude=[34, 68],
        time=[
            np.datetime64(date0),
            np.datetime64(date1),
        ],
        depth=[0.7, 120],
    )
    print("Cropped dataset: ", cropped_dataset.dims)
    try:
        read.crop_dataset(dataset, lon=[23, 45], lat=[34, 68])
    except AssertionError:
        pass
    dataset.close()
    vars = {"temp": "thetao", "sal": "so"}
    dims = {"time": "time", "lon": "longitude", "lat": "latitude", "depth": "depth"}
    coords = dims
    dataset = read.dataset(dims, vars, coords, longitude=[-33, -30], latitude=[40, 42])
    print(dataset["thetao"].values[0, 0, 0, 0])
    print(dataset["thetao"].dtype)
    dataset.close()
    # Read Copernicus dataset.
    cmems = ncRead("./data/reanalysis")
    dataset = cmems.dataset(dims, vars, coords, longitude=[-33, -30], latitude=[40, 42])
    print(dataset["thetao"].dtype)
    print(dataset["thetao"].values[0, 0, 0, 0])
    dataset.close()
