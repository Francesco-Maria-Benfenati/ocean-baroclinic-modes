import xarray as xr
from xarray import DataArray, Dataset
import numpy as np
from numpy.typing import NDArray
import os


class ncWrite:
    """
    This Class is for writing baroclinic modes on NetCDF output file.
    """

    def __init__(self, outpath: str, filename: str = None) -> None:
        """
        Class constructor, given the output path (which may not include the filename).
        """

        if filename is None:
            outfolder, filename = os.path.split(outpath)
            if not filename.endswith(".nc"):
                filename += ".nc"
        else:
            # Check file extension .nc is included
            if not filename.endswith(".nc"):
                filename += ".nc"
            outfolder = outpath
        # Set output path
        self.path = os.path.join(outfolder, filename)
        # Remove old files if having the same name
        if os.path.exists(self.path):
            os.remove(self.path)
            print(f"Existing file {self.path} removed")
        # Create folder if does not exist
        os.makedirs(outfolder, exist_ok=True)

    def create_dataset(
        self, dims: list[str], coords: dict[NDArray], **fields: dict[NDArray]
    ) -> Dataset:
        """
        Create output dataset, given coords and field(s).
        """

        data_arrays = dict()
        for name, val in fields.items():
            data_array = DataArray(data=val, dims=dims, coords=coords)
            data_arrays[name] = data_array

        dataset = Dataset(data_vars=data_arrays)
        return dataset

    def save(self, *datasets: Dataset) -> None:
        """
        Save data on output file.
        """

        for dataset in datasets:
            # If file already exists, add data to the existing one
            if os.path.exists(self.path):
                file_dataset = xr.open_dataset(self.path)
                dataset = dataset.combine_first(file_dataset)
                file_dataset.close()
            dataset.to_netcdf(self.path, mode="w")
            dataset.close()


if __name__ == "__main__":
    import sys

    outpath = "./ncwrite"
    filename = "output"
    ncwrite = ncWrite(outpath, filename=filename)
    dims = ["lon", "lat"]
    coords = {"lon": np.ones(12), "lat": np.ones(24)}
    field = np.ones([12, 24])
    dataset = ncwrite.create_dataset(dims, coords, temp=field * 25, dens=field * 1025)
    print("Output dataset: ", dataset)
    # Save dataset
    ncwrite.save(dataset)
    coords = {"longitude": (dims, field), "latitude": (dims, field)}
    dataset2 = ncwrite.create_dataset(dims, coords, U=field * 0.5, V=field * 0.5)
    print("Output dataset with different coords (2D): ", dataset)
    # Save new dataset in addition
    ncwrite.save(dataset2)
    # # Check the file is deleted if new object is created
    outpath = os.path.join(outpath, filename)
    new_ncwrite = ncWrite(outpath)
    assert not os.path.exists("./ncwrite/output.nc")
    # Test if "save" method works with more datasets
    new_ncwrite.save(dataset, dataset2)
    os.remove("./ncwrite/output.nc")
    os.removedirs(
        "./ncwrite",
    )
