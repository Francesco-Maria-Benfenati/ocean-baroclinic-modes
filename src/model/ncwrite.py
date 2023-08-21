import xarray as xr
import numpy as np
import subprocess
import os


class WriteFile:
    """
    This Class is for writing baroclinic modes on NetCDF output file.
    """

    def __init__(self, outpath: str, filename: str) -> None:
        """
        
        """
        self.path = outpath
        self.filename = filename
        subprocess.run(["mkdir", "-p", outpath], shell=False)

    def save(self) -> None:
        """
        Save data on output file.
        """
        filename = self.filename
        dataset = xr.Dataset(data_vars=self.dataset)
        self.dataset = dataset
        dataset.to_netcdf(self.path + self.filename, mode="w")

        if not filename.endswith(".nc"):
            filename += ".nc"
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Existing file {filename} removed")
        if hasattr(self, "dataset"):
            self.dataset.to_netcdf(filename)
            self.dataset.close()
        else:
            print("No data to write")


if __name__ == "__main__":
    import sys
    outpath = "./ncwrite"
    writefile = WriteFile(outpath, filename="writing_test")
    