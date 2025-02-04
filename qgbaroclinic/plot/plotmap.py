import os, sys
import numpy as np
import xarray as xr
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from ..read import ncRead
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from read import ncRead


class PlotMap:
    """
    Class used for plotting the Rossby Radius 2D map in a region.
    """

    def __init__(
        self, lon_range: list[float], lat_range: list[float], outfolder: str
    ) -> None:
        """
        Class constructor.

        Args:
            lon_range (list[float]): Domain LONGITUDE range
            lat_range (list[float]): Domain LATITUDE range
            outfolder (str): path to output folder
        """
        self.lon_range = np.sort(np.array(lon_range))
        self.lat_range = np.sort(np.array(lat_range))
        self.outfolder = outfolder

    @staticmethod
    def rossrad_from_netcdf_output(path: str) -> tuple[xr.Variable]:
        """
        Read Rossby Radius variable from netcdf file.
        """
        ncread = ncRead(path)
        rossbyrad, lon, lat = ncread.variables("rossrad", "lon", "lat")
        return rossbyrad, lon, lat

    def make_plot(
        self,
        rossby_rad: NDArray,
        lon: NDArray,
        lat: NDArray,
        name: str,
        offset: int = 1,
    ) -> None:
        """
        Create and save plot of Rossby rad 2D map.

        Args:
            rossby_rad (NDArray): _description_
            lon (NDArray): _description_
            lat (NDArray): _description_
            name (str): _description_
        """
        # Create a figure with Equidist-Cylindrical proj.
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(8, 8))
        # Adds coastlines to the current axes
        ax.add_feature(
            cfeature.LAND,
            edgecolor="lightgray",
            facecolor=cfeature.COLORS["land"],
            zorder=0,
        )
        ax.add_feature(cfeature.BORDERS, linewidths=0.5)
        ax.add_feature(cfeature.STATES, linewidths=0.5)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)
        res = "10m"
        ax.coastlines(resolution=res, linewidths=0.5)
        domain_extent = [
            self.lon_range[0] - offset,
            self.lon_range[1] + offset,
            self.lat_range[0] - offset,
            self.lat_range[1] + offset,
        ]
        ax.set_extent(domain_extent)
        # Turn on continent shading
        ax.add_feature(cfeature.LAND.with_scale(res), facecolor="lightgray")
        # Customize cmap.
        mycmap = "gist_rainbow_r"
        vmin, vmax = 10, 100
        # Customize bar ticks position.
        bar_ticks = np.linspace(vmin, vmax, 11)
        levels = np.linspace(vmin, vmax, 100)
        # Contourf-plot data
        scalar = ax.contourf(
            lon,
            lat,
            rossby_rad,
            cmap=mycmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            extend="both",
        )
        cbar = plt.colorbar(scalar, orientation="horizontal", pad=0.1, ticks=bar_ticks)
        cbar.ax.set_title("km", fontsize=15)
        gl = ax.gridlines(
            linewidth=1, color="gray", alpha=0.5, linestyle="--", draw_labels=True
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"fontsize": 15}
        gl.ylabel_style = {"fontsize": 15}
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(
            "First Baroclinic Deformation Radius", style="italic", size=16, pad=0.2
        )
        # Save the current figure to a file
        fig_path = os.path.join(self.outfolder, name + ".png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close()

    def plot_chelton_map(self) -> None:
        """
        Plot Rossby Radius map from Chelton (1998) for comparison.
        """
        rossbyrad, lon, lat = self._read_from_chelton_dat()
        self.make_plot(rossbyrad, lon, lat, "comparison_Chelton_1998", offset=0.0)

    def _read_from_chelton_dat(self) -> tuple[NDArray]:
        """
        Read rossby radius dataset from Chelton (1998).
        """
        lon_range = self.lon_range
        lat_range = self.lat_range
        lon_range[lon_range < 0] = lon_range + 360.0
        chelton_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        chelton_path = os.path.join(chelton_path, "data", "chelton_rossrad.dat")
        df = pd.read_csv(
            chelton_path,
            delimiter=r"\s+",
            header=None,
            usecols=[0, 1, 3],
            names=["lat", "lon", "rossrad"],
            dtype=np.float64,
        )
        lon = df["lon"]
        lat = df["lat"]
        rossrad = df["rossrad"]
        x = np.arange(lon_range[0] - 1, lon_range[1] + 1)
        y = np.arange(lat_range[0] - 1, lat_range[1] + 1)
        rossbyrad = np.ones([len(y), len(x)]) * np.nan
        for i in range(len(lon)):
            id_lon = np.argmin(np.abs((lon[i] - x)))
            id_lat = np.argmin(np.abs((lat[i] - y)))
            rossbyrad[id_lat, id_lon] = rossrad[i]
        return rossbyrad, x, y


if __name__ == "__main__":
    pass
