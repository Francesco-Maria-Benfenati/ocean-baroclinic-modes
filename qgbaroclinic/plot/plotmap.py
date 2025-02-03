import os, sys
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm

try:
    from ..read import ncRead
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from read import ncRead


def read_from_netcdf_output(path: str):
    ncread = ncRead(path)
    rossbyrad, lon, lat = ncread.variables("rossbyrad", "lon", "lat")
    return rossbyrad, lon, lat


def read_from_chelton_dat():
    df = pd.read_csv(
        "./data/chelton_rossrad.dat",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1, 3],
        names=["lat", "lon", "rossrad"],
        dtype=np.float64,
    )
    lon = df["lon"]
    lat = df["lat"]
    rossrad = df["rossrad"]
    y = np.arange(-89.5, 89.5)
    x = np.arange(0, 360)
    rossbyrad = np.ones([len(y), len(x)]) * np.nan
    for i in range(len(lon)):
        id_lon = np.argmin(np.abs((lon[i] - x)))
        id_lat = np.argmin(np.abs((lat[i] - y)))
        rossbyrad[id_lat, id_lon] = rossrad[i]
    return rossbyrad, x, y


def make_plot(rossby_rad: NDArray, lon: NDArray, lat: NDArray, name: str):
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
    ax.set_extent([-80, 0, 0, 50])

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
    cbar.ax.set_title("[ km ]", fontsize=15)
    gl = ax.gridlines(
        linewidth=1, color="gray", alpha=0.5, linestyle="--", draw_labels=True
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"fontsize": 15}
    gl.ylabel_style = {"fontsize": 15}
    cbar.ax.tick_params(labelsize=15)
    ax.set_title("First Baroclinic Rossby Radius", style="italic", size=16, pad=0.2)
    # Save the current figure to a file
    # plt.show()
    fig.savefig(name, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    out_path = "./output/baroclinic_modes_2d_test.nc"
    rossbyrad_out, lon_out, lat_out = read_from_netcdf_output(out_path)
    rossbyrad_out = rossbyrad_out.transpose("lat", "lon", "mode")
    make_plot(rossbyrad_out[:, :, 0], lon_out, lat_out, "north_atlantic_reanalysis")
    chelton_path = "./data/rossrad.dat"
    rossbyrad_chelton, lon_chelton, lat_chelton = read_from_chelton_dat()
    make_plot(rossbyrad_chelton, lon_chelton, lat_chelton, "north_atlantic_chelton_test")
