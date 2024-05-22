# ======================================================================
# MAIN FILE recalling the functions implemented for computing the
# BAROCLINIC ROSSBY RADIUS in a defined region.
# ======================================================================
import sys, os
import time
import logging
import warnings
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from functools import partial
from plumbum import cli, colors

try:
    from .read.ncread import ncRead
    from .tool.eos import Eos
    from .read.config import Config
    from .solve.verticalstructureequation import VerticalStructureEquation
    from .tool.interpolation import Interpolation
    from .tool.bvfreq import BVfreq
    from .write.ncwrite import ncWrite
    from .tool.filter import Filter
    from .tool.utils import Utils
except ImportError:
    from read.ncread import ncRead
    from tool.eos import Eos
    from read.config import Config
    from solve.verticalstructureequation import VerticalStructureEquation
    from tool.interpolation import Interpolation
    from tool.bvfreq import BVfreq
    from write.ncwrite import ncWrite
    from tool.filter import Filter
    from tool.utils import Utils


class QGBaroclinic(cli.Application):
    """
    Application for computing QG baroclinic modes of motion in the Ocean.
    """

    PROGNAME = "qgbaroclinic"
    VERSION = "0.1.0"
    COLOR_GROUPS = {"Switches": colors.blue, "Meta-switches": colors.yellow}
    COLOR_GROUP_TITLES = {
        "Switches": colors.bold | colors.blue,
        "Meta-switches": colors.bold & colors.yellow,
    }

    def main(self):
        # Execute default command if no command is provided
        if not self.nested_command:  # will be "None" if no sub-command follows
            print("No command given")
            return 1  # error exit code


@QGBaroclinic.subcommand("exe")
class Exe(cli.Application):
    """
    Execute QGBaroclinic SOFTWARE.
    """

    @cli.switch(["-c", "--config"], str)
    def set_config(self, config_path: str = None):
        """
        Set config file.
        """
        self.config = Config(os.path.normpath(config_path))

    def read_ocean_variables(self) -> None:
        """
        STORE VARIABLES FROM NETCDF OCE FILE
        """
        logging.info("Reading OCE data ...")
        config = self.config
        oce_path = config.input.oce.path
        read_oce = ncRead(oce_path)
        # OCE dimensions, variables and coordinates
        oce_dims = config.input.oce.dims
        oce_vars = config.input.oce.vars
        oce_coords = config.input.oce.coords
        # check dimension order in config file
        sorted_dims = ("time", "lon", "lat", "depth")
        # try:
        assert (
            tuple(oce_dims.keys()) == sorted_dims
        ), f"Dimensions in config file are expected to be {sorted_dims}."

        oce_domain = {k: v for k, v in zip(oce_dims.values(), config.domain.values())}
        # Convert datetime to numpy friendly
        oce_domain[oce_dims["time"]] = [
            np.datetime64(t) for t in oce_domain[oce_dims["time"]]
        ]
        drop_dims = config.input.oce.drop_dims
        # Read oce dataset
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
        print("temperature", temperature.shape, "salinity", salinity.shape)
        assert (
            temperature.shape == salinity.shape
        ), "Temperature and saliniy do not have the same shape."
        self.temperature = temperature
        self.salinity = salinity
        self.depth = oce_dataset[oce_coords["depth"]]
        self.latitude = oce_dataset[oce_coords["lat"]]
        self.longitude = oce_dataset[oce_coords["lon"]]
        self.oce_domain = oce_domain
        self.oce_dims = oce_dims
        # close datasets
        oce_dataset.close()

    def extract_region_depth(self) -> None:
        """
        STORE SEA FLOOR DEPTH FROM NETCDF BATHYMETRY FILE
        """
        config = self.config
        oce_domain = self.oce_domain
        oce_dims = self.oce_dims
        logging.info("Reading bathymetry dataset ...")
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
        bottom_depth = bathy_dataset[bathy_vars["bottomdepth"]]
        print("bottom depth", bottom_depth.shape)
        # INTERPOLATION OF GEBCO BATHYMETRY
        x = bathy_dataset[bathy_coords["lon"]]
        y = bathy_dataset[bathy_coords["lat"]]
        interp = RegularGridInterpolator(
            (x, y),
            bottom_depth.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        X, Y = np.meshgrid(self.longitude, self.latitude, indexing="ij")
        interp_bathy = interp((X, Y))
        print("Bathymetry after interpolation", interp_bathy.shape)
        max_bottom_depth = np.max(np.abs(interp_bathy))
        print("max bottom depth", max_bottom_depth)
        self.max_bottom_depth = max_bottom_depth
        self.bottom_depth = np.abs(interp_bathy)
        bathy_dataset.close()
        logging.info(f"Mean region depth: {self.max_bottom_depth} m")

    def compute_density(self) -> None:
        """
        Compute Density from Pot. Temperature & Salinity.
        """

        logging.info("Computing density ...")
        config = self.config
        ref_pressure = 0  # reference pressure [dbar]
        salinity = self.salinity
        temperature = self.temperature.values
        # temperature[np.where(np.isnan(temperature.all()))] = -99999
        print("computing pot density")
        if config.input.oce.insitu_temperature is True:
            pot_temperature = Eos.potential_temperature(
                salinity.values,
                temperature,
                self.depth.values,
                ref_press=ref_pressure,
            )
        else:
            pot_temperature = temperature
        eos = Eos(salinity.values, pot_temperature, ref_pressure)
        potdensity = np.nanmean(eos.density, axis=0)
        print("pot density", potdensity.shape)
        # VERTICAL INTERPOLATION (1m grid step)
        logging.info("Vertically interpolating mean density ...")
        grid_step = self.grid_step
        interpolation = Interpolation(self.depth.values, potdensity)
        (interp_dens, self.interp_depth) = interpolation.apply_interpolation(
            -grid_step / 2, self.max_bottom_depth + grid_step, grid_step
        )
        self.interp_dens = interp_dens
        # Interface levels (1m grid step, shifted of 0.5 m respect to depth levels)
        self.interface_depth = self.interp_depth[:-1] + grid_step / 2
        print("after interpolation", interp_dens.shape)

    def compute_bruntvaisala_freq(self) -> None:
        """
        COMPUTE BRUNT-VAISALA FREQUENCY SQUARED
        """
        config = self.config
        logging.info("Computing Brunt-Vaisala frequency ...")
        bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(self.interp_depth, self.interp_dens)
        print("bv_freq", bv_freq_sqrd.shape)

        grid_step = self.grid_step
        max_bottom_depth = self.max_bottom_depth
        # RE-INTERPOLATING BRUNT-VAISALA FREQUENCY SQUARED FOR REMOVING NaNs and < 0 values.
        logging.info("Post-processing Brunt-Vaisala frequency ...")
        grid_step = self.grid_step
        bv_freq_sqrd[np.where(bv_freq_sqrd < 0)] = np.nan
        bv_freq_sqrd[:, :, 0] = 0.0
        bv_freq_sqrd[:, :, -1] = 0.0
        interpolate_bvfreqsqrd = Interpolation(self.interface_depth, bv_freq_sqrd)
        bv_freq_sqrd, interfaces = interpolate_bvfreqsqrd.apply_interpolation(
            0, max_bottom_depth + 2*grid_step, grid_step
        )
        bv_freq = np.sqrt(bv_freq_sqrd)
        print(bv_freq)
        # FILTERING BRUNT-VAISALA FREQUENCY PROFILE WITH A LOW-PASS FILTER.
        if config.filter.filter:
            logging.info("Filtering Brunt-Vaisala frequency ...")
            cutoff_wavelength = config.filter.cutoff_wavelength
            cutoff_depth = config.filter.cutoff_depth
            cutoff_levels = Utils.find_nearvals(self.interface_depth, *cutoff_depth)
            filter = Filter(
                bv_freq,
                grid_step=grid_step,
                type="lowpass",
                order=config.filter.order,
                axis=2,
            )
            cutoff_dict = dict(zip(cutoff_levels, cutoff_wavelength))
            self.bv_freq_filtered = filter.apply_filter(cutoff_dict)
        else:
            logging.info("Brunt-Vaisala frequency has not been filtered...")
            self.bv_freq_filtered = bv_freq
        print("bv freq filtered", self.bv_freq_filtered.shape)

    def compute_baroclinicmodes(self):
        """
        BAROCLINIC MODES & ROSSBY RADII
        """
        config = self.config
        logging.info("Computing baroclinic modes and Rossby radii ...")
        # N° of modes of motion considered (including the barotropic one).
        N_motion_modes = 2
        bv_freq_filtered = self.bv_freq_filtered
        x = np.ones_like(bv_freq_filtered[:, 0, 0])
        z = np.ones_like(bv_freq_filtered[0, 0, :])
        X, latitude, Z = np.meshgrid(
            x, self.latitude, z, indexing="ij"
        )
        print("latitude", latitude.shape)
        # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
        # From config file, specify if structure of vertical velocity should be computed instead of Phi.
        rossby_rad_array = VerticalStructureEquation.multiprofiles(
            bv_freq_filtered, latitude, self.bottom_depth, self.grid_step, n_modes=N_motion_modes
        )
        rossby_rad_array = (
            rossby_rad_array[:, :, 1:] / 1000
        )  # Rossby radius in [km], excluding barotropic mode
        print("rossby rad", rossby_rad_array.shape)

        # WRITE RESULTS ON OUTPUT FILE
        logging.info("Saving Baroclinic Modes dataset to .nc file ...")
        modes_of_motion = np.arange(0, 1)
        lat = self.latitude
        lon = self.longitude
        # Rossby radii dataset
        ncwrite = self.ncwrite
        radii_dataset = ncwrite.create_dataset(
            dims={"lon": lon, "lat": lat, "mode": modes_of_motion},
            coords={"lon": lon, "lat": lat},
            rossbyrad=rossby_rad_array,
        )
        ncwrite.save(radii_dataset)

    def main(self):
        """
        Running the resolution algorithm.
        """
        # Get starting time
        start_time = time.time()
        # load config file
        try:
            self.config
        except AttributeError:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.toml",
            )
            self.config = Config(config_path)
        # RUN MAIN()
        # SET OUTPUT FILE (& LOGGING)
        config = self.config
        self.ncwrite = ncWrite(
            config.output.folder_path, filename=config.output.filename, logfile=True
        )
        logging.info(f"Using config file {config.config_file}")
        try:
            # read ocean variables
            Exe.read_ocean_variables(self)
            # Compute mean region depth
            Exe.extract_region_depth(self)
            # Set new grid step (= 1m)
            self.grid_step = 1  # [m]
            # Compute potential density
            Exe.compute_density(self)
            # Compute Brunt-Vaisala frequency
            Exe.compute_bruntvaisala_freq(self)
            # Compute Baroclinic Modes
            Exe.compute_baroclinicmodes(self)
        except Exception as e:
            # Log any unhandled exceptions
            print("Error: please check the log file for details")
            logging.exception(f"An exception occurred: {e}")
        # Get ending time
        end_time = time.time()
        # Print elapsed time
        elapsed_time = np.round(end_time - start_time, decimals=2)
        logging.info(
            f"Computing Ocean Baroclinic Modes COMPLETED in {elapsed_time} seconds."
        )


if __name__ == "__main__":
    QGBaroclinic.run()
