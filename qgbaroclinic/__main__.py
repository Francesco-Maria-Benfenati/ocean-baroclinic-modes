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
from functools import partial
from plumbum import cli, colors
from numpy.typing import NDArray
try:
    from .model.ocebaroclinicmodes import OceBaroclinicModes
    from .read.ncread import ncRead
    from .tool.eos import EoS
    from .read.config import Config
    from .solve.verticalstructureequation import VerticalStructureEquation
    from .tool.interpolation import Interpolation
    from .tool.bvfreq import BVfreq
    from .write.ncwrite import ncWrite
    from .tool.filter import Filter
    from .tool.utils import Utils
except ImportError:
    from model.ocebaroclinicmodes import OceBaroclinicModes
    from read.ncread import ncRead
    from tool.eos import EoS
    from read.config import Config
    from solve.verticalstructureequation import VerticalStructureEquation
    from tool.interpolation import Interpolation
    from tool.bvfreq import BVfreq
    from write.ncwrite import ncWrite
    from tool.filter import Filter
    from tool.utils import Utils


def read_oce_from_config(config: Config) -> dict[NDArray]:
    """
    STORE VARIABLES FROM NETCDF OCE FILE, based on config file.
    """
    logging.info("Reading OCE data ...")
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
    assert (
        temperature.shape == salinity.shape
    ), "Temperature and saliniy do not have the same shape."
    output_dict = {
        "temperature": temperature,
        "saliniy": salinity,
        "depth": oce_dataset[oce_coords["depth"]],
        "latitude": oce_dataset[oce_coords["lat"]],
    }
    # close datasets
    oce_dataset.close()
    return output_dict

def read_bathy_from_config(config: Config) -> NDArray or float:
    """
    STORE SEA FLOOR DEPTH FROM NETCDF BATHYMETRY FILE, based on config file.
    """
    # Domain defined
    oce_dims = config.input.oce.dims
    oce_domain = {k: v for k, v in zip(oce_dims.values(), config.domain.values())}
    # Read bathy from netcdf
    if config.input.bathy.set_bottomdepth is False:
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
        seafloor_depth = bathy_dataset[bathy_vars["seafloor_depth"]]
        bathy_dataset.close()
    # Read seafloor depth set by the user
    else:
        seafloor_depth = config.input.bathy.set_bottomdepth
    return seafloor_depth


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

    @cli.switch(["-c", "--config"], str)
    def set_config(self, config_path: str = None):
        """
        Set config file.
        """
        QGBaroclinic.config = Config(os.path.normpath(config_path))

    def __init__(self, **domain: dict[list[float]]) -> None:
        """
        Class constructor, given region domain.
        """
        self.domain = domain

    def read(
        self, file: str, *variables: tuple[str], **domain: dict[list[float]]
    ) -> None:
        read = ncRead(file)
        if domain == {}:
            domain = self.domain
        vars = read.variables(*variables, **domain)
        return vars

    def __call__(
        self,
        pot_density: NDArray,
        depth: NDArray,
        latitude: float or NDArray,
        bottom_depth: float or NDArray,
        grid_step: float = 1.0,
        n_modes: int = 4,
    ) -> None:
        """
        Run algorithm.
        """

        # VERTICAL INTERPOLATION (default to 1m grid step)
        interpolation = Interpolation(depth, pot_density)
        (interp_potdens, depth_levels) = interpolation.apply_interpolation(
            -grid_step / 2, bottom_depth, grid_step
        )
        bv_freq = OceBaroclinicModes.compute_bruntvaisala_freq(
            depth_levels, interp_potdens, grid_step
        )
        bv_freq_smoothed = OceBaroclinicModes.smooth(
            bv_freq,
            depth,
            cutoff={0.0: 10.0, 100.0: 100.0},
            grid_step=grid_step,
        )
        rossby_rad, vert_structfunc = OceBaroclinicModes.compute_modes(
            bv_freq_smoothed, latitude, bottom_depth, grid_step, n_modes
        )
        self.rossby_rad = rossby_rad
        self.vert_structfunc = vert_structfunc

    @staticmethod
    def potential_density(
        temperature: NDArray,
        salinity: NDArray,
        depth: NDArray = None,
        insitu_temperature: bool = False,
        ref_pressure: float = 0,  # reference pressure [dbar]
    ) -> tuple[NDArray]:
        """
        Compute Density from Pot. Temperature & Salinity.
        """
        assert (
            temperature.shape == salinity.shape
        ), "Temperature and Salinity arrays have different shapes."
        if insitu_temperature and depth is not None:
            pot_temperature = EoS.potential_temperature(
                salinity,
                temperature,
                depth,
                ref_press=0,
            )
        elif insitu_temperature and depth is None:
            raise ValueError(
                "In order to compute Pot. Temperature, you should provide depth levels."
            )
        else:
            pot_temperature = temperature
        pot_density = EoS.compute_density(salinity, pot_temperature, ref_pressure)
        return pot_density

    @staticmethod
    def compute_bruntvaisala_freq(
        depth_levels: NDArray, interp_potdens: NDArray, grid_step: float = 1
    ) -> NDArray:
        """
        Compute Brunt-Vaisala frequency, given depth levels and pot. density.
        """

        bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(depth_levels, interp_potdens)
        print(bv_freq_sqrd)
        # RE-INTERPOLATING BRUNT-VAISALA FREQUENCY SQUARED FOR REMOVING NaNs and < 0 values.
        bv_freq_sqrd = BVfreq.rm_negvals(bv_freq_sqrd, grid_step=grid_step)
        bv_freq = np.sqrt(bv_freq_sqrd)
        return bv_freq

    @staticmethod
    def smooth(
        profile: NDArray,
        depth: NDArray,
        cutoff: dict[float],
        grid_step: float = 1,
        order: int = 3,
        axis: int = -1,
    ) -> None:
        cutoff_depths = cutoff.keys()
        cutoff_wavelengths = cutoff.values()
        # FILTERING BRUNT-VAISALA FREQUENCY PROFILE WITH A LOW-PASS FILTER.
        cutoff_levels = Utils.find_nearvals(depth, *cutoff_depths)
        filter = Filter(
            profile, grid_step=grid_step, type="lowpass", order=order, axis=axis
        )
        cutoff_dict = dict(zip(cutoff_levels, cutoff_wavelengths))
        profile_filtered = filter.apply_filter(cutoff_dict)
        return profile_filtered

    @staticmethod
    def compute_modes(
        bv_freq: NDArray,
        latitude: NDArray or float,
        bottom_depth: NDArray = None,
        grid_step: float = 1,
        n_modes: int = 4,
    ):
        """
        Compute baroclinic Modes and RossY radii.
        """

        # 1D input array
        if bv_freq.ndim == 1:
            # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
            # From config file, specify if structure of vertical velocity should be computed instead of Phi.
            baroclinicmodes = VerticalStructureEquation(
                bv_freq,
                mean_lat=np.nanmean(latitude),
                grid_step=grid_step,
                n_modes=n_modes,
                vertvel_method=False,
            )
            rossby_rad = baroclinicmodes.rossbyrad  # Rossby radius in [m]
            vert_structfunc = baroclinicmodes.vert_structfunc
        # 3D input array
        if bv_freq.ndim > 1 and bottom_depth is not None:
            rossby_rad, vert_structfunc = VerticalStructureEquation.multiprofiles(
                bv_freq,
                latitude,
                bottom_depth,
                grid_step=grid_step,
                n_modes=n_modes,
            )
        return rossby_rad, vert_structfunc

    def plot(self) -> None:
        pass
    def __init__(self, **domain: dict[list[float]]) -> None:
        """
        Class constructor, given experiment name.
        """
        self.domain = domain

    def main(self):
        # load config file
        try:
            self.config
        except AttributeError:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.toml",
            )
            QGBaroclinic.config = Config(config_path)
        # Execute default command if no command is provided
        if not self.nested_command:  # will be "None" if no sub-command follows
            print("No command given")
            return 1  # error exit code


@QGBaroclinic.subcommand("mean")
class Mean(cli.Application):
    """
    Mean values obtained from mean profiles in the user-defined region.
    """

    def compute_density(self) -> None:
        """
        Compute Density from Pot. Temperature & Salinity.
        """
        logging.info("Computing density ...")
        config = self.config
        ref_pressure = 0  # reference pressure [dbar]
        salinity = self.salinity
        temperature = self.temperature
        pot_density = OceBaroclinicModes.potential_density(
            temperature,
            salinity,
            self.depth,
            insitu_temperature=config.input.oce.insitu_temperature,
            ref_pressure=ref_pressure,
        )
        mean_region_potdensity = pot_density.mean(
            dim=["time", "longitude", "latitude"], skipna=True
        ).values
        # VERTICAL INTERPOLATION (1m grid step)
        logging.info("Vertically interpolating mean density ...")
        grid_step = self.grid_step
        interpolation = Interpolation(self.depth.values, mean_region_potdensity)
        (interp_dens, interp_depth) = interpolation.apply_interpolation(
            -grid_step / 2, self.mean_region_depth, grid_step
        )
        # Interface levels (1m grid step, shifted of 0.5 m respect to depth levels)
        self.interface_depth = interp_depth[:-1] + grid_step / 2
        self.interp_depth = interp_depth
        # New equispatial depth levels.
        depth_levels = interp_depth[1:-1]
        self.depth_levels = depth_levels
        # Save output Density dataset
        self.interp_dens = interp_dens
        logging.info("Saving density dataset to .nc file ...")
        density_dataset = self.ncwrite.create_dataset(
            dims="depth",
            coords={"depth": depth_levels},
            density=interp_dens[1:-1],
        )
        self.ncwrite.save(density_dataset)

    def compute_bruntvaisala_freq(self) -> None:
        """
        COMPUTE BRUNT-VAISALA FREQUENCY SQUARED
        """
        config = self.config
        logging.info("Computing Brunt-Vaisala frequency ...")
        bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(self.interp_depth, self.interp_dens)
        print(bv_freq_sqrd)
        # RE-INTERPOLATING BRUNT-VAISALA FREQUENCY SQUARED FOR REMOVING NaNs and < 0 values.
        logging.info("Post-processing Brunt-Vaisala frequency ...")
        grid_step = self.grid_step
        bv_freq_sqrd = BVfreq.rm_negvals(bv_freq_sqrd, grid_step=self.grid_step)
        print(bv_freq_sqrd)
        bv_freq = np.sqrt(bv_freq_sqrd)
        print(bv_freq)
        # FILTERING BRUNT-VAISALA FREQUENCY PROFILE WITH A LOW-PASS FILTER.
        if config.filter.filter:
            logging.info("Filtering Brunt-Vaisala frequency ...")
            cutoff_wavelength = config.filter.cutoff_wavelength
            cutoff_depth = config.filter.cutoff_depth
            try:
                assert len(cutoff_depth) == len(cutoff_wavelength)
            except AssertionError:
                raise ValueError(
                    "Filter cutoff in not correctly set: cutoff depths and wavelengths are not the same number."
                )
            cutoff_levels = Utils.find_nearvals(self.interface_depth, *cutoff_depth)
            filter = Filter(
                bv_freq,
                grid_step=grid_step,
                type="lowpass",
                order=config.filter.order,
            )
            cutoff_dict = dict(zip(cutoff_levels, cutoff_wavelength))
            self.bv_freq_filtered = filter.apply_filter(cutoff_dict)
        else:
            logging.info("Brunt-Vaisala frequency has not been filtered...")
            self.bv_freq_filtered = bv_freq
        # Save Brunt-Vaisala freq. dataset
        logging.info("Saving Brunt-vaisala freq dataset to .nc file ...")
        bvfreq_dataset = self.ncwrite.create_dataset(
            dims="depth_interface",
            coords={"depth_interface": self.interface_depth},
            bvfreq=self.bv_freq_filtered,
        )
        self.ncwrite.save(bvfreq_dataset)

    def compute_baroclinicmodes(self):
        """
        BAROCLINIC MODES & ROSSBY RADII
        """
        print(self.bv_freq_filtered)
        config = self.config
        logging.info("Computing baroclinic modes and Rossby radii ...")
        # N° of modes of motion considered (including the barotropic one).
        N_motion_modes = config.output.n_modes
        mean_lat = np.mean(self.latitude.values)
        self.mean_lat = mean_lat
        # Warning if the region is too near the equator.
        equator_threshold = 2.0
        lower_condition = (
            -equator_threshold < np.min(self.latitude.values) < equator_threshold
        )
        upper_condition = (
            -equator_threshold < np.max(self.latitude.values) < equator_threshold
        )
        if Utils.andor(lower_condition, upper_condition):
            warnings.warn(
                "The domain area is close to the equator: ! Rossby radii computation might be inaccurate !"
            )
        # Compute baroclinic Rossby radius and vert. struct. function Phi(z).
        # From config file, specify if structure of vertical velocity should be computed instead of Phi.
        baroclinicmodes = VerticalStructureEquation(
            self.bv_freq_filtered,
            mean_lat=mean_lat,
            grid_step=self.grid_step,
            n_modes=N_motion_modes,
            vertvel_method=config.output.vertvel_method,
        )
        rossby_rad = baroclinicmodes.rossbyrad  # Rossby radius in [km]
        self.baroclinicmodes = baroclinicmodes
        logging.info(f"Rossby radii [km]: {rossby_rad/1000}")
        print(f"Rossby radii [km]: {rossby_rad/1000}")
        # WRITE RESULTS ON OUTPUT FILE
        logging.info("Saving Baroclinic Modes dataset to .nc file ...")
        modes_of_motion = np.arange(0, config.output.n_modes)
        # Rossby radii dataset
        ncwrite = self.ncwrite
        radii_dataset = ncwrite.create_dataset(
            dims="mode", coords={"mode": modes_of_motion}, rossbyrad=rossby_rad
        )
        # Vertical structure function dataset
        if config.output.vertvel_method:
            vert_struct_func_dataset = ncwrite.create_dataset(
                dims=["depth_interface", "mode"],
                coords={
                    "mode": modes_of_motion,
                    "depth_interface": self.interface_depth,
                },
                structfunc=baroclinicmodes.structfunc,
            )
        else:
            vert_struct_func_dataset = ncwrite.create_dataset(
                dims=["depth", "mode"],
                coords={"mode": modes_of_motion, "depth": self.depth_levels},
                structfunc=baroclinicmodes.structfunc,
            )
        ncwrite.save(radii_dataset, vert_struct_func_dataset)

    def main(self):
        """
        Running the resolution algorithm.
        """
        # Get starting time
        start_time = time.time()
        # RUN MAIN()
        try:
            # SET OUTPUT FILE (& LOGGING)
            print(QGBaroclinic.config.input)
            exit()
            ncwrite = ncWrite(
                config.output.folder_path, filename=config.output.filename, logfile=True
            )
            logging.info(f"Using config file {config.config_file}")

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


if __name__ == "__main__":
    QGBaroclinic.run()
