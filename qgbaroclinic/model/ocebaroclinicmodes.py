import sys, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Union

try:
    from ..read import ncRead
    from ..tool import Interpolation, EoS, BVfreq, Filter, Utils
    from ..solve import VerticalStructureEquation
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from read import ncRead
    from tool import Interpolation, EoS, BVfreq, Filter, Utils
    from solve import VerticalStructureEquation


class OceBaroclinicModes:
    """
    This class contains high-level functions for Running the QGBaroclinic software.
    """

    def __init__(self, **domain: dict[list[float]]) -> None:
        """
        Class constructor, given (spatial and temporal) domain.

        Args:
            domain (dict, OPTIONAL):
                keys are "coordinate name" and values are [min,max].
                Used to crop the datasets.
        """

        self.domain = domain

    def read(
        self, file: str, *variables: tuple[str], **domain: dict[list[float]]
    ) -> tuple[xr.Variable]:
        """
        Read NetCDF file(s).

        Args:
            file (str): path to netcdf file(s).
            variables (str): name(s) of variable(s) in netcdf file.
            domain (dict, OPTIONAL): dict["coordinate name": [min,max]] to crop the dataset.

        Returns:
            tuple[xr.Variable]: tuple of xarray Variables.
        """

        read = ncRead(file)
        # Default domain is the one provided during instance creation.
        if domain == {}:
            domain = self.domain
        # Extract Variable(s).
        vars = read.variables(*variables, **domain)
        return vars

    @staticmethod
    def potential_density(
        temperature: Union[NDArray, xr.Variable],
        salinity: Union[NDArray, xr.Variable],
        depth: Union[NDArray, xr.Variable, None] = None,
        insitu_temperature: bool = False,
        ref_pressure: float = 0,  # reference pressure [dbar]
    ) -> Union[NDArray, xr.Variable]:
        """
        Compute Potential Density from (Pot.) Temperature & Salinity.

        Args:
            temperature (Union[NDArray, xr.Variable]):
                Temperature or Potential Temperature
            salinity (Union[NDArray, xr.Variable]):
                Salinity
            depth (Union[NDArray, xr.Variable, None], optional):
                Depth grid on which Temperature and Salinity are defined.
                It is used only if "insitu_temperature" = True. Defaults to None.
            insitu_temperature (bool, optional):
                If temperature provided is "in-situ", instead of "potential". Defaults to False.
            ref_pressure (float, optional):
                Reference pressure at sea surface in [dbars].
                It is used only if "insitu_temperature" = True. Defaults to 0.

        Raises:
            ValueError:
                If insitu_temperature = True and depth argument is not provided.

        Returns:
            Union[NDArray, xr.Variable]: Potential Temperature
        """

        assert (
            temperature.shape == salinity.shape
        ), "Temperature and Salinity arrays have different shapes."
        # Compute Potential Temperature from in-situ Temperature.
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
        # Compute Potential Density
        pot_density = EoS.compute_density(salinity, pot_temperature, ref_pressure)
        return pot_density

    def __call__(
        self,
        pot_density: NDArray,
        depth: NDArray,
        latitude: float,
        seafloor_depth: float,
        grid_step: float = 1.0,
        n_modes: int = 4,
    ) -> tuple[NDArray]:
        """
        Compute Ocean QG Baroclinic Modes and Rossby Radii from Pot. Density.

        Args:
            pot_density (NDArray):
                Potential Density (1D).
            depth (NDArray):
                depth (m) array on which Pot. Density is defined (1D).
            latitude (float):
                latitude (°N) at which the Pot. Density profile has been collected.
            seafloor_depth (float):
                seafloor depth (meters) at location where the Pot. Density profile has been collected.
            grid_step (float, optional):
                Grid step (meters) used for interpolating the Pot. Density profile. Defaults to 1 m.
            n_modes (int, optional):
                Number of modes of motion which should be returned as output. Defaults to 4.

        Returns:
            tuple[NDArray]:
                Rossby Radius and NORMALIZED Vertical Structure Function for "n_modes" modes of motion.
        """

        # Vertical Interpolation (and extrapolation) to levels.
        interpolation = Interpolation(depth, pot_density)
        (interp_potdens, depth_levels) = interpolation.apply_interpolation(
            -grid_step / 2, seafloor_depth, grid_step
        )
        # Compute Brunt-Vaisala Frequency at the interfaces.
        bv_freq = OceBaroclinicModes.compute_bruntvaisala_freq(
            interp_potdens, depth_levels, grid_step
        )
        # Smoothing BV frequency using a low-pass filter.
        bv_freq_smoothed = OceBaroclinicModes.smooth(
            bv_freq,
            depth,
            cutoff={0.0: 10.0, 100.0: 100.0},
            grid_step=grid_step,
        )
        # Compute the Baroclinic Modes of motion.
        rossby_rad, vert_structfunc = OceBaroclinicModes.compute_modes(
            bv_freq_smoothed, latitude, grid_step, n_modes
        )
        # Store array of depth_levels
        n_depth_levels = vert_structfunc.shape[0]
        depth = np.array([i * grid_step for i in range(n_depth_levels)])
        self.depth = depth
        # Store and return Rossby Radii and Vertical Structure Functions.
        self.rossby_rad = rossby_rad
        self.vert_structfunc = vert_structfunc
        self.n_modes = n_modes
        return rossby_rad, vert_structfunc

    @staticmethod
    def compute_bruntvaisala_freq(
        pot_density: NDArray, depth: NDArray, grid_step: float = 1
    ) -> NDArray:
        """
        Compute Brunt-Vaisala frequency, given depth levels and Pot. Density.

        Args:
            pot_density (NDArray):
                1D Potential Density
            depth (NDArray):
                1D depth array on which Pot. Density is defined.
            grid_step (float, optional):
                grid step used for the numerical grid. Defaults to 1.

        Returns:
            NDArray: Brunt-Vaisala Frequency profile.
        """

        # Compute BV freq squared.
        bv_freq_sqrd = BVfreq.compute_bvfreq_sqrd(depth, pot_density)
        # Re-interpolating BV freq. squared for removing NaNs and < 0 values.
        bv_freq_sqrd = BVfreq.rm_negvals(bv_freq_sqrd, grid_step=grid_step)
        # Compute square root of squared profile.
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
    ) -> NDArray:
        """
        Smooth BV frequency using a low-pass filter.

        Args:
            profile (NDArray): 1D BV freq. profile
            depth (NDArray): depth grid on which the profile is defined.
            cutoff (dict[float]): cutoff value.
            grid_step (float, optional): numerical grid step. Defaults to 1.
            order (int, optional): filter order. Defaults to 3.
            axis (int, optional):
                axis on which filter should be applied. Defaults to -1.

        Returns:
            NDArray: smoothed BV frequency profile.
        """

        # Cutoff depths, wavelengths and levels.
        cutoff_depths = cutoff.keys()
        cutoff_wavelengths = cutoff.values()
        cutoff_levels = Utils.find_nearvals(depth, *cutoff_depths)
        # Apply low-pass filter.
        filter = Filter(
            profile, grid_step=grid_step, type="lowpass", order=order, axis=axis
        )
        cutoff_dict = dict(zip(cutoff_levels, cutoff_wavelengths))
        # Return filtered profile.
        profile_filtered = filter.apply_filter(cutoff_dict)
        return profile_filtered

    @staticmethod
    def compute_modes(
        bv_freq: NDArray,
        latitude: float,
        grid_step: float = 1,
        n_modes: int = 4,
    ) -> tuple[NDArray]:
        """
        Compute baroclinic Modes and Rossy radii.

        Args:
            bv_freq (NDArray): 1D Brunt-Vaisala freq. profile.
            latitude (float): latitude at which profile is collected.
            grid_step (float, optional): numerical grid step. Defaults to 1.
            n_modes (int, optional): number of modes to be computed. Defaults to 4.

        Returns:
            tuple[NDArray]: Rossby Radii and NORMALIZED Vertical Structure Function profiles.
        """

        # Compute baroclinic Rossby radius and vert. struct. function .
        baroclinicmodes = VerticalStructureEquation(
            bv_freq,
            mean_lat=np.nanmean(latitude),
            grid_step=grid_step,
            n_modes=n_modes,
            vertvel_method=False,
        )
        rossby_rad = baroclinicmodes.rossbyrad  # Rossby Radius in [m]
        vert_structfunc = (
            baroclinicmodes.vert_structfunc
        )  # NORMALIZED Vertical Structure Function
        return rossby_rad, vert_structfunc

    def plot(self, fig_path: str = None) -> None:
        """
        Plot NORMALIZED Vertical Structure Function vs Depth.

        Args:
            fig_path (str, optional):
                path to figure, to be provided if it should be saved.
                Defaults to None.
        """

        # Create fig and plot profile.
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.plot(self.vert_structfunc, -self.depth)
        labels = ["Barotropic Mode 0"]
        for i in range(1, self.n_modes):
            labels.append(
                rf"Mode {i} : $R_{i}$ = {self.rossby_rad[i] / 1000:.1f} km"
            )
        ax.legend(
            labels,
            fontsize=12,
            loc="lower right",
        )
        ax.plot(self.depth * 0.0, -self.depth, "k--", linewidth=0.7)
        ax.grid(True)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel(
            f"Normalized Vertical Structure Function",
            fontsize=14,
            labelpad=20,
        )
        ax.set_ylabel("depth (m)", fontsize=14)
        # Save Figure if a path is provided. Else, show.
        if fig_path is not None:
            fig.savefig(fig_path)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    pass
