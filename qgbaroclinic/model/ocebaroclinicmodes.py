import sys, os
import warnings
import numpy as np
import xarray as xr
from numpy.typing import NDArray

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
    This class contains high-level functions for running the software.
    """

    def __init__(self, **domain: dict[list[float]]) -> None:
        """
        Class constructor, given region domain.
        """
        self.domain = domain

    def read(
        self, file: str, *variables: tuple[str], **domain: dict[list[float]]
    ) -> tuple[xr.Variable]:
        """
        Read NetCDF file.
        """

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
        n_depth_levels = vert_structfunc.shape[0]
        self.depth = np.array([i * grid_step for i in range(n_depth_levels)])

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
        """
        Smooth BV frequency.
        """

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


if __name__ == "__main__":
    pass
