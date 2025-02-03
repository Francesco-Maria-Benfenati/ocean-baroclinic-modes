import numpy as np
import scipy as sp
from numpy.typing import NDArray


class Filter:
    """
    This class is for filtering Brunt-Vaisala freq. profiles.
    """

    def __init__(
        self,
        profile: NDArray,
        grid_step: float = 1.0,
        type: str = "lowpass",
        order=2,
        axis=-1,
    ):
        """
        Set up filter configuration (grid step, type, order)
        """
        self.type = type
        self.filter = getattr(Filter, type)
        self.grid_step = grid_step
        self.order = order
        self.profile = profile
        self.axis = axis

    def apply_filter(self, cutoff_dict: dict[float] = None) -> NDArray:
        """
        Apply different filters to a profile, depending on the depth.

        :params : cutoff_dict = dict of cutoff wavelengths,
                   where keys are indeces corresponding to depth levels.
        """
        depth_levels = np.array(list(cutoff_dict.keys()), dtype=int)
        cutoff_wavelegths = np.array(list(cutoff_dict.values()), dtype=float)
        filtered_profile = np.copy(self.profile)
        for i in range(len(depth_levels)):
            cutoff_slice = cutoff_wavelegths[i]
            filtered_profile[depth_levels[i] :] = self.filter(
                self.profile, self.grid_step, cutoff_slice, self.order, self.axis
            )[depth_levels[i] :]
        return filtered_profile

    @staticmethod
    def lowpass(
        profile: NDArray,
        grid_step: float = 1.0,
        cutoff_wavelength: float = 100.0,
        order: int = 2,
        axis=-1,
    ):
        """
        Low Pass filter.

        Args:
            profile (NDArray): vertical profile
            grid_step (int, optional): profile grid step, considered as the sampling frequency.
            cutoff_wavelength (float, optional): cutoff wavelength (in m). Defaults to 100 m.
            order (int, optional): filter order. Defaults to 2.
        """
        cutoff_frequency = 1 / cutoff_wavelength
        b, a = sp.signal.butter(
            N=order, Wn=cutoff_frequency, btype="low", analog=False, fs=grid_step
        )
        profile_filtered = sp.signal.filtfilt(b, a, profile, axis=axis)
        return profile_filtered


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    degrad = 2 * np.pi / 360
    x = np.linspace(0, 1000, 1000)
    wave = np.cos(x * degrad)
    # wave[900:] = np.nan
    noise = np.random.rand(1000)
    wave_filtered = Filter.lowpass(wave + noise)
    filter = Filter(wave + noise)
    wave_filtered_1 = filter.apply_filter({0: 10, 100: 100, 500: 1000})
    wave_filtered_2 = filter.apply_filter({0: 10})
    wave_filtered_3 = filter.apply_filter({100: 100})
    wave_filtered_4 = filter.apply_filter({500: 1000})
    plt.figure(1)
    plt.plot(wave + noise, -x, "y-")
    plt.plot(wave_filtered, -x, "k--")
    plt.plot(wave_filtered_1, -x, "r-")
    plt.plot(wave_filtered_2, -x, "k.")
    plt.plot(wave_filtered_3, -x, "g.")
    plt.plot(wave_filtered_4, -x, "c.")
    plt.show()
    plt.close()

    try:
        filter = Filter(0, "lowpas")
    except AttributeError:
        pass
