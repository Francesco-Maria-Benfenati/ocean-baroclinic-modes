import numpy as np
import scipy as sp
from numpy.typing import NDArray


class Filter:
    """
    This class is for filtering Brunt-Vaisala freq. profiles.
    """

    @staticmethod
    def lowpass(
        profile: NDArray,
        grid_step: int = 1,
        cutoff_wavelength: float = 100.0,
        order: int = 2,
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
        profile_filtered = sp.signal.filtfilt(b, a, profile)
        return profile_filtered


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    degrad = 2 * np.pi / 360
    x = np.linspace(0, 1000, 1000)
    wave1 = np.cos(x * degrad)
    noise = np.random.rand(1000)
    wave_filtered = Filter.lowpass(wave1 + noise)
    plt.figure()
    plt.plot(wave1 + noise, -x)
    plt.plot(wave_filtered, -x, "k--")
    plt.show()
    plt.close()
