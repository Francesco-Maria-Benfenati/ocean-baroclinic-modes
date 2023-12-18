import numpy as np
import scipy as sp
from numpy.typing import NDArray

try:
    from .interpolation import Interpolation
except ImportError:
    from tool.interpolation import Interpolation


class BVfreq:
    """
    This Class is for computing the Brunt-Vaisala frequency of a density profile.
    """

    def __init__(self, depth: NDArray, density: NDArray) -> None:
        """
        Constructor for computing BVfreq, if depth provided are not interface levels.
        """

        self.depth = depth
        self.density = density
        self.bv_freq = np.sqrt(self.compute_bvfreq_sqrd(depth, density))

    @staticmethod
    def compute_bvfreq_sqrd(depth: NDArray, density: NDArray) -> NDArray:
        """
        Compute Brunt-Vaisala frequency from depth and density.

        Arguments
        ---------
        depth : <class 'numpy.ndarray'>
            depth  [m]
        mean_density : <class 'numpy.ndarray'>
            mean density vertical profile [kg/(m^3)],
            defined on z grid
        Returns
        -------
        bvfreq : <class 'numpy.ndarray'>
            Brunt-Vaisala frequency [(cycles/s)]

        The Brunt-Vaisala frequency N is computed as in Grilli, Pinardi
        (1999) 'Le cause dinamiche della stratificazione verticale nel
        mediterraneo'

        N = (- g/rho_0 * ∂rho_s/∂z)^1/2  with g = 9.806 m/s^2, z<0

        where rho_0 is the reference density.
        Depths are at integer levels 1, 2, 3, 4 ... N
        ** while derivatives are computed at interfaces 1/2, ... k+1/2, ... N+1/2 **
        """

        # Take absolute value of depth with neg sign.
        # Only if values are all positive or all negative.
        if np.all(np.sign(depth) < 0) or np.all(np.sign(depth) > 0):
            depth = np.abs(depth)
        # Defining value of gravitational acceleration.
        g = 9.806  # (m/s^2)
        # Defining value of reference density rho_0.
        rho_0 = 1025.0  # (kg/m^3)
        # rho = density[..., :-1]
        # Compute Brunt-Vaisala frequency
        dz = depth[..., 1:] - depth[..., :-1]
        density_diff = density[..., 1:] - density[..., :-1]
        bvfreq_sqrd = (g / rho_0) * density_diff / dz
        return bvfreq_sqrd

    @staticmethod
    def rm_negvals(bv_freq_sqrd: NDArray) -> NDArray:
        """
        Remove negative values and re-interpolate the profile.
        """

        # Remove possible negative values
        new_arr = np.copy(bv_freq_sqrd)
        new_arr[np.where(new_arr < 0)] = np.nan
        arr_len = bv_freq_sqrd.shape[0]
        depth = np.arange(0, arr_len)
        interpolation = Interpolation(depth, new_arr)
        (interp_arr,) = interpolation.apply_interpolation(0, arr_len, 1)
        return interp_arr


if __name__ == "__main__":
    # Density profiles
    H = 2000  # bottom depth in [m]
    dz = 1  # grid step [m]
    depth = np.arange(-dz / 2, H + (3 / 2) * dz, step=dz)  # depth < 0
    interface = np.arange(0, H + dz, step=dz)
    print("depth levels at: ", depth)
    print("interfaces at: ", interface)
    # NOTE: while density is defined on depth levels, BVfreq is defined on interface levels.
    len_z = len(depth)
    len_int = len(interface)
    print(f"we have {len_z} depth levels while {len_int} interface levels.")
    print(
        "NOTE: the first and last depth levels are used for computing derivatives at z = 0, -H"
    )
    rho_0 = 1025  # kg/m^3
    # Case CONST rho(z)=rho_0 --> N^2 = 0
    const_density = np.full([len_z], rho_0)
    expected_bvfreqsqrd_const = np.full([len_int], 0.0)
    # Case LINEAR rho(z) = a * z/H + rho_0, z < 0 --> N^2 = g*a/(H*rho_0)
    coeff = 10  # kg/m^3
    g = 9.806  # (m/s^2)
    linear_density = coeff * depth / H + rho_0
    expected_bvfreqsqrd_linear = ((g * coeff) / (H * rho_0)) * np.ones(len_int)
    # Case EXPON rho(z) = rho_0 * exp(a*z/H), z < 0 --> N^2 = g*a*exp(a*z)/H
    coeff = 0.01  # kg/m^3
    expon_density = rho_0 * np.exp(coeff * depth / H)
    expected_bvfreqsqrd_expon = g * coeff * (1 / H) * np.exp(coeff * interface / H)

    # Numerical results.
    result_const = BVfreq.compute_bvfreq_sqrd(depth, const_density)
    result_linear = BVfreq.compute_bvfreq_sqrd(depth, linear_density)
    result_expon = BVfreq.compute_bvfreq_sqrd(depth, expon_density)
    assert np.allclose(result_const, expected_bvfreqsqrd_const)
    assert np.allclose(result_linear, expected_bvfreqsqrd_linear, rtol=1e-02)
    print(
        f"For LINEAR case, relative error is of order {np.mean(np.abs(result_linear - expected_bvfreqsqrd_linear)/expected_bvfreqsqrd_linear)}"
    )
    # Remove surface and bottom values, as less accurate method is used there.
    assert np.allclose(result_expon, expected_bvfreqsqrd_expon, rtol=1e-02)
    print(
        f"For EXPON case, relative error is of order {np.mean((np.abs(result_expon - expected_bvfreqsqrd_expon)/expected_bvfreqsqrd_expon))}"
    )
    # Test if the computation works for arrays of density profiles
    density_array = np.array([const_density, linear_density, expon_density])
    print(f"Multidim density array has shape: {density_array.shape}")
    new_result = BVfreq.compute_bvfreq_sqrd(depth, density_array)
    print(f"Multidim BVfreq has shape: {new_result.shape}")
    expected_array = np.array(
        [
            expected_bvfreqsqrd_const,
            expected_bvfreqsqrd_linear,
            expected_bvfreqsqrd_expon,
        ]
    )
    assert np.allclose(new_result, expected_array, rtol=1e-02)
    print("Multidim result has expected accuracy.")
    # Test with 3D array
    density_array = np.array([const_density, linear_density, expon_density])
    density_3d = density_array.reshape((1, 3, len_z))
    assert np.array_equal(density_3d[0, 1, :], linear_density)
    assert np.array_equal(density_3d[0, 2, :], expon_density)
    bvfreq_3d = BVfreq.compute_bvfreq_sqrd(depth, density_3d)
    expected_3d = expected_array.reshape((1, 3, len_int))
    print(bvfreq_3d.shape)
    assert bvfreq_3d.shape == (1, 3, len_int)
    print("OK: computation works with 3D arrays.")
    assert np.allclose(bvfreq_3d, expected_3d[:-1], atol=1e-07)
    print(
        f"Due to interpolation, error is now: {np.mean((np.abs(bvfreq_3d[:,2,:] - expected_bvfreqsqrd_expon)/expected_bvfreqsqrd_expon))}"
    )

    test_arr = np.array([np.nan, np.nan, -1, 2, np.nan, 4, -5, 6])
    processed_arr = BVfreq.rm_negvals(test_arr)
    assert np.allclose(processed_arr, np.arange(-1, 7))
    print("OK post-processing for removing NaN and NEG values.")
    print("test array: ", test_arr)
