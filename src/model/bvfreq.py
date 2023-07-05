import numpy as np
import scipy as sp
from numpy.typing import NDArray

try:
    from .interpolation import Interpolation
except ImportError:
    from interpolation import Interpolation


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
        interface, interp_density = Interpolation.interpolate_at_interface(
            depth, density
        )
        self.bvfreqsqrd = BVfreq.compute_bvfreq_sqrd(interface, interp_density)

    @staticmethod
    def compute_bvfreq_sqrd(depth: NDArray, density: NDArray) -> NDArray:
        """
        Compute Brunt-Vaisala frequency from depth and density.

        Arguments
        ---------
        depth : <class 'numpy.ndarray'>
            depth  [m]
        mean_density : <class 'numpy.ndarray'>
            mean potential density vertical profile [kg/(m^3)],
            defined on z grid

        Raises
        ------
        IndexError
            if input density has not same length as z along depth direction
                            - or -
            if input arrays are empty

        Returns
        -------
        bvfreq : <class 'numpy.ndarray'>
            Brunt-Vaisala frequency [(cycles/s)]

        The Brunt-Vaisala frequency N is computed as in Grilli, Pinardi
        (1999) 'Le cause dinamiche della stratificazione verticale nel
        mediterraneo'

        N = (- g/rho_0 * ∂rho_s/∂z)^1/2  with g = 9.806 m/s^2,

        where rho_0 is the reference density.
        Derivatives are computed at the interfaces z=0, 1/2, ... k+1/2, ... N+1/2
        """

        # Take absolute value of depth.
        depth = np.abs(depth)
        # Defining value of gravitational acceleration.
        g = 9.806  # (m/s^2)
        # Defining value of reference density rho_0.
        rho_0 = 1025.0  # (kg/m^3)
        # Compute Brunt-Vaisala frequency
        dz = depth[..., 1:] - depth[..., :-1]
        density_diff = density[..., 1:] - density[..., :-1]
        bvfreq_sqrd = (g / rho_0) * density_diff / dz
        return bvfreq_sqrd


if __name__ == "__main__":
    # Density profiles
    H = 1000
    dz = 1.0
    depth = -np.arange(0.5, H, step=dz)  # depth < 0
    interface = -np.arange(0, H + dz, step=dz)
    # NOTE: while density is defined on interface levels, BVfreq is defined on depth levels.
    len_z = len(depth)
    rho_0 = 1025  # kg/m^3
    # Case CONST rho(z)=rho_0 --> N^2 = 0
    const_density = np.full([len(interface)], rho_0)
    expected_bvfreqsqrd_const = np.full([len_z], 0.0)
    # Case LINEAR rho(z) = a * z/H + rho_0, z < 0 --> N^2 = g*a/(H*rho_0)
    coeff = 0.12  # kh/m^3
    g = 9.806  # (m/s^2)
    linear_density = -coeff * interface / H + rho_0
    expected_bvfreqsqrd_linear = ((g * coeff) / (H * rho_0)) * np.ones(len_z)
    # Case EXPON rho(z) = rho_0 * exp(a*z/H), z < 0 --> N^2 = g*a*exp(a*z)/H
    expon_density = rho_0 * np.exp(-coeff * interface / H)
    expected_bvfreqsqrd_expon = g * coeff * (1 / H) * np.exp(-coeff * depth / H)
    # Numerical results.
    result_const = BVfreq.compute_bvfreq_sqrd(interface, const_density)
    result_linear = BVfreq.compute_bvfreq_sqrd(interface, linear_density)
    result_expon = BVfreq.compute_bvfreq_sqrd(interface, expon_density)

    assert np.allclose(result_const, expected_bvfreqsqrd_const)
    assert np.allclose(result_linear, expected_bvfreqsqrd_linear)
    print(
        f"For LINEAR case, relative error is of order {np.mean(np.abs(result_linear - expected_bvfreqsqrd_linear)/expected_bvfreqsqrd_linear)}"
    )
    # Remove surface and bottom values, as less accurate method is used there.
    assert np.allclose(result_expon, expected_bvfreqsqrd_expon)
    print(
        f"For EXPON case, relative error is of order {np.mean((np.abs(result_expon - expected_bvfreqsqrd_expon)/expected_bvfreqsqrd_expon))}"
    )
    # Test if the computation works for arrays of density profiles
    density_array = np.array([const_density, linear_density, expon_density])
    print(f"Multidim density array has shape: {density_array.shape}")
    new_result = BVfreq.compute_bvfreq_sqrd(interface, density_array)
    print(f"Multidim BVfreq has shape: {new_result.shape}")
    expected_array = np.array(
        [
            expected_bvfreqsqrd_const,
            expected_bvfreqsqrd_linear,
            expected_bvfreqsqrd_expon,
        ]
    )
    assert np.allclose(new_result, expected_array)
    print("Multidim result has expected accuracy.")
    # Test with 3D array
    const_density = np.full([len(depth)], rho_0)
    linear_density = -coeff * depth / H + rho_0
    expon_density = rho_0 * np.exp(-coeff * depth / H)
    density_array = np.array([const_density, linear_density, expon_density])
    density_3d = density_array.reshape((1, 3, H))
    assert np.array_equal(density_3d[0, 1, :], linear_density)
    assert np.array_equal(density_3d[0, 2, :], expon_density)
    bvfreq_3d = BVfreq(depth, density_3d)
    expected_3d = expected_array.reshape((1, 3, H))
    assert bvfreq_3d.bvfreqsqrd.shape == (1, 3, H)
    print("OK: computation works with 3D arrays.")
    assert np.allclose(bvfreq_3d.bvfreqsqrd, expected_3d, atol=1e-07)
    print(
        f"Due to interpolation, error is now: {np.mean((np.abs(bvfreq_3d.bvfreqsqrd[:,2,:] - expected_bvfreqsqrd_expon)/expected_bvfreqsqrd_expon))}"
    )
