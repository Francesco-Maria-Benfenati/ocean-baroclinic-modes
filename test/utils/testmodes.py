import numpy as np
import scipy as sp
from numpy.typing import NDArray

try:
    from .utils import Utils
except ImportError:
    from utils import Utils


class TestModes(Utils):
    """
    This Class contains analytical solutions from LaCasce (2012), for constant, expontential and Lindzen stratification profiles.
    """

    def __init__(
        self,
        bvfreq_0: float = 1,
        H: float = 1,
        coriolis_param: float = 1,
    ) -> None:
        """
        Class Constructor.

        Args:
            n_levels (int): number of vertical levels
            bvsqrd_0 (float): N^2(z=0), surface value of Brunt-Vaisala frequency profile. Defaults to 1.
            H: bottom depth (can be both positive or negative). Defaults to 1.
            coriolis_param: coriolis_parameter. Defaults to 1.
        """

        self.bvfreq_0 = bvfreq_0
        self.H = abs(H)
        self.coriolis_param = coriolis_param

    def LaCasce_eigenvals(self, profile: str) -> NDArray:
        """
        Theoretical result as in Table 1, from LaCasce (2012). The 0-th eigenval is added.

        NOTE: the values are treated to be used also in DIMENSIONAL case.
        """

        # coefficient for dimensionalization of eigenvalues/gamma values
        dim_coeff = (self.coriolis_param) / (self.bvfreq_0 * self.H)
        match profile:
            case "const":
                eigenvals = np.arange(5) * np.pi
                return eigenvals * dim_coeff
            case "exp2":
                gamma = np.array([0, 4.9107, 9.9072, 14.8875, 19.8628]) / 2
                return gamma * dim_coeff
            case "exp10":
                gamma = np.array([0, 2.7565, 5.9590, 9.1492, 12.3340]) / 2
                return gamma * dim_coeff
            case "lindzen":
                gamma = np.array([0, 1.4214, 2.7506, 4.0950, 5.4448]) / 2
                return gamma * dim_coeff
            case _:
                return None

    def from_gamma_to_eigenvals(self, profile: str, alpha: float) -> float:
        """
        Given coeff alpha and gamma, compute eigenvals.
        """

        gamma = self.LaCasce_eigenvals(profile)
        eigenval = gamma * alpha
        return eigenval**2

    def const_profile(self, n_modes: int = 5) -> NDArray:
        """
        Computes vertical structure func. for N^2(z) = const.
        """

        depth = -np.arange(0.5, self.H, 1)  # new equally spaced depth grid
        n_levels = len(depth)
        # Eigenvalues
        eigenvals = (
            self.LaCasce_eigenvals("const")
            * (self.bvfreq_0) ** 2
            / (self.coriolis_param**2)
        )
        # Theoretical solution.
        struct_func = np.empty([n_levels, n_modes])
        # amplitude
        A = 1
        for i in range(n_modes):
            struct_func[:, i] = A * np.cos(eigenvals[i] * depth / self.H)
            norm = np.sqrt(
                sp.integrate.trapezoid(struct_func[:, i] ** 2, x=-depth) / self.H
            )
            struct_func[:, i] /= norm
        return struct_func

    def expon_profile(self, alpha: float) -> NDArray:
        """
        Computes vertical structure function for exponential N^2(z) = N0^2 * exp(alpha*z).
        """

        # NOTE: Here, gamma = 2*gamma in comparison to LaCasce notation.
        match alpha:
            case 2.0:
                gamma = self.LaCasce_eigenvals("exp2")
            case 10.0:
                gamma = self.LaCasce_eigenvals("exp10")
            case _:
                return None
        n_modes = len(gamma)
        A = 1  # amplitude
        alpha /= self.H
        depth = -np.arange(0.5, self.H, 1)  # new equally spaced depth grid
        n_levels = len(depth)
        # Theoretical Vertical Structure Function Phi(z)
        struct_func = np.empty([n_levels, n_modes])
        for i in range(n_modes):
            struct_func[:, i] = (
                A
                * np.exp(alpha * depth / 2)
                * (
                    sp.special.yn(0, 2 * gamma[i])
                    * sp.special.jn(1, 2 * gamma[i] * np.exp(alpha * depth / 2))
                    - sp.special.jn(0, 2 * gamma[i])
                    * sp.special.yn(1, 2 * gamma[i] * np.exp(alpha * depth / 2))
                )
            )
            norm = np.sqrt(
                sp.integrate.trapezoid(struct_func[:, i] ** 2, x=-depth) / self.H
            )
            struct_func[:, i] /= norm
        return struct_func

    def lindzen_profile(self, D: float) -> NDArray:
        """
        Computes vertical structure function for lindzen profile N^2(z) = N0^2/(1-D*z/H).
        """

        # NOTE: Here, gamma = 2*gamma in comparison to LaCasce notation.
        match D:
            case 10.0:
                gamma = self.LaCasce_eigenvals("lindzen")
            case _:
                return None
        n_modes = len(gamma)
        A = 1  # amplitude
        D /= self.H
        depth = -np.arange(0.5, self.H, 1)  # new equally spaced depth grid
        n_levels = len(depth)
        # Theoretical Vertical Structure Function Phi(z)
        struct_func = np.empty([n_levels, n_modes])
        for i in range(n_modes):
            struct_func[:, i] = -A * (
                sp.special.yn(1, 2 * gamma[i])
                * sp.special.jn(0, 2 * gamma[i] * np.sqrt(1 - D * depth))
                - sp.special.jn(1, 2 * gamma[i])
                * sp.special.yn(0, 2 * gamma[i] * np.sqrt(1 - D * depth))
            )
            norm = np.sqrt(
                sp.integrate.trapezoid(struct_func[:, i] ** 2, x=-depth) / self.H
            )
            struct_func[:, i] /= norm
        return struct_func


if __name__ == "__main__":
    import math

    # Test with LaCasce values (2012). Dimensional Case.
    bvfreq_0 = 2 * 1e-02  # surface BV freq 0.02 1/s
    shallow = True
    if shallow:
        mean_depth = 1 * 1e+02  # mean depth 100 m
    else:
        mean_depth = 3 * 1e+03  # mean depth 3000 km
    f_0 = 1e-04  # coriolis parameter 0.0001
    grid_step = 1  # dz = 1 m
    n_levels = int(mean_depth + 1)
    depth = -np.linspace(0, mean_depth, n_levels)

    testmodes = TestModes(bvfreq_0, mean_depth, f_0)
    # CONST case
    bvfreq_const = np.ones(n_levels) * bvfreq_0
    const_profile = Utils(bvfreq_const**2, f_0, grid_step)
    eigenvals_const = testmodes.LaCasce_eigenvals("const")
    rel_error_const = const_profile.compare_eigenvals(eigenvals_const**2)

    # EXPON case alpha = 2/H, alpha = 10/H, lindzen profile.
    bvfreqsqrd_expon_2 = (bvfreq_0**2) * np.exp(2 * depth / mean_depth)
    bvfreqsqrd_expon_10 = (bvfreq_0**2) * np.exp(10 * depth / mean_depth)
    bvfreqsqrd_lindzen = (bvfreq_0**2) / (1 - 10 * depth / mean_depth)

    expon2_profile = Utils(bvfreqsqrd_expon_2, f_0, grid_step)
    expon10_profile = Utils(bvfreqsqrd_expon_10, f_0, grid_step)
    lindzen_profile = Utils(bvfreqsqrd_lindzen, f_0, grid_step)

    eigenvals_expon2 = testmodes.from_gamma_to_eigenvals("exp2", 2)
    eigenvals_expon10 = testmodes.from_gamma_to_eigenvals("exp10", 10)
    eigenvals_lindzen = testmodes.from_gamma_to_eigenvals("lindzen", 10)

    rel_error_expon2 = expon2_profile.compare_eigenvals(eigenvals_expon2)
    rel_error_expon10 = expon10_profile.compare_eigenvals(eigenvals_expon10)
    rel_error_lindzen = lindzen_profile.compare_eigenvals(eigenvals_lindzen)

    print(f"For Const case, expected eigenvalues are [km^-1]: {eigenvals_const*1000}")
    print(f"Relative error respect to LaCasce const profile: {rel_error_const}")
    print(
        f"For alpha = 2/H case, expected eigenvalues are [km^-1]: {np.sqrt(eigenvals_expon2)*1000}"
    )
    print(f"Relative error respect to LaCasce const profile: {rel_error_expon2}")
    print(
        f"For alpha = 10/H, expected eigenvalues are [km^-1]: {np.sqrt(eigenvals_expon10)*1000}"
    )
    print(f"Relative error respect to LaCasce const profile: {rel_error_expon10}")
    print(
        f"For Lindzen case, expected eigenvalues are [km^-1]: {np.sqrt(eigenvals_lindzen)*1000}"
    )
    print(f"Relative error respect to LaCasce const profile: {rel_error_lindzen}")

    # Plots of Struct Func
    depth_levels = -np.arange(0.5, mean_depth, 1)
    run_plots = False
    if run_plots:
        # Plot Struct Func comparison
        const_profile.plot_struct_func(testmodes.const_profile(), depth_levels)
        expon2_profile.plot_struct_func(testmodes.expon_profile(2), depth_levels)
        expon10_profile.plot_struct_func(testmodes.expon_profile(10), depth_levels)
        lindzen_profile.plot_struct_func(testmodes.lindzen_profile(10), depth_levels)
        # Plot Relative Error
        const_profile.plot_error(testmodes.const_profile(), depth_levels)
        expon2_profile.plot_error(testmodes.expon_profile(2), depth_levels)
        expon10_profile.plot_error(testmodes.expon_profile(10), depth_levels)
        lindzen_profile.plot_error(testmodes.lindzen_profile(10), depth_levels)
