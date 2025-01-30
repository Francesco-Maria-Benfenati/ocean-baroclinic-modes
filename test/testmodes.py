import numpy as np
import scipy as sp
import warnings
from numpy.typing import NDArray

warnings.simplefilter("ignore", RuntimeWarning)

try:
    from .metrics import TestMetrics
except ImportError:
    from metrics import TestMetrics


class TestModes(TestMetrics):
    """
    This Class contains analytical solutions from LaCasce (2012), for constant, exponential and Lindzen stratification profiles.
    """

    def __init__(
        self,
        bvfreq_0: float = 1,
        H: float = 1,
        coriolis_param: float = 1,
        grid_step: float = 1,
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
        self.grid_step = grid_step
        self.dim_coeff = (self.coriolis_param) / (self.bvfreq_0 * self.H)

    def LaCasce_eigenvals(self, profile: str) -> NDArray:
        """
        Theoretical result as in Table 1, from LaCasce (2012). The 0-th eigenval is added.

        NOTE: the values are treated to be used also in DIMENSIONAL case.
        """

        # coefficient for dimensionalization of eigenvalues/gamma values
        match profile:
            case "const":
                eigenvals = np.arange(5) * np.pi
                return (eigenvals * self.dim_coeff) ** 2
            case "exp2":
                gamma = np.array([0, 4.9107, 9.9072, 14.8875, 19.8628]) / 2
                return (gamma * self.dim_coeff) ** 2
            case "exp10":
                gamma = np.array([0, 2.7565, 5.9590, 9.1492, 12.3340]) / 2
                return (gamma * self.dim_coeff) ** 2
            case "lindzen":
                gamma = np.array([0, 1.4214, 2.7506, 4.0950, 5.4448]) / 2
                return (gamma * self.dim_coeff) ** 2
            case _:
                return None

    def from_gamma_to_eigenvals(self, profile: str, alpha: float) -> float:
        """
        Given coeff alpha and gamma, compute eigenvals.
        """

        gamma = self.LaCasce_eigenvals(profile)
        eigenval = gamma * (alpha**2)
        return eigenval

    def const_profile(self, n_modes: int = 5) -> NDArray:
        """
        Computes vertical structure func. for N^2(z) = const.
        """

        depth = -np.arange(
            self.grid_step / 2, self.H, self.grid_step
        )  # new equally spaced depth grid
        max_depth = np.max(np.abs(depth))
        n_levels = len(depth)
        # Non-dimensional eigenvalues
        eigenvals = np.sqrt(self.LaCasce_eigenvals("const")) / self.dim_coeff
        # Theoretical solution.
        struct_func = np.empty([n_levels, n_modes])
        # amplitude
        A = 1
        for i in range(n_modes):
            struct_func[:, i] = A * np.cos(eigenvals[i] * depth / max_depth)
            norm = np.sqrt(
                sp.integrate.trapezoid(struct_func[:, i] ** 2, x=-depth) / max_depth
            )
            struct_func[:, i] /= norm
        return struct_func

    def expon_profile(self, alpha: float) -> NDArray:
        """
        Computes vertical structure function for exponential N^2(z) = N0^2 * exp(alpha*z).
        """

        # Non-dimensional gamma
        # NOTE: Here, gamma = 2*gamma in comparison to LaCasce notation.
        match alpha:
            case 2.0:
                gamma = np.sqrt(self.LaCasce_eigenvals("exp2")) / self.dim_coeff
            case 10.0:
                gamma = np.sqrt(self.LaCasce_eigenvals("exp10")) / self.dim_coeff
            case _:
                return None
        n_modes = len(gamma)
        A = 1  # amplitude
        depth = -np.arange(
            self.grid_step / 2, self.H, self.grid_step
        )  # new equally spaced depth grid
        max_depth = np.max(np.abs(depth))
        alpha /= max_depth
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
                sp.integrate.trapezoid(struct_func[:, i] ** 2, x=-depth) / max_depth
            )
            struct_func[:, i] /= norm
        return struct_func

    def lindzen_profile(self, D: float) -> NDArray:
        """
        Computes vertical structure function for lindzen profile N^2(z) = N0^2/(1-D*z/H).
        """

        # Non-dimensional gamma
        # NOTE: Here, gamma = 2*gamma in comparison to LaCasce notation.
        match D:
            case 10.0:
                gamma = np.sqrt(self.LaCasce_eigenvals("lindzen")) / self.dim_coeff
            case _:
                return None
        n_modes = len(gamma)
        A = 1  # amplitude
        depth = -np.arange(
            self.grid_step / 2, self.H, self.grid_step
        )  # new equally spaced depth grid
        max_depth = np.max(np.abs(depth))
        D /= max_depth
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
                sp.integrate.trapezoid(struct_func[:, i] ** 2, x=-depth) / max_depth
            )
            struct_func[:, i] /= norm
        return struct_func


if __name__ == "__main__":
    import math

    # Test with LaCasce values (2012). Dimensional Case.
    bvfreq_0 = 2 * 1e-02  # surface BV freq 0.02 1/s
    shallow = False
    if shallow:
        mean_depth = 1 * 1e02  # mean depth 100 m
    else:
        mean_depth = 1 * 3e03  # mean depth 3000 km
    f_0 = 1e-04  # coriolis parameter 0.0001
    grid_step = 0.3  # [m]
    interfaces = -np.arange(0, mean_depth + grid_step / 2, grid_step)
    n_interfaces = len(interfaces)

    testmodes = TestModes(bvfreq_0, mean_depth, f_0, grid_step)
    # CONST case
    bvfreq_const = np.ones(n_interfaces) * bvfreq_0
    print("Running CONST case ...")
    const_profile = TestMetrics(bvfreq_const**2, f_0, grid_step)
    eigenvals_const = testmodes.LaCasce_eigenvals("const")
    rel_error_const = const_profile.compare_eigenvals(eigenvals_const)

    # EXPON case alpha = 2/H, alpha = 10/H, lindzen profile.
    bvfreqsqrd_expon_2 = (bvfreq_0**2) * np.exp(2 * interfaces / mean_depth)
    bvfreqsqrd_expon_10 = np.array(
        (bvfreq_0**2) * np.exp(10 * interfaces / mean_depth), dtype=np.float64
    )
    bvfreqsqrd_lindzen = (bvfreq_0**2) / (1 - 10 * interfaces / mean_depth)

    print("Running EXPON 2/H case ...")
    expon2_profile = TestMetrics(bvfreqsqrd_expon_2, f_0, grid_step)
    print("Running EXPON 10/H case ...")
    expon10_profile = TestMetrics(bvfreqsqrd_expon_10, f_0, grid_step)
    print("Running LINDZEN case ...")
    lindzen_profile = TestMetrics(bvfreqsqrd_lindzen, f_0, grid_step)

    eigenvals_expon2 = testmodes.from_gamma_to_eigenvals("exp2", 2)
    eigenvals_expon10 = testmodes.from_gamma_to_eigenvals("exp10", 10)
    eigenvals_lindzen = testmodes.from_gamma_to_eigenvals("lindzen", 10)

    rel_error_expon2 = expon2_profile.compare_eigenvals(eigenvals_expon2)
    rel_error_expon10 = expon10_profile.compare_eigenvals(eigenvals_expon10)
    rel_error_lindzen = lindzen_profile.compare_eigenvals(eigenvals_lindzen)

    print(
        f"For Const case, expected rossby radii are [km]: {(1/(np.sqrt(eigenvals_const))/1000)}"
    )
    print(f"Our result is: {1/np.sqrt(const_profile.eigenvals)/1000}")
    print(f"Relative error respect to LaCasce const profile: {rel_error_const}")
    print(
        f"Absolute error respect to LaCasce: {np.abs(1/(np.sqrt(eigenvals_const))/1000-1/np.sqrt(const_profile.eigenvals)/1000)}"
    )
    print(
        f"For alpha = 2/H case, expected rossby radii are [km]: {1/np.sqrt(eigenvals_expon2)/1000}"
    )
    print(f"Our result is: {1/np.sqrt(expon2_profile.eigenvals)/1000}")
    print(f"Relative error respect to LaCasce expon 2/H profile: {rel_error_expon2}")
    print(
        f"Absolute error respect to LaCasce: {np.abs((1/np.sqrt(expon2_profile.eigenvals)/1000)-(1/np.sqrt(eigenvals_expon2)/1000))}"
    )
    print(
        f"For alpha = 10/H, expected rossby radii are [km]: {1/np.sqrt(eigenvals_expon10)/1000}"
    )
    print(f"Our result is: {1/np.sqrt(expon10_profile.eigenvals)/1000}")
    print(f"Relative error respect to LaCasce expon 10/H profile: {rel_error_expon10}")
    print(
        f"Absolute error respect to LaCasce: {np.abs((1/np.sqrt(expon10_profile.eigenvals)/1000)-(1/np.sqrt(eigenvals_expon10)/1000))}"
    )
    print(
        f"For Lindzen case, expected rossby radii are [km]: {1/np.sqrt(eigenvals_lindzen)/1000}"
    )
    print(f"Our result is: {1/np.sqrt(lindzen_profile.eigenvals)/1000}")
    print(f"Relative error respect to LaCasce Lindzen profile: {rel_error_lindzen}")
    print(
        f"Absolute error respect to LaCasce: {np.abs((1/np.sqrt(lindzen_profile.eigenvals)/1000)-(1/np.sqrt(eigenvals_lindzen)/1000))}"
    )

    # Plots of Struct Func
    depth_levels = -np.arange(grid_step / 2, mean_depth, grid_step)

    run_plots = True
    if run_plots:
        # Plot Struct Func comparison
        const_profile.plot_struct_func(
            testmodes.const_profile(), depth_levels, title="constant"
        )
        expon2_profile.plot_struct_func(
            testmodes.expon_profile(2), depth_levels, title="exp 2/H"
        )
        expon10_profile.plot_struct_func(
            testmodes.expon_profile(10), depth_levels, title="exp 10/H"
        )
        lindzen_profile.plot_struct_func(
            testmodes.lindzen_profile(10), depth_levels, title="Lindzen"
        )
        # Plot Relative Error
        const_profile.plot_error(
            testmodes.const_profile(), depth_levels, title="constant"
        )
        expon2_profile.plot_error(
            testmodes.expon_profile(2), depth_levels, title="exp 2/H"
        )
        expon10_profile.plot_error(
            testmodes.expon_profile(10), depth_levels, title="exp 10/H"
        )
        lindzen_profile.plot_error(
            testmodes.lindzen_profile(10), depth_levels, title="Lindzen"
        )

    print("TEST CONCLUDED SUCCESSFULLY")
