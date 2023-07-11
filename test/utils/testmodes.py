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
        n_levels: int,
        bvsqrd_0: float = 1,
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

        self.bvsqrd_0 = bvsqrd_0
        self.n_levels = n_levels
        self.H = abs(H)
        self.coriolis_param = coriolis_param

    def LaCasce_eigenvals(self, profile: str) -> NDArray:
        """
        Theoretical result as in Table 1, from LaCasce (2012). The 0-th eigenval is added.

        NOTE: the values are treated to be used also in DIMENSIONAL case.
        """

        # coefficient for dimensionalization of eigenvalues/gamma values
        dim_coeff = (np.sqrt(self.bvsqrd_0) * self.H) / (self.coriolis_param)
        match profile:
            case "const":
                eigenvals = np.arange(5) * np.pi
                return eigenvals / dim_coeff
            case "exp2":
                gamma = np.array([0, 4.9107, 9.9072, 14.8875, 19.8628]) / 2
                return gamma / dim_coeff
            case "exp10":
                gamma = np.array([0, 2.7565, 5.9590, 9.1492, 12.3340]) / 2
                return gamma / dim_coeff
            case "lindzen":
                gamma = np.array([0, 1.4214, 2.7506, 4.0950, 5.4448]) / 2
                return gamma / dim_coeff
            case _:
                return None

    def from_eigenvals_to_gamma(self, eigenval: float, alpha: float) -> float:
        """
        Given coeff alpha and eigenvalues, compute gamma.
        """
        dim_coeff = (np.sqrt(self.bvsqrd_0) * self.H) / (self.coriolis_param)
        gamma = np.sqrt(eigenval) * dim_coeff / alpha
        return gamma

    def const_profile(self, n_modes: int = 5) -> NDArray:
        """
        Computes vertical structure func. for N^2(z) = const.
        """

        n_levels = self.n_levels
        depth = -np.linspace(0, self.H, n_levels)  # new equally spaced depth grid
        # Eigenvalues
        eigenvals = (
            self.LaCasce_eigenvals("const") * self.bvsqrd_0 / (self.coriolis_param**2)
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
        n_levels = self.n_levels
        alpha /= self.H
        depth = -np.linspace(0, self.H, n_levels)  # new equally spaced depth grid
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
        n_levels = self.n_levels
        depth = -np.linspace(0, self.H, n_levels)  # new equally spaced depth grid
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
    import matplotlib.pyplot as plt

    try:
        from ...src.model.baroclinicmodes import BaroclinicModes
    except ImportError:
        import sys, os

        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from src.model.baroclinicmodes import BaroclinicModes

    H = 1
    n = 5000
    z = -np.linspace(0, H, n)
    f = BaroclinicModes.coriolis_param(42.6)
    #S = 1.0e04 * np.exp(10 * z / H)
    Z = -np.array(
        [
            0,
            5,
            8,
            10,
            12,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
        ]
    )
    N = np.array(
        [
            0.13,
            0.25,
            0.45,
            0.5,
            0.45,
            0.3,
            0.2,
            0.17,
            0.15,
            0.14,
            0.13,
            0.12,
            0.1,
            0.09,
            0.08,
            0.07,
            0.06,
            0.065,
            0.05,
            0.0475,
            0.045,
            0.0425,
            0.04,
        ]
    )
    f = sp.interpolate.interp1d(Z, N, fill_value="extrapolate")
    interp_N = f(z)
    #S = (interp_N) ** 2  # /(1e-04)**2
    #S = np.exp(2*z/H)
    #s_param = np.asarray(S, dtype=np.float64)
    #s_param = np.exp(10*z/H)
    s_param = np.ones(n)
    # S = 1/(1-10*z/H)
    # S = np.exp(10 * z / H)
    # S = np.ones(H+1)*1e+04
    # S = 4.6e-04/BaroclinicModes.coriolis_param(45)**2 * np.ones(H+1)
    testmodes = TestModes(n, 1, H, 1)
    #  profile = testmodes.expon_profile(10)
    depth = -np.linspace(0, testmodes.H, testmodes.n_levels)
    # print([sp.integrate.trapezoid(profile[:,i]*profile[:,i-1], x=-depth) for i in range(1,profile.shape[1])])
    grid_step = H/n
    eigenvals, structfunc = BaroclinicModes.compute_baroclinicmodes(
        s_param, grid_step= H/n, generalized_method=True
    )
    print((np.sqrt(eigenvals) - testmodes.LaCasce_eigenvals("const"))/testmodes.LaCasce_eigenvals("const"))
    # print([sp.integrate.trapezoid(structfunc[100:,i]*structfunc[100:,i], x=-z[100:]/H) for i in range(1,structfunc.shape[1])])
    # gamma = testmodes.LaCasce_eigenvals("const")
    #   print(testmodes.from_eigenvals_to_gamma(eigenvals, 10), gamma)
    # print(profile[-1,:]-profile[-2,:], structfunc[0,:])
    plt.figure(figsize=(7, 8))
    plt.grid(visible=True)
    # plt.plot(profile, depth, "k--")
    plt.plot(structfunc[:, :4], z)
    plt.show()
    plt.figure(figsize=(7, 8))
    plt.plot(interp_N, z)
    #  plt.show()
    plt.close()
