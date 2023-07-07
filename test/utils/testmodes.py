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
        Theoretical result as in Table 1, from LaCasce (2012).

        NOTE: the values are treated to be used also in DIMENSIONAL case.
        """

        # coefficient for dimensionalization of eigenvalues/gamma values
        dim_coeff = ((np.sqrt(self.bvsqrd_0)*self.H)/(self.coriolis_param))
        match profile:
            case "const":
                eigenvals = np.arange(5) * np.pi
                return eigenvals/dim_coeff
            case "exp2":
                gamma =  np.array([0, 4.9107, 9.9072, 14.8875, 19.8628])/2
                return gamma/dim_coeff
            case "exp10":
                gamma = (
                    np.array([0, 2.7565, 5.9590, 9.1492, 12.3340])/2
                )
                return gamma/dim_coeff
            case "lindzen":
                gamma = np.array([0, 1.4214, 2.7506, 4.0950, 5.4448])/2
                return gamma/dim_coeff
            case _:
                return None              
    
    def from_eigenvals_to_gamma(self, eigenval: float, alpha: float) -> float:
        """
        Given coeff alpha and eigenvalues, compute gamma.
        """
        dim_coeff = ((np.sqrt(self.bvsqrd_0)*self.H)/(self.coriolis_param))
        gamma = np.sqrt(eigenval)*dim_coeff/alpha
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
                    sp.special.yv(0, 2*gamma[i])
                    * sp.special.jv(1, 2*gamma[i] * np.exp(alpha * depth / 2))
                    - sp.special.jv(0, 2*gamma[i])
                    * sp.special.yv(1, 2*gamma[i] * np.exp(alpha * depth / 2))
                )
            )
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
            struct_func[:, i] = A * (
                sp.special.yn(1, 2*gamma[i])
                * sp.special.jv(0, 2*gamma[i] * np.sqrt(1 - D * depth))
                - sp.special.jv(1, 2*gamma[i])
                * sp.special.yn(0, 2*gamma[i] * np.sqrt(1 - D * depth))
            )
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

    H = 3000
    z = -np.linspace(0, H, H + 1)
    f = BaroclinicModes.coriolis_param(42.6)
    S = 1.0e+04*np.exp(10*z/H)
    #S = np.exp(10 * z / H)
    #S = np.ones(H+1)*1e+04
    # S = 4.6e-04/BaroclinicModes.coriolis_param(45)**2 * np.ones(H+1)
    testmodes = TestModes(1000, 1e-04, H, 1e-04)
    profile = testmodes.expon_profile(10)
    eigenvals, structfunc = BaroclinicModes.compute_baroclinicmodes(S, grid_step = 1)
    gamma = testmodes.LaCasce_eigenvals("const")
    print(testmodes.from_eigenvals_to_gamma(eigenvals, 10), gamma)
    depth = -np.linspace(0, testmodes.H, testmodes.n_levels)
