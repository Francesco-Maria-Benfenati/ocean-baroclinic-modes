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

    H = 100
    n = 1000
    z = -np.linspace(0, H, n)
    f_0 = BaroclinicModes.coriolis_param(42.6)
    print(f_0)
    #S = 1.0e04 * np.exp(10 * z / H)
     # Test Kundu (1975)
    Z = -np.array(
        [
            0,
            3,
            5,
            6,
            8,
            10,
            12,
            13,
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
    N_carnation = np.array(
        [
            0.135,
            0.18,
            0.25,
            0.47,
            0.5,
            0.47,
            0.40,
            0.36,
            0.3,
            0.22,
            0.185,
            0.165,
            0.145,
            0.13,
            0.12,
            0.105,
            0.095,
            0.085,
            0.077,
            0.069,
            0.067,
            0.06,
            0.05,
            0.045,
            0.0425,
            0.04,
        ]
    )
    N_db7 = np.array(
        [
            0.1,
            0.12,
            0.15,
            0.17,
            0.24,
            0.32,
            0.36,
            0.37,
            0.34,
            0.25,
            0.185,
            0.15,
            0.135,
            0.125,
            0.11,
            0.105,
            0.095,
            0.085,
            0.077,
            0.069,
            0.067,
            0.06,
            0.055,
            0.055,
            0.05,
            0.045,
        ]
    )
    f = sp.interpolate.interp1d(Z, N_carnation, fill_value="extrapolate")
    interp_N = f(z)
    S = (interp_N*2*np.pi/60)**2 /(f_0**2)
    #S = np.exp(2*z/H)
    s_param = np.asarray(S, dtype=np.float64)
    #s_param = np.exp(10*z/H)
    #s_param = np.ones(n)
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
        s_param, grid_step= H/n, generalized_method=False
    )
    print(f"Kundu eigenvalues [km^-1]: {1000*np.sqrt(eigenvals)}")
    # print([sp.integrate.trapezoid(structfunc[100:,i]*structfunc[100:,i], x=-z[100:]/H) for i in range(1,structfunc.shape[1])])
    # gamma = testmodes.LaCasce_eigenvals("const")
    #   print(testmodes.from_eigenvals_to_gamma(eigenvals, 10), gamma)
    # print(profile[-1,:]-profile[-2,:], structfunc[0,:])
    # plt.figure(figsize=(7, 8))
    # plt.grid(visible=True)
    # # plt.plot(profile, depth, "k--")
    # plt.plot(structfunc[:, :4], z[1:])
    # plt.show()
    # plt.figure(figsize=(7, 8))
    # plt.plot(interp_N, z)
    # #  plt.show()
    # plt.close()
    # BAROCLINIC MODES
    fig1, ax1 = plt.subplots(figsize=(7, 8))
    im1 = plt.imread("/mnt/d/Physics/ocean-baroclinic-modes/src/OBM/kundu_modes.png")
    im1 = ax1.imshow(im1, extent=[-50, 50, -100, 0])
    ax1.grid(visible=True)
    structfunc[:,3] *=-1
    ax1.plot(structfunc[:, :3] / 3 * 50, z[:-1], "r")
    ax1.plot(structfunc[:, 3] / 3 * 50, z[:-1], "r--")
    ax1.set_xlabel(
        "NORMALIZED MODE AMPLITUDE AT CARNATION,\n from Kundu, Allen, Smith (1975)",
        labelpad=15,
        fontsize=14,
    )
    ax1.set_yticks(np.linspace(-100, 0, 11))
    ax1.set_yticklabels(
        ["-100", None, "-80", None, "-60", None, "-40", None, "-20", None, "0"]
    )
    ax1.set_xticks(np.linspace(-50, 50, 7))
    ax1.set_xticklabels(["-3", "-2", "1", "0", "1", "2", "3"])
    ax1.yaxis.set_tick_params(width=1, length=7)
    ax1.xaxis.set_tick_params(width=1, length=7)
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    ax1.set_ylabel("DEPTH (m)", fontsize=14)
    ax1.set_xlim(-55, 55)
    ax1.set_ylim(-100, 0)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.spines["left"].set_position("center")
    ax1.spines["right"].set_color("none")
    ax1.legend(["Kundu (1975)", "numerical results REPLICA"])
    plt.show()
    plt.close()

