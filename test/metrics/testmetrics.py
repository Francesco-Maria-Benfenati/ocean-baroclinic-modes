import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.typing import NDArray

try:
    from ...qgbaroclinic.solve.verticalstructureequation import VerticalStructureEquation
    from ...qgbaroclinic.tool.interpolation import Interpolation
except ImportError:
    import sys, os

    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from qgbaroclinic.solve.verticalstructureequation import VerticalStructureEquation
    from qgbaroclinic.tool.interpolation import Interpolation


class TestMetrics:
    """
    This Class contains utils for testings.
    """

    def __init__(
        self,
        bvfreq_sqrd: NDArray,
        coriolis_param: float,
        grid_step: float,
        depth: NDArray = None,
    ) -> None:
        """
        Computing eigenvalues/eigenvectors.
        """

        if depth is not None:
            interpolation = Interpolation(depth, bvfreq_sqrd / (coriolis_param**2))
            z_0 = depth[0] - grid_step / 2
            z_N = depth[-1] + grid_step
            s_param = interpolation.apply_interpolation(z_0, z_N, grid_step)[0]
        else:
            s_param = bvfreq_sqrd / (coriolis_param**2)
        print("According to numpy, input values have precision:", s_param.dtype)
        self.eigenvals, self.structfunc = VerticalStructureEquation.compute_baroclinicmodes(
            s_param, grid_step
        )
        print("After using scipy, output values have precision:", self.eigenvals.dtype)

    def compare_eigenvals(self, ref_eigenvals: NDArray) -> NDArray:
        """
        Compare eigenvalues with reference values and return error.
        """

        rel_error = TestMetrics.rel_error(self.eigenvals, ref_eigenvals)
        return rel_error

    def compare_structfunc(self, ref_structfunc: NDArray) -> NDArray:
        """
        Compare structure funcs with reference ones and return error.
        """

        rel_error = TestMetrics.rel_error(self.eigenvals, ref_structfunc)
        return rel_error

    @staticmethod
    def rel_error(a: float, b: float) -> float:
        """
        Return relative error of a respect to b.
        """

        return np.abs(a - b) / np.abs(b)

    def plot_bvfreq(self, bvfreq: NDArray, depth: NDArray, legend: str = None) -> None:
        """
        Plot Brunt-Vaisala Freq profile.
        """

        fig, ax = plt.subplots(figsize=(7, 8))
        ax.plot(bvfreq, depth, "k")
        ax.legend([legend], fontsize=12)
        ax.grid(True)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel(
            r"Brunt-Vaisala frequency squared [$1/s^2$]", fontsize=14, labelpad=20
        )
        ax.set_ylabel("depth (m)", fontsize=14)
        plt.show()
        plt.close()

    def plot_struct_func(self, ref_struct_func: NDArray, depth: NDArray) -> None:
        """
        Plot Structure Function compared to expected values.
        """

        struct_func = self.structfunc
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.plot(np.nan, 0, "k--")
        ax.plot(struct_func[:, :4], depth)
        ax.plot(struct_func[:, :4], depth, "k--")
        ax.legend(
            ["Analytical sol", "Mode 0", "Mode 1", "Mode 2", "Mode 3"],
            fontsize=12,
            loc="lower right",
        )
        ax.grid(True)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel(
            "Normalized Vertical Structure Function", fontsize=14, labelpad=20
        )
        ax.set_ylabel("depth (m)", fontsize=14)
        plt.show()
        plt.close()

    @staticmethod
    def error_eigenvecs(a: float, b: float) -> float:
        """
        Return relative error of a respect to b.
        """

        return np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2)

    def plot_error(self, ref_struct_func: NDArray, depth: NDArray) -> None:
        """
        Plot error profile.
        """

        error = TestMetrics.error_eigenvecs(self.structfunc, ref_struct_func)

        fig, ax = plt.subplots(figsize=(7, 8))
        ax.plot(np.nan, 0)
        ax.plot(error[:, 1:4], depth)
        ax.grid(True)
        ax.legend(
            ["Mode 0", "Mode 1", "Mode 2", "Mode 3"],
            fontsize=12,
            loc="lower right",
        )
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlim(
            -0.05,
        )
        ax.set_xlabel(
            "Vertical Structure Function \n Relative Error", fontsize=14, labelpad=20
        )
        ax.set_ylabel("depth (m)", fontsize=14)
        plt.show()
        plt.close()


if __name__ == "__main__":
    pass
