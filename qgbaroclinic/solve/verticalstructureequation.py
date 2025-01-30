import sys, os
import numpy as np
import scipy as sp
from tqdm import tqdm
from numpy.typing import NDArray

try:
    from . import EigenProblem
except ImportError:
    from eigenproblem import EigenProblem


class VerticalStructureEquation:
    """
    This class is for solving the Vertical Structure Equation and
    computing the ocean QG Baroclinic Modes of motion and baroclinic rossby radii.
    """

    def __init__(
        self,
        bvfreq: NDArray,
        mean_lat: float = None,
        grid_step: float = None,
        n_modes: int = 5,
        vertvel_method: bool = False,
    ) -> None:
        """
        Constructor for OceBaroclModes objects

        Args:
            bvfreq (NDArray): brunt vaisala frequency profile
            mean_lat (float): region mean latitude. Defaults to None
        """

        if mean_lat is None:
            s_param = bvfreq**2
        else:
            s_param = VerticalStructureEquation.compute_problem_sparam(bvfreq, mean_lat)
        (
            self.eigenvals,
            self.vert_structfunc,
        ) = VerticalStructureEquation.compute_baroclinicmodes(
            s_param,
            grid_step,
            n_modes,
            vertvel_method=vertvel_method,
        )
        # Rossby deformation radius
        self.rossbyrad = VerticalStructureEquation.compute_rossby_rad(self.eigenvals)

    @staticmethod
    def compute_baroclinicmodes(
        s_param: NDArray,
        grid_step: float = None,
        n_modes: int = 5,
        vertvel_method: bool = False,
    ) -> tuple[NDArray]:
        """
        Compute Baroclinic Rossby rad and vertical structure function.

        Args:
            grid_step (float): vertical grid step 'dz'. Defaults to None.
            n_modes (int, optional): Num. of modes of motion to be computed. Defaults to 5.
            vertvel_method (bool, optional): If the modal structure for vertical velocity method should be used
                                                 instead of the standard one. Defaults to False.

        Returns:
            eigenvalues, vertical structure function (or modal structure of vertical velocity,
                                                        depending on 'vertvel_method')
        """

        # Number of vertical levels
        n_levels = s_param.shape[0]
        if grid_step is None:
            dz = 1 / n_levels
        else:
            dz = grid_step
        # COMPUTE MATRIX & SOLVE EIGEN PROBLEM
        if vertvel_method:  # generalized case
            lhs_matrix = VerticalStructureEquation.lhs_matrix_generalizedprob(
                n_levels, dz
            )
            rhs_matrix = VerticalStructureEquation.rhs_matrix_generalizedprob(s_param)
            eigenprob = EigenProblem(
                lhs_matrix, rhs_matrix, grid_step=dz, n_modes=n_modes
            )
            # Add BC values to eigenvectors (w=0 where z = 0,-H)
            eigenprob.eigenvecs = np.insert(
                eigenprob.eigenvecs, (0, -1), np.zeros(n_modes), axis=0
            )
        else:
            # compute tridiagonal matrix (standard case)
            matrix = VerticalStructureEquation.tridiag_matrix_standardprob(s_param, dz)
            eigenprob = EigenProblem(matrix, n_modes=n_modes)
        # Return eigenvalues and vertical structure function
        eigenvalues = eigenprob.eigenvals
        vert_structfunc = eigenprob.eigenvecs
        if not vertvel_method:
            # Check sign
            vert_structfunc = VerticalStructureEquation.check_sign_eigenvectors(
                vert_structfunc
            )
        # Normalization of vertical structure function
        normalized_vert_structfunc = VerticalStructureEquation.normalize_eigenfunc(
            vert_structfunc, dz
        )
        return eigenvalues, normalized_vert_structfunc

    @staticmethod
    def normalize_eigenfunc(eigenfunc: NDArray, dz: float) -> NDArray:
        """
        Normalize eigenfunction(s) so that (1/H)* integral_0^H(funf*func dz) = 1.
        """

        H = dz * eigenfunc.shape[0]  # depth
        norm = np.sqrt(sp.integrate.trapezoid(eigenfunc**2, dx=dz, axis=0) / H)
        normalized_eigenfuncs = eigenfunc / norm
        return normalized_eigenfuncs

    @staticmethod
    def check_sign_eigenvectors(eigenvectors: NDArray) -> NDArray:
        """
        Check if the eigenvectors have correct sign.
        """

        for i in range(eigenvectors.shape[1]):
            if eigenvectors[0, i] < 0:
                eigenvectors[:, i] *= -1
        return eigenvectors

    @staticmethod
    def from_w_to_vertstructfunc(
        eigenvectors: NDArray, s_param: NDArray, dz: float
    ) -> NDArray:
        """
        Compute vertical structure function from eigenvectors (if using modal structure for vertical velocity method).
        """

        n_modes = eigenvectors.shape[1]
        # Define integration constant phi_0 = phi(z = 0) = 1 as BC.
        phi_barotropic = 1
        phi_0 = np.ones(n_modes) * phi_barotropic
        vert_structfunc = np.empty_like(eigenvectors)
        for i in range(n_modes):
            # Obtain Phi integrating eigenvectors * S.
            integral_argument = s_param * eigenvectors[:, i]
            for j in range(vert_structfunc.shape[0]):
                vert_structfunc[j, i] = (
                    sp.integrate.trapezoid(integral_argument[:j], dx=dz) + phi_0[i]
                )
        return vert_structfunc

    @staticmethod
    def compute_problem_sparam(bvfreq: NDArray, mean_lat: float) -> NDArray:
        """
        Compute problem parameter S = N^2/f^2, given brunt vaisala frequency and latitude.
        """

        if mean_lat is None:
            coriolis_param = 1e-04
        else:
            coriolis_param = VerticalStructureEquation.coriolis_param(mean_lat)
        s_param = bvfreq**2 / coriolis_param**2
        return s_param

    @staticmethod
    def compute_rossby_rad(eigenvalue: float) -> float:
        """
        Compute Rossby Radius given the eigenvalue(s).
        """

        rossby_rad = 1 / (np.sqrt(eigenvalue))
        return rossby_rad

    @staticmethod
    def coriolis_param(mean_lat: float) -> float:
        """
        Compute Coriolis parameter given the region mean latitude.
        """

        # earth angular velocity (1/s)
        earth_angvel = 7.29 * 1e-05
        # coriolis parameter (1/s)
        coriolis_param = 2 * earth_angvel * np.sin(mean_lat * np.pi / 180)
        return coriolis_param

    @staticmethod
    def tridiag_matrix_standardprob(s_param: NDArray, dz: float) -> NDArray:
        """
        Compute finite difference tridiagonal matrix corresponding to LHS matrix of STANDARD eigenproblem.

        :params s_param : S parameter BV freq**2 / coriolis_param**2
        :params dz : grid_step
        """

        S = s_param
        n = S.shape[0] - 1
        M = np.zeros([n, n])

        for k in range(2, n):
            M[k - 1, k] = 1 / S[k] * (-1)
            M[k - 1, k - 1] = (-1 / S[k - 1] - 1 / S[k]) * (-1)
            M[k - 1, k - 2] = 1 / S[k - 1] * (-1)
        # B.C.s
        M[0, 0] = -1 / S[1] * (-1)
        M[0, 1] = 1 / S[1] * (-1)
        M[-1, -2] = 1 / S[-2] * (-1)
        M[-1, -1] = -1 / S[-2] * (-1)

        M *= 1 / (dz**2)
        return M

    @staticmethod
    def lhs_matrix_generalizedprob(n: int, dz: float) -> NDArray:
        """
        Compute L.H.S. matrix in the GENERALIZED eigenvalues/eigenvectors problem.

        Arguments
        ---------
        n : 'int'
            number of vertical levels
        dz : 'float'
            vertical grid step

        Returns
        -------
        A : <class 'numpy.ndarray'>
            L.H.S. finite difference matrix

                        | 0      0    0     0   . . . . .  0 |
                        | 12    -24   12     0   . . . . . 0 |
                        | -1    16   -30    16  -1  0 . .  0 |
        A = (1/12dz^2) * | .     -1    16  -30   16  -1  .  0 |
                        | .        .      .   .     .    .   |
                        | .       0   -1    16  -30   16  -1 |
                        | .            0    0    12  -24  12 |
                        | 0      0    0     0   . . . . .  0 |

        where dz is the grid step (= 1m).
        Boundary Conditions are implemented setting the first and
        last lines of A = 0.
        """

        # Create matrix (only null values).
        A = np.zeros([n, n])  # finite difference matrix.
        # Fill matrix with values (centered finite difference).
        for i in range(3, n - 1):
            A[i - 1, i - 3] = -1 / (12 * dz**2)
            A[i - 1, i - 2] = 16 / (12 * dz**2)
            A[i - 1, i - 1] = -30 / (12 * dz**2)
            A[i - 1, i] = 16 / (12 * dz**2)
            A[i - 1, i + 1] = -1 / (12 * dz**2)
        # Set Boundary Conditions.
        A[1, 0] = 1 / (dz**2)
        A[1, 1] = -2 / (dz**2)
        A[1, 2] = 1 / (dz**2)
        A[n - 2, n - 1] = 1 / (dz**2)
        A[n - 2, n - 2] = -2 / (dz**2)
        A[n - 2, n - 3] = 1 / (dz**2)
        # Delete BCs rows which may be not considered.
        A = np.delete(A, n - 1, axis=0)
        A = np.delete(A, 0, axis=0)
        A = np.delete(A, n - 1, axis=1)
        A = np.delete(A, 0, axis=1)
        return A

    @staticmethod
    def rhs_matrix_generalizedprob(s_param: NDArray) -> NDArray:
        """
        Comput R.H.S. matrix in the GENERALIZED eigenvalues/eigenvectors problem.

        Arguments
        ---------
        S: <class 'numpy.ndarray'>
        problem parameter S = N2/f_0^2

        Returns
        -------
        B : <class 'numpy.ndarray'>
            R.H.S. S-depending matrix

                        | -S_0    0    0     . . .   0 |
                        | 0     -S_1   0     0   . . 0 |
                        | 0      0   -S_2    0 . . . 0 |
        B =              | .      0     0     .       . |
                        | .        .      .      .   . |
                        | .          0     0      -S_n |

        where dz is the grid step (= 1m).
        """

        S = s_param
        n = S.shape[0]
        # Create matrix (only null values).
        B = np.zeros([n, n])  # right side S-depending matrix.
        # Fill matrix with values (S).
        for i in range(3, n - 1):
            B[i - 1, i - 1] = -S[i - 1]
        # Set Boundary Conditions.
        B[0, 0] = -S[0]  # first row
        B[1, 1] = -S[1]
        B[n - 1, n - 1] = -S[n - 1]  # last row
        B[n - 2, n - 2] = -S[n - 2]  # last row
        # Delete BCs rows which may be not considered.
        B = np.delete(B, n - 1, axis=0)
        B = np.delete(B, 0, axis=0)
        B = np.delete(B, n - 1, axis=1)
        B = np.delete(B, 0, axis=1)
        return B


if __name__ == "__main__":
    dz = 0.5
    # -----------------------------
    #  Testing _compute_matrix_A() of generalized problem
    # -----------------------------
    A = (1 / (12 * dz**2)) * np.array(
        [
            [-24, 12, 0, 0, 0, 0],
            [16, -30, 16, -1, 0, 0],
            [-1, 16, -30, 16, -1, 0],
            [0, -1, 16, -30, 16, -1],
            [0, 0, -1, 16, -30, 16],
            [0, 0, 0, 0, 12, -24],
        ]
    )
    computed_A = VerticalStructureEquation.lhs_matrix_generalizedprob(8, dz)
    assert np.allclose(A, computed_A)
    # -----------------------------
    #  Testing _compute_matrix_B() of generalized problem
    # -----------------------------
    n = 1000
    S = np.arange(n)
    B = np.diag(-S[1:-1])
    computed_B = VerticalStructureEquation.rhs_matrix_generalizedprob(S)
    assert np.allclose(B, computed_B)

    # Test eigenvalues for nondim constant profile
    n_modes = 3
    H = 2000
    N2_const = np.full(H, 1.0)
    obm = VerticalStructureEquation(N2_const)
    eigval = obm.eigenvals
    expected_eigenvals = (np.arange(0, 5) * np.pi) ** 2
    print("For const nondimensional case, eigenvalues are lambda = n*pi (n=0,1,2...).")
    print(
        f"Computed lambdas are {eigval}, with relative errors {(eigval-expected_eigenvals)/expected_eigenvals}"
    )
    print(eigval - expected_eigenvals, (1 / 2000))
    # Test modal structure for vertical velocity method
    (
        eigvals_generalized_case,
        vert_structfunc,
    ) = VerticalStructureEquation.compute_baroclinicmodes(N2_const, vertvel_method=True)
    print("On the other hand, solving generalized leads to relative error:")
    print((eigvals_generalized_case - expected_eigenvals) / expected_eigenvals)

    # Test tridiag matrix
    new_s_param = np.ones(10)
    new_s_param[0], new_s_param[-1] = 0, 0
    new_tridiag = VerticalStructureEquation.tridiag_matrix_standardprob(new_s_param, 1)
    print("Tridiagonal matrix is of type:\n", new_tridiag)
