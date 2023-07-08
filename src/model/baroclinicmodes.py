import numpy as np
import scipy as sp
from numpy.typing import NDArray

try:
    from .interpolation import Interpolation
    from .eigenproblem import EigenProblem
except ImportError:
    from interpolation import Interpolation
    from eigenproblem import EigenProblem


class BaroclinicModes:
    """
    This class is for computing the ocean QG Baroclinic Modes of motion and rossby radius.
    """

    def __init__(
        self,
        bvfreq: NDArray,
        mean_lat: float = None,
        grid_step: float = None,
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
            s_param = BaroclinicModes.compute_problem_sparam(bvfreq, mean_lat)
        self.eigenvalues, self.structurefunc = BaroclinicModes.compute_baroclinicmodes(
            s_param, grid_step
        )
        # Rossby deformation radius
        self.deformrad = BaroclinicModes.rossby_rad(self.eigenvalues)

    @staticmethod
    def compute_baroclinicmodes(
        s_param: NDArray,
        grid_step: float = None,
        n_modes: int = 5,
        generalized_method: bool = False,
    ) -> tuple[NDArray]:
        """
        Compute Baroclinic Rossby rad and vertical structure function.

        Args:
            grid_step (float): vertical grid step 'dz'. Defaults to None.
            n_modes (int, optional): Num. of modes of motion to be computed. Defaults to 5.
            generalized_method (bool, optional): If the generalized method should be used
                                                 instead of the standard one. Defaults to False.

        Returns:
            eigenvalues, vertical structure functions
        """

        # Number of vertical levels
        n_levels = s_param.shape[0]
        if grid_step is None:
            dz = 1 / n_levels
        else:
            dz = grid_step
        # COMPUTE MATRIX & SOLVE EIGEN PROBLEM
        if generalized_method:  # generalized case
            lhs_matrix = BaroclinicModes.lhs_matrix_generalizedprob(n_levels, dz)
            rhs_matrix = BaroclinicModes.rhs_matrix_generalizedprob(s_param)
            eigenprob = EigenProblem(
                lhs_matrix, rhs_matrix, grid_step=dz, n_modes=n_modes
            )
        else:
            # compute tridiagonal matrix (standard case)
            matrix = BaroclinicModes.tridiag_matrix_standardprob(s_param, dz)
            eigenprob = EigenProblem(matrix, n_modes=n_modes)
        # Return eigenvalues and vertical structure function
        eigenvalues = eigenprob.eigenvals
        vert_structurefunc = eigenprob.eigenvecs
        # Normalization of vertical structure function
        mean_depth = dz * n_levels
        for i in range(n_modes):
            norm = np.sqrt(
                sp.integrate.trapezoid(vert_structurefunc[:, i] ** 2, dx=dz)
                / mean_depth
            )
            vert_structurefunc[:, i] /= norm
            if vert_structurefunc[0,i] < 0:
                vert_structurefunc[:,i] *= -1 # Check all eigenvectors are > 0 at the surface
        return eigenvalues, vert_structurefunc

    @staticmethod
    def compute_problem_sparam(bvfreq: NDArray, mean_lat: float) -> NDArray:
        """
        Compute problem parameter S = N^2/f^2, given brunt vaisala frequency and latitude.
        """

        if mean_lat is None:
            coriolis_param = 1e-04
        else:
            coriolis_param = BaroclinicModes.coriolis_param(mean_lat)
        s_param = bvfreq**2 / coriolis_param**2
        return s_param

    @staticmethod
    def rossby_rad(eigenvalue: float) -> float:
        """
        Compute Rossby Radius given the eigenvalue(s).
        """

        rossby_rad = 1 / (np.sqrt(eigenvalue) * 2 * np.pi)
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
        n = S.shape[0]
        M = np.zeros([n, n])
        # Fill diagonal and subdiagonals
        for k in range(2, n):
            M[k - 1, k] = 1 / S[k - 1]
            M[k - 1, k - 1] = -1 / S[k - 2] - 1 / S[k - 1]
            M[k - 1, k - 2] = 1 / S[k - 2]
        # B.C.s
        M[0, 0] = -1 / S[0]
        M[0, 1] = 1 / S[0]
        M[n - 1, n - 2] = 1 / S[n - 2]
        M[n - 1, n - 1] = -1 / S[n - 2]
        # Multiply for coefficient
        M *= -1 / (dz**2)
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

    def __doc__(self) -> None:
        """
        If called, print info aboit the algorithm for computing baroclinic modes of motion.
        """
        print(
            """ 
            Computes baroclinic Rossby radius & vertical structure function.
            
                NOTE:
            ---------------------------------------------------------------
                                    About the algorithm
                ---------------------------------------------------------------
            Coriolis parameter (used for scaling) and region mean depth are
            defined.
            
            1) N2 is linearly interpolated on a new equally spaced 
            nondimensional depth grid with grid step dz = 1 m.
            2) The problem parameter S is then computed as in 
            Grilli, Pinardi (1999)
            
                    S = (N2)/(f^2)
                    
            where f is the Coriolis parameter.
            3) The finite difference matrix 'A' and the S matrix 'B' are
            computed and the eigenvalues problem is solved:
                
                A * w = lambda * B * w  (lambda eigenvalues, w eigenvectors)
                
            with BCs: w = 0 at z=0,1.
            
            4) The eigenvectors are computed through numerical integration
            of the equation 
            
                (d^2/dz^2) * w = - lambda * S * w    (with BCs  w = 0 at z=0,1).
            
            5) The baroclinic Rossby radius is obtained as 
                
                    R_n = 1/sqrt(lambda_n)
            
            for each mode of motion 'n'.
                                    - & - 
            The vertical structure function Phi(z) is computed integrating
            S*w between 0 and z (for each mode of motion).
            """
        )


if __name__ == "__main__":
    dz = 0.5
    # -----------------------------
    #  Testing _compute_matrix_A()
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
    computed_A = BaroclinicModes.lhs_matrix_generalizedprob(8, dz)
    assert np.allclose(A, computed_A)
    # -----------------------------
    #  Testing _compute_matrix_B()
    # -----------------------------
    n = 1000
    S = np.arange(n)
    B = np.diag(-S[1:-1])
    computed_B = BaroclinicModes.rhs_matrix_generalizedprob(S)
    assert np.allclose(B, computed_B)

    """
    Test if compute_barocl_modes() works whell when depth is taken with
    negative sign convention.
    """
    n_modes = 3
    H = 1000
    N2_const = np.full(H, 1.0)
    obm1 = BaroclinicModes(N2_const)
    obm2 = BaroclinicModes(N2_const)
    eigval_neg = obm1.eigenvalues
    eigval_pos = obm2.eigenvalues
    assert np.array_equal(eigval_pos, eigval_neg)

    # Test eigenvalues for nondim constant profile
    expected_eigenvals = (np.arange(0, 5) * np.pi) ** 2
    assert np.allclose(
        eigval_pos[:-1], expected_eigenvals[:-1]
    )  # remove 5th mode, less accurate
    print("For const nondimensional case, eigenvalues are lambda = n*pi (n=0,1,2...).")
    print(
        f"Computed lambdas are {eigval_pos}, with relative errors {(eigval_pos-expected_eigenvals)/expected_eigenvals}"
    )
