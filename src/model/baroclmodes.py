import numpy as np
import scipy as sp
from numpy.typing import NDArray

try:
    from .interpolation import Interpolation
    from .eigenproblem import EigenProblem
except ImportError:
    from interpolation import Interpolation
    from eigenproblem import EigenProblem


class BaroclModes:
    """
    This class is for computing the ocean Baroclinic Modes of motion and rossby radius.
    """

    def __init__(
        self,
        depth: NDArray,
        bvfreq_sqrd: NDArray,
        mean_lat: float = None,
        mean_depth: float = None,
    ) -> None:
        """
        Constructor for OceBaroclModes objects

        Args:
            depth (NDArray): region depth array
            bvfreq_sqrd (NDArray): brunt vaisala frequency squared profile
            mean_lat (float): region mean latitude. Defaults to None
            mean_depth (float, optional): region mean depth. Defaults to None
            n_modes (int, optional): Num. of baroclinic modes to be computed. Defaults to 4
        """

        self.depth = np.abs(depth)
        self.bvfreq_sqrd = bvfreq_sqrd
        if mean_lat is not None:
            self.coriolis_param = self.__coriolis_param(mean_lat)
        else:
            self.coriolis_param = 1.0e-04
        if mean_depth is not None:
            self.mean_depth = abs(mean_depth)
        else:
            self.mean_depth = np.max(self.depth)

    def compute_baroclmodes(
        self, dz: float = 1.0, n_modes: int = 4, generalized_method: bool = False
    ) -> tuple[NDArray]:
        """
        Compute Baroclinic Rossby rad and vertical structure function.

        Args:
            dz (float): vertical grid step. Defaults to 1 m.
            n_modes (int, optional): Num. of modes of motion to be computed. Defaults to 4.
            generalized_method (bool, optional): If the generalized method should be used
                                                 instead of the standard one. Defaults to False.

        Returns:
            rossby radii, vertical structure functions
        """

        interp = Interpolation(self.depth, self.bvfreq_sqrd)
        interp_bvfreq = interp.apply_interpolation(dz / 2, self.mean_depth, dz)[0]
        # Compute S(z) parameter.
        s_param = interp_bvfreq / self.coriolis_param**2
        # COMPUTE MATRIX & SOLVE EIGEN PROBLEM
        if generalized_method:  # generalized case
            lhs_matrix = self.lhs_matrix_generalizedprob(s_param.shape[0], dz)
            rhs_matrix = self.rhs_matrix_generalizedprob(s_param)
            eigenprob = EigenProblem(lhs_matrix, rhs_matrix, dz, n_modes)
            # add matrix to attributes
            self.matrix = {"lhs": lhs_matrix, "rhs": rhs_matrix}
        else:
            # compute tridiagonal matrix (standard case)
            matrix = self.tridiag_matrix_standardprob(s_param, dz)
            eigenprob = EigenProblem(matrix, n_modes=n_modes)
            self.matrix = matrix
        # Add eigenvals, verticalstructure function and Rossby rad as attributes
        self.eigenvals = np.sqrt(eigenprob.eigenvals)
        self.structfunc = eigenprob.eigenvecs
        self.rossbyrad = BaroclModes.compute_rossby_rad(self.eigenvals)
        # Add S param as attribute
        self.s_param = s_param
        return self.rossbyrad, self.structfunc

    @staticmethod
    def compute_rossby_rad(eigenvalue: float) -> float:
        """
        Compute Rossby Radius given the eigenvalue(s).
        """
        rossby_rad = 1 / (eigenvalue * 2 * np.pi)
        return rossby_rad

    def __coriolis_param(self, mean_lat: float) -> float:
        """
        Compute Coriolis parameter given the region mean latitude.
        """

        # earth angular velocity (1/s)
        earth_angvel = 7.29 * 1e-05
        # coriolis parameter (1/s)
        coriolis_param = 2 * earth_angvel * np.sin(mean_lat * np.pi / 180)
        return coriolis_param

    def tridiag_matrix_standardprob(self, s_param: NDArray, dz: float) -> NDArray:
        """
        Compute tridiagonal matrix corresponding to LHS matrix of STANDARD eigenproblem.

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

    def lhs_matrix_generalizedprob(self, n: int, dz: float) -> NDArray:
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

    def rhs_matrix_generalizedprob(self, s_param: NDArray) -> NDArray:
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

    def __doc__(self):
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
    obm = BaroclModes(0, 0)
    obm.__doc__()
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
    computed_A = obm.lhs_matrix_generalizedprob(8, dz)
    assert np.allclose(A, computed_A)
    # -----------------------------
    #  Testing _compute_matrix_B()
    # -----------------------------
    n = 1000
    S = np.arange(n)
    B = np.diag(-S[1:-1])
    computed_B = obm.rhs_matrix_generalizedprob(S)
    assert np.allclose(B, computed_B)

    """
    Test compute_barocl_modes() gives ValueError when input arrays
    have different lengths.
    """
    n_modes = 3
    H = 5000
    z = np.linspace(0.5, H, 50)
    N2 = np.full(len(z) + 1, 2.0)
    obm = BaroclModes(z, np.sqrt(N2))
    try:
        obm.compute_baroclmodes()
    except ValueError:
        assert True
    else:
        assert False

    """
    Test if compute_barocl_modes() works whell when depth is taken with
    negative sign convention.
    """
    n_modes = 3
    len_z = 50
    H = 3000
    z_neg = -np.linspace(0.5, H, len_z)
    z_pos = np.linspace(0.5, H, len_z)
    N2 = np.full(len_z, 2.0)
    obm1 = BaroclModes(z_neg, np.sqrt(N2))
    obm2 = BaroclModes(z_pos, np.sqrt(N2))
    obm1.compute_baroclmodes()
    obm2.compute_baroclmodes()
    eigval_pos = obm1.eigenvals
    eigval_neg = obm2.eigenvals
    assert np.array_equal(eigval_pos, eigval_neg)
