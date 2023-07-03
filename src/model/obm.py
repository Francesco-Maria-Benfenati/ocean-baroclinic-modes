import numpy as np
import scipy as sp
from numpy.typing import NDArray

try:
    from .interpolation import Interpolation
    from .eigenproblem import EigenProblem
except ImportError:
    from interpolation import Interpolation
    from eigenproblem import EigenProblem


class Obm:
    """
    This class is for computing the Ocean Baroclinic Modes of motion and rossby radius.
    """

    def __init__(
        self,
        depth: NDArray,
        bvfreq: NDArray,
        mean_lat: float = None,
        mean_depth: float = None,
        n_modes: int = 4,
    ) -> None:
        """
        Constructor for OBM objects

        Args:
            depth (NDArray): region depth array
            bvfreq (NDArray): brunt vaisala frequency array
            mean_lat (float): region mean latitude. Defaults to None
            mean_depth (float, optional): region mean depth. Defaults to None
            n_modes (int, optional): Num. of baroclinic modes to be computed. Defaults to 4
        """

        self.depth = np.abs(depth)
        self.bvfreq = bvfreq
        if mean_lat is not None:
            self.coriolis_param = self.__coriolis_param(mean_lat)
        else:
            self.coriolis_param = 1.0e-04
        if mean_depth is not None:
            self.mean_depth = abs(mean_depth)
        else:
            self.mean_depth = np.max(self.depth)
        self.n_modes = n_modes

    def compute_obm(self) -> None:
        interp = Interpolation(self.depth, self.bvfreq)
        dz = 1  # (m)
        interp_bvfreq = interp.apply_interpolation(0.5, self.mean_depth, dz)[0]
        # Compute S(z) parameter.
        s_param = interp_bvfreq**2 / self.coriolis_param**2
        # compute tridiagonal matrix
        matrix = self.tridiag_matrix(len(interp_bvfreq), dz, s_param)
        eigenprob = EigenProblem(matrix)
        eigenvals = eigenprob.eigenvals
        return np.sqrt(eigenvals)

    def __coriolis_param(self, mean_lat: float) -> float:
        """
        Compute Coriolis parameter given the region mean latitude.
        """

        # earth angular velocity (1/s)
        earth_angvel = 7.29 * 1e-05
        # coriolis parameter (1/s)
        coriolis_param = 2 * earth_angvel * np.sin(mean_lat * np.pi / 180)
        return coriolis_param

    def tridiag_matrix(self, n, dz, S):
        """
        Compute tridiagonal matrix M.
        """
        M = np.zeros([n, n])

        for k in range(2, n):
            M[k - 1, k] = 1 / S[k - 1]
            M[k - 1, k - 1] = -1 / S[k - 2] - 1 / S[k - 1]
            M[k - 1, k - 2] = 1 / S[k - 2]

        # B.C.s
        M[0, 0] = -1 / S[0]
        M[0, 1] = 1 / S[0]

        M[n - 1, n - 2] = 1 / S[n - 2]
        M[n - 1, n - 1] = -1 / S[n - 2]

        M *= -1 / (dz**2)

        return M

    def compute_lhs_matrix(self, n, dz):
        """
        Compute L.H.S. matrix in the eigenvalues/eigenvectors problem.

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

        where dz is the scaled grid step (= 1m/H).
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

    def compute_rhs_matrix(self, n, S):
        """
        Comput R.H.S. matrix in the eigenvalues/eigenvectors problem.

        Arguments
        ---------
        n : 'int'
            number of vertical levels
        dz : 'float'
            vertical grid step
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

        where dz is the scaled grid step (= 1m/H).
        """

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

    Arguments
    ---------
    depth : <class 'numpy.ndarray'>
            depth variable (1D)
    mean_depth : <class 'numpy.ndarray'> or 'float'
        mean depth of the considered region (m)
    N2 : <class 'numpy.ndarray'>
         Brunt-Vaisala frequency squared (along depth, 1D)
    n_modes : 'int'
              number of modes of motion to be considered
  
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
    obm = Obm(0, 0)
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
    computed_A = Obm.compute_lhs_matrix(8, dz)
    assert np.allclose(A, computed_A)

    # -----------------------------
    #  Testing _compute_matrix_B()
    # -----------------------------
    n = 1000
    S = np.arange(n)
    B = np.diag(-S[1:-1])
    computed_B = Obm.compute_rhs_matrix(n,S)
    assert np.allclose(B, computed_B)

    """
    Test compute_barocl_modes() gives ValueError when input arrays
    have different lengths.
    """

    n_modes = 3
    H = 5000
    z = np.linspace(0.5, H, 50)
    N2 = np.full(len(z) + 1, 2.0)
    obm = Obm(z, np.sqrt(N2))
    try:
        obm.compute_obm()
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
    H = 5000
    z_neg = - np.linspace(0.5, H, len_z)
    z_pos = np.linspace(0.5, H, len_z)
    N2 = np.full(len_z, 2.0)
    obm1 = Obm(z_neg, np.sqrt(N2))
    obm2 = Obm(z_pos, np.sqrt(N2))
    eigval_pos = obm2.compute_obm()
    eigval_neg = obm1.compute_obm()
    assert np.allclose(eigval_pos, eigval_neg, rtol=1e-05, atol=1e-08, 
                                      equal_nan=True)
