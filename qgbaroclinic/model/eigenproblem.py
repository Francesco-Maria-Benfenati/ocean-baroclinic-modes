import numpy as np
import scipy as sp
from numpy.typing import NDArray


class EigenProblem:
    """
    This class is for computing eigenvalues/eigenvectors.
    """

    def __init__(
        self,
        lhs_matrix: NDArray,
        rhs_matrix: NDArray = None,
        grid_step: float = None,
        n_modes: int = 4,
    ) -> None:
        """
        Constructor for eigen problem object.

        Args:
            lhs_matrix (NDArray): L.H.S. matrix of eigenvalues/eigenvectors problem
            rhs_matrix (NDArray, optional): R.H.S. matrix of eigenvalues/eigenvectors problem. Defaults to None.
            grid_step (NDArray, optional): grid step used for computing L.H.S matrix. Defaults to None.
            n_modes (int, optional): _description_. Defaults to 4.
            numerov_method : if numerov method should be applied for computing eigenvectors. Default is True.
        """

        self.lhs_matrix = lhs_matrix
        self.rhs_matrix = rhs_matrix
        self.grid_step = grid_step
        self.n_modes = n_modes
        self.solve_eigenprob()

    def solve_eigenprob(self) -> None:
        """
        Solve eigenproblem.

        NOTE: if only LHS matrix is provided, solves the problem for a tridiagonal matrix.
              Else, solve the generalized problem.
        """

        if self.rhs_matrix is not None:
            # Solve generalized problem
            self.eigenvals, eigenvecs = self.eigenvals_generalizedprob()
            if self.grid_step is not None:
                self.eigenvecs = self.eigenvecs_with_numerov()
            else:
                self.eigenvecs = eigenvecs
        else:
            # solve eigenproblem for tridiagonal matrix
            self.eigenvals, self.eigenvecs = self.tridiag_eigensolver()

    def tridiag_eigensolver(self) -> tuple[NDArray]:
        """
        Compute as many eigenvalues as n_modes, for a tridiagonal matrix.

        Exploits scipy algorithm.
        """

        tridiagmatrix = self.lhs_matrix
        n_modes = self.n_modes
        # extract diagonal and subdiagonal
        d = np.diagonal(tridiagmatrix, offset=0).copy()
        e = np.diagonal(tridiagmatrix, offset=1).copy()
        # Compute eigenvalues using scipy
        eigenvalues, eigenvectors = sp.linalg.eigh_tridiagonal(
            d,
            e,
            lapack_driver="auto",
            select_range=(0,),
        )
        # Take real part of eigenvalues and sort them in ascending order.
        eigenvalues = np.real(eigenvalues)
        sort_index = np.argsort(eigenvalues)
        eigenvals = eigenvalues[sort_index]
        eigenvecs = eigenvectors[:, sort_index]
        return eigenvals[:n_modes], eigenvecs[:, :n_modes]

    def eigenvals_generalizedprob(self) -> NDArray:
        """
        Compute as many eigenvalues as n_modes, solving the generalized eigenvalues/eigenvectors problem.

        NOTE: The solved problem is

            A * w = lambda * B * w  (lambda eigenvalues, w eigenvectors)

            with BCs: w = 0 at z=0,1 .

        Here, a scipy algorithm is used.
        """

        A: NDArray = self.lhs_matrix
        B: NDArray = self.rhs_matrix
        n_modes: int = self.n_modes
        # Change sign to matrices (for consistency with scipy algorithm).
        A *= -1
        B *= -1
        # Compute smallest Eigenvalues.
        val, vecs = sp.sparse.linalg.eigs(
            A,
            k=n_modes - 1,
            M=B,
            sigma=0,
            which="LM",
        )
        # Take real part of eigenvalues and sort them in ascending order.
        eigenvalues = np.real(val)
        sort_index = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = vecs[:, sort_index]
        # Add null eigenvalue, not obtained using shift-invert mode
        eigenvalues = np.insert(eigenvalues, 0, 0.0)
        eigenvectors = np.insert(eigenvectors, 0, np.ones(A.shape[0]), axis=1)
        return eigenvalues, eigenvectors

    def eigenvecs_with_numerov(self) -> NDArray:
        """
        Compute eigenvectors using Numerov's method.
        """

        n_modes = self.n_modes
        S = self.rhs_matrix.diagonal()
        n = S.shape[0]
        eigenvalues = self.eigenvals
        dz = self.grid_step
        # Define integration constant phi_0 = phi(z = 0) = 1 as BC.
        phi_barotropic = 1
        phi_0 = np.ones(n_modes) * phi_barotropic
        # Define f_n(z).
        f_n = np.empty([n, n_modes])
        for i in range(n_modes):
            f_n[:, i] = -eigenvalues[i] * S[:]
        # Define dw_0 = dw/dz at z=0.
        dw_0 = -eigenvalues * phi_0
        # Define BCs: w_0 = w(z=0) = 0 ; w_n = w(z=1) = 0.
        w_0 = 0
        w_n = 0
        # Compute eigenvectors through Numerov's Algortihm.
        w = np.empty([n, n_modes])
        for i in range(n_modes):
            w[:, i] = EigenProblem.numerov_method(dz, f_n[:, i], dw_0[i], w_0, w_n)
        return w

    @staticmethod
    def numerov_method(dz, f, dw_0, w_0, w_n) -> NDArray:
        """
        Compute eigenvectors through Numerov's method.

        Solves the generic ODE:
        -------------------------------
        | d^2/dz^2 w(z) = f(z) * w(z)  | .
        --------------------------------

        Parameters
        ----------
        dz : 'float'
            vertical grid step
        f : <class 'numpy.ndarray'>
            problem parameter, depending on z
        dw_0 : 'float'
            first derivative of w: dw/dz computed at z = 0.
        w_0, w_N : 'float'
                BCs, respectively at z = 0,1

        Returns
        -------
        w : <class 'numpy.ndarray'>
            problem eigenvectors

        The problem eigenvectors are computed through Numerov's numerical
        method, integrating the equation

            (d^2/dz^2) * w = - lambda * S * w    (with BCs  w = 0 at z=0,1).

        The first value is computed as
            w(0 + dz)= (- eigenvalues*phi_0)*dz/(1+lambda*S[1]*(dz**2)/6)
        where phi_0 = phi(z=0) set = 1 as BC.
        """

        # Number of vertical levels.
        n = len(f)
        # Store array for eigenvectors ( w(0) = w(n-1) = 0 for BCs).
        w = np.empty([n])
        w[0] = w_0
        w[n - 1] = w_n
        # Define constant k
        k = (dz**2) / 12
        # First value computed from BCs and dw/dz|z=0 .
        w[1] = (w[0] + dz * dw_0 + (1 / 3) * (dz**2) * f[0] * w[0]) / (
            1 - k * 2 * f[1]
        )
        # Numerov's algorithm.
        for j in range(2, n - 1):
            w[j] = (1 / (1 - k * f[j])) * (
                (2 + 10 * k * f[j - 1]) * w[j - 1] - (1 - k * f[j - 2]) * w[j - 2]
            )
        return w


if __name__ == "__main__":
    from baroclinicmodes import BaroclinicModes

    test_plot = False

    def test_compute_eigenvals_simple_problem():
        """
        Test if eigenvalues are computed correctly for a simple problem
        taken from a linear algebra book.
        """
        A = np.array([[8.0, -18.0, 9.0], [3.0, -7.0, 3.0], [0.0, 0.0, -1.0]])
        B = np.diag(np.ones(3))
        expected_eigenvals = np.array([0, -1, -1, 2])
        eigenprob = EigenProblem(A, B, n_modes=3)
        out_eigenvals = eigenprob.eigenvals
        assert np.allclose(out_eigenvals, expected_eigenvals)

    def test_harmonic_oscillator():
        """
        Test if _Numerov_method() gives correct result for the
        harmonic oscillator problem.
        """
        k = 1  # N/m
        m = 1  # kg
        omega = k / m  # np.sqrt(k/m)
        A = 1
        n = 1000
        t_end = 1
        t = np.linspace(0, t_end, n)
        dt = abs(t[1] - t[0])
        # Theor solution
        theor_sol = A * np.cos(omega * t)
        # Compute eigenvectors with Numerov's
        dw_0 = 0
        f = -(omega**2) * np.ones(n)
        w_0 = A
        w_n = A * np.cos(omega * t_end)
        num_sol = EigenProblem.numerov_method(dt, f, dw_0, w_0, w_n)
        # Check if num and theor solution are the same (considering error)
        assert np.allclose(num_sol, theor_sol)

    def test_stationary_wave():
        """
        Test if eigenvalues are computed correctly for a simple problem
        with well-know resolution: the stationary wave.
        """
        n = 1000
        L = 100
        expected_eigenvals = (np.arange(4) * np.pi / L) ** 2
        dx = L / n
        lhs_tridiag = BaroclinicModes.tridiag_matrix_standardprob(np.ones(n), dx)
        lhs_general = BaroclinicModes.lhs_matrix_generalizedprob(n, dx)
        rhs_general = BaroclinicModes.rhs_matrix_generalizedprob(np.ones(n))
        eigenprob_tridiag = EigenProblem(lhs_tridiag, n_modes=4)
        eigenprob_general = EigenProblem(lhs_general, rhs_general, n_modes=4)
        out_eigenvals_tridiag = eigenprob_tridiag.eigenvals
        out_eigenvals_general = eigenprob_general.eigenvals
        # Check tridiag eigenvals are computed correctly
        tridiag_rel_error = (
            expected_eigenvals - out_eigenvals_tridiag
        ) / expected_eigenvals
        print(
            f"For standing wave problem, uncertainty associated to standard method is: {tridiag_rel_error}"
        )
        # Check generalized problem eigenvals are computed correctly
        general_rel_error = (
            expected_eigenvals - out_eigenvals_general
        ) / expected_eigenvals
        print(
            f"On the other hand, uncertainty associated to generalized method is: {general_rel_error}"
        )

    def test_stationary_wave_inapipe():
        """
        Test if _Numerov_method() gives correct eigenvectors for the
        stationary waves problem in a pipe (1D).
        """

        # Problem parameters
        A = 1
        L = 9.5
        N = 1000
        x = np.linspace(0, L, N)
        dx = abs(x[1] - x[0])
        # Problem matrices
        lhs_matrix = BaroclinicModes.lhs_matrix_generalizedprob(N + 2, dx)
        rhs_matrix = BaroclinicModes.rhs_matrix_generalizedprob(np.ones(N + 2))
        eigenprob_general = EigenProblem(lhs_matrix, rhs_matrix, grid_step=None)
        eigenprob_general.eigenvecs = BaroclinicModes.normalize_eigenfunc(
            eigenprob_general.eigenvecs, dx
        )
        for n in range(1, 4):
            # change sign of SCIPY eigenvectors due to LAPACK algorithm
            if eigenprob_general.eigenvecs[0, n] < 0:
                eigenprob_general.eigenvecs[:, n] *= -1
            eigenvals = n * np.pi / L
            # Theoretical solution (normalized using BaroclinicModes method)
            theor_sol = BaroclinicModes.normalize_eigenfunc(np.sin(eigenvals * x), dx)
            # Numerical solution
            w_0 = 0
            w_n = 0
            dw_0 = n * np.pi / L
            f_val = -(eigenvals**2)
            f = np.full(N, f_val)
            num_sol = EigenProblem.numerov_method(dx, f, dw_0, w_0, w_n)
            num_sol = BaroclinicModes.normalize_eigenfunc(num_sol, dx)
            assert np.allclose(num_sol, theor_sol)
            error = np.nanmean(num_sol - theor_sol)
            print(f"Mean absolute error associated to Numerov's method is: {error}")
            # Test if eigenvectors obtained through scipy
            # are the same obtained through Numerov's.
            scipy_error = np.nanmean((eigenprob_general.eigenvecs[:, n] - theor_sol))
            print(
                f"On the other hand, mean absolute error associated to Scipy method is: {scipy_error}"
            )
            if test_plot:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.title(
                    f"Numerov's (coloured) VS theoretical (dashed) solutions.\n Mode of motion *{n}*"
                )
                plt.plot(theor_sol, -x, "k--")
                plt.plot(num_sol, -x)
                plt.figure()
                plt.title(
                    f"Scipy (coloured) VS theoretical (dashed) solutions.\n Mode of motion *{n}*"
                )
                plt.plot(theor_sol, -x, "k--")
                plt.plot(eigenprob_general.eigenvecs[:, n], -x)
                plt.show()
                plt.close()

    # Run test functions.
    test_compute_eigenvals_simple_problem()
    test_harmonic_oscillator()
    test_stationary_wave()
    test_stationary_wave_inapipe()
    print("Tests SUCCEDED.")
