import numpy as np
import scipy as sp
from numpy.typing import NDArray


class EigenProblem:
    """
    This class is for computing eigenvalues/eigenvectors.
    """

    def __init__(self, lhs_matrix: NDArray, rhs_matrix: NDArray = None, n_modes: int = 4) -> None:
        self.lhs_matrix = lhs_matrix
        self.rhs_matrix = rhs_matrix
        self.n_modes = n_modes
        self.eigenvals = self.apply_compute_eigenvals()

    def apply_compute_eigenvals(self) -> tuple[NDArray]:
        if self.rhs_matrix is not None:
            eigenvals = EigenProblem.compute_eigenvals(self.lhs_matrix, self.rhs_matrix, self.n_modes)
        else:
            eigenvals, eigenvecs = EigenProblem.tridiag_eigenvals(self.lhs_matrix)[:self.n_modes]
        return eigenvals

    @staticmethod
    def compute_eigenvals(A, B, n_modes):
        """
        Compute eigenvalues solving the eigenvalues/eigenvectors problem.
            
        Parameters
        ----------
        A : <class 'numpy.ndarray'>
            L.H.S. finite difference matrix
        B : <class 'numpy.ndarray'>
            R.H.S. S-depending matrix
        n_modes : 'int'
                number of modes of motion to be considered

        Returns
        -------
        eigenvalues : <class 'numpy.ndarray'>
                    problem eigenvalues 'lambda'
        
        The eigenvalues/eigenvectors problem is 
        
            A * w = lambda * B * w  (lambda eigenvalues, w eigenvectors)

            with BCs: w = 0 at z=0,1 .
            
        Here, a scipy algorithm is used.
        """
        
        # Change sign to matrices (for consistency with scipy algorithm).
        A *= -1
        B *= -1
        # Compute smallest Eigenvalues.
        val = sp.sparse.linalg.eigs(A, k = n_modes, M=B, sigma=0, which='LM', 
                                                    return_eigenvectors=False)
        
        # Take real part of eigenvalues and sort them in ascending order.
        eigenvalues = np.real(val)
        sort_index = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_index]
        
        return eigenvalues
    
    @staticmethod
    def tridiag_eigenvals(M):
        """
        Compute eigenvalues of tridiagonal matrix M.
        """
        d = np.diagonal(M, offset=0).copy()
        e = np.diagonal(M, offset=1).copy()
        eigenvalues, eigenvectors = sp.linalg.eigh_tridiagonal(d, e)
        # Take real part of eigenvalues and sort them in ascending order.
        eigenvalues = np.real(eigenvalues)
        sort_index = np.argsort(eigenvalues)
        eigenvals = eigenvalues[sort_index]
        eigenvecs = eigenvectors[sort_index]
        return eigenvals, eigenvecs

    @staticmethod
    def numerov_method(dz, f, dw_0, w_0, w_N):
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
        w[n-1] = w_N
        # Define constant k
        k = (dz**2) / 12
        # First value computed from BCs and dw/dz|z=0 .
        w[1] = (w[0] + dz*dw_0 + (1/3)*(dz**2)*f[0]*w[0])/(1-k*2*f[1])
        # Numerov's algorithm.
        for j in range(2,n-1):  
            w[j] = (1/(1 - k*f[j])) * ((2 + 10*k*f[j-1])*w[j-1] 
                                        - (1 - k*f[j-2])* w[j-2])
        
        return w

if __name__ == "__main__":
    from obm import Obm
    """
    Test if eigenvalues are computed correctly for a simple problem
    taken from a linear algebra book.
    """
    
    n_modes = 3
    A = np.array([[8.0,-18.0,9.0],[3.0,-7.0,3.0],[0.0,0.0,-1.0]])
    B = np.diag(np.ones(3))
    expected_eigenvals = np.array([-1, -1, 2])
    eigenprob = EigenProblem(A, B, n_modes=3)
    out_eigenvals = eigenprob.eigenvals
    
    assert np.allclose(out_eigenvals, expected_eigenvals)

    """
    Test if eigenvalues are computed correctly for a simple problem
    with well-know resolution: the stationary wave (non-dimensional).
    """
    n = 1000
    expected_integers = np.arange(1,4) 
    dz = 1/n 
    obm = Obm(0,0)
    A = obm.compute_lhs_matrix(n, dz)
    B = np.diag(- np.ones(n-2))
    eigenprob = EigenProblem(A, B, n_modes=3)
    out_eigenvals = eigenprob.eigenvals
    out_integers = (np.sqrt(out_eigenvals)/(np.pi))
    assert np.allclose(out_integers, expected_integers, 
                                                   rtol= 1e-02, atol = 1e-3)

    """
    Test if _Numerov_method() gives correct eigenvectors for the
    harmonic oscillator problem.
    """
    
    k = 1 #N/m
    m = 1 #kg
    omega = k/m #np.sqrt(k/m)
    A = 1
    n = 1000
    L = 100
    x = np.linspace(0,L,n)
    dx = abs(x[1]-x[0])
    theor_sol = A*np.cos(omega * x)
    dw_0 = 0
    f = -omega*np.ones(n)
    w_0 = A * 1
    w_N = A * np.cos(omega * L)
    num_sol = EigenProblem.numerov_method(dx, f, dw_0, w_0, w_N)
    error = (dx**4) * A # Numerov Method error O(dx^4)
    
    assert np.allclose(num_sol, theor_sol, atol = error)
   

    """
    Test if _Numerov_method() gives correct eigenvectors for the
    stationary waves problem in a pipe (1D).
    n is the integer corresponding to mode of motion 'n'.
    """

    # Problem parameters
    A = 1
    L = 9.5
    N = 1000
    eigenvals = n * np.pi / L
    x = np.linspace(0,L,N)
    dx = abs(x[1]-x[0])
    # Theoretical solution
    theor_sol = np.sin(eigenvals * x)
    # Numerical solution
    w_0 = 0
    w_N = 0
    dw_0 = n*np.pi/L
    f_val = - eigenvals**2
    f = np.full(N, f_val)
    num_sol = EigenProblem.numerov_method(dx, f, dw_0, w_0, w_N )
    error = (dx**4) * A # Numerov Method error O(dx^4)
    assert np.allclose(num_sol, theor_sol, atol=error) 
