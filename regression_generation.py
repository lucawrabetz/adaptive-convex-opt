import numpy as np
import pdb
from scipy.stats import ortho_group


# Generate least squares data (and minimizers)
def gen_ls_data(m, n, ni, kappa, k):
    # m, number of functions
    # n, number of variables (or features)
    # ni, number of observations for i-th regression problem (same for all i)
    # k is desired condition number
    A_list = []  # Design matrix
    b_list = []  # Right-hand side
    x_list = []  # Minimizers
    for i in range(0, m):
        # Creating A = U \Sigma V
        # Generate orthonormal matrices for SVD
        U = ortho_group.rvs(ni)
        V = ortho_group.rvs(n)

        # Create diagonal matrix
        Sigma = np.zeros((ni, n))

        if i < k:
            d = np.array([np.sqrt(kappa) for j in range(n)])
        else:
            d = np.ones(n)
            d[0] = np.sqrt(kappa)
        Sigma[0:n, :] = np.diag(d)
        A = np.matmul(U, np.matmul(Sigma, V))
        A = np.round(
            A, 8
        )  # Rounding is needed to limit the precision of A^TA so that the numpys eigenvector solver finds real eigenvalues
        # Add A to matrix list
        A_list.append(A)
        # Construct random minimizer
        x = np.random.normal(0, 1, (n, 1))
        x_list.append(x)
        # Construct b to ensure x is minimizer
        b = np.matmul(A, x)
        b_list.append(b)
    return A_list, b_list, x_list


# Least squares class
class least_squares:
    def __init__(self, A_list, b_list, x_list):
        self.A_list = A_list
        self.b_list = b_list
        self.x_list = x_list
        constants = self._constant_calculate()
        self.lipschitz_constants = constants[0]  # smoothness
        self.convexity_constants = constants[1]  # strong convexity
        self.k = constants[2]  # Largest condition number

    def _constant_calculate(self):
        """
        Calculates the lipschitz and convexity constants, returns a list with the Lipschitz constants first and convexity constants second
        parameters:
        self is used to get the A_list and B_list

        """
        # Initialize list of lipschitz constants
        n = len(self.b_list)
        A_list = self.A_list
        L = []
        mu = []
        kappa = 0
        for i in range(0, n):
            # Calculate the matrix for eigenvalue calculation
            Eig_Calc = np.matmul(np.transpose(A_list[i]), A_list[i])
            Eigenvalues = np.linalg.eig(Eig_Calc)[0]
            L.append(max(Eigenvalues))
            mu.append(min(Eigenvalues))
            k_temp = max(Eigenvalues) / min(Eigenvalues)
            if k_temp > kappa:
                kappa = k_temp
        return [L, mu, kappa]

    def obj(self, x):
        A = self.A
        b = self.b
        return 0.5 * np.linalg.norm(np.matmul(A, x) - b) ** 2


if __name__ == "__main__":
    # Number of functions
    m = 20
    # Number of variables,
    n = 3
    # Number of observations per machine (should be greater than n)
    ni = 5
    # Desired condition number
    kappa = 30
    k = 2

    # Generate data
    A_list, b_list, x_list = gen_ls_data(m, n, ni, kappa, k)

    # Create least squares class
    prob = least_squares(A_list, b_list, x_list)
