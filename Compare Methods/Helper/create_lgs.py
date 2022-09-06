import numpy as np
from scipy.sparse import diags


def create_block_triag(n):
    k = [np.ones(n-1)*(-1),2*np.ones(n),np.ones(n-1)*(-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()
    b = np.ones((n))
    return A, b


def band_matrix(n):
    """
    This function sets up the matrices A and b using the first order central differences
    stencil to discretise Poisson's equation in 2D.
    """
    N = n**2 # Number of points
    h = 1./(n+1) # gridspacing
    A = np.zeros([N, N]) # initialise A

    #Diagonals
    lead_diag = np.diag(np.ones(N)*-4, 0)
    outer_diags = np.ones(N-1)
    for i in range(n-1, N-1, n):
        outer_diags[i] = 0
    outer_diags = np.diag(outer_diags, 1) + np.diag(outer_diags, -1)

    #Diagonals dependent on n
    n_diags = np.diag(np.ones(N-n), n) + np.diag(np.ones(N-n), -n)

    #Populate A matrix
    A += lead_diag + outer_diags + n_diags
    A = A/(h**2)
    #Populate the RHS b matrix
    b=np.zeros(N)
    b[int((N-1)/2)]=2

    return A,b

def create_upper_triangular(n, dens=0.25):
    from scipy.sparse import random
    from scipy.stats import rv_continuous
    from numpy.random import default_rng
    class CustomDistribution(rv_continuous):
        def _rvs(self,  size=None, random_state=None):
            return random_state.standard_normal(size)

    rng = default_rng()
    X = CustomDistribution(seed=rng)
    Y = X()  # get a frozen version of the distribution
    S = random(n, n, density=dens, random_state=rng, data_rvs=Y.rvs)
    A = S.A

    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j]+= sum(abs(A[i]))
                A[i][j]+= 1

    A = np.triu(A,k=0)

    B = random(n, 1, density=dens, random_state=rng, data_rvs=Y.rvs)
    b = B.A

    #A=np.random.randint(low=-20,high=20,size=(n,n))
    #b = np.ones((n))
    #b=np.random.randint(low=-20,high=20,size=(n,1))
    if  help.characteristics(A) == 0: print("Determinante ist Null")

    return A, b