import numpy as np

tol = 1e-15


def explicit_jacobi(A,b,x,tol=tol):
    # explicit_gseidel solves the system Ax = b
    # Inputs
    #       A: n-by-n matrix
    #       U: n-by-n Upper triangular matrix
    #       L: n-by-n Lower triangular matrix
    #       D: n-by-n Diagonal matrix
    #       x: initial solution vector
    #       b: n-by-1 coefficient vector
    #       tol: tolerance for stopping criteria
    # Outputs
    #       x: n-by-1 solution vector
    #      RES: residual
    maxiter=100000
    N=np.shape(A)[0]
    b=np.reshape(b,[N,1])
    RES =np.zeros([maxiter,1])
    res=1
    x=np.reshape(x,[N,1])
    k=0
    while res>tol and k<maxiter:
        k=k+1
        for j in np.arange(0,N):
            x[j] = (b[j] - A[j,j+1::].dot(x[j+1::]) - A[j,0:j].dot(x[0:j]))/A[j,j]
        res = np.linalg.norm(b.reshape(N,1) - (A.dot(x)).reshape(N,1))
        RES[k-1,0]=res
    return x,k


def explicit_gseidel(A,b,x,tol=tol):
    # explicit_gseidel solves the system Ax = b
    # Inputs
    #       A: n-by-n matrix
    #       U: n-by-n Upper triangular matrix
    #       L: n-by-n Lower triangular matrix
    #       D: n-by-n Diagonal matrix
    #       x: initial solution vector
    #       b: n-by-1 coefficient vector
    #       tol: tolerance for stopping criteria
    # Outputs
    #       x: n-by-1 solution vector
    #      RES: residual
    maxiter=100000
    N=np.shape(A)[0]
    b=np.reshape(b,[N,1])
    RES =np.zeros([maxiter,1])
    res=1
    x=np.reshape(x,[N,1])
    k=0
    while res>tol and k<maxiter:
        k=k+1
        for j in np.arange(0,N):
            x[j] = (b[j] - A[j,j+1::].dot(x[j+1::]) - A[0:j,j].T.dot(x[0:j]))/A[j,j]
        res = np.linalg.norm(b.reshape(N,1)- (A.dot(x)).reshape(N,1))
        RES[k-1,0]=res

    return x, k

def explicit_sor(A,b,x,w,tol=tol):
    # explicit_gseidel solves the system Ax = b
    # Inputs
    #       A: n-by-n matrix
    #       U: n-by-n Upper triangular matrix
    #       L: n-by-n Lower triangular matrix
    #       D: n-by-n Diagonal matrix
    #       x: initial solution vector
    #       b: n-by-1 coefficient vector
    #       tol: tolerance for stopping criteria
    # Outputs
    #       x: n-by-1 solution vector
    #      RES: residual
    maxiter=100000
    N=np.shape(A)[0]
    b=np.reshape(b,[N,1])
    RES =np.zeros([maxiter,1])
    res=1
    x=np.reshape(x,[N,1])
    k=0
    while res>tol and k<maxiter:
        k=k+1
        for j in np.arange(0,N):
            x[j] = w*(b[j] - A[j,j+1::].dot(x[j+1::]) - A[0:j,j].T.dot(x[0:j]))/A[j,j] + (1-w)*x[j]
        res = np.linalg.norm(b.reshape(N,1)- (A.dot(x)).reshape(N,1))
        RES[k-1,0]=res

    return x, k