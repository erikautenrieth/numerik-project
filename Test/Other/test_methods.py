import numpy as np
from numpy.linalg import inv
from numpy.linalg import *


def gauss_elimination(A,b):
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x

def gaussElim(A,B):
    a = A.copy() # copy original matrix A
    b = B.copy() #copy original vector b
    n = len(b)
    c=0
    # Elimination phase
    for k in range(0,n-1):
        c +=1
        for i in range(k+1,n):
            c+=1
            if a[i,k] != 0.0:
                #if not null define λ
                lam = a [i,k]/a[k,k]
                #we calculate the new row of the matrix
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                #we update vector b
                b[i] = b[i] - lam*b[k]
                # backward substitution
    for k in range(n-1,-1,-1):
        c+=1
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]

    return b, c


def gauss_seidel_num(A, b, tol=10e-4, iter=10000):
    x = np.zeros_like(b, dtype=np.double)
    for k in range(iter):
        x_prev  = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_prev[(i+1):])) / A[i ,i]

        if norm(x - x_prev, ord=np.inf) < tol: # / norm(x, ord=np.inf)
            break
    return x, k


def jacobi_num(A, b, tol=10e-4, iter=10000):
    n = A.shape[0]
    x0 = np.zeros((n))
    x = x0.copy()
    x_prev = x0.copy()
    k = 0
    rel_diff = tol * 2
    while (rel_diff > tol) and (k < iter):
        for i in range(0, n):
            subs = 0.0
            for j in range(0, n):
                if i != j: subs += A[i,j] * x_prev[j]
            x[i] = (b[i] - subs ) / A[i,i]
        k += 1
        rel_diff = norm(x - x_prev) / norm(x)
        x_prev = x.copy()

    return x, k


def gauss_seidel_num(A, b, iter=10000):
    x = np.zeros_like(b, dtype=np.double)
    for k in range(iter):
        x_prev  = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_prev[(i+1):])) / A[i ,i]
        if termin(x , x_prev):
            break
    return x, k


def jacobi(A, b, iter=10000):
    #erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = np.copy(x)
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    B = np.dot(D_inv,D - A)
    g = np.dot(D_inv,b)
    for k in range(iter):
        #erg.append(x)
        x = np.add(np.dot(B,x),g)
        if termin(x , x0 ):
            return x, k
        x0 = np.copy(x)
    return x, k


def sor(A, b,w, iter=10000):
    k=0
    n = b.shape
    x0 =  np.zeros((n))
    x = np.copy(x0)
    for step in range (1, iter):
        for i in range(n[0]):
            k+=1
            new_values_sum = np.dot(A[i, :i], x[:i])
            old_values_sum = np.dot(A[i, i+1 :], x0[ i+1: ])
            x[i] = (b[i] - (old_values_sum + new_values_sum)) / A[i, i]
            x[i] = np.dot(x[i], w) + np.dot(x0[i], (1 - w))
        if termin(x, x0):
            break
        x0 = x
    return x, k

def termin(x , x0 , tol=10e-10):
    return norm(np.subtract(x , x0), ord=1)  < tol