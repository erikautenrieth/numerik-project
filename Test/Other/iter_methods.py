import time
import  math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import *
import numpy.linalg as npl
import numpy.random as npr
import pandas as pd
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import linalg
from scipy.sparse import diags
from sympy import *
import sympy as sym

def gauss_elimination(A,b):
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    #print("piv:", piv)
    #print("lu:", lu)

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


