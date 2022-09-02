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
sym.init_printing()
from IPython.display import display, Math

def gaussElim(A,B):
    a = A.copy() # copy original matrix A
    b = B.copy() #copy original vector b
    n = len(b)
    c=0
    # Elimination phase
    start = time.process_time()
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

    end = time.process_time()
    gauss_elim_time = '{:5.4f}s'.format(end-start)

    return b, gauss_elim_time, c


def gauss_elimination(A,b):
    from scipy.linalg import lu_factor, lu_solve
    start = time.process_time()
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)

    end = time.process_time()
    gauss_elim_time = '{:5.4f}s'.format(end-start)

    #print("piv:", piv)
    #print("lu:", lu)

    return x, gauss_elim_time

def jacobi_num(A, b, tol=10e-4, iter=10000):
    n = A.shape[0]
    x0 = np.zeros((n))
    x = x0.copy()
    x_prev = x0.copy()
    k = 0
    rel_diff = tol * 2
    start = time.process_time()
    while (rel_diff > tol) and (k < iter):
        for i in range(0, n):
            subs = 0.0
            for j in range(0, n):
                if i != j: subs += A[i,j] * x_prev[j]
            x[i] = (b[i] - subs ) / A[i,i]
        k += 1
        rel_diff = norm(x - x_prev) / norm(x)
        x_prev = x.copy()
    end = time.process_time()
    jacobi_time = '{:5.4f}s'.format(end-start)

    return x, jacobi_time, k


def gauss_seidel_num(A, b, tol=1e-4, max_iterations=10000):
    x = np.zeros_like(b, dtype=np.double)
    start = time.process_time()
    for k in range(max_iterations):
        x_prev  = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_prev[(i+1):])) / A[i ,i]

        if norm(x - x_prev, ord=np.inf) / norm(x, ord=np.inf) < tol:
            break
    end = time.process_time()
    gauss_seidel_time = '{:5.4f}s'.format(end-start)
    return x, gauss_seidel_time, k


def jacobi(A, b, tol=10e-4, iter=10000):
    erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = x.copy()
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    B = np.dot(D_inv,D - A)
    g = np.dot(D_inv,b)
    start = time.process_time()
    for k in range(iter):
        erg.append(x)
        x = np.dot(B,x)+g

        if norm(x - x0, ord=np.inf) < tol: # / norm(x, ord=np.inf)
            end = time.process_time()
            jacobi_time = '{:5.4f}s'.format(end-start)
            return erg, jacobi_time, k

        x0 = x.copy()

    end = time.process_time()
    jacobi_time = '{:5.4f}s'.format(end-start)

    return erg, jacobi_time, k


def gauss_seidel(A, b, tol=10e-4, iter=10000):
    erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = x.copy()
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D = np.diag(np.diag(A))
    B = -(np.linalg.inv(D + L)) @ R
    g =   (np.linalg.inv(D + L)) @ b
    start = time.process_time()
    for k in range(iter):
        erg.append(x)
        x = np.dot(B,x)+g

        if norm(x - x0, ord=np.inf)  < tol: # / norm(x, ord=np.inf)
            end = time.process_time()
            gauss_seidel_time = '{:5.4f}s'.format(end-start)
            return erg, gauss_seidel_time, k
        x0 = x.copy()

    end = time.process_time()
    gauss_seidel_time =  '{:5.4f}s'.format(end-start)
    return erg, gauss_seidel_time, k


def sor(A, b, w_in=None,  tol=10e-4, iter=10000):
    erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = x.copy()
    R = np.triu(A,k=1)  # C2
    L = np.tril(A,k=-1) # C1
    D = np.diag(np.diag(A))
    Jm = np.dot(-np.linalg.inv(D), (L + R))
    p = np.linalg.norm(Jm, 2)
    #print("Spektralradius der Jacobi-Matrix: " +str(p))
    if p > 1: w = 1
    else:
        result = (2*(1 - np.sqrt(1-p**2))) / p**2
        w = result
    if w_in is not None: w = w_in
    #print("omega =",w)
    start = time.process_time()
    for k in range(iter):
        erg.append(x)
        x = -(np.linalg.inv(D + np.dot(w,L))) @ (np.dot(w,R) - np.dot(1-w,D)) @ x +   np.dot( w,np.linalg.inv(D + np.dot(w,L))) @ b

        if norm(x - x0, ord=np.inf)< tol: # / norm(x, ord=np.inf)
            end = time.process_time()
            sor_time = '{:5.4f}s'.format(end-start)
            return erg, sor_time, k
        x0 = x.copy()

    end = time.process_time()
    sor_time = '{:5.4f}s'.format(end-start)
    return erg, sor_time, k
