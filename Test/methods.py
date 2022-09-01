import time
import  math
import numpy as np
from numpy.linalg import inv
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


def jacobi(A, b, iter=30, error=10e-4):
    start = time.process_time()
    erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = x.copy()

    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    B = np.dot(D_inv,D - A)
    g = np.dot(D_inv,b)

    for k in range(iter):
        erg.append(x)
        x = np.dot(B,x)+g

        var = abs(x0-x)
        for el in var:
            if el <= error:
                #print(f"Jacobi nach {k} -Iterationen fertig.")
                end = time.process_time()
                jacobi_time = '{:5.4f}s'.format(end-start)
                return erg, jacobi_time, k
        x0 = x.copy()


    end = time.process_time()
    jacobi_time = '{:5.4f}s'.format(end-start)

    print('Jacobi Time: ',jacobi_time)
    return erg, jacobi_time, k


def gauss_seidel(A, b, iter=30, error=10e-4):
    start = time.process_time()
    erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = x.copy()
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D = np.diag(np.diag(A))

    B = -(np.linalg.inv(D + L)) @ R
    g =   (np.linalg.inv(D + L)) @ b

    for k in range(iter):
        erg.append(x)
        x = np.dot(B,x)+g

        var = abs(x0-x)
        for el in var:
            if el <= error:
                #print(f"Gauss-Seidel nach {k} -Iterationen fertig.")
                end = time.process_time()
                gauss_seidel_time = '{:5.4f}s'.format(end-start)
                return erg, gauss_seidel_time, k
        x0 = x.copy()

    end = time.process_time()
    gauss_seidel_time =  '{:5.4f}s'.format(end-start)
    print('Gauß/Seidel: ', gauss_seidel_time)
    return erg, gauss_seidel_time, k


def sor(A, b, w_in=None, iter=30, error=10e-4):
    start = time.process_time()
    erg = []
    [m, n] = np.shape(A)
    x = np.zeros((m))
    x0 = x.copy()
    R = np.triu(A,k=1)  # C2
    L = np.tril(A,k=-1) # C1
    D = np.diag(np.diag(A))
    Jm = np.dot(-np.linalg.inv(D), (L + R))
    p = np.linalg.norm(Jm, 2)
    print("Spektralradius der Jacobi-Matrix: " +str(p))
    if p > 1:
        w = 1
    else:
        result = (2*(1 - math.sqrt(1-p**2))) / p**2
        w = result
    if w_in is not None:
        w = w_in
    print("w =",w)
    for k in range(iter):
        erg.append(x)
        x = -(np.linalg.inv(D + np.dot(w,L))) @ (np.dot(w,R) - np.dot(1-w,D)) @ x +   np.dot( w,np.linalg.inv(D + np.dot(w,L))) @ b
        #Hw = np.dot(np.linalg.inv((D + (w * L))), ((1 - w) * D - w * R))
        #x = np.dot(Hw, x) + np.dot(w * (np.linalg.inv(D + (w * L))), b)

        var = abs(x0-x)
        for el in var:
            if el <= error:
                #print(f"SOR nach {k} -Iterationen fertig.")
                end = time.process_time()
                sor_time = '{:5.4f}s'.format(end-start)
                return erg, sor_time, k
        x0 = x.copy()


    end = time.process_time()
    sor_time = '{:5.4f}s'.format(end-start)
    #print('SOR Time: ',sor_time)
    return erg, sor_time, k


def absolut_error(x_true, x):
    return np.abs(x_true - x)

def relativ_error(x_true, x):
    return np.abs(x_true - x) / np.abs(x)