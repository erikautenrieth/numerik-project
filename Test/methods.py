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


def gauss(A,B):
    from numpy import array, zeros, fabs
    n = len(B)
    x = zeros(n, float)
    a = A.copy() # copy original matrix A
    b = B.copy()
    c = 0
    #first loop specifys the fixed row
    for k in range(n-1):
        if fabs(a[k,k]) < 1.0e-12:
            for i in range(k+1, n):
                c+=1
                if fabs(a[i,k]) > fabs(a[k,k]):
                    a[[k,i]] = a[[i,k]]
                    b[[k,i]] = b[[i,k]]
                    break
        #applies the elimination below the fixed row
        for i in range(k+1,n):
            if a[i,k] == 0:continue

            factor = a[k,k]/a[i,k]
            for j in range(k,n):
                c+=1
                a[i,j] = a[k,j] - a[i,j]*factor
                #we also calculate the b vector of each row
            b[i] = b[k] - b[i]*factor

    x[n-1] = b[n-1] / a[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += a[i,j] * x[j]
            c+=1
        x[i] = (b[i] - sum_ax) / a[i,i]

    return x, c


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


def jacobi_2(A, b, iter=10000):
    x0 = np.zeros_like(b, dtype=np.double)
    D = np.diag(A)
    R = A - np.diagflat(D)

    for k in range(iter):
        x = (b - np.dot(R,x0))/ D
        if termin(x , x0 ):
            return x, k
        x0 = np.copy(x)
    return x, k


def gauss_seidel(A, b, iter=10000):
    x = np.zeros_like(b, dtype=np.double)
    x0 = np.copy(x)
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D = np.diag(np.diag(A))
    B = -(np.linalg.inv(D + L)) @ R
    g =   (np.linalg.inv(D + L)) @ b
    for k in range(iter):
        x = np.add(np.dot(B,x),g)
        if termin(x , x0 ):
            return x, k
        x0 = np.copy(x)

    return x, k


def gauss_seidel_num(A, b, iter=10000):
    x = np.zeros_like(b, dtype=np.double)
    for k in range(iter):
        x_prev  = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_prev[(i+1):])) / A[i ,i]

        if termin(x , x_prev ):
            break
    return x, k


def sor(A, b,w=1.25, iter=10000):
    n = b.shape
    x = np.zeros((n))
    x0 = np.copy(x)
    R = np.triu(A,k=1)  # C2
    L = np.tril(A,k=-1) # C1
    D = np.diag(np.diag(A))
    for k in range(iter):
        x = -(np.linalg.inv(np.add(D , np.dot(w,L)))) @ (np.subtract(np.dot(w,R) , np.dot(1-w,D))) @ x +   np.dot(np.dot( w,np.linalg.inv(D + np.dot(w,L))) , b)
        if termin(x , x0 ):
            return x, k
        x0 = np.copy(x)

    return x, k


def comp_w(A):
    R = np.triu(A,k=1)  # C2
    L = np.tril(A,k=-1) # C1
    D = np.diag(np.diag(A))
    Jm = np.dot(-np.linalg.inv(D), (L + R))
    p = np.linalg.norm(Jm, 2)
    w = (2*(1 - np.sqrt(1-p**2))) / p**2
    print("Spektralradius der Jacobi-Matrix: " +str(p))
    print("omega =",w)
    return w


def SOR_num(A, b,w, iter=10000):
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
        if termin(x , x0 ):
            break
        x0 = x
    return x, k


def termin(x , x0 , tol=10e-10):
    #  # if (np.linalg.norm(np.dot(A, x)-b ) < tol):  / norm(x, ord=np.inf)
    return norm(np.subtract(x , x0), ord=1)   < tol


def error(algo_sol, true_sol):
    return "{:.3E}".format((norm(np.subtract(algo_sol , true_sol),ord=1)))