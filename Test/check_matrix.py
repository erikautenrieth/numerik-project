import numpy as np
import pandas as pd
from scipy.sparse import diags
from numpy.linalg import *

def isDDM(m, n) :
    for i in range(0, n) :
        # for each column, finding
        # sum of each row.
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(m[i][j])
            # removing the
        # diagonal element.
        sum = sum - abs(m[i][i])
        # checking if diagonal
        # element is less than
        # sum of non-diagonal
        # element.
        if (abs(m[i][i]) < sum) :
            return False
    return True

def characteristics(M):
    #print("shape:", np.shape(M))
    #print("det:  ", np.linalg.det(M))
    #print("norm: ", np.linalg.norm(M))
    #print("rank: ", np.linalg.matrix_rank(M))
    return np.linalg.det(M)


def create_block_triag(n):
    k = [np.ones(n-1)*(-1),2*np.ones(n),np.ones(n-1)*(-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()
    return A


def runntime_compare_df(mat, jac_it, gasei_it, sor_it, gael_it,jac_time, gasei_time, sor_time_arr, gael_time, label = "Zeit (sek)"):
    mux = pd.MultiIndex.from_product([["Jacobi", "Gauß Seidel", "SOR", "Gauß Eliminierung"], ['Iterationen',f'{label}']])
    df = pd.DataFrame( columns=mux)
    df["Jacobi","Iterationen"] = jac_it
    df["Gauß Seidel","Iterationen"] = gasei_it
    df["SOR","Iterationen"] = sor_it
    df["Gauß Eliminierung","Iterationen"] = gael_it
    df["Jacobi",label] = jac_time
    df["Gauß Seidel",label] = gasei_time
    df["SOR",label] = sor_time_arr
    df["Gauß Eliminierung",label] = gael_time
    df["Matrixgröße"] = mat
    df.set_index("Matrixgröße", inplace=True)

    latex_name =f"vergleich{label}.tex"
    df.to_latex("C:\\Users\\eaut2\\DataspellProjects\\Numerik-Projekt\\Test\\Latex\\"f"{latex_name}",
                index=True, bold_rows=True, caption="Laufzeitvergleich.", position="h!", label="laufzeit")

    return df


def create_mat(n,add=4):
    from scipy.sparse import random
    from scipy.stats import rv_continuous
    from numpy.random import default_rng
    #S = random(n, n, density=0.75, random_state=rng, data_rvs=Y.rvs)
    #A = S.A
    class CustomDistribution(rv_continuous):
        def _rvs(self,  size=None, random_state=None):
            return random_state.standard_normal(size)
    rng = default_rng()
    X = CustomDistribution(seed=rng)
    Y = X()  # get a frozen version of the distribution

    A=np.random.randint(low=-1,high=3,size=(n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j]+= sum(abs(A[i]))+1

    A[0][0] += 0.5
    A[n-1][n-1] += 0.5

    if  characteristics(A) == 0:
        A = create_block_triag(n)
        print("Determinante ist Null")
    if isDDM(m=A,n=n):
        return A
    else:
        print("Nicht DDM")