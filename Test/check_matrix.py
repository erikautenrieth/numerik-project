import numpy as np
from scipy.sparse import diags


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