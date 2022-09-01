import numpy as np

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