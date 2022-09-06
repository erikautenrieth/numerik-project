import numpy as np
import pandas as pd

def isDDM(m, n) :
    for i in range(0, n) :
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(m[i][j])
        sum = sum - abs(m[i][i])
        if (abs(m[i][i]) < sum) :
            return False
    return True

def characteristics(M):
    #print("shape:", np.shape(M))
    #print("det:  ", np.linalg.det(M))
    #print("norm: ", np.linalg.norm(M))
    #print("rank: ", np.linalg.matrix_rank(M))
    return np.linalg.det(M)

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

    latex_name =f"compare_{label}.tex"
    df.to_latex("C:\\Users\\eaut2\\DataspellProjects\\Numerik-Projekt\\Compare Methods\\Latex\\"f"{latex_name}",
                index=True, bold_rows=True, caption="Laufzeitvergleich.", position="h!", label="laufzeit")

    return df

def print_lgs(A,b):
    from sympy import symbols, Matrix
    from IPython.display import display, Math
    import sympy as sym
    l1, l2, l3,l4, la = symbols("x1 x2 x3 x.. x")
    l = Matrix([l1,l2,l3,l4])
    r = Matrix(np.round(b,4))
    a = Matrix(np.round(A,4))
    display(Math('\ \\text{  }%s   \\text{*} %s \\text{= } %s' % (sym.latex(sym.simplify(a)) , (sym.latex(sym.simplify(l))), (sym.latex(sym.simplify(r))))))