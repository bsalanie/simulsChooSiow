"""
Equilibrium on a marriage market
Takes about 2 seconds for nmen = nwomen = 100
            10 seconds                    200
            50 seconds                    400
            250 seconds                   800
"""

import pulp
import numpy as np
import pandas as pd
import time
from Timer import Timer
from expand_grid import expand_grid


def get_eq_marriage(surplus_mat):
    nmen1, nwomen1 = surplus_mat.shape

    # create a pandas data frame
    im = np.arange(nmen1)
    iw = np.arange(nwomen1)
    df_marriage = expand_grid({'i': im, 'j': iw})
    df_marriage['surplus'] = surplus_mat.ravel()
    df_marriage.set_index(['i', 'j'], inplace=True)

    x = pulp.LpVariable.dicts("x", ((i, j) for i, j in df_marriage.index),
                              0.0, 1.0)

    marriage_prob = pulp.LpProblem("Marriage market", pulp.LpMaximize)

    #  first the objective function
    marriage_prob += pulp.lpSum(
        [x[i, j] * df_marriage.loc[(i, j), 'surplus'] for i, j in df_marriage.index])

    # then the margin constraints
    for i in range(1, nmen1):
        marriage_prob += pulp.lpSum(x[i, j] for j in iw) == 1.0

    for j in range(1, nwomen1):
        marriage_prob += pulp.lpSum(x[i, j] for i in im) == 1.0

    # and we don't want '0' to match with '0'
    marriage_prob += x[0, 0] == 0.0

    marriage_prob.solve()

    status_string = pulp.LpStatus[marriage_prob.status]

    if status_string == "Optimal":

        # create output DataFrame
        output = []
        for i, j in x:
            var_output = {
                'm': i,
                'w': j,
                'Probability': x[(i, j)].varValue,
            }
            output.append(var_output)

        output_df = pd.DataFrame.from_records(
            output).sort_values(['m', 'w'])

        output_df.set_index(['m', 'w'], inplace=True)

        return marriage_prob, status_string, output_df
    
    else:
        return marriage_prob, status_string, None


def print_solution(prob):
    val_obj = pulp.value(prob.objective)
    print(f"\n\n The value of the equilibrium total  surplus is {val_obj}\n")
    print("\n     the equilibrium matches are as follows:\n")
    for v in prob.variables():
        v_val = v.varValue
        if v_val > 1e-6:
            print(f"{v.name} = {v_val}")


def marriage_patterns(matching_probas: pd.DataFrame, x, y):
    ncatX, ncatY = np.max(x), np.max(y)
    ncatX1, ncatY1 = ncatX + 1, ncatY + 1
    nmen, nwomen = x.size, y.size
    nmen1, nwomen1 = nmen + 1, nwomen + 1
    mu = np.empty((ncatX1, ncatY1))
    cat_x = np.zeros(matching_probas.size, dtype=int)
    for m in range(1, nmen1):
        cat_x[m*nwomen1:(m+1)*nwomen1] = x[m-1]
    cat_y = np.zeros_like(cat_x)
    for w in range(1, nwomen1):
        cat_y[w::nmen1] = y[w-1]
    matching_patterns = pd.DataFrame(
        {'proba': matching_probas['Probability'].values, 'x': cat_x, 'y': cat_y})
    # print(matching_patterns.head())
    for xy, data_xy in matching_patterns.groupby(['x', 'y']):
        # print(f"xy: {xy}")
        # print(data_xy.head())
        mu[xy] = data_xy['proba'].sum()
    mux0 = mu[1:, 0]
    mu0y = mu[0, 1:]
    muxy = mu[1:, 1:]
    return muxy, mux0, mu0y


if __name__ == "__main__":
    nmen = nwomen = 10

    nmen1, nwomen1 = nmen + 1, nwomen + 1

    surplus_mat = np.array(np.arange(nmen1*nwomen1)).reshape((nmen1, nwomen1))
    surplus_mat[0, :] = 0.0
    surplus_mat[:, 0] = 0.0

    with Timer() as t:
        marriage_eq, status_string, marriage_probas = get_eq_marriage(surplus_mat)

    print(
        f"Computing the equilibrium with {nmen} men and {nwomen} women  took {t.elapsed:.3f} seconds")
    print(f"Status was: {status_string}")
    
    if status_string == "Optimal":
        print_solution(marriage_eq)
        ncatX = 2
        ncatY = 3
        x = np.random.randint(1, ncatX + 1, size=nmen)
        y = np.random.randint(1, ncatY + 1, size=nwomen)
        print(f"\n the types of men are:\n {x}")
        print(f"\n the types of women are:\n {y}")
        muxy, mux0, mu0y = marriage_patterns(marriage_probas, x, y)
        print(f"\nmuxy:\n {muxy}")
        print(f"\nmux0:\n {mux0}")
        print(f"\nmu0y:\n {mu0y}")
