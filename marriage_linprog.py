
"""
solving for equilibrium in a marriage market:
  scipy.optimize.linprog version
"""

import numpy as np
import scipy.stats as sts
import scipy.optimize as spopt


def linprog_marriage_eq(Phi, nx, my):
    ncatX, ncatY = nx.size, my.size
    nmen, nwomen = Phi.shape
    # set uo constraints of linear program
    nsum = nmen + nwomen
    nprod = nmen * nwomen
    ntot = nprod + nsum
    A_eq = np.full((nsum, ntot), 0.0)
    for i in range(nmen):
        for j in range(nwomen):
            ij = j * nmen + i
            A_eq[i, ij] = 1.0
            A_eq[nmen + j, ij] = 1.0
    for i in range(nmen):
        A_eq[i, nprod + i] = 1.0
    for j in range(nwomen):
        A_eq[nmen + j, nprod + nmen + j] = 1.0
    b_eq = np.full((nsum), 1.0)

    c = np.empty((ntot))
    c[:nprod] = Phi_rand.ravel(order='F')
    c[nprod:(nprod + nmen)] = Phi_single_men
    c[(nprod + nmen):] = Phi_single_women

    opt_matching = spopt.linprog(-c, A_eq=A_eq, b_eq=b_eq, method='simplex')
    # print(opt_matching)

    opt_probs = opt_matching.x
    x_marr = opt_probs[:nprod].reshape((nmen, nwomen), order='F')
    x_single_men = opt_probs[nprod:(nprod + nmen)]
    x_single_women = opt_probs[(nprod + nmen):]

    print("Check on optimal matching:")
    iproblem = False
    eps = 1e-6
    single_men = np.argwhere(x_single_men > 1.0 - eps).flatten()
    single_women = np.argwhere(x_single_women > 1.0 - eps).flatten()

    # for (i, j) in marriages:
    # print(f"  {i + 1} and {j + 1} are married")
    for i in single_men:
        # print(f"   Man {i + 1} is single")
        if np.sum(x_marr[i, :]) > eps:
            print(" Man {i + 1} is single and also married!")
            iproblem = True
    for j in single_women:
        # print(f"   Woman {j + 1} is single")
        if np.sum(x_marr[:, j]) > eps:
            print(" Woman {j + 1} is single and also married!")
            iproblem = True
        marg_err_i = np.sum(x_marr, 1) + x_single_men - 1.0
        marg_err_j = np.sum(x_marr, 0) + x_single_women - 1.0
        err_i = np.argwhere(abs(marg_err_i) > eps).flatten()
        err_j = np.argwhere(abs(marg_err_j) > eps).flatten()
        if err_i.size > 0:
            iproblem = True
        for i in err_i:
            print(f"Margin error of {marg_err_i[i]} for man {i+1}")
        if err_j.size > 0:
            iproblem = True
        for j in err_j:
            print(f"Margin error of {marg_err_j[j]} for woman {j+1}")

        if not iproblem:
            print("  all good in this matching.")


if __name__ == "__main__":
    nmen = 100
    nwomen = 100
    ncatX = ncatY = 2
    x = np.random.randint(1, ncatX + 1, size=nmen)
    y = np.random.randint(1, ncatY + 1, size=nwomen)
    # generate a surplus matrix
    Phi_rand = np.empty((nmen, nwomen))
    for i in range(nmen):
        xi = x[i]
        for j in range(nwomen):
            yj = y[j]
            Phi_rand[i, j] = Phi_mat[xi - 1, yj - 1] + \
                eps_vals[i, yj] + eta_vals[j, xi]
    Phi_single_men = eps_vals[:, 0]
    Phi_single_women = eta_vals[:, 0]
