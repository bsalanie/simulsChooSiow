import numpy as np
import scipy.stats as sts

"""
simulate draws and surplus for a population for a Choo and Siow marriage market
"""


def CS_draws(nmen, nwomen, ncatX, ncatY, typedist):
    """
    simulate draws for a population
    """

    eps_vals = typedist.rvs(size=nmen * (ncatY + 1)
                            ).reshape((nmen, ncatY + 1))
    eta_vals = typedist.rvs(size=nwomen * (ncatX + 1)
                            ).reshape((nwomen, ncatX + 1))
    return eps_vals, eta_vals


def CS_random_surplus(Phi_sys, x, y, eps_vals, eta_vals):
    nmen, nwomen = x.size, y.size
    surplus_mat = np.empty((nmen + 1, nwomen + 1))
    for m in range(nmen):
        ix = x[m]
        for w in range(nwomen):
            iy = y[w]
            surplus_mat[m + 1, w + 1] = Phi_sys[ix - 1, iy - 1] + \
                eps_vals[m, iy] + eta_vals[w, ix]
    surplus_mat[1:, 0] = eps_vals[:, 0]
    surplus_mat[0, 1:] = eta_vals[:, 0]
    surplus_mat[0, 0] = 0.0
    return surplus_mat


def CS_random_surplus_direct(Phi_sys, x, y, typedist):
    ncatX, ncatY = Phi_sys.shape
    nmen, nwomen = x.size, y.size
    eps_vals, eta_vals = CS_draws(nmen, nwomen, ncatX, ncatY, typedist)
    return CS_random_surplus(Phi_sys, x, y, eps_vals, eta_vals)
