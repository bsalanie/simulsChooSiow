import numpy as np
import scipy.stats as sts
from scipy.optimize import linprog
from math import sqrt

# numbers of X and Y categories
ncatX = 2
ncatY = 2

# numbers of men and women
nmen = 400
nwomen = 400

# surplus Phi matrix
Phi11, Phi12, Phi21, Phi22 = 0.5, 1.0, 1.0, 1.6
Phi_mat = np.array([[Phi11, Phi12], [Phi21, Phi22]])

typeIEV = sts.gumbel_r()


# simulate one man's draws
def new_man():
    eps_vals = typeIEV.rvs(size=ncatY + 1)
    return eps_vals


# simulate one woman's draws
def new_woman():
    eta_vals = typeIEV.rvs(size=ncatX + 1)
    return eta_vals


# simulate a population
def new_popu():
    # types of men and women
    x = np.random.randint(1, ncatX + 1, nmen)
    y = np.random.randint(1, ncatY + 1, nwomen)
    # their draws
    eps_vals = typeIEV.rvs(size=nmen * (ncatY + 1)).reshape((nmen, ncatY + 1))
    eta_vals = typeIEV.rvs(size=nwomen * (ncatX + 1)).reshape((nwomen, ncatX + 1))
    return x, y, eps_vals, eta_vals


def ipfp_solve(Phi, nx, my, eps_diff=1e-6):
    ncatX = nx.shape[0]
    ncatY = my.shape[0]
    assert Phi.shape == (ncatX, ncatY)
    ephi2 = np.exp(Phi / 2.0)
    ephi2T = ephi2.T
    nindivs = np.sum(nx) + np.sum(my)
    bigc = sqrt(nindivs / (ncatX + ncatY + np.sum(ephi2)))
    txi = np.full((ncatX), bigc)
    tyi = np.full((ncatY), bigc)
    err_diff = bigc
    while err_diff > eps_diff:
        sx = ephi2 @ tyi
        tx = (np.sqrt(sx * sx + 4.0 * nx) - sx) / 2.0
        sy = ephi2T @ tx
        ty = (np.sqrt(sy * sy + 4.0 * my) - sy) / 2.0
        err_x = np.max(np.abs(tx - txi))
        err_y = np.max(np.abs(ty - tyi))
        err_diff = err_x + err_y
        # print(f"Errors {err_x} and {err_y}")
        txi = tx
        tyi = ty
    mux0 = tx * tx
    mu0y = ty * ty
    muxy = ephi2 * np.sqrt(np.outer(mux0, mu0y))
    marg_err_x = mux0 + np.sum(muxy, 1) - nx
    marg_err_y = mu0y + np.sum(muxy, 0) - my
    # print(f"Margin error on x: {np.max(np.abs(marg_err_x))}")
    # print(f"Margin error on y: {np.max(np.abs(marg_err_y))}")
    return muxy, mux0, mu0y, marg_err_x, marg_err_y

x, y, eps_vals, eta_vals = new_popu()

nx = np.empty(ncatX)
my = np.empty(ncatY)
for ix in range(ncatX):
    nx[ix] = np.sum(x == ix + 1)
for iy in range(ncatY):
    my[iy] = np.sum(y == iy + 1)

muxy, mux0, mu0y, marg_err_x, marg_err_y = ipfp_solve(Phi_mat, nx, my)
Phi_est = np.log(muxy * muxy / np.outer(mux0, mu0y))

print("After IPFP: Phi_est == Phi_mat?")
print(Phi_mat)
print(Phi_est)
print("margin errors:")
print(marg_err_x)
print(marg_err_y)