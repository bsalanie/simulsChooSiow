
import numpy as np
import scipy.stats as sts
from math import sqrt, floor

"""
solve for equilibrium in a Choo and Siow market given systematic surplus and margins
"""


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


if __name__ == "__main__":

    # numbers of X and Y categories
    ncatX = 2
    ncatY = 2

    # numbers of men and women
    nmen = 100
    nwomen = 100

    # surplus Phi matrix
    Phi11, Phi12, Phi21, Phi22 = 0.5, 1.0, 1.0, 1.6
    Phi_mat = np.array([[Phi11, Phi12], [Phi21, Phi22]])

    typeIEV = sts.gumbel_r()

    # types of men and women
    x = np.empty(nmen)
    nmen_cat = floor(nmen/ncatX)
    for ix in range(ncatX):
        x[ix*nmen_cat:(ix+1)*nmen_cat] = ix + 1
    # just in case
    x[ncatX*nmen_cat:] = ncatX
    y = np.empty(nwomen)
    nwomen_cat = floor(nwomen/ncatY)
    for iy in range(ncatY):
        y[iy*nwomen_cat:(iy+1)*nwomen_cat] = iy + 1
    # just in case
    y[ncatY*nwomen_cat:] = ncatY
    # compute the margins
    nx = np.empty(ncatX)
    my = np.empty(ncatY)
    for ix in range(ncatX):
        nx[ix] = np.sum(x == ix + 1)
    for iy in range(ncatY):
        my[iy] = np.sum(y == iy + 1)

    muxy, mux0, mu0y, marg_err_x, marg_err_y = ipfp_solve(Phi_mat, nx, my)
    Phi_est = np.log(muxy * muxy / np.outer(mux0, mu0y))

    print("After IPFP: Phi_est == Phi_mat?")
    print("   here is Phi_mat:")
    print(Phi_mat)
    print("   and here is Phi_est:")
    print(Phi_est)
    print("margin errors for men:")
    print(marg_err_x)
    print("margin errors for women:")
    print(marg_err_y)

