import numpy as np
import scipy.stats as sts
from marriage_Market import get_eq_marriage, marriage_patterns
from math import sqrt, floor
from CS_draws import CS_random_surplus_direct
from Timer import Timer


if __name__ == "__main__":

    # numbers of X and Y categories
    ncatX = ncatY = 2

    # numbers of men and women
    nmen = nwomen = 400

    # surplus Phi matrix
    Phi11, Phi12, Phi21, Phi22 = 0.5, 1.0, 1.0, 1.6
    Phi_sys = np.array([[Phi11, Phi12], [Phi21, Phi22]])

    typeIEV = sts.gumbel_r()

    # types of men and women
    x = np.empty(nmen, dtype=int)
    nmen_cat = floor(nmen/ncatX)
    for ix in range(ncatX):
        x[ix*nmen_cat:(ix+1)*nmen_cat] = ix + 1
    # just in case
    x[ncatX*nmen_cat:] = ncatX
    y = np.empty(nwomen, dtype=int)
    nwomen_cat = floor(nwomen/ncatY)
    for iy in range(ncatY):
        y[iy*nwomen_cat:(iy+1)*nwomen_cat] = iy + 1
    # just in case
    y[ncatY*nwomen_cat:] = ncatY

    # random surplus
    Phi_rand = CS_random_surplus_direct(Phi_sys, x, y, typeIEV)

    # print(Phi_rand)

    with Timer() as t:
        marriage_eq, marriage_probas = get_eq_marriage(Phi_rand)

    print(
        f"Computing the equilibrium with {nmen} men and {nwomen} women  took {t.elapsed:.3f} seconds")

    muxy, mux0, mu0y = marriage_patterns(marriage_probas, x, y)

    print(f"\nmuxy:\n {muxy}")
    print(f"\nmux0:\n {mux0}")
    print(f"\nmu0y:\n {mu0y}")

    Phi_est = np.log(muxy * muxy / np.outer(mux0, mu0y))

    print(f"\n\nPhi_est: {Phi_est}")
