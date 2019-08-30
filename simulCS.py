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

# simulate a population (stratified by types)
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



# generate surplus matrix
Phi_rand = np.empty((nmen, nwomen))
for i in range(nmen):
    xi = x[i]
    for j in range(nwomen):
        yj = y[j]
        Phi_rand[i, j] = Phi_mat[xi - 1, yj - 1] + eps_vals[i, yj] + eta_vals[j, xi]
Phi_single_men = eps_vals[:, 0]
Phi_single_women = eta_vals[:, 0]

c = np.empty((ntot))
c[:nprod] = Phi_rand.ravel(order='F')
c[nprod:(nprod + nmen)] = Phi_single_men
c[(nprod + nmen):] = Phi_single_women


opt_matching = linprog(-c, A_eq=A_eq, b_eq=b_eq, method='simplex')
# print(opt_matching)

opt_probs = opt_matching.x
x_marr = opt_probs[:nprod].reshape((nmen, nwomen), order='F')
x_single_men = opt_probs[nprod:(nprod + nmen)]
x_single_women = opt_probs[(nprod + nmen):]

print("Check on optimal matching:")
iproblem = False
eps = 1e-6
marriages = np.argwhere(x_marr > 1.0 - eps)
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
marg_err_i = np.sum(x_marr, 1) +  x_single_men - 1.0
marg_err_j = np.sum(x_marr, 0) +  x_single_women - 1.0
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
