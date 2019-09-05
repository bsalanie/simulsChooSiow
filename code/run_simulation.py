#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ipfp_routines import ipfp_solve
import numpy as np
import scipy.stats as sts
from marriage_Market import get_eq_marriage, marriage_patterns
from math import sqrt, floor
from CS_draws import CS_random_surplus_direct
from Timer import Timer


# In[2]:


def equal_cats(nmen, ncatX):
    x = np.empty(nmen, dtype=int)
    nmen_cat = floor(nmen/ncatX)
    for ix in range(ncatX):
        x[ix*nmen_cat:(ix+1)*nmen_cat] = ix + 1
    # just in case
    x[ncatX*nmen_cat:] = ncatX
    return x


# In[3]:


# numbers of X and Y categories
ncatX = ncatY = 2

# numbers of men and women
nmen = nwomen = 100

# surplus Phi matrix
Phi11, Phi12, Phi21, Phi22 = 0.5, 1.0, 1.0, 1.6
Phi_sys = np.array([[Phi11, Phi12], [Phi21, Phi22]])

typeIEV = sts.gumbel_r()

# types of men and women
x = equal_cats(nmen, ncatX)
y = equal_cats(nwomen, ncatY)

# random surplus
Phi_rand = CS_random_surplus_direct(Phi_sys, x, y, typeIEV)

# print(Phi_rand)


# In[4]:


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


# In[5]:


def estimate_Phi():
    # random surplus
    Phi_rand = CS_random_surplus_direct(Phi_sys, x, y, typeIEV)
    _, marriage_probas = get_eq_marriage(Phi_rand)
    muxy, mux0, mu0y = marriage_patterns(marriage_probas, x, y)
    Phi_est = np.log(muxy * muxy / np.outer(mux0, mu0y))
    return Phi_est


def Phi_comp(Phi):
    return Phi[0, 0] + Phi[1, 1] - Phi[0, 1] - Phi[1, 0]


# In[6]:

Phi_comp(estimate_Phi())


# In[ ]:


# numbers of men and women
nmen = nwomen = 400

# numbers of X and Y categories
ncatX = ncatY = 2

# surplus Phi matrix
Phi11, Phi12, Phi21, Phi22 = 0.5, 1.0, 1.0, 1.6
Phi_sys = np.array([[Phi11, Phi12], [Phi21, Phi22]])

typeIEV = sts.gumbel_r()

# types of men and women
x = equal_cats(nmen, ncatX)
y = equal_cats(nwomen, ncatY)


nsimuls = 50
Phi_est_sim = np.zeros((nsimuls, ncatX, ncatY))
Phi_comp_true = Phi_comp(Phi_sys)
print(f"\nTrue complementarity = {Phi_comp_true}")
Phi_comp_sim = np.zeros(nsimuls)
for isimul in range(nsimuls):
    Phi_est = estimate_Phi()
    Phi_est_sim[isimul, :, :] = Phi_est
    Phi_comp_sim[isimul] = Phi_comp(Phi_est)
    print(f"Sample {isimul}: estimated complementarity {Phi_comp_sim[isimul]}")

np.save("Phi_est_sim.npy", Phi_est_sim)
np.save("Phi_comp_sim.npy", Phi_comp_sim)

print(
    f"The mean and standard error of the estimated complementarity (true value {Phi_comp_true})  are")

# In[19]:


print(f"{Phi_comp_sim.mean()} and {Phi_comp_sim.std()}")


# In[ ]:

nx = np.full(ncatX, nmen/ncatX)
my = np.full(ncatY, nmen/ncatY)
muxy, mux0, mu0y, _, _ = ipfp_solve(Phi_sys, nx, my)
I1 = I2 = nmen/ncatX
p1_1 = muxy[0, 0]/I1
p1_2 = muxy[0, 1]/I1
p2_1 = muxy[1, 0]/I2
p2_2 = muxy[1, 1]/I2
Phi_comp_std_linh = sqrt(
    (4.0/I1)*(1.0/p1_1 + 1.0/p1_2) + (4.0/I2)*(1.0/p2_1 + 1.0/p2_2))

print(f"The Linh standard error is {Phi_comp_std_linh}.")
