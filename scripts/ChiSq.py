import numpy as np
import math
from scipy.special import gamma
from Systematics import *

# compute chi squared without penalty or jacobian or analytic prior for jacobian 
def ChiSq(analysis, syst, N_dat):
    sim = analysis.sim
    syst_ls = analysis.systNames
    syst_reg = analysis.systRegistry
    N_mod = sim._BF_rates_weighted_binned
    assert(N_dat.shape == N_mod.shape)
    syst_shift = 0

    for i, sname in enumerate(syst_ls):
        apply_fn, diff_fn = syst_reg[sname]
        syst_shift += apply_fn(syst[i], sim)
    N_mod_syst = N_mod * (1 + syst_shift)

    X2 = 2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst))
    print("Total Chi Squared is ", np.sum(X2))
    return X2