from Simulation import Simulation
from Analysis import Analysis
from ChiSq import *
from scipy.optimize import minimize

# set up the analysis
livetime = 5 * 365 * 24 * 60 * 60 # assume 10 years
csv = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/NeutrinoTelescopeOscillationAnalysis/datafiles/IC/neutrino_mc.csv"
this_IC = '../datafiles/IC/neutrino_mc.csv'
ORCA = '../datafiles/ORCA/ORCA_MC.parquet'
Analysis = Analysis(experiment = "IC", livetime = livetime, filename = this_IC)

# verify with some set of parameters
s2t12_bf, s2t13_bf, s2t23_bf, m21_bf, m31_bf, dCP_bf = 0.303, 0.022, 0.572, 7.41e-5, 2.511e-3, 1.36 * np.pi
t12_bf = np.arcsin(np.sqrt(s2t12_bf))
t13_bf = np.arcsin(np.sqrt(s2t13_bf))
t23_bf = np.arcsin(np.sqrt(s2t23_bf))

s2t12, s2t13, s2t23, m21, m31, dCP = 0.303, 0.03, 0.572, 7.41e-5, 2.511e-3, 1.36 * np.pi
t12 = np.arcsin(np.sqrt(s2t12))
t13 = np.arcsin(np.sqrt(s2t13))
t23 = np.arcsin(np.sqrt(s2t23))

nominal_syst = np.array(Analysis.systNominal)
obs_weights = Analysis.sim.GetOscillatedRate(t12, t23_bf, t23, m21, m31, dCP)
N_mod = Analysis.sim._BF_rates_weighted_binned
N_dat = Analysis.sim.BinWeightedRate3DFlatten(obs_weights)
print(np.sum(N_mod - N_dat))

# # now compare the rates
# old = np.load("../datafiles/OldCode/N_mod_sterile.npz")
# old_N_mod = old["N_mod"]
# old_N_dat = old["N_dat"]
# print("difference between N dat is ", np.sum(N_dat - old_N_dat))
# print("difference between N mod is ", np.sum(N_mod - old_N_mod))

simple = ChiSq_only_no_prior(Analysis, nominal_syst, N_dat)
print("simple chisq is ", simple)
exit(0)

X2, JX2 = ChiSq_Jac_with_penalty(Analysis, nominal_syst, N_dat)
statOnly = ChiSq_only_no_prior(Analysis, nominal_syst, N_dat)
print("the statistics only chi squared is ", statOnly)
print("without minimization, the computed chisq is ", X2)
print("and jacobian is ", JX2)



exit(0)
# now we minimize chisq and find the best values of N_mod and N_dat
# 1) objective wrapper that only depends on syst
def obj(syst):
    return ChiSq_Jac_with_penalty(Analysis, syst, N_dat)

Analysis.SystPrior, bounds = syst_penalty_prior(Analysis, nominal_syst, N_dat)
# Combined chi^2 minimization
tol = max(1e-6,np.sqrt(statOnly)*1e-6)

# 4) call the minimizer
res = minimize(obj,
               Analysis.SystPrior,
               method='L-BFGS-B',   # or 'Nelder-Mead', 'trust-constr', etc.
               jac = True,
               bounds=bounds,
               options={'ftol':tol, 'disp':True})

if res.success:
    best_syst = res.x
    print("Converged!  χ²_min =", res.fun)
    print("Best-fit syst =", best_syst)
else:
    print("Minimization failed:", res.message)
