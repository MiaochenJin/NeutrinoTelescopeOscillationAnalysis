from Simulation import Simulation
from Analysis import Analysis
from ChiSq import *
from scipy.optimize import minimize
import numpy as np

# set up the analysis
livetime = 157700000 # Set to match config_sterile.yaml for consistency
# livetime = 1.39 # Set to match config_sterile.yaml for consistency

csv = "../datafiles/IC/neutrino_mc.csv"
orca = '../datafiles/ORCA/ORCA_MC_dataverse_with_muons.parquet'

analysis = Analysis(experiment = "IC", livetime = livetime, filename = csv, config = "../config/config_sterile.yaml")
# analysis = Analysis(experiment = "ORCA", livetime = 1.39, filename = orca, config = "../config/config_sterile.yaml")

# verify with some set of parameters
s2t12, s2t13, s2t23, m21, m31, dCP = 0.303, 0.03, 0.572, 7.41e-5, 2.511e-3, 1.36 * np.pi
s2t14, s2t24, s2t34, m41, dCP24 = 0, 1e-2, 0, 1e-4, 0
t12 = np.arcsin(np.sqrt(s2t12))
t13 = np.arcsin(np.sqrt(s2t13))
t23 = np.arcsin(np.sqrt(s2t23))
t14 = np.arcsin(np.sqrt(s2t14))
t24 = np.arcsin(np.sqrt(s2t24))
t34 = np.arcsin(np.sqrt(s2t34))

nominal_syst = np.array(analysis.systNominal)
# Compute N_mod (Best-Fit Rates)
# analysis.sim.ComputeBFRatesSterile(t12_bf, t13_bf, t23_bf, t14_bf, t24_bf, t34_bf, m21_bf, m31_bf, m41_bf, dCP_bf, dCP24_bf)
# analysis.sim.BinBFRatesEnergyZenith() # Assuming "3DEZ" binning from config_sterile.yaml
N_mod = analysis.sim.ReturnBFBinned()
print("lenth of N mod is ",len(N_mod))
# Compute N_dat (Hypothesis Rates)
N_dat = analysis.Compute_N_dat_sterile(sin2t12=s2t12, sin2t13=s2t13, sin2t23=s2t23,
                                sin2t14=s2t14, sin2t24=s2t24, sin2t34=s2t34,
                                dm21=m21, dm31=m31, dm41=m41, dcp=dCP, dcp24=dCP24)

print("N_mod shape:", N_mod.shape)
print("N_dat shape:", N_dat.shape)
print(np.sum(N_mod))
print(np.sum(N_dat))
print("Sum of N_mod - N_dat:", np.sum(N_mod - N_dat))

# exit(0)
# X2, JX2 = ChiSq_Jac_no_prior(analysis, nominal_syst, N_dat)
statOnly = ChiSq_only_no_prior(analysis, nominal_syst, N_dat)
print("The statistics only chi squared is ", statOnly)
# print("Without minimization, the computed chisq is ", X2)
# print("and jacobian is ", JX2)
exit(0)
# now we minimize chisq and find the best values of N_mod and N_dat
# 1) objective wrapper that only depends on syst
def obj(syst):
    return ChiSq_Jac_with_penalty(analysis, syst, N_dat)

analysis.SystPrior, bounds = syst_penalty_prior(analysis, nominal_syst, N_dat)
# Combined chi^2 minimization
tol = max(1e-6,np.sqrt(statOnly)*1e-6)

# 4) call the minimizer
res = minimize(obj,
               analysis.SystPrior,
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