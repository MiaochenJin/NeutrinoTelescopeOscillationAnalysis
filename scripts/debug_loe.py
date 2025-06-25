from Simulation import Simulation
from Analysis import Analysis
from ChiSq import *
from scipy.optimize import minimize
import time

# set up the analysis
livetime = 1.39
orca = '../datafiles/ORCA/ORCA_MC.parquet'
Analysis = Analysis(experiment = "ORCA", livetime = livetime, filename = orca, config = "../config/config_orca.yaml")
# verify with some set of parameters
s2t12_bf, s2t13_bf, s2t23_bf, m21_bf, m31_bf, dCP_bf = 0.303, 0.022, 0.572, 7.41e-5, 2.511e-3, 1.36 * np.pi
t12_bf = np.arcsin(np.sqrt(s2t12_bf))
t13_bf = np.arcsin(np.sqrt(s2t13_bf))
t23_bf = np.arcsin(np.sqrt(s2t23_bf))

s2t12, s2t13, s2t23, m21, m31, dCP = 0.303, 0.022, 0.8, 7.41e-5, 2.511e-3, 1.36 * np.pi
t12 = np.arcsin(np.sqrt(s2t12))
t13 = np.arcsin(np.sqrt(s2t13))
t23 = np.arcsin(np.sqrt(s2t23))

nominal_syst = np.array(Analysis.systNominal)
N_mod = Analysis.sim.ReturnBFBinned()
N_dat = Analysis.Compute_N_dat(sin2t12 = s2t12, sin2t13 = s2t13, sin2t23 = s2t23, 
                                dm21 = m21, dm31 = m31, dcp = dCP)
print(np.sum(N_mod - N_dat))
X2, JX2 = ChiSq_Jac_with_penalty(Analysis, nominal_syst, N_dat)
statOnly = ChiSq_only_no_prior(Analysis, nominal_syst, N_dat)
print("the statistics only chi squared is ", statOnly)
print("without minimization, the computed chisq is ", X2)
print("and jacobian is ", JX2)

X2 = ChiSq_with_penalty_with_error(Analysis, nominal_syst, N_dat)
print("with the BB method implemented, the new X2 is ", X2)
exit(0)
res = Analysis.FitSystematics(nominal_syst = nominal_syst, N_dat = N_dat, tol = 1e-5, 
                        stat_chisq_fn = ChiSq_only_no_prior, 
                        full_chisq_fn = ChiSq_with_penalty_with_error)

best_syst = res.x
chi2 = res.fun
if res.success:
    best_syst = res.x
    print("Converged!  χ²_min =", res.fun)
    print("Best-fit syst =", best_syst)
else:
    print("Minimization failed:", res.message)
