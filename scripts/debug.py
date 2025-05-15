from Simulation import Simulation
from Analysis import Analysis
from ChiSq import *

# set up the analysis
livetime = 10 * 365 * 24 * 60 * 60 # assume 10 years
Analysis = Analysis(experiment = "IC", livetime = livetime, filename = '../datafiles/IC/neutrino_mc.csv')

# verify with some set of parameters
s2t12_bf, s2t13_bf, s2t23_bf, m21_bf, m31_bf, dCP_bf = 0.303, 0.022, 0.572, 7.41e-5, 2.511e-3, 1.36 * np.pi
t12_bf = np.arcsin(np.sqrt(s2t12_bf))
t13_bf = np.arcsin(np.sqrt(s2t13_bf))
t23_bf = np.arcsin(np.sqrt(s2t23_bf))
nominal_syst = [1., 1., 1., 0., 1., 1., 0., 0.] # only flux systematics are implemented at the moment
nominal_weights = Analysis.sim.GetOscillatedRate(t12_bf, t13_bf, t23_bf, m21_bf, m31_bf, dCP_bf)

# you can get the reco energy distribution 
E_dist = Analysis.sim.BinWeightedRateRecoEnergy(nominal_weights)
print(E_dist.shape)
# you can get the chisq
N_dat = Analysis.sim.BinWeightedRate3DFlatten(nominal_weights)
ChiSq(Analysis, nominal_syst, N_dat)