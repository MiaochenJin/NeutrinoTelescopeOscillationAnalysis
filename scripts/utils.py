import nuSQuIDS as nsq
import nuflux
import numpy as np
from matplotlib import pyplot as plt

units = nsq.Const()
interactions = False
flux = nuflux.makeFlux('IPhonda2014_sk_solmin')

neutrino_flavors = 3
# some earth parameters to compute L/E
R_E = 6371 # KM

# Set up mixing parameters as in nu-fit5.0
t12_bf = np.arcsin(np.sqrt(0.303))
t13_bf = np.arcsin(np.sqrt(0.02203))
t23_bf = np.arcsin(np.sqrt(0.572))
m21_bf = 7.41e-5
m31_bf = 2.511e-3
dCP_bf = 4.3

# define a dictionary for nutrino type and nuflux
f_dict = dict({"nue": 0, "numu": 1, "nutau": 2})
t_dict = dict({"nu": 0, "nubar": 1})
nuflux_label = dict({"nue_nu": nuflux.NuE, "nue_nubar": nuflux.NuEBar, 
					 "numu_nu": nuflux.NuMu, "numu_nubar": nuflux.NuMuBar, 
					 "nutau_nu": nuflux.NuTau, "nutau_nubar": nuflux.NuTauBar})
# define dict to grab flux from nuflux
nuflux_dict = dict({12: (0, 0), -12: (1,0),
                    14: (0,1), -14: (1,1),
                    16: (0,2), -16: (1,2)})