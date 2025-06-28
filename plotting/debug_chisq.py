import numpy as np
import pandas as pd
from Simulation import Simulation
from Analysis import Analysis
from ChiSq import *
from scipy.optimize import minimize
import itertools
from utils import *
import os

# This script is designed to replicate a single point of the data-fitting analysis
# to trigger and capture the detailed debugging output from ChiSq.py.

# --- Configuration ---
# Use the same parameters that were causing the error.
# You can adjust the grid point index if needed.
POINT_INDEX = 50 
CONFIG_FILE = '../config/config_orca_datafit.yaml'
TOLERANCE = 1e-5

# --- Parameter Grid Setup (copied from run_analysis_datafit.py) ---
s2t23_range = (0.45, 0.7, 10)
dm31_range = (2.4e-3, 2.6e-3, 10)

sin2t12_vals = [s2t12_bf]
sin2t13_vals = [s2t13_bf]
sin2t23_vals = np.linspace(s2t23_range[0], s2t23_range[1], int(s2t23_range[2]))
dm21_vals = [m21_bf]
dm31_vals = np.linspace(dm31_range[0], dm31_range[1], int(dm31_range[2]))
dcp_vals = [dCP_bf]

param_grid = list(itertools.product(sin2t12_vals, sin2t13_vals, sin2t23_vals, dm21_vals, dm31_vals, dcp_vals))

try:
    sin2t12, sin2t13, sin2t23, dm21, dm31, dcp = param_grid[POINT_INDEX]
except IndexError:
    raise ValueError(f"Point index {POINT_INDEX} is out of range for grid size {len(param_grid)}")

# --- Analysis Setup ---
print("--- Initializing Analysis for Debugging ---")
orca_mc_file = '../datafiles/ORCA/ORCA_MC_dataverse.parquet'
analysis = Analysis(experiment="ORCA", livetime=1.39, filename=orca_mc_file, config=CONFIG_FILE)
nominal_syst = np.array(analysis.systNominal)
print("--- Analysis Initialized ---")


# --- Run The Problematic Calculation ---
print(f"\n--- Running Grid Point #{POINT_INDEX} ---")
print(f"Parameters: sin2theta23={sin2t23:.5f}, dm31={dm31:.6e}")

# This block replicates the logic from run_analysis_datafit.py
N_dat = analysis.N_dat_fixed
t12 = np.arcsin(np.sqrt(sin2t12))
t13 = np.arcsin(np.sqrt(sin2t13))
t23 = np.arcsin(np.sqrt(sin2t23))
mod_weights_unbinned = analysis.sim.GetOscillatedRate(t12, t13, t23, dm21, dm31, dcp)
N_mod, N_mod_err = analysis.sim.BinHypothesis(mod_weights_unbinned)
analysis.sim._hypothesis_rates = mod_weights_unbinned
chisq_kwargs = {'N_mod_hypo': N_mod, 'N_mod_hypo_err': N_mod_err}

print("\n--- Calling FitSystematics ---")
# The error should be triggered inside this function call
res = analysis.FitSystematics(nominal_syst=nominal_syst, N_dat=N_dat, tol=TOLERANCE,
                              chisq_kwargs=chisq_kwargs)

del analysis.sim._hypothesis_rates

print("\n--- Debugging Run Finished ---")
print(f"Point #{POINT_INDEX} complete with minimized chi squared = {res.fun}.")
