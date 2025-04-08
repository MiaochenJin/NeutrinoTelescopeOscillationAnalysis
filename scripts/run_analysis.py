import numpy as np
import pandas as pd
from SimReader import Reader
from Systematics import Systematics
from Sensitivity import get_minimized_chi2
import itertools
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Run sensitivity analysis over oscillation parameter grid")
parser.add_argument("--theta23", nargs=3, type=float, metavar=('MIN', 'MAX', 'N'), help="sin^2(theta_23) range")
parser.add_argument("--dm31", nargs=3, type=float, metavar=('MIN', 'MAX', 'N'), help="delta m^2_31 range (eV^2)")
parser.add_argument("--outfile", type=str, default="sensitivity_output.csv", help="Output CSV file")
args = parser.parse_args()

# --- Oscillation parameter grid ---
t23_vals = np.linspace(args.theta23[0], args.theta23[1], int(args.theta23[2]))
dm31_vals = np.linspace(args.dm31[0], args.dm31[1], int(args.dm31[2]))
param_grid = list(itertools.product(t23_vals, dm31_vals))

# --- Set up Reader and Asimov dataset ---
print("[INFO] Initializing simulation and Asimov dataset...")
ORCA = Reader()
ORCA.SetInitialFlux()

# Best-fit oscillation parameters (used for Asimov)
t12_bf, t13_bf, t23_bf, m21_bf, m31_bf, dCP_bf = 0.304, 0.022, 0.572, 7.42e-5, 2.514e-3, 1.36 * np.pi
weights_bf = ORCA.GetOscillatedRate(t12_bf, t13_bf, t23_bf, m21_bf, m31_bf, dCP_bf)
binned_asimov = ORCA.BinWeightedRate(weights_bf)

# Use no systematics for Asimov
syst_nominal = Systematics(*([0]*7 + [0]*5))  # All systematics set to 0
asimov_binned = syst_nominal.apply_systematics(binned_asimov, ORCA._E_true_bins, ORCA._E_true_centers, ORCA._cth_bin_centers)

# --- Run sensitivity scan ---
results = []
print("[INFO] Starting grid scan...")
for i, (t23, dm31) in enumerate(param_grid):
    print(f"  --> [{i+1}/{len(param_grid)}] t23={t23:.5f}, dm31={dm31:.6e}")
    weights = ORCA.GetOscillatedRate(t12_bf, t13_bf, t23, m21_bf, dm31, dCP_bf)
    binned = ORCA.BinWeightedRate(weights)

    chi2, best_fit_syst = get_minimized_chi2(
        binned, ORCA._E_true_bins, ORCA._E_true_centers, ORCA._cth_bin_centers,
        asimov_binned, syst_nominal
    )

    results.append({"sin2_theta23": t23, "dm31": dm31, "chi2": chi2})

# --- Save results ---
pd.DataFrame(results).to_csv(args.outfile, index=False)
print(f"[INFO] Analysis complete. Results saved to {args.outfile}")
