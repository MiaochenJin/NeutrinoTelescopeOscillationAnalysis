import numpy as np
import pandas as pd
from SimReader import Reader
from Systematics import Systematics
from Sensitivity import get_minimized_chi2
import itertools
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Run a single grid point for sensitivity analysis")
parser.add_argument("--sin2theta23", nargs=3, type=float, default = (0.45, 0.7, 10), metavar=('MIN', 'MAX', 'N'), help="theta_23 range (in radians)")
parser.add_argument("--dm31", nargs=3, type=float, default = (2.4e-3, 2.6e-3, 10), metavar=('MIN', 'MAX', 'N'), help="delta m^2_31 range (eV^2)")
parser.add_argument("--point", type=int, default = 50, help="Index of point in the parameter grid to run")
parser.add_argument("--sin2theta12", nargs=3, type=float, default = None, help="theta_12 range (rad)")
parser.add_argument("--sin2theta13", nargs=3, type=float, default = None, help="theta_13 range (rad)")
parser.add_argument("--dm21", nargs=3, type=float, default = None, help="delta m^2_21 range")
parser.add_argument("--dcp", nargs=3, type=float, default = None, help="delta CP range (rad)")
parser.add_argument("--outfile", type=str, default="foo_point.csv", help="Output CSV filename for this point")
parser.add_argument("--newBF", type=bool, default=False, help="Whether to generate binned best fit values for a new best fit point")

args = parser.parse_args()

def parse_grid(arglist, default_val):
    if arglist is None:
        return [default_val]  # No scan → use single value (best-fit)
    return np.linspace(arglist[0], arglist[1], int(arglist[2]))

# Build full parameter grid
sin2t12_vals = parse_grid(args.sin2theta12, s2t12_bf)
sin2t13_vals = parse_grid(args.sin2theta13, s2t13_bf)
sin2t23_vals = parse_grid(args.sin2theta23, s2t23_bf)
dm21_vals = parse_grid(args.dm21, m21_bf)
dm31_vals = parse_grid(args.dm31, m31_bf)
dcp_vals = parse_grid(args.dcp. dcp_bf)

param_grid = list(itertools.product(sin2t12_vals, sin2t13_vals, sin2t23_vals, dm21_vals, dm31_vals, dcp_vals))


# Extract this job’s grid point
# Extract this job’s grid point
try:
    sin2t12, sin2t13, sin2t23, dm21, dm31, dcp = param_grid[args.point]
except IndexError:
    raise ValueError(f"Point index {args.point} is out of range for grid size {len(param_grid)}")

# Initialize Reader and Asimov
ORCA = Reader()
ORCA.SetInitialFlux()
s2t12_bf, s2t13_bf, s2t23_bf, m21_bf, m31_bf, dCP_bf = 0.303, 0.022, 0.572, 7.41e-5, 2.511e-3, 1.36 * np.pi
t12_bf = np.arcsin(np.sqrt(s2t12_bf))
t13_bf = np.arcsin(np.sqrt(s2t13_bf))
t23_bf = np.arcsin(np.sqrt(s2t23_bf))

# Best-fit parameters for Asimov dataset
if args.newBF:
    weights_bf = ORCA.GetOscillatedRate(t12_bf, t13_bf, t23_bf, m21_bf, m31_bf, dCP_bf)
    weights_non_osci = ORCA.GetOscillatedRate(0, 0, 0, 0, 0, 0)
    binned_asimov = ORCA.BinWeightedRate(weights_bf)
    binned_non_osci = ORCA.BinWeightedRate(weights_non_osci)
    np.savez("best_fit.npz", osc = binned_asimov, non_osci = binned_non_osci)
    syst_nominal = Systematics(*syst_bf)
    asimov_binned = syst_nominal.apply_systematics(binned_asimov, ORCA._E_true_bins, ORCA._E_true_centers, ORCA._cth_bin_centers)
    non_osci_binned = syst_nominal.apply_systematics(binned_non_osci, ORCA._E_true_bins, ORCA._E_true_centers, ORCA._cth_bin_centers)
    plot_reco_energy_distribution(asimov_binned, non_osci_binned, ORCA._E_reco_bins)
else:
    data = np.load('best_fit.npz')
    binned_asimov = data["osc"]
    binned_non_osci = data["non_osci"]

syst_nominal = Systematics(*syst_bf)
asimov_binned = syst_nominal.apply_systematics(binned_asimov, ORCA._E_true_bins, ORCA._E_true_centers, ORCA._cth_bin_centers)

# Run this grid point
print(f"[INFO] Running grid point #{args.point}: sin2theta23={sin2t23:.5f}, dm31={dm31:.6e}")
t12 = np.arcsin(np.sqrt(sin2t12))
t13 = np.arcsin(np.sqrt(sin2t13))
t23 = np.arcsin(np.sqrt(sin2t23))
weights = ORCA.GetOscillatedRate(t12, t13, t23, dm21, dm31, dCP)
binned = ORCA.BinWeightedRate(weights)

chi2, best_fit_syst = get_minimized_chi2(
    binned, ORCA._E_true_bins, ORCA._E_true_centers, ORCA._cth_bin_centers,
    asimov_binned, syst_nominal
)


# Save to CSV
row = pd.DataFrame([{
    "sin2theta12": sin2t12, "sin2theta13": sin2t13, "sin2theta23": sin2t23,
    "dm21": dm21, "dm31": dm31, "dcp": dcp,
    "chi2": chi2,
    "f_all": best_fit_syst[0],
    "f_HPT": best_fit_syst[1],
    "f_S": best_fit_syst[2],
    "f_HE": best_fit_syst[3],
    "f_tauCC": best_fit_syst[4],
    "f_NC": best_fit_syst[5],
    "s_mu_mubar": best_fit_syst[6],
    "s_e_ebar": best_fit_syst[7],
    "s_e_mu": best_fit_syst[8],
    "delta_gamma": best_fit_syst[9],
    "delta_theta": best_fit_syst[10]
}])
row.to_csv(args.outfile, index=False)

print(f"[INFO] Point #{args.point} complete. Result saved to {args.outfile}")