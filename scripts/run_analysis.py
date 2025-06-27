import numpy as np
import pandas as pd
from Simulation import Simulation
from Analysis import Analysis
from ChiSq import *
from scipy.optimize import minimize
import itertools
import argparse
from utils import *
import os
parser = argparse.ArgumentParser(description="Run a single grid point for sensitivity analysis")
parser.add_argument("--sin2theta23", nargs=3, type=float, default = (0.45, 0.7, 10), metavar=('MIN', 'MAX', 'N'), help="theta_23 range (in radians)")
parser.add_argument("--dm31", nargs=3, type=float, default = (2.4e-3, 2.6e-3, 10), metavar=('MIN', 'MAX', 'N'), help="delta m^2_31 range (eV^2)")
parser.add_argument("--point", type=int, default = 50, help="Index of point in the parameter grid to run")
parser.add_argument("--sin2theta12", nargs=3, type=float, default = None, help="theta_12 range (rad)")
parser.add_argument("--sin2theta13", nargs=3, type=float, default = None, help="theta_13 range (rad)")
parser.add_argument("--dm21", nargs=3, type=float, default = None, help="delta m^2_21 range")
parser.add_argument("--dcp", nargs=3, type=float, default = None, help="delta CP range (rad)")
parser.add_argument("--livetime", type=float, default = 5, help="livetime")
parser.add_argument("--config", type=str, default = '../config/config_orca.yaml', help="config file")
parser.add_argument("--tol", type=float, default = 1e-5, help="tolerance when minimizing")
parser.add_argument("--outfile", type=str, default="foo_point.csv", help="Output CSV filename for this point")

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
dcp_vals = parse_grid(args.dcp, dCP_bf)

param_grid = list(itertools.product(sin2t12_vals, sin2t13_vals, sin2t23_vals, dm21_vals, dm31_vals, dcp_vals))

# Extract this job's grid point
try:
    sin2t12, sin2t13, sin2t23, dm21, dm31, dcp = param_grid[args.point]
except IndexError:
    raise ValueError(f"Point index {args.point} is out of range for grid size {len(param_grid)}")

# set up analysis object
orca = '../datafiles/ORCA/ORCA_MC.parquet'
config = "../config/config_orca.yaml"
Analysis = Analysis(experiment = "ORCA", livetime = 1.39, filename = orca, config = config)
nominal_syst = np.array(Analysis.systNominal)

# Run this grid point
print(f"[INFO] Running Experiment {Analysis.sim._experiment}")
print(f"[INFO] Using set of systematics {Analysis.systNames}")
print(f"[INFO] Running grid point #{args.point}: sin2theta23={sin2t23:.5f}, dm31={dm31:.6e}")

print("[INFO] Sensitivity mode")
# Sensitivity mode: N_mod is fixed (best-fit), N_dat varies with grid point
N_dat = Analysis.Compute_N_dat(sin2t12=sin2t12, sin2t13=sin2t13, sin2t23=sin2t23,
                               dm21=dm21, dm31=dm31, dcp=dcp)
res = Analysis.FitSystematics(nominal_syst=nominal_syst, N_dat=N_dat, tol=args.tol)

best_syst = res.x
chi2 = res.fun
sys_names = Analysis.systNames    

data = {
    "sin2theta12": sin2t12,
    "sin2theta13": sin2t13,
    "sin2theta23": sin2t23,
    "dm21":         dm21,
    "dm31":         dm31,
    "dcp":          dcp,
    "chi2":         chi2
}

for name, val in zip(sys_names, best_syst):
    data[name] = val

row = pd.DataFrame([data])
outdir = f"../results/{args.outfile}"
os.makedirs(os.path.dirname(outdir), exist_ok = True)
row.to_csv(outdir, index=False)

print(f"[INFO] Point #{args.point} complete with minmized chi squred = {chi2}. Result saved to {outdir}")