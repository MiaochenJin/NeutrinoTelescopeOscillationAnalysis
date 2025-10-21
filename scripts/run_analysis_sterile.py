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
parser.add_argument("--sin2theta24", nargs=3, type=float, default = (2e-3, 5e-1, 20),  help="theta_23 range (in radians)")
parser.add_argument("--dm41", nargs=3, type=float, default =(1e-4, 1e-1, 20), metavar=('MIN', 'MAX', 'N'), help="delta m^2_31 range (eV^2)")
parser.add_argument("--point", type=int, default = 50, help="Index of point in the parameter grid to run")
parser.add_argument("--sin2theta12", nargs=3, type=float, default = None, help="theta_12 range (rad)")
parser.add_argument("--sin2theta13", nargs=3, type=float, default = None, help="theta_13 range (rad)")
parser.add_argument("--sin2theta23", nargs=3, type=float, default = None, help="theta_23 range (in radians)")
parser.add_argument("--sin2theta14", nargs=3, type=float, default = None, help="theta_23 range (in radians)")
parser.add_argument("--sin2theta34", nargs=3, type=float, default = None, help="theta_23 range (in radians)")
parser.add_argument("--dm21", nargs=3, type=float, default = None, help="delta m^2_21 range")
parser.add_argument("--dm31", nargs=3, type=float, default = None, metavar=('MIN', 'MAX', 'N'), help="delta m^2_31 range (eV^2)")
parser.add_argument("--dcp", nargs=3, type=float, default = None, help="delta CP range (rad)")
parser.add_argument("--dcp24", nargs=3, type=float, default = None, help="delta CP range (rad)")
parser.add_argument("--livetime", type=float, default = 5, help="livetime")
parser.add_argument("--config", type=str, default = '../config/config_sterile.yaml', help="config file")
parser.add_argument("--tol", type=float, default = 1e-5, help="tolerance when minimizing")
parser.add_argument("--infile", type=str, default = '../datafiles/IC/neutrino_mc.csv', help="input csv")
parser.add_argument("--outfile", type=str, default="foo_point.csv", help="Output CSV filename for this point")
parser.add_argument("--newBF", type=bool, default=False, help="Whether to generate binned best fit values for a new best fit point")

args = parser.parse_args()

def parse_grid(arglist, default_val, spacing = 'log', exist_bf =False ):
    if arglist is None:
        return [default_val]  # No scan → use single value (best-fit)
    if spacing == 'log':
        res = np.logspace(np.log10(arglist[0]), np.log10(arglist[1]), int(arglist[2]))
    else:
        res = np.linspace(arglist[0], arglist[1], int(arglist[2]))
    if exist_bf:
        if default_val in res: return res
        else: return np.append(res, np.array([default_val]))
    else:
        return res

# Set up mixing parameters
bfpoint = yaml.safe_load(open(args.config, 'r'))["BestFit"]

t12_bf = np.arcsin(np.sqrt(bfpoint["s2t12"]))
t13_bf = np.arcsin(np.sqrt(bfpoint['s2t13']))
t23_bf = np.arcsin(np.sqrt(bfpoint['s2t23']))
s2t12_bf = bfpoint["s2t12"]
s2t13_bf = bfpoint['s2t13']
s2t23_bf = bfpoint['s2t23']
m21_bf = bfpoint['m21']
m31_bf = bfpoint['m31']
dCP_bf = bfpoint['dCP']

try:
    t14_bf = np.arcsin(np.sqrt(bfpoint["s2t14"]))
    t24_bf = np.arcsin(np.sqrt(bfpoint['s2t24']))
    t34_bf = np.arcsin(np.sqrt(bfpoint['s2t34']))
    s2t14_bf = bfpoint["s2t14"]
    s2t24_bf = bfpoint['s2t24']
    s2t34_bf = bfpoint['s2t34']
    m41_bf = bfpoint['m41']
    dCP24_bf = bfpoint['dCP24']
except:
    pass

# Build full parameter grid
sin2t12_vals = parse_grid(args.sin2theta12, s2t12_bf, 'lin')
sin2t13_vals = parse_grid(args.sin2theta13, s2t13_bf, 'lin')
sin2t23_vals = parse_grid(args.sin2theta23, s2t23_bf, 'lin')
sin2t14_vals = parse_grid(args.sin2theta14, s2t14_bf, 'log')
sin2t24_vals = parse_grid(args.sin2theta24, s2t24_bf, 'log', exist_bf = False)
sin2t34_vals = parse_grid(args.sin2theta34, s2t34_bf, 'log')

dm21_vals = parse_grid(args.dm21, m21_bf, 'lin')
dm31_vals = parse_grid(args.dm31, m31_bf, 'lin')
dm41_vals = parse_grid(args.dm41, m41_bf, 'log', exist_bf = False)

dcp_vals = parse_grid(args.dcp, dCP_bf, 'lin')
dcp24_vals = parse_grid(args.dcp24, dCP24_bf, 'log')
param_grid = list(itertools.product(sin2t12_vals, sin2t13_vals, sin2t23_vals, sin2t14_vals,\
                                    sin2t24_vals, sin2t34_vals, dm21_vals, dm31_vals, dm41_vals,\
                                    dcp_vals, dcp24_vals))

# Extract this job’s grid point
try:
    sin2t12, sin2t13, sin2t23, sin2t14, sin2t24, sin2t34, dm21, dm31, dm41, dcp, dcp24 = param_grid[args.point]
except IndexError:
    raise ValueError(f"Point index {args.point} is out of range for grid size {len(param_grid)}")

# set up analysis object
config = yaml.safe_load(open(args.config, 'r'))
experiment = config["Experiment"]
filepath = config['filepath']
livetime = config['livetime']
Analysis = Analysis(experiment = experiment, livetime = livetime, filename = filepath, config = args.config)
nominal_syst = np.array(Analysis.systNominal)

# Run this grid point
print(f"[INFO] Running Experiment {Analysis.sim._experiment}")
print(f"[INFO] Using set of systematics {Analysis.systNames}")
print(f"[INFO] Running grid point #{args.point}: sin2theta24={sin2t24:.5f}, dm41={dm41:.6e}")

N_dat = Analysis.Compute_N_dat_sterile(sin2t12=sin2t12, sin2t13=sin2t13, sin2t23=sin2t23,
                                sin2t14=sin2t14, sin2t24=sin2t24, sin2t34=sin2t34,
                                dm21=dm21, dm31=dm31, dm41=dm41, dcp=dcp, dcp24=dcp24)

res = Analysis.FitSystematics(nominal_syst=nominal_syst, N_dat=N_dat, tol=args.tol)


best_syst = res.x
chi2 = res.fun
sys_names = Analysis.systNames    

data = {
    "sin2theta12": sin2t12,
    "sin2theta13": sin2t13,
    "sin2theta23": sin2t23,
    "sin2theta14": sin2t14,
    "sin2theta24": sin2t24,
    "sin2theta34": sin2t34,

    "dm21":         dm21,
    "dm31":         dm31,
    "dm41":         dm41,

    "dcp":          dcp,
    "dcp24":        dcp24,

    "chi2":         chi2
}

for name, val in zip(sys_names, best_syst):
    data[name] = val

row = pd.DataFrame([data])
outdir = f"../results/{args.outfile}"
os.makedirs(os.path.dirname(outdir), exist_ok = True)
row.to_csv(outdir, index=False)

print(f"[INFO] Point #{args.point} complete with minmized chi squred = {chi2}. Result saved to {outdir}")