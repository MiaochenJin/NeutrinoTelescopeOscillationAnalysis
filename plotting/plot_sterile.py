import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import glob
import os
import sys
import yaml


# Add the scripts directory to Python path for sibling imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'scripts'))
from Simulation import Simulation
from Analysis import Analysis

parser = argparse.ArgumentParser(description="Plot sensitivity contours for sterile oscillations")
parser.add_argument("--input", "-i",  type=str, required=True, help="Name of the results directory (e.g., '0914_sterile_smbf') containing the point_*.csv files. This will also be used as the output folder name for plots.")
parser.add_argument("--config", "-c", type=str, default="../config/config_sterile.yaml", help="Path to the analysis YAML config file for sterile oscillations")
args = parser.parse_args()

pointsdir = f"../results/{args.input}"
files = glob.glob(f"{pointsdir}/point_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(f"{pointsdir}/sensitivity_grid_sterile.csv", index=False)

# --- Setup Output Directory ---
outdir = os.path.join(base_dir, 'results', 'plots', args.input)
os.makedirs(outdir, exist_ok=True)
print(f"[INFO] Saving plots to: {outdir}")

# --- Find best-fit and delta chi2 ---
min_chi2 = df['chi2'].min()
df['delta_chi2'] = df['chi2'] - min_chi2
best_fit_row = df.loc[df['chi2'] == min_chi2]

# --- Interpolate for smoother contours ---
# Original grid for sin2theta24 and dm41
sin2_t24_orig = np.sort(df['sin2theta24'].unique())
dm41_vals_orig = np.sort(df['dm41'].unique())

# Pivot the dataframe to get chi2 values on a 2D grid for interpolation
try:
    chi2_pivot = df.pivot(index='dm41', columns='sin2theta24', values='delta_chi2')
    chi2_grid_orig = chi2_pivot.values
except ValueError:
    print("[WARNING] Duplicate points found in the grid. Averaging them for interpolation.")
    df_agg = df.groupby(['dm41', 'sin2theta24']).mean().reset_index()
    chi2_pivot = df_agg.pivot(index='dm41', columns='sin2theta24', values='delta_chi2')
    chi2_grid_orig = chi2_pivot.values

# Create the interpolation function. interp2d is great for gridded data.
interp_func = interp2d(sin2_t24_orig, dm41_vals_orig, chi2_grid_orig, kind='linear', bounds_error=False, fill_value=None)

# Create a finer grid for interpolation
n_points = 400
sin2_t24_fine = np.linspace(sin2_t24_orig.min(), sin2_t24_orig.max(), n_points)
dm41_vals_fine = np.linspace(dm41_vals_orig.min(), dm41_vals_orig.max(), n_points)

# Evaluate the function on the fine grid
chi2_grid_fine = interp_func(sin2_t24_fine, dm41_vals_fine)

# Apply a Gaussian filter for extra smoothness.
chi2_grid_fine = gaussian_filter(chi2_grid_fine, sigma=1.5)

# --- Calculate 1-sigma uncertainties (for marginal plots) ---
profile_s24  = np.nanmin(chi2_grid_orig, axis=0)
profile_dm41 = np.nanmin(chi2_grid_orig, axis=1)

s24_bf = float(best_fit_row['sin2theta24'])
dm41_bf = float(best_fit_row['dm41'])

# Find 1-sigma points (delta chi2 = 1.0 for 1 parameter)
# This is a simplification for 2D contours, but useful for marginals
# Need to handle cases where 1.0 is not directly in profile_s24/dm41
s24_1sigma_low = np.interp(1.0, profile_s24[sin2_t24_orig <= s24_bf], sin2_t24_orig[sin2_t24_orig <= s24_bf]) if np.any(profile_s24[sin2_t24_orig <= s24_bf] <= 1.0) else s24_bf
s24_1sigma_high = np.interp(1.0, profile_s24[sin2_t24_orig >= s24_bf], sin2_t24_orig[sin2_t24_orig >= s24_bf]) if np.any(profile_s24[sin2_t24_orig >= s24_bf] <= 1.0) else s24_bf

dm41_1sigma_low = np.interp(1.0, profile_dm41[dm41_vals_orig <= dm41_bf], dm41_vals_orig[dm41_vals_orig <= dm41_bf]) if np.any(profile_dm41[dm41_vals_orig <= dm41_bf] <= 1.0) else dm41_bf
dm41_1sigma_high = np.interp(1.0, profile_dm41[dm41_vals_orig >= dm41_bf], dm41_vals_orig[dm41_vals_orig >= dm41_bf]) if np.any(profile_dm41[dm41_vals_orig >= dm41_bf] <= 1.0) else dm41_bf

s24_err_low = np.abs(s24_bf - s24_1sigma_low)
s24_err_high = np.abs(s24_bf - s24_1sigma_high)
dm41_err_low = np.abs(dm41_bf - dm41_1sigma_low)
dm41_err_high = np.abs(dm41_bf - dm41_1sigma_high)

# --- Elegant Best-Fit Parameter Printout ---
print("\n" + "="*40)
print("      Best-Fit Parameter Summary (Sterile)")
print("="*40)
print(f"  Minimum Chi-Squared: {min_chi2:.4f}")
print(f"  {'Parameter':<20} | {'Best-Fit Value':<20}")
print("-"*40)
print(f"  {'sin^2(theta_24)':<20} | {s24_bf:<.4e} +{s24_err_high:<.4e}/-{s24_err_low:<.4e}")
print(f"  {'delta_m^2_41 (eV^2)':<20} | {dm41_bf:<.6e} +{dm41_err_high:<.6e}/-{dm41_err_low:<.6e}")
print("\n--- Full Best-Fit Row ---")
for col, val in best_fit_row.iloc[0].items():
    if 'sin2theta24' not in col and 'dm41' not in col and 'chi2' not in col:
        print(f"  {col:<20} | {val}")
print("="*40 + "\n")

# --- Contour Levels ---
cl2_68    = chi2.ppf(0.68, df=2) # 68% CL for 2 degrees of freedom
cl2_90    = chi2.ppf(0.90, df=2) # 90% CL for 2 degrees of freedom
cl1_1sig  = chi2.ppf(0.68, df=1) # 1-sigma for 1 degree of freedom (for marginals)
cl1_3sig  = chi2.ppf(0.997, df=1) # 3-sigma for 1 degree of freedom (for marginals)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",    size="25%", pad=0.1, sharex=ax)
ax_right = divider.append_axes("right",  size="25%", pad=0.1, sharey=ax)

levels = np.linspace(0, cl1_3sig * 1.5, 30) # Zoom into 3-sigma region
cf = ax.contourf(sin2_t24_fine, dm41_vals_fine, chi2_grid_fine, levels=levels, cmap='Blues')
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$\Delta\chi^2$")

CS = ax.contour(sin2_t24_fine, dm41_vals_fine, chi2_grid_fine,
                levels=[cl2_68, cl2_90],
                colors=['C0','C1'], linewidths=1.5)
fmt = {cl2_90:"90% CL", cl2_68:"68% CL"}
ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

ax.plot(best_fit_row['sin2theta24'], best_fit_row['dm41'],
        marker='*', color='red', markersize=12, label='Best Fit', linestyle='None')
ax.legend()

ax.set_xlabel(r"$\sin^2\theta_{24}$")
ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_xscale('log')
ax.set_yscale('log')

# Set limits based on the data range, or a reasonable default for sterile parameters
ax.set_xlim(sin2_t24_orig.min() * 0.8, sin2_t24_orig.max() * 1.2)
ax.set_ylim(dm41_vals_orig.min() * 0.8, dm41_vals_orig.max() * 1.2)


ax_top.plot(sin2_t24_orig, profile_s24, lw=1.5)
ax_top.axhline(cl1_3sig, color='k', ls='--', lw=1, label=r"$3\sigma$")
ax_top.axhline(cl1_1sig, color='k', ls='-.', lw=1, label=r"$1\sigma$")

ax_top.legend(loc='upper right', frameon=False)
ax_top.grid(which = 'major')
ax_top.tick_params(labelbottom=False)
ax_top.locator_params(axis='y', nbins=4)

ax_right.plot(profile_dm41, dm41_vals_orig, lw=1.5)
ax_right.axvline(cl1_3sig, color='k', ls='--', lw=1)
ax_right.axvline(cl1_1sig, color='k', ls='-.', lw=1)

ax_right.grid(which = 'major')
ax_right.tick_params(labelleft=False)
ax_right.locator_params(axis='x', nbins=4)

plt.tight_layout()
outpath = os.path.join(outdir, "sterile_sensitivity_contours.png")
plt.savefig(outpath)
print(f"[INFO] Saved sterile sensitivity contours to {outpath}")

# The rest of the plotting code (binned distributions, L/E ratio) is commented out
# as it's not directly requested for the sensitivity contour plot and would require
# significant adaptation for sterile parameters and data handling.
# If these plots are needed, they should be implemented separately.