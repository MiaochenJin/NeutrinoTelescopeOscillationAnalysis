import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import glob
import os
import sys

# Add the scripts directory to Python path for sibling imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'scripts'))
from Simulation import Simulation

parser = argparse.ArgumentParser(description="Plot sensitivity contours with larger marginals")
parser.add_argument("--input", "-i",  type=str, default='0609', help="CSV file with scan results")
parser.add_argument("--outfile", "-o", type=str, default="0609", help="Output plot filename")
args = parser.parse_args()

pointsdir = f"../results/{args.input}"
files = glob.glob(f"{pointsdir}/point_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(f"{pointsdir}/sensitivity_grid.csv", index=False)

# --- Find best-fit and delta chi2 ---
min_chi2 = df['chi2'].min()
df['delta_chi2'] = df['chi2'] - min_chi2
best_fit_row = df.loc[df['chi2'] == min_chi2]

sin2_t23 = np.sort(df['sin2theta23'].unique())
dm31_vals = np.sort(df['dm31'].unique())
X, Y = np.meshgrid(sin2_t23, dm31_vals)
chi2_grid = griddata((df['sin2theta23'], df['dm31']), df['delta_chi2'], (X, Y), method='linear')
if np.any(np.isnan(chi2_grid)):
    chi2_grid = griddata((df['sin2theta23'], df['dm31']), df['delta_chi2'], (X, Y), method='nearest')

cl2_68    = chi2.ppf(0.68, df=2)
cl2_90    = chi2.ppf(0.90, df=2)
print(cl2_68, cl2_90)
cl1_3sig  = 9.
cl1_1sig  = 1.

profile_s23  = np.nanmin(chi2_grid, axis=0)
profile_dm31 = np.nanmin(chi2_grid, axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",    size="25%", pad=0.1, sharex=ax)
ax_right = divider.append_axes("right",  size="25%", pad=0.1, sharey=ax)

levels = np.linspace(np.nanmin(chi2_grid), np.nanmax(chi2_grid), 30)
cf = ax.contourf(sin2_t23, dm31_vals, chi2_grid, levels=levels, cmap='Blues')
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$\Delta\chi^2$")

CS = ax.contour(sin2_t23, dm31_vals, chi2_grid,
                levels=[cl2_68, cl2_90],
                colors=['C0','C1'], linewidths=1.5)
fmt = {cl2_90:"90% CL", cl2_68:"68% CL"}
ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

ax.plot(best_fit_row['sin2theta23'], best_fit_row['dm31'], 
        marker='*', color='red', markersize=12, label='Best Fit', linestyle='None')
ax.legend()

ax.set_xlabel(r"$\sin^2\theta_{23}$")
ax.set_ylabel(r"$\Delta m^2_{31}$ [eV$^2$]")

ax_top.plot(sin2_t23, profile_s23, lw=1.5)
ax_top.axhline(cl1_3sig, color='k', ls='--', lw=1, label=r"$3\sigma$")
ax_top.axhline(cl1_1sig, color='k', ls='-.', lw=1, label=r"$1\sigma$")

ax_top.legend(loc='upper right', frameon=False)
ax_top.grid(which = 'major')

ax_top.tick_params(labelbottom=True)
ax_top.locator_params(axis='y', nbins=4)
ax_top.tick_params(labelbottom=False)

ax_right.plot(profile_dm31, dm31_vals, lw=1.5)
ax_right.axvline(cl1_3sig, color='k', ls='--', lw=1)
ax_right.axvline(cl1_1sig, color='k', ls='-.', lw=1)

ax_right.grid(which = 'major')

ax_right.tick_params(labelleft=True)
ax_right.locator_params(axis='x', nbins=4)
ax_right.tick_params(labelleft=False)
plt.tight_layout()
outpath = f"../results/plots/{args.outfile}"
plt.savefig(outpath)
print(f"[INFO] Saved with enlarged marginals to {outpath}")

#########################################
# --- SECOND PLOT: Best-fit vs Data Ratio ---
print("[INFO] Generating best-fit vs data comparison plot...")

# --- Setup Simulation to calculate rates ---
data_dir = os.path.join(base_dir, 'datafiles', 'ORCA')
sim = Simulation(
    experiment='ORCA',
    livetime=1.39,
    filename=os.path.join(data_dir, 'ORCA_MC.parquet'),
    mode='Standard'
)
sim.SetInitialFlux()
sim._analysis_binning = "LoE"

# --- Calculate No-Oscillation Baseline ---
no_osc_unbinned = sim.GetOscillatedRate(0, 0, 0, 0, 0, 0)
N_no_osc, _ = sim.BinHypothesis(no_osc_unbinned)

# --- Calculate Best-Fit Model Rates ---
bf_s2t12 = float(best_fit_row.get('sin2theta12', 0.307))
bf_s2t13 = float(best_fit_row.get('sin2theta13', 0.022))
bf_dm21 = float(best_fit_row.get('dm21', 7.5e-5))
bf_dcp = float(best_fit_row.get('dcp', 0))

t12 = np.arcsin(np.sqrt(bf_s2t12))
t13 = np.arcsin(np.sqrt(bf_s2t13))
t23 = np.arcsin(np.sqrt(float(best_fit_row['sin2theta23'])))
dm21 = bf_dm21
dm31 = float(best_fit_row['dm31'])
dcp = bf_dcp

best_fit_unbinned = sim.GetOscillatedRate(t12, t13, t23, dm21, dm31, dcp)
N_best_fit, _ = sim.BinHypothesis(best_fit_unbinned)
ratio_best_fit = N_best_fit / N_no_osc

# --- Calculate literature best fit for comparison ---
lit_t12 = np.arcsin(np.sqrt(0.307))
lit_t13 = np.arcsin(np.sqrt(0.022))
lit_t23 = np.arcsin(np.sqrt(0.57))
lit_dm21 = 7.5e-5
lit_dm31 = 2.5e-3
lit_dcp = 0
lit_bf_unbinned = sim.GetOscillatedRate(lit_t12, lit_t13, lit_t23, lit_dm21, lit_dm31, lit_dcp)
N_lit_bf, _ = sim.BinHypothesis(lit_bf_unbinned)
ratio_lit_bf = N_lit_bf / N_no_osc

# --- Load Original Data Ratios ---
original_ratio_files = {0: "Shower_median.csv", 1: "HPT_median.csv", 2: "LPT_median.csv"}
original_ratios_all = np.array([])
for i in range(sim._num_morphology):
    ratio_df = pd.read_csv(os.path.join(data_dir, original_ratio_files[i]), header=None)
    original_ratios_all = np.append(original_ratios_all, ratio_df.iloc[:, 1].values)

# --- Create the plot ---
fig_ratio, axes_ratio = plt.subplots(sim._num_morphology, 1, figsize=(6, 15), sharex=True)
morph_labels = ['Shower', 'HPT', 'LPT']
bin_centers = 0.5 * (sim._loe_bins[:-1] + sim._loe_bins[1:])
bin_widths = sim._loe_bins[1:] - sim._loe_bins[:-1]
n_bins = len(bin_centers)

for i in range(sim._num_morphology):
    ax = axes_ratio[i]
    start_idx = i * n_bins
    end_idx = (i + 1) * n_bins

    # Plot Original Data Ratio as errorbar crosses
    data_ratio = original_ratios_all[start_idx:end_idx]
    n_no_osc_slice = N_no_osc[start_idx:end_idx]

    # Calculate vertical error for the data ratio
    # y_err = sqrt(N_data) / N_no_osc = sqrt(ratio * N_no_osc) / N_no_osc = sqrt(ratio / N_no_osc)
    y_err = np.zeros_like(data_ratio)
    non_zero_mask = n_no_osc_slice > 0
    # To avoid sqrt of negative numbers if data_ratio has some weird values, although it shouldn't
    safe_ratio = np.maximum(data_ratio[non_zero_mask], 0)
    y_err[non_zero_mask] = np.sqrt(safe_ratio / n_no_osc_slice[non_zero_mask])

    ax.errorbar(bin_centers, data_ratio,
                yerr=y_err, xerr=bin_widths / 2.0,
                fmt='ko', markersize=3, ecolor='black', capsize=2,
                label='Original Data Ratio', linestyle='none')

    # Plot Best-Fit Model Ratio
    model_ratio_slice = ratio_best_fit[start_idx:end_idx]
    # To make the step plot extend to the very edge of the last bin,
    # we use 'pre' and provide x-coordinates as the bin edges.
    # y must have the same length as x, so we duplicate the last value.
    y_step = np.append(model_ratio_slice, model_ratio_slice[-1])
    ax.step(sim._loe_bins, y_step, where='pre', 
            label='Best-Fit Model Ratio', color='red')

    # Plot literature best fit model
    lit_model_slice = ratio_lit_bf[start_idx:end_idx]
    y_step_lit = np.append(lit_model_slice, lit_model_slice[-1])
    ax.step(sim._loe_bins, y_step_lit, where='pre',
            label='Literature Best Fit', color='blue', linestyle='--')

    ax.set_ylabel('Ratio to No Oscillation')
    ax.set_title(f'Morphology: {morph_labels[i]}')
    ax.grid(True, linestyle=':', alpha=0.7)

axes_ratio[-1].set_xlabel('L/E [km/GeV]')
axes_ratio[0].set_xscale('log')
fig_ratio.suptitle('Comparison of Best-Fit Model vs. Original Data', fontsize=16)
fig_ratio.tight_layout(rect=[0, 0, 1, 0.96])

outpath_ratio = f"../results/plots/{args.outfile}_ratio_comparison.png"
plt.savefig(outpath_ratio)
print(f"[INFO] Saved ratio comparison plot to {outpath_ratio}")