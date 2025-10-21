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

parser = argparse.ArgumentParser(description="Plot sensitivity contours with larger marginals")
parser.add_argument("--input", "-i",  type=str, required=True, help="Name of the results directory (e.g., '0609_run') containing the point_*.csv files. This will also be used as the output folder name for plots.")
parser.add_argument("--config", "-c", type=str, default="../config/config_orca.yaml", help="Path to the analysis YAML config file")
parser.add_argument("--official-data", "-o",  type=str, help="Path to the official ORCA chi2 landscape CSV file for comparison.")
args = parser.parse_args()

pointsdir = f"../results/{args.input}"
files = glob.glob(f"{pointsdir}/point_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(f"{pointsdir}/sensitivity_grid.csv", index=False)

# --- Setup Output Directory ---
outdir = os.path.join(base_dir, 'results', 'plots', args.input)
os.makedirs(outdir, exist_ok=True)
print(f"[INFO] Saving plots to: {outdir}")

# --- Find best-fit and delta chi2 ---
min_chi2 = df['chi2'].min()
df['delta_chi2'] = df['chi2'] - min_chi2
best_fit_row = df.loc[df['chi2'] == min_chi2]
# --- Interpolate for smoother contours ---
# Original grid
sin2_t23_orig = np.sort(df['sin2theta23'].unique())
dm31_vals_orig = np.sort(df['dm31'].unique())

# Pivot the dataframe to get chi2 values on a 2D grid for interpolation
# This is necessary for interp2d and is more robust than meshgrid.
try:
    chi2_pivot = df.pivot(index='dm31', columns='sin2theta23', values='delta_chi2')
    chi2_grid_orig = chi2_pivot.values
except ValueError:
    # Handle cases with duplicate points by averaging them
    print("[WARNING] Duplicate points found in the grid. Averaging them for interpolation.")
    df_agg = df.groupby(['dm31', 'sin2theta23']).mean().reset_index()
    chi2_pivot = df_agg.pivot(index='dm31', columns='sin2theta23', values='delta_chi2')
    chi2_grid_orig = chi2_pivot.values

# --- Calculate 1-sigma uncertainties ---
profile_s23  = np.nanmin(chi2_grid_orig, axis=0)
profile_dm31 = np.nanmin(chi2_grid_orig, axis=1)
s23_1sigma = np.interp(1.0, profile_s23, sin2_t23_orig)
dm31_1sigma = np.interp(1.0, profile_dm31, dm31_vals_orig)

s23_bf = float(best_fit_row['sin2theta23'])
dm31_bf = float(best_fit_row['dm31'])

s23_err = np.abs(s23_bf - s23_1sigma)
dm31_err = np.abs(dm31_bf - dm31_1sigma)

# --- Elegant Best-Fit Parameter Printout ---
print("\n" + "="*40)
print("      Best-Fit Parameter Summary")
print("="*40)
print(f"  Minimum Chi-Squared: {min_chi2:.4f}")
print(f"  {'Parameter':<20} | {'Best-Fit Value':<20}")
print("-"*40)
print(f"  {'sin^2(theta_23)':<20} | {s23_bf:<.4f} +/- {s23_err:<.4f}")
print(f"  {'delta_m^2_31 (eV^2)':<20} | {dm31_bf:<.6e} +/- {dm31_err:<.6e}")
print("\n--- Full Best-Fit Row ---")
# Print other parameters that might not have uncertainties calculated
for col, val in best_fit_row.iloc[0].items():
    if 'sin2theta23' not in col and 'dm31' not in col and 'chi2' not in col:
        print(f"  {col:<20} | {val}")
print("="*40 + "\n")

# --- Load and process official data for comparison, if provided ---
if args.official_data:
    print(f"[INFO] Loading official data for comparison from: {args.official_data}")
    df_official = pd.read_csv(args.official_data)
    df_official["sin2_theta23"] = (np.sin(df_official["theta23"] / 180 * np.pi)) ** 2
    min_chi2_official = df_official['chi2'].min()
    df_official['delta_chi2'] = df_official['chi2'] - min_chi2_official
    bf_official = df_official.loc[df_official['chi2'] == min_chi2_official]

    print("--- Official Best-Fit vs. Your Best-Fit ---")
    print(f"  Your Chi2:      {min_chi2:.4f}")
    print(f"  Official Chi2:  {min_chi2_official:.4f}")
    print("="*40 + "\n")

    # Interpolate official data for contour overlay
    sin2_t23_off = np.sort(df_official['sin2_theta23'].unique())
    dm31_vals_off = np.sort(df_official['dm31'].unique())
    X_off, Y_off = np.meshgrid(sin2_t23_off, dm31_vals_off)
    chi2_grid_off = griddata((df_official['sin2_theta23'], df_official['dm31']), df_official['delta_chi2'], (X_off, Y_off), method='linear')

    interp_func = interp2d(sin2_t23_off, dm31_vals_off, chi2_grid_off, kind='linear', bounds_error=False, fill_value=None)

    # Create a finer grid for interpolation
    n_points = 400 # Increased for more smoothness
    sin2_t23_off_fine = np.linspace(sin2_t23_off.min(), sin2_t23_off.max(), n_points)
    dm31_vals_off_fine = np.linspace(dm31_vals_off.min(), dm31_vals_off.max(), n_points)

    # Evaluate the function on the fine grid
    chi2_grid_off_fine = interp_func(sin2_t23_off_fine, dm31_vals_off_fine)

    # Apply a Gaussian filter for extra smoothness. Sigma is the std dev of the filter.
    # This helps to remove any minor artifacts from the interpolation.
    chi2_grid_off_fine = gaussian_filter(chi2_grid_off_fine, sigma=1.5)

    profile_s23_off  = np.nanmin(chi2_grid_off_fine, axis=0)
    profile_dm31_off = np.nanmin(chi2_grid_off_fine, axis=1)

# Create the interpolation function. interp2d is great for gridded data.
interp_func = interp2d(sin2_t23_orig, dm31_vals_orig, chi2_grid_orig, kind='linear', bounds_error=False, fill_value=None)

# Create a finer grid for interpolation
n_points = 400 # Increased for more smoothness
sin2_t23_fine = np.linspace(sin2_t23_orig.min(), sin2_t23_orig.max(), n_points)
dm31_vals_fine = np.linspace(dm31_vals_orig.min(), dm31_vals_orig.max(), n_points)

# Evaluate the function on the fine grid
chi2_grid_fine = interp_func(sin2_t23_fine, dm31_vals_fine)

# Apply a Gaussian filter for extra smoothness. Sigma is the std dev of the filter.
# This helps to remove any minor artifacts from the interpolation.
chi2_grid_fine = gaussian_filter(chi2_grid_fine, sigma=1.5)

cl2_68    = chi2.ppf(0.68, df=2)
cl2_90    = chi2.ppf(0.90, df=2)
cl1_3sig  = 9.
cl1_1sig  = 1.

profile_s23  = np.nanmin(chi2_grid_orig, axis=0)
profile_dm31 = np.nanmin(chi2_grid_orig, axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",    size="25%", pad=0.1, sharex=ax)
ax_right = divider.append_axes("right",  size="25%", pad=0.1, sharey=ax)

levels = np.linspace(np.nanmin(chi2_grid_fine), np.nanmax(chi2_grid_fine), 30)
cf = ax.contourf(sin2_t23_fine, dm31_vals_fine, chi2_grid_fine, levels=levels, cmap='Blues')
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$\Delta\chi^2$")

CS = ax.contour(sin2_t23_fine, dm31_vals_fine, chi2_grid_fine,
                levels=[cl2_68, cl2_90],
                colors=['C0','C1'], linewidths=1.5)
fmt = {cl2_90:"90% CL", cl2_68:"68% CL"}
ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

# --- Overlay official data if provided ---
if args.official_data:
    # 90% CL contour
    ax.contour(sin2_t23_off_fine, dm31_vals_off_fine, chi2_grid_off_fine,
               levels=[chi2.ppf(0.90, df=2)],
               colors=['black'], linewidths=1.0, linestyles='--')
    # Best-fit point
    ax.plot(bf_official['sin2_theta23'], bf_official['dm31'],
            marker='*', color='black', markersize=10, label='Official Best Fit', linestyle='None')

ax.plot(best_fit_row['sin2theta23'], best_fit_row['dm31'], 
        marker='*', color='red', markersize=12, label='Best Fit', linestyle='None')
ax.legend()

ax.set_xlabel(r"$\sin^2\theta_{23}$")
ax.set_xlim(0.3, 0.7)
ax.set_ylim(1.5e-3, 3e-3)

ax.set_ylabel(r"$\Delta m^2_{31}$ [eV$^2$]")

ax_top.plot(sin2_t23_orig, profile_s23, lw=1.5)
ax_top.axhline(cl1_3sig, color='k', ls='--', lw=1, label=r"$3\sigma$")
ax_top.axhline(cl1_1sig, color='k', ls='-.', lw=1, label=r"$1\sigma$")

ax_top.legend(loc='upper right', frameon=False)
ax_top.grid(which = 'major')

ax_top.tick_params(labelbottom=True)
ax_top.locator_params(axis='y', nbins=4)
ax_top.tick_params(labelbottom=False)

ax_right.plot(profile_dm31, dm31_vals_orig, lw=1.5)
ax_right.axvline(cl1_3sig, color='k', ls='--', lw=1)
ax_right.axvline(cl1_1sig, color='k', ls='-.', lw=1)

ax_right.grid(which = 'major')

ax_right.tick_params(labelleft=True)
ax_right.locator_params(axis='x', nbins=4)
ax_right.tick_params(labelleft=False)
plt.tight_layout()
outpath = os.path.join(outdir, "sensitivity_contours.png")
plt.savefig(outpath)
print(f"[INFO] Saved with enlarged marginals to {outpath}")

exit(0)
#########################################
# --- BINNED DISTRIBUTION PLOTS ---
print("[INFO] Generating binned data distribution plots...")

# --- Load Config File ---
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# --- Setup Simulation and get binning info ---
data_dir = os.path.join(base_dir, 'datafiles', 'ORCA')
orca = '../datafiles/ORCA/ORCA_MC_dataverse_with_muons.parquet'
analysis_config = "../config/config_orca_datafit.yaml"
analysis = Analysis(experiment = "ORCA", livetime = 1., filename = orca, config = analysis_config)
nominal_syst = np.array(analysis.systNominal)
sim = analysis.sim

# sim.SetInitialFlux()
E_reco_bins = sim._E_reco_bins
cosT_reco_bins = sim._cosT_reco_bins
num_morphology = sim._num_morphology
morph_labels = ['Shower', 'HPT', 'LPT']

# --- Load the real data from the dataverse parquet file ---
data_parquet_path = os.path.join(data_dir, 'ORCA_data_dataverse.parquet')
data_df = pd.read_parquet(data_parquet_path)

# --- Calculate Model Rates for comparison ---
def get_unweighted_rates(sim_instance, osc_params, from_config=False):
    t12 = np.arcsin(np.sqrt(osc_params.get('s2t12', 0)))
    t13 = np.arcsin(np.sqrt(osc_params.get('s2t13', 0)))
    t23 = np.arcsin(np.sqrt(osc_params.get('s2t23', 0)))
    dm21 = osc_params.get('m21', 0)
    dm31 = osc_params.get('m31', 0)
    # The key for dcp is 'dCP' in config, but 'dcp' in the output CSVs
    dcp_val = osc_params.get('dcp', osc_params.get('dCP', 0))
    if from_config:
        dcp_val *= np.pi
    return sim_instance.GetOscillatedRate(t12, t13, t23, dm21, dm31, dcp_val)

def get_weighted_rates(sim_instance, unweighted_rates):
    return unweighted_rates * sim_instance._mc_weights * sim_instance._livetime * sim_instance._unit_norm

# Helper function to safely get params from DataFrame or config
def get_param(name_in_df, name_in_config, df_row, config_dict):
    if name_in_df in df_row:
        return float(df_row[name_in_df])
    else:
        print(f"[INFO] '{name_in_df}' not in results file, falling back to value from config.")
        return config_dict[name_in_config]

# print(best_fit_row)
# Best Fit (from chi2 scan)
bf_params = {
    's2t12': float(best_fit_row.get('sin2theta12', config['BestFit']['s2t12'])),
    's2t13': float(best_fit_row.get('sin2theta13', config['BestFit']['s2t13'])),
    's2t23': float(best_fit_row['sin2theta23']),
    'm21': float(best_fit_row.get('dm21', config['BestFit']['m21'])),
    'm31': float(best_fit_row['dm31']),
    'dCP': float(best_fit_row.get('dcp', config['BestFit']['dCP']))
}
unweighted_rate_bf = get_unweighted_rates(sim, bf_params)
weighted_rate_bf = get_weighted_rates(sim, unweighted_rate_bf)

# NuFit Best Fit (from config file)
nufit_params = config['BestFit']
unweighted_rate_nufit = get_unweighted_rates(sim, nufit_params, from_config=True)
weighted_rate_nufit = get_weighted_rates(sim, unweighted_rate_nufit)

# No Oscillation
no_osc_params = {}
unweighted_rate_no_osc = get_unweighted_rates(sim, no_osc_params)
weighted_rate_no_osc = get_weighted_rates(sim, unweighted_rate_no_osc)
before = np.sum(np.sum(weighted_rate_bf))
print(f"[INFO] before applying systematics, total number before fit is {before}")

# Apply systematics to the best-fit oscillated flux
print("[INFO] Applying best-fit systematics to the best-fit oscillated model...")
for syst_obj in analysis.syst_objects:
    pull = float(best_fit_row[syst_obj.name])
    if hasattr(syst_obj, 'shift_rate'):
        print(f"[INFO] applying systematics {syst_obj.name} with best fit value {pull}")
        weighted_rate_bf = syst_obj.shift_rate(pull, sim, weighted_rate_bf)
        weighted_rate_nufit = syst_obj.shift_rate(pull, sim, weighted_rate_nufit)
        weighted_rate_no_osc = syst_obj.shift_rate(pull, sim, weighted_rate_no_osc)
        after = np.sum(np.sum(weighted_rate_bf))
        print(f"[INFO] the total number of event after scaling is {after}")


# --- Plot 1 & 2: Reconstructed Energy and Zenith in one figure ---
fig, axes = plt.subplots(num_morphology, 2, figsize=(10, 15), sharex='col', constrained_layout=True)
fig.suptitle('Binned Reconstructed Distributions', fontsize=16)

plot_order = [1, 2, 0] # Plot order: HPT, LPT, Shower

for plot_idx, morph_idx in enumerate(plot_order):
    ax_en = axes[plot_idx, 0]
    ax_zen = axes[plot_idx, 1]
    
    # -- Energy Plot (Left Column) --
    # Data
    data_morph_mask = (data_df['pid'] == morph_idx)
    data_counts, _ = np.histogram(data_df['reco_energy'][data_morph_mask], bins=E_reco_bins, weights=data_df['weight'][data_morph_mask])
    y_err = np.sqrt(data_counts)
    bin_centers_en = 0.5 * (E_reco_bins[:-1] + E_reco_bins[1:])
    bin_widths_en = E_reco_bins[1:] - E_reco_bins[:-1]
    ax_en.errorbar(bin_centers_en, data_counts, yerr=y_err, xerr=bin_widths_en / 2.0, fmt='ko', markersize=3, ecolor='black', capsize=2, label='Data', linestyle='none')

    # Models
    mc_morph_mask = (sim._mc_morphology == morph_idx)
    ax_en.hist(sim._mc_ereco[mc_morph_mask], bins=E_reco_bins, weights=weighted_rate_no_osc[mc_morph_mask], histtype='step', color='gray', lw=1.5, ls=':', label='No Oscillation')
    ax_en.hist(sim._mc_ereco[mc_morph_mask], bins=E_reco_bins, weights=weighted_rate_nufit[mc_morph_mask], histtype='step', color='b', lw=1.5, label='NuFit')
    ax_en.hist(sim._mc_ereco[mc_morph_mask], bins=E_reco_bins, weights=weighted_rate_bf[mc_morph_mask], histtype='step', color='r', lw=1.5, label='Best Fit')
    
    ax_en.set_ylabel('Events')
    ax_en.set_title(f'Morphology: {morph_labels[morph_idx]}')
    ax_en.set_xscale('log')
    ax_en.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_en.legend()

    # -- Zenith Plot (Right Column) --
    # Data
    data_reco_cos_zenith = np.cos(data_df['reco_zenith'][data_morph_mask])
    data_counts_zen, _ = np.histogram(data_reco_cos_zenith, bins=cosT_reco_bins, weights=data_df['weight'][data_morph_mask])
    y_err_zen = np.sqrt(data_counts_zen)
    bin_centers_zen = 0.5 * (cosT_reco_bins[:-1] + cosT_reco_bins[1:])
    bin_widths_zen = cosT_reco_bins[1:] - cosT_reco_bins[:-1]
    ax_zen.errorbar(bin_centers_zen, data_counts_zen, yerr=y_err_zen, xerr=bin_widths_zen / 2.0, fmt='ko', markersize=3, ecolor='black', capsize=2, label='Data', linestyle='none')

    # Models
    ax_zen.hist(sim._mc_cthreco[mc_morph_mask], bins=cosT_reco_bins, weights=weighted_rate_no_osc[mc_morph_mask], histtype='step', color='gray', lw=1.5, ls=':')
    ax_zen.hist(sim._mc_cthreco[mc_morph_mask], bins=cosT_reco_bins, weights=weighted_rate_nufit[mc_morph_mask], histtype='step', color='b', lw=1.5)
    ax_zen.hist(sim._mc_cthreco[mc_morph_mask], bins=cosT_reco_bins, weights=weighted_rate_bf[mc_morph_mask], histtype='step', color='r', lw=1.5)

    ax_zen.grid(True, linestyle=':', alpha=0.7)

axes[-1, 0].set_xlabel('Reconstructed Energy [GeV]')
axes[-1, 1].set_xlabel('Reconstructed cos($\\theta_{zenith}$)')
fig.tight_layout(rect=[0, 0, 1, 0.96])
outpath_dist = os.path.join(outdir, "distributions.png")
plt.savefig(outpath_dist)
print(f"[INFO] Saved distribution plot to {outpath_dist}")


# #########################################
# # --- PLOT 3: Best-fit vs Data L/E Ratio ---
# print("[INFO] Generating best-fit vs data L/E ratio plot...")

# # --- Re-configure simulation for L/E binning ---
# sim._analysis_binning = "LoE"
# R = 6371.0 # Earth radius in km

# # --- Bin all models in L/E ---
# def bin_in_loe(sim_instance, unweighted_rates):
#     return sim_instance.BinHypothesis(unweighted_rates)[0]

# N_no_osc_loe = bin_in_loe(sim, unweighted_rate_no_osc)
# N_bf_loe = bin_in_loe(sim, unweighted_rate_bf)
# N_nufit_loe = bin_in_loe(sim, unweighted_rate_nufit)

# ratio_best_fit = np.divide(N_bf_loe, N_no_osc_loe, out=np.zeros_like(N_bf_loe), where=N_no_osc_loe!=0)
# ratio_nufit = np.divide(N_nufit_loe, N_no_osc_loe, out=np.zeros_like(N_nufit_loe), where=N_no_osc_loe!=0)

# # --- Bin real data in L/E ---
# data_reco_cos_zenith = np.cos(data_df['reco_zenith'])
# data_LoE = np.divide(-2.0 * R * data_reco_cos_zenith, data_df['reco_energy'],
#                        out=np.full_like(data_df['reco_energy'], np.inf), where=data_df['reco_energy']!=0)
# data_ratios_all = np.array([])

# for i in range(num_morphology):
#     morph_mask = (data_df['pid'] == i)
#     data_counts, _ = np.histogram(data_LoE[morph_mask], bins=sim._loe_bins, weights=data_df['weight'][morph_mask])
    
#     n_bins_loe = len(sim._loe_bins) - 1
#     start_idx = i * n_bins_loe
#     end_idx = (i + 1) * n_bins_loe
#     n_no_osc_slice = N_no_osc_loe[start_idx:end_idx]

#     ratio = np.divide(data_counts, n_no_osc_slice, out=np.zeros_like(data_counts), where=n_no_osc_slice!=0)
#     data_ratios_all = np.append(data_ratios_all, ratio)

# # --- Create the L/E ratio plot ---
# fig_ratio, axes_ratio = plt.subplots(num_morphology, 1, figsize=(6, 18), sharex=True, constrained_layout=True)
# bin_centers_loe = 0.5 * (sim._loe_bins[:-1] + sim._loe_bins[1:])
# bin_widths_loe = sim._loe_bins[1:] - sim._loe_bins[:-1]
# n_bins_loe = len(bin_centers_loe)

# for plot_idx, morph_idx in enumerate(plot_order):
#     ax = axes_ratio[plot_idx]
#     start_idx = morph_idx * n_bins_loe
#     end_idx = (morph_idx + 1) * n_bins_loe

#     # Plot Data Ratio as errorbar crosses
#     data_ratio_slice = data_ratios_all[start_idx:end_idx]
#     n_no_osc_slice = N_no_osc_loe[start_idx:end_idx]
    
#     y_err = np.zeros_like(data_ratio_slice)
#     non_zero_mask = n_no_osc_slice > 0
#     safe_ratio = np.maximum(data_ratio_slice[non_zero_mask], 0)
#     y_err[non_zero_mask] = np.sqrt(safe_ratio / n_no_osc_slice[non_zero_mask])

#     ax.errorbar(bin_centers_loe, data_ratio_slice, yerr=y_err, xerr=bin_widths_loe / 2.0, fmt='ko', markersize=3, ecolor='black', capsize=2, label='Data Ratio', linestyle='none')

#     # Plot Models
#     ax.axhline(1.0, label='No Oscillation', color='gray', linestyle=':')
    
#     nufit_ratio_slice = ratio_nufit[start_idx:end_idx]
#     ax.hist(bin_centers_loe, bins=sim._loe_bins, weights=nufit_ratio_slice,
#             histtype='step', color='b', ls='--', label='NuFit')

#     bf_ratio_slice = ratio_best_fit[start_idx:end_idx]
#     ax.hist(bin_centers_loe, bins=sim._loe_bins, weights=bf_ratio_slice,
#             histtype='step', color='r', label='Best Fit')

#     ax.set_ylabel('Ratio to No Oscillation')
#     ax.set_title(f'Morphology: {morph_labels[morph_idx]}')
#     ax.grid(True, linestyle=':', alpha=0.7)
#     ax.legend()

# axes_ratio[-1].set_xlabel('L/E [km/GeV]')
# axes_ratio[0].set_xscale('log')
# fig_ratio.suptitle('Comparison of Best-Fit Model vs. Data L/E Ratio', fontsize=16)

# outpath_ratio = os.path.join(outdir, "loe_ratio_comparison.png")
# plt.savefig(outpath_ratio)
# print(f"[INFO] Saved L/E ratio comparison plot to {outpath_ratio}")