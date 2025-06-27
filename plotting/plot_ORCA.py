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
import yaml

# Add the scripts directory to Python path for sibling imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'scripts'))
from Simulation import Simulation

parser = argparse.ArgumentParser(description="Plot sensitivity contours with larger marginals")
parser.add_argument("--input", "-i",  type=str, required=True, help="Name of the results directory (e.g., '0609_run') containing the point_*.csv files. This will also be used as the output folder name for plots.")
parser.add_argument("--config", "-c", type=str, default="../config/config_orca.yaml", help="Path to the analysis YAML config file")
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
outpath = os.path.join(outdir, "sensitivity_contours.png")
plt.savefig(outpath)
print(f"[INFO] Saved with enlarged marginals to {outpath}")


#########################################
# --- BINNED DISTRIBUTION PLOTS ---
print("[INFO] Generating binned data distribution plots...")

# --- Load Config File ---
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# --- Setup Simulation and get binning info ---
data_dir = os.path.join(base_dir, 'datafiles', 'ORCA')
sim = Simulation(
    experiment='ORCA',
    livetime=1., # Livetime from config is used for rate calculation
    filename=os.path.join(data_dir, 'ORCA_MC_dataverse.parquet'), # MC file for setup
    mode='Standard'
)
sim.SetInitialFlux()
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


#########################################
# --- PLOT 3: Best-fit vs Data L/E Ratio ---
print("[INFO] Generating best-fit vs data L/E ratio plot...")

# --- Re-configure simulation for L/E binning ---
sim._analysis_binning = "LoE"
R = 6371.0 # Earth radius in km

# --- Bin all models in L/E ---
def bin_in_loe(sim_instance, unweighted_rates):
    return sim_instance.BinHypothesis(unweighted_rates)[0]

N_no_osc_loe = bin_in_loe(sim, unweighted_rate_no_osc)
N_bf_loe = bin_in_loe(sim, unweighted_rate_bf)
N_nufit_loe = bin_in_loe(sim, unweighted_rate_nufit)

ratio_best_fit = np.divide(N_bf_loe, N_no_osc_loe, out=np.zeros_like(N_bf_loe), where=N_no_osc_loe!=0)
ratio_nufit = np.divide(N_nufit_loe, N_no_osc_loe, out=np.zeros_like(N_nufit_loe), where=N_no_osc_loe!=0)

# --- Bin real data in L/E ---
data_reco_cos_zenith = np.cos(data_df['reco_zenith'])
data_LoE = np.divide(-2.0 * R * data_reco_cos_zenith, data_df['reco_energy'],
                       out=np.full_like(data_df['reco_energy'], np.inf), where=data_df['reco_energy']!=0)
data_ratios_all = np.array([])

for i in range(num_morphology):
    morph_mask = (data_df['pid'] == i)
    data_counts, _ = np.histogram(data_LoE[morph_mask], bins=sim._loe_bins, weights=data_df['weight'][morph_mask])
    
    n_bins_loe = len(sim._loe_bins) - 1
    start_idx = i * n_bins_loe
    end_idx = (i + 1) * n_bins_loe
    n_no_osc_slice = N_no_osc_loe[start_idx:end_idx]

    ratio = np.divide(data_counts, n_no_osc_slice, out=np.zeros_like(data_counts), where=n_no_osc_slice!=0)
    data_ratios_all = np.append(data_ratios_all, ratio)

# --- Create the L/E ratio plot ---
fig_ratio, axes_ratio = plt.subplots(num_morphology, 1, figsize=(6, 18), sharex=True, constrained_layout=True)
bin_centers_loe = 0.5 * (sim._loe_bins[:-1] + sim._loe_bins[1:])
bin_widths_loe = sim._loe_bins[1:] - sim._loe_bins[:-1]
n_bins_loe = len(bin_centers_loe)

for plot_idx, morph_idx in enumerate(plot_order):
    ax = axes_ratio[plot_idx]
    start_idx = morph_idx * n_bins_loe
    end_idx = (morph_idx + 1) * n_bins_loe

    # Plot Data Ratio as errorbar crosses
    data_ratio_slice = data_ratios_all[start_idx:end_idx]
    n_no_osc_slice = N_no_osc_loe[start_idx:end_idx]
    
    y_err = np.zeros_like(data_ratio_slice)
    non_zero_mask = n_no_osc_slice > 0
    safe_ratio = np.maximum(data_ratio_slice[non_zero_mask], 0)
    y_err[non_zero_mask] = np.sqrt(safe_ratio / n_no_osc_slice[non_zero_mask])

    ax.errorbar(bin_centers_loe, data_ratio_slice, yerr=y_err, xerr=bin_widths_loe / 2.0, fmt='ko', markersize=3, ecolor='black', capsize=2, label='Data Ratio', linestyle='none')

    # Plot Models
    ax.axhline(1.0, label='No Oscillation', color='gray', linestyle=':')
    
    nufit_ratio_slice = ratio_nufit[start_idx:end_idx]
    y_step_nufit = np.append(nufit_ratio_slice, nufit_ratio_slice[-1])
    ax.step(sim._loe_bins, y_step_nufit, where='pre', label='NuFit', color='blue', linestyle='--')

    bf_ratio_slice = ratio_best_fit[start_idx:end_idx]
    y_step_bf = np.append(bf_ratio_slice, bf_ratio_slice[-1])
    ax.step(sim._loe_bins, y_step_bf, where='pre', label='Best Fit', color='red')

    ax.set_ylabel('Ratio to No Oscillation')
    ax.set_title(f'Morphology: {morph_labels[morph_idx]}')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

axes_ratio[-1].set_xlabel('L/E [km/GeV]')
axes_ratio[0].set_xscale('log')
fig_ratio.suptitle('Comparison of Best-Fit Model vs. Data L/E Ratio', fontsize=16)

outpath_ratio = os.path.join(outdir, "loe_ratio_comparison.png")
plt.savefig(outpath_ratio)
print(f"[INFO] Saved L/E ratio comparison plot to {outpath_ratio}")