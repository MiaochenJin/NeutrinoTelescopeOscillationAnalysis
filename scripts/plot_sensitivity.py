#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import argparse, glob, os

parser = argparse.ArgumentParser(
    description="Concatenate multiple result‐folders, compute 2D χ²‐contour in "
                "(sin²θ₂₄, Δm²₄₁) marginalized over (sin²θ₂₃, Δm²₃₁), and save to a new folder."
)
parser.add_argument(
    "--input", "-i",
    required=True,
    nargs="+",
    help="One or more folders under ../results/ containing point_*.csv"
)
parser.add_argument(
    "--output", "-o",
    required=True,
    help="Output folder under ../results/ where combined and plots will be saved"
)
parser.add_argument(
    "--mode", "-m",
    required=True,
    choices=["Standard","Sterile"],
    help="Plotting mode: Standard or Sterile"
)
parser.add_argument(
    "--cbar-scale", "-c",
    choices=["linear","log"], default="log",
    help="Colorbar scale for Δχ² (default=log)"
)
parser.add_argument(
    "--dof", "-d", type=int, default=2,
    help="Degrees of freedom for 2D CL contours (default=2)"
)
parser.add_argument(
    "--save-marginals", action="store_true",
    help="If set, write out the marginalized minima table to disk"
)
parser.add_argument(
    "--force-new-read", "-f", default=False,
    help="If set, will read input csvs regardless of existence"
)
args = parser.parse_args()

# =============================================================================
# 1) Gather all CSV files from all input folders, concatenate into a single DataFrame
# =============================================================================
try:
    assert(not args.force_new_read)
    outdir = os.path.join("..", "results", args.output)
    parquet_path = os.path.join(outdir, "combined.parquet")
    df_full = pd.read_parquet(parquet_path)
    print("[INFO] Found existing combined parquet file")
except:
    all_files = []
    for folder in args.input:
        print("[INFO] parsing files in folder", folder)
        results_dir = os.path.join("..", "results", folder)
        pattern = os.path.join(results_dir, "point_*.csv")
        matched = glob.glob(pattern)
        if not matched:
            raise RuntimeError(f"No files matched {pattern}")
        all_files.extend(matched)

    # Concatenate all CSVs
    print("[INFO] concatenating all files")
    df_full = pd.concat([pd.read_csv(f) for f in sorted(all_files)], ignore_index=True)
    print(f"[INFO] Loaded {len(all_files)} CSV files into one DataFrame")

    # Make sure output directory exists
    outdir = os.path.join("..", "results", args.output)
    os.makedirs(outdir, exist_ok=True)

# =============================================================================
# 2) Save combined DataFrame as Parquet in output folder
# =============================================================================
parquet_path = os.path.join(outdir, "combined.parquet")
df_full.to_parquet(parquet_path)
print(f"[INFO] Saved combined DataFrame to {parquet_path}")

# =============================================================================
# 3) Compute global Δχ² and global‐best‐fit over all four parameters
# =============================================================================
df_full['delta_chi2'] = df_full['chi2'] - df_full['chi2'].min()
global_best_idx = df_full['chi2'].idxmin()
global_best_row = df_full.loc[global_best_idx]
gb_sin2theta24 = global_best_row['sin2theta24']
gb_sin2theta23 = global_best_row['sin2theta23']
gb_dm31        = global_best_row['dm31']
gb_dm41        = global_best_row['dm41']
gb_chi2        = global_best_row['chi2']

# =============================================================================
# 4) “Collapse”: for each (sin2theta24, dm41), pick row with min χ²
# =============================================================================
grouped = df_full.groupby(['sin2theta24','dm41'], sort=True)
rows = []
for (xval, yval), subdf in grouped:
    if subdf['chi2'].dropna().empty:
        print("[WARNING] nan chi2 found for ", (xval, yval))
        rows.append({
            'sin2theta24':      xval,
            'dm41':             yval,
            'best_sin2theta23': np.nan,
            'best_dm31':        np.nan,
            'best_chi2':        np.nan
        })
    else:
        best_idx = subdf['chi2'].idxmin()
        best_row = subdf.loc[best_idx]
        rows.append({
            'sin2theta24':      best_row['sin2theta24'],
            'dm41':             best_row['dm41'],
            'best_sin2theta23': best_row['sin2theta23'],
            'best_dm31':        best_row['dm31'],
            'best_chi2':        best_row['chi2']
        })

df_marg = pd.DataFrame(rows)
df_marg['delta_chi2'] = df_marg['best_chi2'] - df_full['chi2'].min()
df_marg_clean    = df_marg.dropna(subset=['best_chi2','delta_chi2'])

if args.save_marginals:
    marg_csv = os.path.join(outdir, "marginalized_minima.csv")
    df_marg.to_csv(marg_csv, index=False)
    print(f"[INFO] Wrote marginalized minima to {marg_csv}")

# =============================================================================
# 5) Build fine-grid contour in (sin2theta24, dm41)
# =============================================================================
unique_x = np.sort(df_marg_clean['sin2theta24'].unique())
unique_y = np.sort(df_marg_clean['dm41'].unique())
Znodes   = df_marg_clean.pivot(
    index='dm41', columns='sin2theta24', values='delta_chi2'
).values

if args.mode == "Standard":
    use_log_axes = False
    xlabel = r"$\sin^2\theta_{23}$"
    ylabel = r"$\Delta m^2_{31}\,$[eV$^2$]"
else:
    use_log_axes = True
    xlabel = r"$\sin^2\theta_{24}$"
    ylabel = r"$\Delta m^2_{41}\,$[eV$^2$]"

if use_log_axes:
    CX = np.log10(unique_x)
    CY = np.log10(unique_y)
else:
    CX = unique_x
    CY = unique_y

Ngrid = 200
xi = np.linspace(CX.min(), CX.max(), Ngrid)
yi = np.linspace(CY.min(), CY.max(), Ngrid)
Xg, Yg = np.meshgrid(xi, yi)

Xi_nodes, Yi_nodes = np.meshgrid(CX, CY)
points = np.column_stack([Xi_nodes.ravel(), Yi_nodes.ravel()])
values = Znodes.ravel()

Zg = griddata(points, values, (Xg, Yg), method='linear')
mask = np.isnan(Zg)
if mask.any():
    Zg_nn = griddata(points, values, (Xg, Yg), method='nearest')
    Zg[mask] = Zg_nn[mask]

if use_log_axes:
    x_plot = 10**xi
    y_plot = 10**yi
else:
    x_plot = xi
    y_plot = yi

# =============================================================================
# 6) Draw figure with top/right marginals
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",    size="25%", pad=0.1, sharex=ax)
ax_right = divider.append_axes("right",  size="25%", pad=0.1, sharey=ax)

if use_log_axes:
    ax.set_xscale('log');     ax.set_yscale('log')
    ax_top.set_xscale('log'); ax_right.set_yscale('log')
else:
    ax.set_xscale('linear');  ax.set_yscale('linear')
    ax_top.set_xscale('linear'); ax_right.set_yscale('linear')

# =============================================================================
# 7) Choose color mapping (log vs. linear)
# =============================================================================
if args.cbar_scale == "log":
    pos = Zg[Zg > 0]
    vmin = float(pos.min()) if pos.size > 0 else 1e-3
    eps  = vmin / 10.0
    Zplot = Zg.copy()
    Zplot[Zplot <= 0] = eps
    vmax = float(np.nanmax(Zplot))
    levels = np.logspace(np.log10(eps), np.log10(vmax), 30)
    norm = LogNorm(vmin=eps, vmax=vmax)

    cf = ax.contourf(x_plot, y_plot, Zplot,
                     levels=levels, norm=norm, cmap='Blues')
    cbar = fig.colorbar(cf, ax=ax, format="%.1e")
    cbar.set_label(r"$\Delta\chi^2$ (log scale)")
else:
    lo = float(np.nanmin(Zg))
    hi = float(np.nanmax(Zg))
    levels = np.linspace(lo, hi, 30)
    cf = ax.contourf(x_plot, y_plot, Zg,
                     levels=levels, cmap='Blues')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\Delta\chi^2$ (linear scale)")

# =============================================================================
# 8) Overlay 2D CL lines
# =============================================================================
cl2_68 = float(chi2.ppf(0.68, df=args.dof))
cl2_90 = float(chi2.ppf(0.90, df=args.dof))
CS = ax.contour(x_plot, y_plot, Zg,
                levels=sorted([cl2_68, cl2_90]),
                colors=['C0','C1'], linewidths=1.5)
ax.clabel(CS, inline=True, fmt={cl2_68:"68% CL", cl2_90:"90% CL"}, fontsize=10)

# =============================================================================
# 9) Mark the global best-fit (over all four parameters)
# =============================================================================
ax.scatter([gb_sin2theta24], [gb_dm41],
           marker='x', s=100, c='red', lw=2,
           label="Global best fit")

# =============================================================================
# 10) Text box below with all best-fit parameters
# =============================================================================
txt = (
    f"Global best-fit:\n"
    f"  sin²θ₂₄ = {gb_sin2theta24:.3g},  sin²θ₂₃ = {gb_sin2theta23:.3g}\n"
    f"  Δm²₃₁  = {gb_dm31:.3g},  Δm²₄₁  = {gb_dm41:.3g}\n"
    f"  χ²ₘᵢₙ  = {gb_chi2:.2f}"
)
plt.tight_layout(rect=[0, 0.18, 1, 1])
fig.text(0.5, 0.07, txt,
         ha='center', va='bottom',
         fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5"))

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

# =============================================================================
# 11) Plot top/right marginals
# =============================================================================
profile_x = np.nanmin(Zg, axis=0)
profile_y = np.nanmin(Zg, axis=1)

ax_top.plot(x_plot, profile_x, lw=1.5)
ax_top.axhline(1.0, color='k', ls='-.', lw=1, label="1σ (1-dof)")
ax_top.axhline(9.0, color='k', ls='--', lw=1, label="3σ (1-dof)")
ax_top.grid(which='major', linestyle=':')
ax_top.tick_params(labelbottom=False)
ax_top.legend(loc='upper right', frameon=False)
ax_top.set_ylim(0, 20)

ax_right.plot(profile_y, y_plot, lw=1.5)
ax_right.axvline(1.0, color='k', ls='-.', lw=1)
ax_right.axvline(9.0, color='k', ls='--', lw=1)
ax_right.grid(which='major', linestyle=':')
ax_right.tick_params(labelleft=False)
ax_right.set_xlim(0, 20)

# =============================================================================
# 12) Save final figure
# =============================================================================
outfile = os.path.join(outdir, "contour_marginalized.png")
plt.savefig(outfile)
plt.close(fig)
print(f"[INFO] Saved → {outfile}")