#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import argparse, os

parser = argparse.ArgumentParser(
    description="Load combined results, compute 2D χ²‐contour in "
                "(sin²θ₂₃, Δm²₃₁) marginalized over (sin²θ₂₄, Δm²₄₁) or for fixed sterile parameters."
)
parser.add_argument(
    "--input", "-i",
    required=True,
    help="Path to combined.parquet file"
)
parser.add_argument(
    "--output", "-o",
    required=True,
    help="Output folder under ../results/ where plots will be saved"
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
    '--fix-sterile',
    nargs=2,
    type=float,
    default=None,
    metavar=('SIN2TH24', 'DM41'),
    help='Fix sterile parameters to specific values (sin2th24, dm41) instead of marginalizing.'
)
args = parser.parse_args()

# =============================================================================
# 1) Load the combined DataFrame from the Parquet file
# =============================================================================
outdir = os.path.join("..", "results", args.output)
os.makedirs(outdir, exist_ok=True)

print(f"[INFO] Reading {args.input}")
df_full = pd.read_parquet(args.input)
print("[INFO] Loaded DataFrame")

# =============================================================================
# 2) Compute true global Δχ² and global‐best‐fit over all four parameters
# =============================================================================
true_global_min_chi2 = df_full['chi2'].min()
true_global_best_idx = df_full['chi2'].idxmin()
true_global_best_row = df_full.loc[true_global_best_idx]
tgb_sin2theta24 = true_global_best_row['sin2theta24']
tgb_sin2theta23 = true_global_best_row['sin2theta23']
tgb_dm31        = true_global_best_row['dm31']
tgb_dm41        = true_global_best_row['dm41']
tgb_chi2        = true_global_best_row['chi2']


# =============================================================================
# 3) Optional: Filter DataFrame for fixed sterile parameter values
# =============================================================================
if args.fix_sterile:
    sin2th24_fixed, dm41_fixed = args.fix_sterile
    print(f"[INFO] Fixing sterile parameters to sin2th24={sin2th24_fixed}, dm41={dm41_fixed}")
    # Find the closest values in the dataframe
    closest_sin2th24 = df_full['sin2theta24'].unique()[np.abs(df_full['sin2theta24'].unique() - sin2th24_fixed).argmin()]
    closest_dm41 = df_full['dm41'].unique()[np.abs(df_full['dm41'].unique() - dm41_fixed).argmin()]
    
    df_plot = df_full[
        (df_full['sin2theta24'] == closest_sin2th24) &
        (df_full['dm41'] == closest_dm41)
    ].copy()
    
    if df_plot.empty:
        raise RuntimeError("No data points found for the specified fixed sterile parameters.")
    
    print(f"[INFO] Using closest sterile parameters: sin2th24={closest_sin2th24}, dm41={closest_dm41}")
    print(f"[INFO] Filtered DataFrame has {len(df_plot)} rows")

else:
    # If not fixing, we are marginalizing over sterile parameters.
    # The df for plotting is the full one.
    df_plot = df_full

# =============================================================================
# 4) “Collapse”: for each (sin2theta23, dm31), pick row with min χ²
# If marginalizing, this collapses over the sterile parameters.
# If fixing sterile params, this ensures uniqueness for the pivot.
# =============================================================================
# Drop points where chi2 is NaN, as they can't be compared.
df_plot_clean = df_plot.dropna(subset=['chi2'])
if len(df_plot_clean) < len(df_plot):
    print(f"[WARNING] Dropped {len(df_plot) - len(df_plot_clean)} rows with NaN chi2")

# Find index of minimum chi2 for each group
idx = df_plot_clean.groupby(['sin2theta23', 'dm31'])['chi2'].idxmin()
df_marg = df_plot_clean.loc[idx].copy()

# Rename columns for clarity
df_marg = df_marg.rename(columns={'chi2': 'best_chi2',
                                  'sin2theta24': 'best_sin2theta24',
                                  'dm41': 'best_dm41'})

# Find best fit point in the (potentially marginalized) plane
plane_best_idx = df_marg['best_chi2'].idxmin()
plane_best_row = df_marg.loc[plane_best_idx]
pb_sin2theta23 = plane_best_row['sin2theta23']
pb_dm31        = plane_best_row['dm31']
pb_chi2        = plane_best_row['best_chi2']

df_marg['delta_chi2'] = df_marg['best_chi2'] - pb_chi2
df_marg_clean = df_marg.dropna(subset=['best_chi2','delta_chi2'])


if args.save_marginals:
    if args.fix_sterile:
        s24_str = f"s24_{closest_sin2th24:.2e}".replace('.', 'p')
        dm41_str = f"dm41_{closest_dm41:.2e}".replace('.', 'p')
        csv_fname = f"marginalized_minima_standard_fixed_{s24_str}_{dm41_str}.csv"
    else:
        csv_fname = "marginalized_minima_standard_marginalized.csv"
    marg_csv = os.path.join(outdir, csv_fname)
    df_marg.to_csv(marg_csv, index=False)
    print(f"[INFO] Wrote marginalized minima to {marg_csv}")

# =============================================================================
# 5) Build fine-grid contour in (sin2theta23, dm31)
# =============================================================================
unique_x = np.sort(df_marg_clean['sin2theta23'].unique())
unique_y = np.sort(df_marg_clean['dm31'].unique())
Znodes   = df_marg_clean.pivot(
    index='dm31', columns='sin2theta23', values='delta_chi2'
).values

use_log_axes = False
xlabel = r"$\sin^2\theta_{23}$"
ylabel = r"$\Delta m^2_{31}\,$[eV$^2$]"

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
# 9) Mark the best-fit point in this plane
# =============================================================================
ax.scatter([pb_sin2theta23], [pb_dm31],
           marker='x', s=100, c='red', lw=2,
           label="Best fit in plane")

# =============================================================================
# 10) Text box below with all best-fit parameters
# =============================================================================
if args.fix_sterile:
    txt = (
        f"Fixed sterile parameters:\n"
        f"  sin²θ₂₄ = {closest_sin2th24:.3g}, Δm²₄₁ = {closest_dm41:.3g}\n\n"
        f"Best-fit in this plane (χ²={pb_chi2:.2f}):\n"
        f"  sin²θ₂₃ = {pb_sin2theta23:.3g}, Δm²₃₁ = {pb_dm31:.3g}\n\n"
        f"Global best-fit (χ²={tgb_chi2:.2f}):\n"
        f"  sin²θ₂₃ = {tgb_sin2theta23:.3g}, Δm²₃₁ = {tgb_dm31:.3g}\n"
        f"  sin²θ₂₄ = {tgb_sin2theta24:.3g}, Δm²₄₁ = {tgb_dm41:.3g}"
    )
else: # Marginalized
    txt = (
        f"Global best-fit (χ²={tgb_chi2:.2f}, marginalized):\n"
        f"  sin²θ₂₃ = {tgb_sin2theta23:.3g}, Δm²₃₁ = {tgb_dm31:.3g}\n"
        f"  sin²θ₂₄ = {tgb_sin2theta24:.3g}, Δm²₄₁ = {tgb_dm41:.3g}"
    )

plt.tight_layout(rect=[0, 0.3, 1, 1])
fig.text(0.5, 0.15, txt,
         ha='center', va='center',
         fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5"))

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_ylim(0.0024, 0.0026)
ax.legend()

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
if args.fix_sterile:
    s24_str = f"s24_{closest_sin2th24:.2e}".replace('.', 'p')
    dm41_str = f"dm41_{closest_dm41:.2e}".replace('.', 'p')
    fname = f"contour_standard_fixed_{s24_str}_{dm41_str}.png"
else:
    fname = "contour_standard_marginalized.png"

outfile = os.path.join(outdir, fname)
plt.savefig(outfile, bbox_inches='tight')
plt.close(fig)
print(f"[INFO] Saved → {outfile}")