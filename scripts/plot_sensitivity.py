import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import glob
import os

parser = argparse.ArgumentParser(description="Plot sensitivity contours from chi2 scan")
parser.add_argument("--input", "-i",  type=str, required=True, help="CSV file with scan results")
parser.add_argument("--mode", "-m",  type=str, required=True, help="Whether plotting the standard or sterile case")
args = parser.parse_args()

# --- choose keys & labels ---
if args.mode == "Standard":
    xkey, ykey = "sin2theta23", "dm31"
    xlabel, ylabel = r"$\sin^2\theta_{23}$", r"$\Delta m^2_{31}$ [eV$^2$]"
    use_log = False
else:
    xkey, ykey = "sin2theta24", "dm41"
    xlabel, ylabel = r"$\sin^2\theta_{24}$", r"$\Delta m^2_{41}$ [eV$^2$]"
    use_log = True

# --- load all scan points into one DataFrame ---
files = glob.glob(f"../results/{args.input}/point_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# --- compute Δχ² ---
df['delta_chi2'] = df['chi2'] - df['chi2'].min()

# --- prepare coordinates for interpolation ---
if use_log:
    coords_x = np.log10(df[xkey].values)
    coords_y = np.log10(df[ykey].values)
    x_vals = np.sort(np.log10(df[xkey].unique()))
    y_vals = np.sort(np.log10(df[ykey].unique()))
else:
    coords_x = df[xkey].values
    coords_y = df[ykey].values
    x_vals = np.sort(df[xkey].unique())
    y_vals = np.sort(df[ykey].unique())

# mesh in (possibly log) space
X, Y = np.meshgrid(x_vals, y_vals)

# interpolate Δχ² onto grid
chi2_grid = griddata(
    (coords_x, coords_y),
    df['delta_chi2'],
    (X, Y),
    method='linear'
)
if np.any(np.isnan(chi2_grid)):
    chi2_grid = griddata(
        (coords_x, coords_y),
        df['delta_chi2'],
        (X, Y),
        method='nearest'
    )

# --- confidence levels for 2D contours ---
cl2_68 = chi2.ppf(0.68, df=2)
cl2_90 = chi2.ppf(0.90, df=2)
cl1_1sig, cl1_3sig = 1.0, 9.0

# --- profile curves ---
profile_x  = np.nanmin(chi2_grid, axis=0)
profile_y  = np.nanmin(chi2_grid, axis=1)

# --- recover real coordinates for plotting if needed ---
if use_log:
    x_plot = 10**x_vals
    y_plot = 10**y_vals
else:
    x_plot = x_vals
    y_plot = y_vals

# --- start figure with marginal axes ---
fig, ax = plt.subplots(figsize=(8, 6))
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",    size="25%", pad=0.1, sharex=ax)
ax_right = divider.append_axes("right",  size="25%", pad=0.1, sharey=ax)

# --- set log scales if requested ---
if use_log:
    ax.set_xscale('log')
    ax.set_yscale('log')

# --- filled contour ---
lo, hi = np.nanmin(chi2_grid), np.nanmax(chi2_grid)
levels = np.linspace(lo, hi, 30)
cf = ax.contourf(
    x_plot, y_plot, chi2_grid,
    levels=levels, cmap='Blues'
)
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$\Delta\chi^2$")

# --- 68% & 90% CL lines ---
CS = ax.contour(
    x_plot, y_plot, chi2_grid,
    levels=sorted([cl2_68, cl2_90]),
    colors=['C0','C1'], linewidths=1.5
)
fmt = {cl2_90: "90% CL", cl2_68: "68% CL"}
ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

# --- top marginal (x) ---
ax_top.plot(x_plot, profile_x, lw=1.5)
ax_top.axhline(cl1_1sig, color='k', ls='-.', lw=1)
ax_top.axhline(cl1_3sig, color='k', ls='--', lw=1)
ax_top.legend([r"$1\sigma$", r"$3\sigma$"], loc='upper right', frameon=False)
ax_top.grid(True)
ax_top.tick_params(labelbottom=False)
ax_top.locator_params(axis='y', nbins=4)

# --- right marginal (y) ---
ax_right.plot(profile_y, y_plot, lw=1.5)
ax_right.axvline(cl1_1sig, color='k', ls='-.', lw=1)
ax_right.axvline(cl1_3sig, color='k', ls='--', lw=1)
ax_right.grid(True)
ax_right.tick_params(labelleft=False)
ax_right.locator_params(axis='x', nbins=4)

plt.tight_layout()

# --- save figure ---
outfile = f"../results/{args.input}_contour.png"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
plt.savefig(outfile)
print(f"[INFO] Plot saved to {outfile}")