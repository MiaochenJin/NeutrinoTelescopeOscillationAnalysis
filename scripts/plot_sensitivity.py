import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import glob

parser = argparse.ArgumentParser(description="Plot sensitivity contours from chi2 scan")
parser.add_argument("--input", "-i",  type=str, required=True, help="CSV file with scan results")
parser.add_argument("--mode", "-m",  type=str, required=True, help="Whether plotting the standard or sterile case")

args = parser.parse_args()
if args.mode == "Standard":
    xkey = "sin2theta23"
    ykey = "dm31"
    xlabel = r"$\sin^2\theta_{23}$"
    ylabel = r"$\Delta m^2_{31}$ [eV$^2$]"
else:
    xkey = "sin2theta24"
    ykey = "dm41"
    xlabel = r"$\sin^2\theta_{24}$"
    ylabel = r"$\Delta m^2_{41}$ [eV$^2$]"
# --- Load scan results and combine into one dataframe---
files = glob.glob(f"../results/{args.input}/point_*.csv")
df = pd.concat([pd.read_csv(f) for f in files])
# --- Find best-fit and delta chi2 ---
min_chi2 = df['chi2'].min()
df['delta_chi2'] = df['chi2'] - min_chi2

# --- Grid for 2D contour ---
df[xkey] = np.log10(df[xkey])
df[ykey] = np.log10(df[ykey])
sin2_t23 = np.sort((df[xkey]).unique())
dm31_vals = np.sort((df[ykey]).unique())
X, Y = np.meshgrid(sin2_t23, dm31_vals)
chi2_grid = griddata((df[xkey], df[ykey]), df['delta_chi2'], (X, Y), method='linear')
if np.any(np.isnan(chi2_grid)):
    chi2_grid = griddata((df[xkey], df[ykey]), df['delta_chi2'], (X, Y), method='nearest')

# --- Confidence levels ---
cl2_68    = chi2.ppf(0.68, df=2)
cl2_90    = chi2.ppf(0.90, df=2)
print(cl2_68, cl2_90)
# cl2_95    = chi2.ppf(0.95, df=2)
# cl2_99    = chi2.ppf(0.99, df=2)
cl1_3sig  = 9.
cl1_1sig  = 1.



# --- Profile curves ---
profile_s23  = np.nanmin(chi2_grid, axis=0)
profile_dm31 = np.nanmin(chi2_grid, axis=1)

# --- Figure with larger marginals ---
fig, ax = plt.subplots(figsize=(8, 6))
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",    size="25%", pad=0.1, sharex=ax)
ax_right = divider.append_axes("right",  size="25%", pad=0.1, sharey=ax)

# --- Main 2D filled contour + lines ---
# levels = np.linspace(np.nanmin(chi2_grid), np.nanmax(chi2_grid), 30)
lo = np.nanmin(chi2_grid)
hi = np.nanmax(chi2_grid)
levels = np.linspace(lo, hi, 30)
levels = np.sort(levels)      # enforce monotonic increase
cf = ax.contourf(sin2_t23, dm31_vals, chi2_grid,
                 levels=levels,
                 cmap='Blues')
# cf = ax.contourf(sin2_t23, dm31_vals, chi2_grid, levels=levels, cmap='Blues')
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$\Delta\chi^2$")

CS = ax.contour(sin2_t23, dm31_vals, chi2_grid,
                levels=[cl2_68, cl2_90],
                colors=['C0','C1'], linewidths=1.5)
fmt = {cl2_90:"90% CL", cl2_68:"68% CL"}
ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

# --- Top marginal: Δχ² vs sin2θ23 ---
ax_top.plot(sin2_t23, profile_s23, lw=1.5)
ax_top.axhline(cl1_3sig, color='k', ls='--', lw=1, label=r"$3\sigma$")
ax_top.axhline(cl1_1sig, color='k', ls='-.', lw=1, label=r"$1\sigma$")

ax_top.legend(loc='upper right', frameon=False)
ax_top.grid(which = 'major')

# show tick labels and add more ticks
ax_top.tick_params(labelbottom=True)
# ax_top.locator_params(axis='x', nbins=6)
ax_top.locator_params(axis='y', nbins=4)
ax_top.tick_params(labelbottom=False)

# --- Right marginal: Δχ² vs Δm²31 ---
ax_right.plot(profile_dm31, dm31_vals, lw=1.5)
ax_right.axvline(cl1_3sig, color='k', ls='--', lw=1)
ax_right.axvline(cl1_1sig, color='k', ls='-.', lw=1)

ax_right.grid(which = 'major')

# show tick labels and add more ticks
ax_right.tick_params(labelleft=True)
# ax_right.locator_params(axis='y', nbins=6)
ax_right.locator_params(axis='x', nbins=4)
ax_right.tick_params(labelleft=False)
plt.tight_layout()
outfile = f"../results/{args.input}_contour.png"
plt.savefig(outfile)
print(f"[INFO] Plot saved to {outfile}")