import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
import argparse

parser = argparse.ArgumentParser(description="Plot sensitivity contours from chi2 scan")
parser.add_argument("--input", "-i",  type=str, required=True, help="CSV file with scan results")
parser.add_argument("--outfile", type=str, default="sensitivity_contour.png", help="Output plot filename")
args = parser.parse_args()

# --- Load scan results ---
df = pd.read_csv(args.input)

# --- Find best-fit point and delta chi2 ---
min_chi2 = df['chi2'].min()
df['delta_chi2'] = df['chi2'] - min_chi2

# --- Reshape for contour plotting ---
sin2_t23 = np.sort(df['sin2theta23'].unique())
dm31_vals = np.sort(df['dm31'].unique())
print(sin2_t23)
print(dm31_vals)

X, Y = np.meshgrid(sin2_t23, dm31_vals)

chi2_grid = griddata((df['sin2theta23'], df['dm31']), df['delta_chi2'],
                     (X, Y), method='linear')

print(chi2_grid)

if chi2_grid is None or np.any(np.isnan(chi2_grid)):
    print("[WARNING] NaNs detected in chi2 grid. Falling back to nearest interpolation.")
    chi2_grid = griddata((df['sin2theta23'], df['dm31']), df['delta_chi2'], (X, Y), method='nearest')

# --- Define CL thresholds for 2 parameters ---
cl_95 = chi2.ppf(0.95, df=2)  # ~5.99
cl_99 = chi2.ppf(0.99, df=2)  # ~9.21

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))
# Compute levels explicitly between the minimum and maximum of chi2_grid
levels = np.linspace(np.nanmin(chi2_grid), np.nanmax(chi2_grid), 30)
c = ax.contourf(sin2_t23, dm31_vals, chi2_grid, levels=levels, cmap='Blues')
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r"$\Delta\chi^2$")

# Contours
CS = ax.contour(sin2_t23, dm31_vals, chi2_grid, levels=[cl_95, cl_99], colors=['C0', 'C1'], linewidths=1.5)
fmt = {cl_95: "95% CL", cl_99: "99% CL"}
ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

# Labels
ax.set_xlabel(r"$\sin^2\theta_{23}$")
ax.set_ylabel(r"$\Delta m^2_{31}$ [eV$^2$]")
ax.set_title("Sensitivity Contours")

plt.tight_layout()
plt.savefig(args.outfile)
print(f"[INFO] Plot saved to {args.outfile}")