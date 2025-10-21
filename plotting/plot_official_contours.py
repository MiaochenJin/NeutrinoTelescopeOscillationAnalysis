import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import chi2
import argparse
import os

def plot_official_contours(filepath, outdir):
    """
    Plots the chi-squared contours from the official ORCA data release.
    """
    os.makedirs(outdir, exist_ok=True)
    print(f"[INFO] Loading official data from: {filepath}")
    df = pd.read_csv(filepath)
    df['sin2_theta23'] = np.sin(df['theta23'] / 180 * np.pi)**2
    print(np.unique(df['sin2_theta23']))

    # --- Find best-fit and delta chi2 ---
    min_chi2 = df['chi2'].min()
    df['delta_chi2'] = df['chi2'] - min_chi2
    best_fit_row = df.loc[df['chi2'] == min_chi2]

    print("\n" + "="*40)
    print("  Official ORCA Best-Fit Parameter Summary")
    print("="*40)
    print(f"  Minimum Chi-Squared: {min_chi2:.4f}")
    print(f"  sin^2(theta_23): {float(best_fit_row['sin2_theta23']):.4f}")
    print(f"  delta_m^2_31 (eV^2): {float(best_fit_row['dm31']):.6e}")
    print("="*40 + "\n")

    # --- Interpolate for smoother contours ---
    sin2_t23_orig = np.sort(df['sin2_theta23'].unique())
    dm31_vals_orig = np.sort(df['dm31'].unique())
    X_orig, Y_orig = np.meshgrid(sin2_t23_orig, dm31_vals_orig)
    
    # Use griddata; it's robust for scattered or incomplete data.
    chi2_grid_orig = griddata((df['sin2_theta23'], df['dm31']), df['delta_chi2'], (X_orig, Y_orig), method='linear')
    
    # Fallback for any remaining NaNs
    if np.any(np.isnan(chi2_grid_orig)):
        chi2_grid_orig = griddata((df['sin2_theta23'], df['dm31']), df['delta_chi2'], (X_orig, Y_orig), method='nearest')

    n_points = 200
    sin2_t23_fine = np.linspace(sin2_t23_orig.min(), sin2_t23_orig.max(), n_points)
    dm31_vals_fine = np.linspace(dm31_vals_orig.min(), dm31_vals_orig.max(), n_points)
    X_fine, Y_fine = np.meshgrid(sin2_t23_fine, dm31_vals_fine)
    chi2_grid_fine = griddata((df['sin2_theta23'], df['dm31']), df['delta_chi2'], (X_fine, Y_fine), method='cubic')

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cl2_68 = chi2.ppf(0.68, df=2)
    cl2_90 = chi2.ppf(0.90, df=2)
    
    levels = np.linspace(np.nanmin(chi2_grid_fine), np.nanmax(chi2_grid_fine), 30)
    cf = ax.contourf(X_fine, Y_fine, chi2_grid_fine, levels=levels, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\Delta\chi^2$")

    CS = ax.contour(X_fine, Y_fine, chi2_grid_fine,
                    levels=[cl2_68, cl2_90],
                    colors=['darkorange', 'yellow'], linewidths=1.5)
    fmt = {cl2_90: "90% CL", cl2_68: "68% CL"}
    ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

    ax.plot(best_fit_row['sin2_theta23'], best_fit_row['dm31'],
            marker='*', color='red', markersize=12, label='Official Best Fit', linestyle='None')
    
    ax.set_xlabel(r"$\sin^2\theta_{23}$")
    ax.set_ylabel(r"$\Delta m^2_{31}$ [eV$^2$]")
    ax.set_title("Official ORCA $\chi^2$ Landscape (NO)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_path = os.path.join(outdir, "official_contours.png")
    plt.savefig(output_path)
    print(f"[INFO] Saved official contour plot to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot official ORCA chi-squared contours.")
    parser.add_argument("--filepath", type=str, default = "../datafiles/ORCA/chi2_landscape_NO.csv", help="Full path to the official chi2_landscape_NO.csv file.")
    parser.add_argument("--outdir", type=str, default = "../results/plots/official", help="Directory to save the output plot.")
    
    cli_args = parser.parse_args()
    plot_official_contours(cli_args.filepath, cli_args.outdir)