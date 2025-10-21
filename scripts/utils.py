import nuSQuIDS as nsq
import nuflux
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import yaml
import os

units = nsq.Const()
interactions = False
flux = nuflux.makeFlux('IPhonda2014_spl_solmin') # Changed to match original codebase

neutrino_flavors = 3
# some earth parameters to compute L/E
R_E = 6371 # KM

# define a dictionary for nutrino type and nuflux
f_dict = dict({"nue": 0, "numu": 1, "nutau": 2})
t_dict = dict({"nu": 0, "nubar": 1})
nuflux_label = dict({"nue_nu": nuflux.NuE, "nue_nubar": nuflux.NuEBar, 
					 "numu_nu": nuflux.NuMu, "numu_nubar": nuflux.NuMuBar, 
					 "nutau_nu": nuflux.NuTau, "nutau_nubar": nuflux.NuTauBar})
# define dict to grab flux from nuflux
nuflux_dict = dict({12: (0, 0), -12: (1,0),
                    14: (0,1), -14: (1,1),
                    16: (0,2), -16: (1,2)})




def plot_reco_energy_distribution(binned_events_osci, binned_events_non_osci, reco_energy_bins):
    """
    Plot the reconstructed energy distribution for different morphologies (cascades, tracks, low-purity tracks).

    Parameters:
    - binned_events: 8D numpy array (oscillated event counts).
    - reco_energy_bins: Reconstructed energy bin edges.
    - savename: Optional string for saving the figure.
    """

    # Morphology labels: (0 = Cascades, 1 = High Purity Tracks, 2 = Low Purity Tracks)
    morphologies = ["Cascades", "High Purity Tracks", "Low Purity Tracks"]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 18))

    for pid in range(3):
        # Sum over all dimensions except Reco Energy and Morphology
        reco_energy_dist_osci = np.sum(binned_events_osci[:, :, :, :, :, :, :, pid], axis=(0, 2, 3, 4, 5, 6))
        reco_energy_dist_non_osci = np.sum(binned_events_non_osci[:, :, :, :, :, :, :, pid], axis=(0, 2, 3, 4, 5, 6))

        # Plot results
        ax[pid].hist(reco_energy_bins[:-1], bins=reco_energy_bins, weights=reco_energy_dist_osci, histtype='step', linewidth=2, label = 'Oscillation')
        ax[pid].hist(reco_energy_bins[:-1], bins=reco_energy_bins, weights=reco_energy_dist_non_osci, histtype='step', linewidth=2, label = 'Non-Oscillation')

        if pid == 2: ax[pid].set_xlabel('Reconstructed Energy (GeV)')
        ax[pid].set_ylabel('Event Rate')
        ax[pid].set_xscale("log")
        ax[pid].set_title(f'{morphologies[pid]}')
        print(f"number of unoscillated and oscillated events in pid {pid} is {np.sum(reco_energy_dist_non_osci)} and {np.sum(reco_energy_dist_osci)}")

    ax[0].legend()

    plt.tight_layout()
    plt.savefig(f"./Asimov_Distribution.png", bbox_inches='tight')

def LoadBinnedData(directory, file_pattern="*.csv"):
    """
    Loads and concatenates binned data from CSV files in a given directory
    that match a specific pattern.
    Assumes each CSV contains a column with the binned event counts.
    The column name is inferred if it's 'N_dat', 'count', or 'counts'.
    """
    all_files = sorted(glob.glob(os.path.join(directory, file_pattern)))
    if not all_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in directory: {directory}")
    
    df_list = [pd.read_csv(f) for f in all_files]
    
    # Try to find the data column
    sample_df = df_list[0]
    data_col = None
    possible_cols = ['N_dat', 'count', 'counts']
    for col in possible_cols:
        if col in sample_df.columns:
            data_col = col
            break
            
    if data_col is None:
        raise KeyError(f"Data column (e.g., 'N_dat', 'count') not found in CSV files in {directory}")

    # Concatenate the data column from all files
    full_data = pd.concat([df[data_col] for df in df_list], ignore_index=True)
    
    return full_data.values

if __name__ == "__main__":
    files = glob.glob("results/point_*.csv")
    df_all = pd.concat([pd.read_csv(f) for f in files])
    df_all.to_csv("sensitivity_grid.csv", index=False)