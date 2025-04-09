import nuSQuIDS as nsq
import nuflux
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd

units = nsq.Const()
interactions = False
flux = nuflux.makeFlux('IPhonda2014_sk_solmin')

neutrino_flavors = 3
# some earth parameters to compute L/E
R_E = 6371 # KM

# Set up mixing parameters as in nu-fit5.0
t12_bf = np.arcsin(np.sqrt(0.303))
t13_bf = np.arcsin(np.sqrt(0.02203))
t23_bf = np.arcsin(np.sqrt(0.572))
m21_bf = 7.41e-5
m31_bf = 2.511e-3
dCP_bf = 4.3
syst_bf = [0.11, -0.08, -0.08, 0.59, -0.08, -0.14, 0, 0.01, -0.004, -0.019, -0.005]

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

if __name__ == "__main__":
    files = glob.glob("results/point_*.csv")
    df_all = pd.concat([pd.read_csv(f) for f in files])
    df_all.to_csv("sensitivity_grid.csv", index=False)