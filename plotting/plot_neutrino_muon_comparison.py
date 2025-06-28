import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

# Add the scripts directory to Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(base_dir, 'scripts')
sys.path.insert(0, scripts_dir)
from Analysis import Analysis

def plot_neutrino_muon_comparison():
    """
    Generates a plot comparing the binned neutrino signal and atmospheric muon background
    at the best-fit oscillation parameters.
    """
    print("--- Generating Neutrino vs. Muon Background Comparison Plot ---")

    # --- Configuration ---
    config_file = '../config/config_orca_datafit.yaml'
    output_filename = 'neutrino_muon_comparison.png'
    output_path = os.path.join(os.path.dirname(__file__), output_filename)

    # --- Load Analysis and Simulation ---
    # We initialize the Analysis object, which in turn sets up the Simulation
    print(f"Loading analysis from config: {config_file}")
    analysis = Analysis(experiment="ORCA", livetime=1.39, 
                        filename='../datafiles/ORCA/ORCA_MC_dataverse_with_muons.parquet', 
                        config=config_file)
    sim = analysis.sim
    
    # --- Ensure Best-Fit Rates are Calculated ---
    # The Analysis constructor in data-fitting mode doesn't compute a BF model,
    # so we do it manually here.
    print("Calculating best-fit neutrino rates...")
    bf_params = analysis.bf
    if sim._BF_rates is None:
        analysis.sim.ComputeBFRates(
            np.arcsin(np.sqrt(bf_params['s2t12'])),
            np.arcsin(np.sqrt(bf_params['s2t13'])),
            np.arcsin(np.sqrt(bf_params['s2t23'])),
            bf_params['m21'],
            bf_params['m31'],
            bf_params['dCP'] * np.pi
        )
    
    # --- Bin the Rates ---
    # This will bin the neutrinos and add the already-binned muons
    print("Binning total best-fit model (neutrinos + muons)...")
    total_binned_model, _ = sim.BinHypothesis(sim._BF_rates)

    # --- Separate Neutrino and Muon Components ---
    # The muon background is stored and already cut by the binning mask
    muon_binned = sim._muon_bkg_binned[sim._cut_bins]
    
    # The neutrino component is the total minus the background
    neutrino_binned = total_binned_model - muon_binned

    # --- Reshape to 3D for Energy Spectrum Projection ---
    # We need to create a full-size array and fill it using the cut mask
    # to correctly reshape the data.
    num_morph = sim._num_morphology
    num_ebins = len(sim._E_reco_bins) - 1
    num_zbins = len(sim._cosT_reco_bins) - 1
    total_bins_unmasked = num_morph * num_ebins * num_zbins

    full_neutrino = np.zeros(total_bins_unmasked)
    full_neutrino[sim._cut_bins] = neutrino_binned
    reshaped_neutrino = full_neutrino.reshape((num_morph, num_ebins, num_zbins))

    full_muon = np.zeros(total_bins_unmasked)
    full_muon[sim._cut_bins] = muon_binned
    reshaped_muon = full_muon.reshape((num_morph, num_ebins, num_zbins))

    # --- Project onto Energy Axis ---
    # Sum over the first (PID) and third (Zenith) axes
    neutrino_energy_spectrum = np.sum(reshaped_neutrino, axis=(0, 2))
    muon_energy_spectrum = np.sum(reshaped_muon, axis=(0, 2))

    # --- Plotting ---
    print("Creating plot...")
    # plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 7))

    energy_bins = sim._E_reco_bins
    
    ax.step(energy_bins, np.append(neutrino_energy_spectrum, neutrino_energy_spectrum[-1]), 
            where='post', label='Neutrino Signal (Best-Fit)')
    ax.step(energy_bins, np.append(muon_energy_spectrum, muon_energy_spectrum[-1]), 
            where='post', label='Atmospheric Muon Background')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reconstructed Energy [GeV]', fontsize=14)
    ax.set_ylabel('Binned Event Rate', fontsize=14)
    ax.set_title('Best-Fit Model Components: Neutrino vs. Muon Background', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- Save Figure ---
    plt.savefig(output_path)
    print(f"Plot saved successfully to: {output_path}")

if __name__ == '__main__':
    plot_neutrino_muon_comparison()