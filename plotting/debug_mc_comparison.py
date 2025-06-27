import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
import pandas as pd

# Add the scripts directory to Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(base_dir, 'scripts')
sys.path.insert(0, scripts_dir)

from Analysis import Analysis

def load_and_bin_events(config_path, mc_filename, ltime):
    """Helper function to load MC data and bin it."""
    analysis = Analysis(
        experiment='ORCA',
        livetime=ltime,
        filename=mc_filename,
        config=config_path
    )

    # # Manually compute best-fit rates
    # bf = analysis.bf
    # analysis.sim.ComputeBFRates(
    #     np.arcsin(np.sqrt(bf['s2t12'])),
    #     np.arcsin(np.sqrt(bf['s2t13'])),
    #     np.arcsin(np.sqrt(bf['s2t23'])),
    #     bf['m21'],
    #     bf['m31'],
    #     bf['dCP'] * np.pi
    # )

    # Get binned events for oscillated and no-oscillation cases
    binned_events_osc, _ = analysis.sim.BinWeightedRate2DLoE(analysis.sim._BF_rates, return_error=True)
    no_osc_rates = analysis.sim.GetOscillatedRate(0, 0, 0, 0, 0, 0, Ordering='normal')
    binned_events_noosc, _ = analysis.sim.BinWeightedRate2DLoE(no_osc_rates, return_error=True)
    
    # Reshape
    n_morph = analysis.sim._num_morphology
    n_bins = len(analysis.sim._loe_bins) - 1
    binned_events_osc = binned_events_osc.reshape(n_morph, n_bins)
    binned_events_noosc = binned_events_noosc.reshape(n_morph, n_bins)
    
    return binned_events_osc, binned_events_noosc, analysis.sim._loe_bins, n_morph

def main():
    config_path = os.path.join(base_dir, 'config', 'config_orca.yaml')
    
    # File paths
    old_mc_file = os.path.join(base_dir, 'datafiles', 'ORCA', 'ORCA_MC.parquet')
    new_mc_file = os.path.join(base_dir, 'datafiles', 'ORCA', 'ORCA_MC_dataverse.parquet')

    print("Loading and processing OLD MC file...")
    osc_old, noosc_old, loe_bins_old, n_morph_old = load_and_bin_events(config_path, old_mc_file, 1.39)
    
    print("\nLoading and processing NEW MC file...")
    osc_new, noosc_new, loe_bins_new, n_morph_new = load_and_bin_events(config_path, new_mc_file, 1.)

    # --- Sanity Checks ---
    print("\n--- Sanity Checks ---")
    print(f"Old MC - Osc events shape: {osc_old.shape}")
    print(f"New MC - Osc events shape: {osc_new.shape}")
    print(f"Old MC - Total Events (No Osc): {np.sum(noosc_old)}")
    print(f"New MC - Total Events (No Osc): {np.sum(noosc_new)}")
    print(f"Old MC - Zero Bins (No Osc): {np.sum(noosc_old == 0)} out of {noosc_old.size}")
    print(f"New MC - Zero Bins (No Osc): {np.sum(noosc_new == 0)} out of {noosc_new.size}")
    print(f"Old MC - Zero Bins (Osc): {np.sum(osc_old == 0)} out of {osc_old.size}")
    print(f"New MC - Zero Bins (Osc): {np.sum(osc_new == 0)} out of {osc_new.size}")
    
    # --- Plotting ---
    assert np.array_equal(loe_bins_old, loe_bins_new), "L/E bins are different between the two files!"
    assert n_morph_old == n_morph_new, "Number of morphologies is different!"
    
    loe_bins = loe_bins_old
    n_morph = n_morph_old
    bin_centers = 0.5 * (loe_bins[:-1] + loe_bins[1:])
    
    for m in range(n_morph):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot No-Oscillation Comparison
        ax1.step(loe_bins[:-1], noosc_old[m], where='post', label='Old MC', color='blue')
        ax1.step(loe_bins[:-1], noosc_new[m], where='post', label='New MC (Dataverse)', color='red', linestyle='--')
        ax1.set_ylabel('Events')
        ax1.set_title(f'Morphology {m}: No-Oscillation Events Comparison')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # Plot Oscillation Comparison
        ax2.step(loe_bins[:-1], osc_old[m], where='post', label='Old MC', color='blue')
        ax2.step(loe_bins[:-1], osc_new[m], where='post', label='New MC (Dataverse)', color='red', linestyle='--')
        ax2.set_xlabel('L/E [km/GeV]')
        ax2.set_ylabel('Events')
        ax2.set_title(f'Morphology {m}: Best-Fit Oscillation Events Comparison')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'../results/plots/debug_mc_comparison_morph_{m}.png', dpi=300)
        print(f"\nSaved comparison plot for morphology {m} to ../results/plots/debug_mc_comparison_morph_{m}.png")

if __name__ == '__main__':
    main() 