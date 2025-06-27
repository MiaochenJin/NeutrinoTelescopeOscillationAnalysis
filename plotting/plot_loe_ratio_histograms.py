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

def main():
    # Paths
    config_path = os.path.join(base_dir, 'config', 'config_orca.yaml')
    
    # Initialize Analysis object for ORCA with config_orca.yaml
    # This automatically sets up Simulation, SetInitialFlux, ComputeBFRates, and BinBFRatesLoE
    analysis = Analysis(
        experiment='ORCA',
        livetime=1.39,  # ORCA livetime
        # filename='../datafiles/ORCA/ORCA_MC.parquet',
        filename='../datafiles/ORCA/ORCA_MC_dataverse.parquet',
        config=config_path
    )
    
    # # Manually compute best-fit rates, as this is not done in data-fitting mode
    # bf = analysis.bf
    # analysis.sim.ComputeBFRates(
    #     np.arcsin(np.sqrt(bf['s2t12'])),
    #     np.arcsin(np.sqrt(bf['s2t13'])),
    #     np.arcsin(np.sqrt(bf['s2t23'])),
    #     bf['m21'],
    #     bf['m31'],
    #     bf['dCP'] * np.pi
    # )
    
    # Temporarily clear the cut_bins to get full uncut data for plotting
    original_cut_bins = analysis.sim._cut_bins.copy() if hasattr(analysis.sim, '_cut_bins') else []
    analysis.sim._cut_bins = []
    
    # Get the oscillated L/E data (best fit)
    binned_events_osc, binned_errors_osc = analysis.sim.BinWeightedRate2DLoE(analysis.sim._BF_rates, return_error=True)
    
    # Get the no-oscillation L/E data (all oscillation parameters = 0)
    # For no oscillations, we need to set all mixing angles to 0 and mass differences to 0
    no_osc_rates = analysis.sim.GetOscillatedRate(0, 0, 0, 0, 0, 0, Ordering='normal')
    binned_events_noosc, binned_errors_noosc = analysis.sim.BinWeightedRate2DLoE(no_osc_rates, return_error=True)
    
    # Restore the original cut_bins
    analysis.sim._cut_bins = original_cut_bins
    
    loe_bins = analysis.sim._loe_bins
    n_morph = analysis.sim._num_morphology
    n_bins = len(loe_bins) - 1

    # Reshape to (n_morph, n_bins)
    binned_events_osc = binned_events_osc.reshape(n_morph, n_bins)
    binned_errors_osc = np.sqrt(binned_errors_osc.reshape(n_morph, n_bins))
    binned_events_noosc = binned_events_noosc.reshape(n_morph, n_bins)
    binned_errors_noosc = np.sqrt(binned_errors_noosc.reshape(n_morph, n_bins))

    # Calculate ratios and their errors
    # Use np.where to avoid division by zero
    ratios = np.where(binned_events_noosc > 0, binned_events_osc / binned_events_noosc, 0)
    
    # Error propagation for ratio: sqrt((σ_a/a)² + (σ_b/b)²) * (a/b)
    ratio_errors = np.where(
        (binned_events_noosc > 0) & (binned_events_osc > 0),
        ratios * np.sqrt((binned_errors_osc/binned_events_osc)**2 + (binned_errors_noosc/binned_events_noosc)**2),
        0
    )

    # Morphology labels (customize as needed)
    morph_labels = [f"Morphology {i}" for i in range(n_morph)]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create subplots for each morphology
    fig, axes = plt.subplots(1, n_morph, figsize=(24, 8))
    if n_morph == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    bin_centers = 0.5 * (loe_bins[:-1] + loe_bins[1:])
    
    for m in range(n_morph):
        ax = axes[m]
        
        # For step histograms, we need to extend the data to show the last bin
        # Add the last bin value to the end so the step extends properly
        ratios_extended = np.append(ratios[m], ratios[m][-1])
        
        # Plot step histogram for ratios
        ax.step(loe_bins, ratios_extended, where='post', 
                color=colors[m % len(colors)], linewidth=2, 
                label=f'{morph_labels[m]} Ratio')
        
        # Add error bars
        ax.errorbar(bin_centers, ratios[m], yerr=ratio_errors[m],
                   fmt='none', color='black', capsize=3, capthick=1)
        
        # Add dashed line at ratio = 1
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Oscillation')
        
        ax.set_xlabel('L/E [km/GeV]')
        ax.set_ylabel('Ratio (Oscillated / No Oscillation)')
        ax.set_xscale('log')
        ax.set_title(f'{morph_labels[m]} Oscillation Ratio')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('../results/plots/loe_ratio_histograms_dataverse.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main() 