import numpy as np
import matplotlib.pyplot as plt
import yaml
from Analysis import Analysis
import os

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config_orca.yaml')
    
    # Initialize Analysis object for ORCA with config_orca.yaml
    # This automatically sets up Simulation, SetInitialFlux, ComputeBFRates, and BinBFRatesLoE
    analysis = Analysis(
        experiment='ORCA',
        livetime=1.39,  # ORCA livetime
        filename='../datafiles/ORCA/ORCA_MC.parquet',
        config=config_path
    )
    
    # Temporarily clear the cut_bins to get full uncut data for plotting
    # The cut_bins were created for a different binning scheme and don't match L/E dimensions
    original_cut_bins = analysis.sim._cut_bins.copy() if hasattr(analysis.sim, '_cut_bins') else []
    analysis.sim._cut_bins = []
    
    # Get the full uncut L/E data for plotting
    binned_events, binned_errors = analysis.sim.BinWeightedRate2DLoE(analysis.sim._BF_rates, return_error=True)
    
    # Restore the original cut_bins
    analysis.sim._cut_bins = original_cut_bins
    
    loe_bins = analysis.sim._loe_bins
    n_morph = analysis.sim._num_morphology
    n_bins = len(loe_bins) - 1

    # Reshape to (n_morph, n_bins)
    binned_events = binned_events.reshape(n_morph, n_bins)
    binned_errors = np.sqrt(binned_errors.reshape(n_morph, n_bins))  # MC error is sqrt(variance)

    # Morphology labels (customize as needed)
    morph_labels = [f"Morphology {i}" for i in range(n_morph)]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create subplots for each morphology
    fig, axes = plt.subplots(1, n_morph, figsize=(15, 5))
    if n_morph == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    bin_centers = 0.5 * (loe_bins[:-1] + loe_bins[1:])
    bin_widths = np.diff(loe_bins)
    
    for m in range(n_morph):
        ax = axes[m]
        
        # Plot histogram with error bars
        ax.bar(bin_centers, binned_events[m], width=bin_widths, 
               alpha=0.7, color=colors[m % len(colors)], 
               label=morph_labels[m])
        
        # Add error bars on top of histogram bars
        ax.errorbar(bin_centers, binned_events[m], yerr=binned_errors[m],
                   fmt='none', color='black', capsize=3, capthick=1)
        
        ax.set_xlabel('L/E [km/GeV]')
        ax.set_ylabel('Weighted Event Count')
        ax.set_xscale('log')
        ax.set_title(f'{morph_labels[m]} (Best Fit)')
        ax.grid(True, alpha=0.3)
    
    # plt.suptitle('L/E Event Distributions by Morphology (Best Fit)', fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../results/loe_histograms.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main() 