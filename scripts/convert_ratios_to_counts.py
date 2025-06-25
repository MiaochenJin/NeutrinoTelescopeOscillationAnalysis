import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the scripts directory to Python path for sibling imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'scripts'))

from Simulation import Simulation # Use Simulation directly
from utils import LoadBinnedData

def verify_ratios():
    """
    This script provides a definitive check on the data loading and processing pipeline.
    It generates absolute event counts from source ratio files, saves them,
    reloads them, and plots all three versions of the data ratio (original,
    generated-in-memory, and reloaded-from-disk) to ensure they are identical.
    If all lines on the plot overlap, the data conversion and I/O is correct.
    """
    print("--- Starting Data Verification ---")
    
    # --- Configuration ---
    data_dir = os.path.join(base_dir, 'datafiles', 'ORCA')
    
    original_ratio_files = {
        0: "Shower_median.csv", 1: "HPT_median.csv", 2: "LPT_median.csv"
    }

    # --- Initialize Simulation Object Directly to Avoid Systematics ---
    print("Initializing Simulation object...")
    sim = Simulation(
        experiment='ORCA',
        livetime=1.39,
        filename=os.path.join(data_dir, 'ORCA_MC.parquet'),
        mode='Standard'
    )
    sim.SetInitialFlux()
    sim._analysis_binning = "LoE" # Manually set the binning

    # --- Calculate a PURE no-oscillation baseline ---
    print("Calculating raw no-oscillation baseline...")
    no_osc_unbinned = sim.GetOscillatedRate(0, 0, 0, 0, 0, 0)
    N_no_osc, _ = sim.BinHypothesis(no_osc_unbinned)
    print("...baseline calculation complete.")

    # --- Load original ratios and generate N_dat on the fly for verification ---
    print("Loading original '*_median.csv' files and generating counts on the fly...")
    n_morph = sim._num_morphology
    n_bins = len(sim._loe_bins) - 1
    N_dat_generated_from_ratio = np.array([])
    original_ratios_all = np.array([])

    for i in range(n_morph):
        # Load original ratio data
        original_df = pd.read_csv(os.path.join(data_dir, original_ratio_files[i]), header=None)
        original_ratio_morph = original_df.iloc[:, 1].values
        original_ratios_all = np.append(original_ratios_all, original_ratio_morph)

        # Generate the "correct" N_dat for this morphology
        no_osc_counts_morph = N_no_osc.reshape(n_morph, n_bins)[i]
        N_dat_morph = original_ratio_morph * no_osc_counts_morph
        N_dat_generated_from_ratio = np.append(N_dat_generated_from_ratio, N_dat_morph)

    # This ratio is calculated from the on-the-fly data and should perfectly match the input
    calculated_ratio = N_dat_generated_from_ratio / N_no_osc
    print(N_dat_generated_from_ratio)
    print(calculated_ratio)

    # --- Save the generated counts to CSV files ---
    print("\n--- Saving generated counts to files ---")
    
    # Correct mapping from morphology index to output file name
    output_files = {
        0: "counts_hpt.csv", 
        1: "counts_lpt.csv", 
        2: "counts_shower.csv"
    }
    
    generated_counts_reshaped = N_dat_generated_from_ratio.reshape(n_morph, n_bins)

    for i in range(n_morph):
        # The data in generated_counts_reshaped[i] corresponds to original_ratio_files[i]
        # So we use the same index `i` to get the correct output file name.
        output_filename = os.path.join(data_dir, output_files[i])
        output_df = pd.DataFrame({'N_dat': generated_counts_reshaped[i]})
        output_df.to_csv(output_filename, index=False)
        print(f"  Saved counts for morphology {i} to: {output_filename}")

    # --- Load the newly saved data and calculate its ratio for plotting ---
    print("\nLoading newly saved counts files for verification plot...")
    N_dat_from_file = LoadBinnedData(data_dir, file_pattern="counts_*.csv")
    reloaded_ratio = N_dat_from_file / N_no_osc

    # --- Plotting ---
    print("Generating self-consistency plot...")
    
    fig, axes = plt.subplots(n_morph, 1, figsize=(12, 18), sharex=True)
    morph_labels = ['HPT', 'LPT', 'Shower']
    bin_centers = 0.5 * (sim._loe_bins[:-1] + sim._loe_bins[1:])
    
    for i in range(n_morph):
        ax = axes[i]
        start_idx = i * n_bins
        end_idx = (i + 1) * n_bins

        # Plot the ratio calculated from the on-the-fly generated counts
        ax.step(bin_centers, calculated_ratio[start_idx:end_idx], where='mid', 
                label='On-the-fly Ratio (Generated from source)', color='red', lw=4, alpha=0.7)

        # Plot the original ratio from the source _median.csv files
        ax.step(bin_centers, original_ratios_all[start_idx:end_idx], where='mid', 
                label='Original Ratio (from *_median.csv)', color='blue', linestyle='--', lw=2)

        # Plot the ratio from the reloaded files to confirm save/load integrity
        ax.step(bin_centers, reloaded_ratio[start_idx:end_idx], where='mid',
                label='Saved & Reloaded Ratio', color='limegreen', linestyle=':', lw=3, zorder=10)

        ax.set_ylabel('Ratio to No Oscillation')
        ax.set_title(f'Morphology: {morph_labels[i]} (Self-Consistency Check)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        ax.set_ylim(0, 1.5)

    axes[-1].set_xlabel('L/E [km/GeV]')
    axes[0].set_xscale('log')
    fig.suptitle('Verification: Original Data vs. On-the-fly Processed Ratio', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    outpath = os.path.join(base_dir, 'results', 'plots', 'verification_plot.png')
    plt.savefig(outpath, dpi=300)
    print(f"--- Verification plot saved to: {outpath} ---")
    
    # --- Final comparison: Print the generated counts vs the file counts ---
    print("\n--- Numerical Comparison of Event Counts ---")
    N_dat_from_file = LoadBinnedData(data_dir, file_pattern="counts_*.csv")
    
    for i in range(n_morph):
        start_idx = i * n_bins
        end_idx = (i + 1) * n_bins
        print(f"\n--- Morphology: {morph_labels[i]} ---")
        
        comparison_df = pd.DataFrame({
            'Generated_Counts_on_the_fly': N_dat_generated_from_ratio[start_idx:end_idx],
            'Counts_from_file': N_dat_from_file[start_idx:end_idx],
        })
        comparison_df['Difference'] = comparison_df['Generated_Counts_on_the_fly'] - comparison_df['Counts_from_file']
        print(comparison_df)

if __name__ == '__main__':
    verify_ratios() 