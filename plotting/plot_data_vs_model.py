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

    # Initialize Analysis object. This will set up the simulation and, if in
    # data-fitting mode with a parquet file specified, it will NOT load N_dat_fixed.
    # We will load it manually for plotting purposes.
    analysis = Analysis(
        experiment='ORCA',
        livetime=1,
        filename='../datafiles/ORCA/ORCA_MC_dataverse.parquet', # MC for model calcs
        config=config_path
    )

    # --- Load The Real Data (N_dat) in L/E bins for plotting ---
    data_settings = analysis.data_fitting_settings
    data_parquet_relative = data_settings.get('data_parquet_file')
    if not data_parquet_relative:
        print("Error: 'data_parquet_file' not specified in the config under DataFitting.")
        sys.exit(1)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    data_parquet_abs = os.path.join(config_dir, data_parquet_relative)
    
    # Use the simulation object to bin the real data into L/E bins
    print(f"Loading and binning real data from {data_parquet_abs}...")
    binned_data_loe = analysis.sim.BinDataFromParquet(data_parquet_abs, binning_type='LoE')

    # --- Calculate a Best-Fit Model (N_mod) for comparison ---
    print("Calculating best-fit model...")
    bf = analysis.bf
    # 1. Get unweighted rates for best-fit oscillation parameters
    bf_rates = analysis.sim.GetOscillatedRate(
        np.arcsin(np.sqrt(bf['s2t12'])),
        np.arcsin(np.sqrt(bf['s2t13'])),
        np.arcsin(np.sqrt(bf['s2t23'])),
        bf['m21'], bf['m31'], bf['dCP'] * np.pi
    )
    # 2. Bin the rates into L/E bins
    binned_model_loe, binned_model_loe_err = analysis.sim.BinWeightedRate2DLoE(bf_rates, return_error=True)

    # --- Reshape and Plot ---
    loe_bins = analysis.sim._loe_bins
    n_morph = analysis.sim._num_morphology
    n_bins = len(loe_bins) - 1

    binned_data_loe = binned_data_loe.reshape(n_morph, n_bins)
    binned_model_loe = binned_model_loe.reshape(n_morph, n_bins)
    binned_model_loe_err = np.sqrt(np.abs(binned_model_loe_err.reshape(n_morph, n_bins))) # Use abs for safety

    bin_centers = 0.5 * (loe_bins[:-1] + loe_bins[1:])
    
    for m in range(n_morph):
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot Data
        ax.step(loe_bins[:-1], binned_data_loe[m], where='post', label='Data (from Parquet)', color='black', linewidth=2)
        
        # Plot Model
        ax.step(loe_bins[:-1], binned_model_loe[m], where='post', label='Best-Fit Model (MC)', color='red', linestyle='--')
        
        # Add model statistical error
        ax.fill_between(bin_centers, 
                        binned_model_loe[m] - binned_model_loe_err[m],
                        binned_model_loe[m] + binned_model_loe_err[m],
                        step='mid', color='red', alpha=0.3, label='Model Stat. Error')


        ax.set_xlabel('L/E [km/GeV]')
        ax.set_ylabel('Events')
        ax.set_title(f'Morphology {m}: Data vs. Best-Fit Model')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        output_path = os.path.join(base_dir, 'results', 'plots', f'data_vs_model_loe_morph_{m}.png')
        plt.savefig(output_path, dpi=300)
        print(f"\nSaved comparison plot for morphology {m} to {output_path}")

if __name__ == '__main__':
    main() 