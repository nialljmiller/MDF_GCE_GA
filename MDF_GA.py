#!/usr/bin/env python3.8
################################
# Author: N Miller, M Joyce
################################

# Importing required libraries
import matplotlib.pyplot as plt
import warnings
import numpy as np
import sys
import argparse
from scipy.interpolate import CubicSpline
from deap import base, creator, tools
import random
import Gal_GA_PP as Gal_GA
import pandas as pd
import os
import checkpoint  # checkpointing utilities
# Import plotting module
import mdf_plotting


def load_bensby_data(file_path='data/Bensby_Data.tsv'):
    obs_age_data = pd.read_csv(file_path, sep='\t')
    print(f"Loaded Bensby data with shape: {obs_age_data.shape}")
    print(f"Columns available: {list(obs_age_data.columns)}")
    return obs_age_data

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Adding custom paths
sys.path.append('../')

# Create argument parser
parser = argparse.ArgumentParser(description='Run MDF Genetic Algorithm with optional plotting only')
parser.add_argument('--plot-only', action='store_true', help='Skip computation and only generate plots')
parser.add_argument('--results-file', type=str, default='GA/simulation_results.csv', 
                   help='CSV file containing results (for plot-only mode)')
args = parser.parse_args()

# Parse parameters from the 'bulge_pcard.txt' file
params = Gal_GA.parse_inlist('bulge_pcard.txt')

# Assign parsed parameters to variables
obs_file = params['obs_file']
iniab_header = params['iniab_header']
sn1a_header = params['sn1a_header']
sigma_2_list = params['sigma_2_list']
tmax_1_list = params['tmax_1_list']
tmax_2_list = params['tmax_2_list']
infall_timescale_1_list = params['infall_timescale_1_list']
infall_timescale_2_list = params['infall_timescale_2_list']
comp_array = params['comp_array']
sfe_array = params['sfe_array']
imf_array = params['imf_array']
imf_upper_limits = params['imf_upper_limits']
sn1a_assumptions = params['sn1a_assumptions']
stellar_yield_assumptions = params['stellar_yield_assumptions']
mgal_values = params['mgal_values']
nb_array = params['nb_array']
sn1a_rates = params['sn1a_rates']
timesteps = params['timesteps']
A2 = params['A2']
A1 = params['A1']
physical_constraints_freq = params['physical_constraints_freq']
delta_sfe_array = params['delta_sfe_array']
exploration_steps = params['exploration_steps']
popsize = params['popsize']
generations = params['generations']
crossover_probability = params['crossover_probability']
mutation_probability = params['mutation_probability']
tournament_size = params['tournament_size']
selection_threshold = params['selection_threshold']

obs_age_data_loss_metric = params['obs_age_data_loss_metric']
mdf_vs_age_weight = params['mdf_vs_age_weight']






output_interval = params.get('output_interval')

loss_metric = params['loss_metric']
fancy_mutation = params['fancy_mutation']
shrink_range = params['shrink_range']

# Parameters controlling mutation and augmentation scales
gaussian_sigma_scale = params.get('gaussian_sigma_scale', 0.01)
crossover_noise_fraction = params.get('crossover_noise_fraction', 0.05)
perturbation_strength = params.get('perturbation_strength', 0.1)


# Load and normalize observational data
feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
normalized_count = count / count.max()  # Normalize count for comparison



# Load the data
obs_age_data = load_bensby_data('data/Bensby_Data.tsv')

# Display basic info about the dataset
print("\nDataset Info:")
print(f"Number of stars: {len(obs_age_data)}")
print(f"Number of columns: {len(obs_age_data.columns)}")

print("\nFirst few rows:")
print(obs_age_data.head())

print("\nColumn data types:")
print(obs_age_data.dtypes)

# Example usage - accessing specific columns
print(f"\nExample access:")
print(f"Joyce ages: {obs_age_data['Joyce_age'].head()}")
print(f"Bensby ages: {obs_age_data['Bensby'].head()}")
print(f"[Fe/H] values: {obs_age_data['[Fe/H]'].head()}")
print(f"[Mg/Fe] values: {obs_age_data['[Mg/Fe]'].head()}")

# Show some basic statistics
print(f"\nBasic statistics for key columns:")
key_columns = ['Joyce_age', 'Bensby', '[Fe/H]', '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]']
for col in key_columns:
    if col in obs_age_data.columns:
        print(f"{col}: mean={obs_age_data[col].mean():.2f}, std={obs_age_data[col].std():.2f}, range=[{obs_age_data[col].min():.2f}, {obs_age_data[col].max():.2f}]")



# Global GalGA object to be used for both computation and plotting
GalGA = None

os.makedirs('GA', exist_ok=True)
os.makedirs('GA/loss', exist_ok=True)
os.makedirs('GA/analysis', exist_ok=True)

# Save/load walker history
def save_walker_history():
    if not hasattr(GalGA, 'walker_history'):
        return

    np.savez_compressed(
        'GA/walker_history.npz',
        walker_ids=np.array(list(GalGA.walker_history.keys()), dtype=np.int32),
        histories=[np.array(h) for h in GalGA.walker_history.values()],
        mdf_data=np.array(GalGA.mdf_data, dtype=object),      # your [Fe/H] vs count
        alpha_data=np.array(GalGA.alpha_data, dtype=object)   # your α-distributions
    )

    print("Walker history saved")

def load_walker_history():
    if not os.path.exists('GA/walker_history.npz'):
        return {}
        
    data = np.load('GA/walker_history.npz', allow_pickle=True)
    walker_ids = data['walker_ids']
    histories = data['histories']
    
    walker_history = {}
    for i, walker_id in enumerate(walker_ids):
        walker_history[int(walker_id)] = histories[i]
    
    print("Walker history loaded")
    return walker_history




def run_ga(cp_manager):
    """Run the genetic algorithm with optional checkpointing."""
    global GalGA
    
    # Initialize the Galactic Evolution Genetic Algorithm class with parsed parameters
    GalGA = Gal_GA.GalacticEvolutionGA(
        iniab_header=iniab_header,
        sn1a_header=sn1a_header,
        sigma_2_list=sigma_2_list,
        tmax_1_list=tmax_1_list,
        tmax_2_list=tmax_2_list,    
        infall_timescale_1_list=infall_timescale_1_list,
        infall_timescale_2_list=infall_timescale_2_list,
        comp_array=comp_array,
        imf_array=imf_array,
        sfe_array=sfe_array,
        delta_sfe_array=delta_sfe_array,
        imf_upper_limits=imf_upper_limits,
        sn1a_assumptions=sn1a_assumptions,
        stellar_yield_assumptions=stellar_yield_assumptions,
        mgal_values=mgal_values,
        nb_array=nb_array,
        sn1a_rates=sn1a_rates,
        timesteps=timesteps,
        A1=A1,
        A2=A2,
        feh=feh,
        normalized_count=normalized_count,
        obs_age_data=obs_age_data,
        loss_metric=loss_metric,
        obs_age_data_loss_metric = obs_age_data_loss_metric,
        mdf_vs_age_weight = mdf_vs_age_weight,
        fancy_mutation=fancy_mutation,
        shrink_range=shrink_range,
        gaussian_sigma_scale=gaussian_sigma_scale,
        crossover_noise_fraction=crossover_noise_fraction,
        perturbation_strength=perturbation_strength,
        tournament_size=tournament_size,
        threshold=selection_threshold,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        physical_constraints_freq=physical_constraints_freq,
        exploration_steps=exploration_steps,
        PP=True
    )

    # Initialize Genetic Algorithm population and toolbox
    genal_population, genal_toolbox = GalGA.init_GenAl(population_size=popsize)

    # Check for existing checkpoint
    cp_data = cp_manager.load()
    start_gen = 0
    if cp_data:
        genal_population = cp_data['population']
        GalGA.__dict__.update(cp_data['ga_state'])
        start_gen = cp_data['generation'] + 1

    # Run the GA with checkpointing support
    GalGA.GenAl(
        population_size=popsize,
        num_generations=generations,
        population=genal_population,
        toolbox=genal_toolbox,
        checkpoint_manager=cp_manager,
        start_gen=start_gen,
        output_interval=output_interval,
    )

    # Define column names based on the structure of GalGA.results
    col_names = [
        'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
        'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
        'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
        'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber',
        'cosine', 'log_cosh', 'fitness'
    ]

    # Create DataFrame from results
    results_df = pd.DataFrame(GalGA.results, columns=col_names)

    # Use the chosen loss metric to define a loss column
    results_df['loss'] = results_df[loss_metric]

    # Sort the results DataFrame by loss (lowest first) and reset index
    results_df.sort_values('loss', inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Save the results to a CSV file
    results_file = 'GA/simulation_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    # The best model is now the first row in the sorted DataFrame
    best_model = results_df.iloc[0]
    print("Best model from results dataframe:")
    print(best_model)
    
    return results_file

def load_ga_for_plotting():
    """Load GA object for plotting only"""
    global GalGA
    
    # For plot-only mode, we create a minimal GalGA object that has the properties 
    # needed for plotting, but doesn't run any computations
    
    print(f"Loading existing results from {args.results_file}")
    
    # Make sure results file exists
    import os
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found")
        sys.exit(1)
    
    # Initialize a basic GalGA object
    GalGA = Gal_GA.GalacticEvolutionGA(
        iniab_header=iniab_header,
        sn1a_header=sn1a_header,
        sigma_2_list=sigma_2_list,
        tmax_1_list=tmax_1_list,
        tmax_2_list=tmax_2_list,    
        infall_timescale_1_list=infall_timescale_1_list,
        infall_timescale_2_list=infall_timescale_2_list,
        comp_array=comp_array,
        imf_array=imf_array,
        sfe_array=sfe_array,
        delta_sfe_array=delta_sfe_array,
        imf_upper_limits=imf_upper_limits,
        sn1a_assumptions=sn1a_assumptions,
        stellar_yield_assumptions=stellar_yield_assumptions,
        mgal_values=mgal_values,
        nb_array=nb_array,
        sn1a_rates=sn1a_rates,
        timesteps=timesteps,
        A1=A1,
        A2=A2,
        feh=feh,
        normalized_count=normalized_count,
        obs_age_data=obs_age_data,
        loss_metric=loss_metric,
        obs_age_data_loss_metric = obs_age_data_loss_metric,
        mdf_vs_age_weight = mdf_vs_age_weight,
        fancy_mutation=fancy_mutation,
        shrink_range=shrink_range,
        gaussian_sigma_scale=gaussian_sigma_scale,
        crossover_noise_fraction=crossover_noise_fraction,
        perturbation_strength=perturbation_strength,
        tournament_size=tournament_size,
        threshold=selection_threshold,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        physical_constraints_freq=physical_constraints_freq,
        exploration_steps=exploration_steps,        
        PP=False  # Don't use parallel processing for plot-only
    )
    
    # Load results from CSV
    try:
        df = pd.read_csv(args.results_file)
        
        # Extract results from the dataframe
        GalGA.results = df.values.tolist()
        
        # We need to generate some placeholder data for plotting functions
        # that require mdf_data and labels
        x_vals = np.linspace(-2, 1, 100)
        y_vals = np.zeros_like(x_vals)
        GalGA.mdf_data = [(x_vals, y_vals)]
        GalGA.labels = ["Placeholder"]
        
        # Create an empty walker_history
        GalGA.walker_history = {}
        
        # Check if log files or other data sources might have the actual MDFs
        # and walker history data, but this is beyond the scope of this example
        
        print(f"Loaded {len(df)} model results")
    
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    return args.results_file

if __name__ == "__main__":

    results_file = 'GA/simulation_results.csv'

    make_history = True
    if make_history:
        results_file = checkpoint.run_with_checkpoint(run_ga)
        save_walker_history()
    else:
        load_ga_for_plotting()
        GalGA.walker_history = load_walker_history()
    
    # Generate all plots using the plotting module
    mdf_plotting.generate_all_plots(GalGA, feh, normalized_count, results_file)
