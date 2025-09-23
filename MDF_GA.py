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
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")  # avoid BLAS thread explosion per process
os.environ.setdefault("MPLBACKEND", "Agg")  # non-GUI backend, prevents stray GUI state

import checkpoint  # checkpointing utilities
# Import plotting module
import mdf_plotting
from multiprocessing import cpu_count
import shutil
import os
import numpy as _np, random as _random, os as _os


def load_bensby_data(file_path='data/Bensby_Data.tsv'):
    obs_age_data = pd.read_csv(file_path, sep='\t')
    print(f"Loaded Bensby data with shape: {obs_age_data.shape}")
    print(f"Columns available: {list(obs_age_data.columns)}")
    return obs_age_data

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Adding custom paths
sys.path.append('../')


# Parse parameters from the 'bulge_pcard.txt' file
params = Gal_GA.parse_inlist('bulge_pcard.txt')

# Assign parsed parameters to variables
output_path = params['output_path']
os.makedirs(output_path, exist_ok=True)
shutil.copy('bulge_pcard.txt', os.path.join(output_path, 'bulge_pcard.txt'))

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
if popsize < 0:
    popsize = cpu_count() * (popsize * -1)

generations = params['generations']
crossover_probability = params['crossover_probability']
mutation_probability = params['mutation_probability']
tournament_size = params['tournament_size']
selection_threshold = params['selection_threshold']

obs_age_data_loss_metric = params['obs_age_data_loss_metric']
obs_age_data_target = params['obs_age_data_target']
mdf_vs_age_weight = params['mdf_vs_age_weight']

rand_seed = params['seed']
if rand_seed > 0:
    _random.seed(rand_seed)
    _np.random.seed(rand_seed)
    _os.environ['PYTHONHASHSEED'] = str(rand_seed)


# Create argument parser
parser = argparse.ArgumentParser(description='Run MDF Genetic Algorithm with optional plotting only')
parser.add_argument('--plot-only', action='store_true', help='Skip computation and only generate plots')
parser.add_argument('--results-file', type=str, default=output_path + 'simulation_results.csv', 
                   help='CSV file containing results (for plot-only mode)')
args = parser.parse_args()




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
try:
    obs_age_data = load_bensby_data('data/Bensby_Data.tsv')
except:
    obs_age_data = load_bensby_data('/project/galacticbulge/MDF_GCE_GA/data/Bensby_Data.tsv')

# Global GalGA object to be used for both computation and plotting
GalGA = None

os.makedirs('GA', exist_ok=True)
os.makedirs(output_path + 'loss', exist_ok=True)
os.makedirs(output_path + 'analysis', exist_ok=True)

# Save/load walker history
def save_walker_history():
    if not hasattr(GalGA, 'walker_history'):
        return

    np.savez_compressed(
        output_path + 'walker_history.npz',
        walker_ids=np.array(list(GalGA.walker_history.keys()), dtype=np.int32),
        histories=[np.array(h) for h in GalGA.walker_history.values()],
        mdf_data=np.array(GalGA.mdf_data, dtype=object),      # your [Fe/H] vs count
        alpha_data=np.array(GalGA.alpha_data, dtype=object)   # your α-distributions
    )

    print("Walker history saved")

def load_walker_history():
    if not os.path.exists(output_path + 'walker_history.npz'):
        return {}
        
    data = np.load(output_path + 'walker_history.npz', allow_pickle=True)
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
    import numpy as _np

    # 0) Build GA object
    GalGA = Gal_GA.GalacticEvolutionGA(
        output_path=output_path,
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
        obs_age_data_loss_metric=obs_age_data_loss_metric,
        obs_age_data_target=obs_age_data_target,
        mdf_vs_age_weight=mdf_vs_age_weight,
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

    # 1) Init population & toolbox
    init_population, toolbox = GalGA.init_GenAl(population_size=popsize)

    # helpers
    def _invalidate(ind):
        try:
            if getattr(ind.fitness, "valid", False):
                del ind.fitness.values
        except Exception:
            pass

    def _tiny_jitter(ind, frac=1e-3):
        # jitter continuous genes (indices 5..14)
        for gi in range(5, len(ind)):
            try:
                x = float(ind[gi])
            except Exception:
                continue
            span = max(abs(x), 1.0) * frac
            ind[gi] = x + _np.random.normal(0.0, span)


    cp_data = cp_manager.load()
    start_gen = 0
    population = None
    num_generations = generations  # may bump if checkpoint is beyond target

    if cp_data:
        cp_gen   = int(cp_data.get("generation", -1))
        ga_state = dict(cp_data.get("ga_state", {}))
        full_pop = list(cp_data.get("population", []) or [])

        # Apply GA state EXACTLY as saved (do NOT scrub results/mdf/labels/etc.)
        GalGA.__dict__.update(ga_state)

        # Keep the full checkpoint population available for plotting/analysis
        # (nothing is thrown away)
        GalGA.checkpoint_population = full_pop[:]  # optional, for transparency/tools

        # --- MINIMAL CHANGE: choose ACTIVE walkers (top-by-fitness), else pad ---
        def _fit(ind):
            try:
                if getattr(ind.fitness, "valid", False) and hasattr(ind.fitness, "values"):
                    return float(ind.fitness.values[0])  # lower is better
            except Exception:
                pass
            return float("inf")  # invalid/unknown fitness sorted last

        if len(full_pop) >= popsize:
            ranked = sorted(full_pop, key=_fit)  
            print(f'reducing from {len(full_pop)} to {popsize}')
            population = ranked[:popsize]                # take best popsize
        else:
            ranked = sorted(full_pop, key=_fit)
            population = ranked[:]                       # take what exists (maybe 0..popsize-1)
            if population:
                seed = population[0]                     # best individual
                while len(population) < popsize:
                    clone = toolbox.clone(seed)
                    # tiny jitter on continuous genes so clones aren’t byte-identical
                    for gi in range(5, len(clone)):
                        try:
                            xv = float(clone[gi]); span = max(abs(xv), 1.0) * 1e-4
                            clone[gi] = xv + _np.random.normal(0.0, span)
                        except Exception:
                            pass
                    # mark clone for re-eval so at least something runs
                    try:
                        if getattr(clone.fitness, "valid", False):
                            del clone.fitness.values
                    except Exception:
                        pass
                    population.append(clone)
            else:
                # empty checkpoint population; fall back to initializer
                population = init_population


        # Size walker history to the ACTIVE population only
        GalGA.walker_history = {i: [] for i in range(len(population))}

        # Resume one step beyond the saved generation; ensure at least one gen will run
        start_gen = cp_gen + 1
        if start_gen >= num_generations:
            num_generations = start_gen + 1
            print(f"Extending generations to {num_generations} to ensure ≥1 generation runs after resume.")

    else:
        # Fresh run
        population = init_population
        GalGA.walker_history = {i: [] for i in range(len(population))}
        start_gen = 0


    # 3) RUN the GA (this always executes; no exit paths)
    GalGA.GenAl(
        population_size=popsize,
        num_generations=num_generations,
        population=population,
        toolbox=toolbox,
        checkpoint_manager=cp_manager,
        start_gen=start_gen,
        output_interval=output_interval,
    )

    # 4) Save final results
    col_names = [
        'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
        'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
        'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
        'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber',
        'cosine', 'log_cosh', 'fitness', 'age_meta_fitness', 'physics_penalty'
    ]

    results_df = pd.DataFrame(GalGA.results, columns=col_names) if GalGA.results else pd.DataFrame(columns=col_names)
    if 'loss' not in results_df.columns and not results_df.empty:
        results_df['loss'] = results_df[loss_metric]
        results_df.sort_values('loss', inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    results_file = output_path + 'simulation_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    if not results_df.empty:
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
        output_path=output_path,        
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
        obs_age_data_target = obs_age_data_target,
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

    results_file = output_path + 'simulation_results.csv'

    make_history = True
    if make_history:
        results_file = checkpoint.run_with_checkpoint(run_ga, output_path)
        save_walker_history()
    else:
        load_ga_for_plotting()
        GalGA.walker_history = load_walker_history()
    
    # Generate all plots using the plotting module
    mdf_plotting.generate_all_plots(GalGA, feh, normalized_count, results_file)
