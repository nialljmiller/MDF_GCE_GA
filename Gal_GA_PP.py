#!/usr/bin/env python3.8
################################
# Author: N Miller, M Joyce, (ChatGPT 4o for delint things)
################################
#import imp
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
#testing jesus
#from sklearn import preprocessing
sys.path.append('../')

import gc
from scipy.interpolate import CubicSpline
from matplotlib import cm
from matplotlib.lines import *
from matplotlib.patches import *
from JINAPyCEE import omega_plus
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

from deap import base, creator, tools
import random

from loss import *

# Function to find the index of the nearest value in an array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# Function to model inflow rates
def two_inflow_fn(t, exp_inflow):
    if t < exp_inflow[1][1]:
        return exp_inflow[0][0] * np.exp(-t / exp_inflow[0][2])
    else:
        return (exp_inflow[0][0] * np.exp(-t / exp_inflow[0][2]) +
                exp_inflow[1][0] * np.exp(-(t - exp_inflow[1][1]) / exp_inflow[1][2]))


# Function to parse the inlist file into a dictionary
def parse_inlist(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed_value = eval(value)  # Try to evaluate if it's a list or expression
            except:
                parsed_value = value  # Keep original string if eval fails
            params[key] = parsed_value
    return params



def print_population(GA, population, generation):
    """Helper function to print population details."""
    print(f"\nGeneration {generation+1}:")
    for i, individual in enumerate(population):
        print(f"Individual {i}: {individual}, Fitness: {individual.fitness.values if individual.fitness.valid else 'Not evaluated'}")






class GalacticEvolutionGA:

    def __init__(self, sn1a_header, iniab_header, sigma_2_list, tmax_1_list, tmax_2_list, infall_timescale_1_list, infall_timescale_2_list, comp_array, imf_array, sfe_array, imf_upper_limits, 
                 sn1a_assumptions, stellar_yield_assumptions, mgal_values, nb_array, sn1a_rates, timesteps,A1, A2, feh, normalized_count, loss_metric='huber', fancy_mutation = 'gaussian', shrink_range = False, tournament_size = 3, lambda_diversity = 0.01, threshold = -1, cxpb=0.5, mutpb=0.5,  PP = False):
        # Initialize parameters as instance variables
        self.sn1a_header = sn1a_header
        self.iniab_header = iniab_header
        self.sigma_2_list = sigma_2_list
        self.tmax_1_list = tmax_1_list
        self.tmax_2_list = tmax_2_list
        self.infall_timescale_1_list = infall_timescale_1_list
        self.infall_timescale_2_list = infall_timescale_2_list        
        self.comp_array = comp_array
        self.imf_array = imf_array
        self.sfe_array = sfe_array
        self.imf_upper_limits = imf_upper_limits
        self.sn1a_assumptions = sn1a_assumptions
        self.stellar_yield_assumptions = stellar_yield_assumptions
        self.mgal_values = mgal_values
        self.nb_array = nb_array
        self.sn1a_rates = sn1a_rates
        self.timesteps = timesteps
        self.A1 = A1
        self.A2 = A2        
        self.feh = feh
        self.normalized_count = normalized_count
        self.placeholder_sigma_array = np.zeros(len(normalized_count)) + 1  # Assume all sigmas are 1
        self.fancy_mutation = fancy_mutation
        self.PP = PP
        self.quant_individuals = False
        self.model_count = 0
        self.mdf_data = []
        self.results = []
        self.labels = []
        self.MDFs = []
        self.alpha_data = []
        self.model_numbers = []
        self.shrink_range = shrink_range
        # Min and max values for sigma_2, t_2, and infall_2
        self.sigma_2_min, self.sigma_2_max = min(sigma_2_list), max(sigma_2_list)
        self.t_2_min, self.t_2_max = min(tmax_2_list), max(tmax_2_list)
        self.infall_2_min, self.infall_2_max = min(infall_timescale_2_list), max(infall_timescale_2_list)

        self.cxpb=cxpb
        self.mutpb=mutpb
        
        print('############################')
        print(f'Doing {self.fancy_mutation} mutations with {loss_metric} loss and parallel processing is {self.PP}')
        print('############################')
        
        # Define available loss metrics
        self.loss_functions = {
            'wrmse': compute_wrmse,
            'mae': compute_mae,
            'mape': compute_mape,
            'huber': compute_huber,
            'cosine_similarity': compute_cosine_similarity,
            'ks': compute_ks_distance,
            'ensemble': compute_ensemble_metric,
            'log_cosh': compute_log_cosh
        }

        # Select the loss function based on user input
        if loss_metric not in self.loss_functions:
            raise ValueError(f"Invalid loss metric. Available options are: {list(self.loss_functions.keys())}")
        
        self.selected_loss_function = self.loss_functions[loss_metric]

        self.all_gene_values_successful = []
        self.all_gene_values_unsuccessful = []
        self.all_losses_successful = []
        self.all_losses_unsuccessful = []
        self.gene_bounds = []
        self.global_min_loss = None
        self.global_max_loss = None
        
        self.threshold = threshold
        self.tournament_size = tournament_size
        self.lambda_diversity = lambda_diversity #A higher value places more emphasis on diversity.


        # Define which indices are categorical vs continuous
        self.categorical_indices = [0, 1, 2, 3, 4]  # comp, imf, sn1a, stellar_yield, sn1a_rate
        self.continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13]  # sigma_2, t_1, t_2, etc.
        
        # Map from index to parameter name (for getting bounds dynamically)
        self.index_to_param_map = {
            0: 'comp_array',
            1: 'imf_array',
            2: 'sn1a_assumptions',
            3: 'stellar_yield_assumptions',
            4: 'sn1a_rates',
            5: 'sigma_2',
            6: 'tmax_1',
            7: 'tmax_2',
            8: 'infall_timescale_1',
            9: 'infall_timescale_2',
            10: 'sfe',
            11: 'imf_upper_limits',
            12: 'mgal_values',
            13: 'nb_array'
        }




    def init_GenAl(self, population_size):
        # DEAP framework setup for Genetic Algorithm (GA)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Toolbox to define how individuals (solutions) are created and evolve
        toolbox = base.Toolbox()

        # Register attribute generators for all parameters
        # Truly discrete choices (categorical parameters)
        toolbox.register("comp_attr", lambda: random.randint(0, len(self.comp_array) - 1))
        toolbox.register("imf_attr", lambda: random.randint(0, len(self.imf_array) - 1))
        toolbox.register("sn1a_attr", lambda: random.randint(0, len(self.sn1a_assumptions) - 1))
        toolbox.register("sy_attr", lambda: random.randint(0, len(self.stellar_yield_assumptions) - 1))
        toolbox.register("sn1a_rate_attr", lambda: random.randint(0, len(self.sn1a_rates) - 1))
        
        # Continuous parameters
        toolbox.register("sigma_2_attr", random.uniform, min(self.sigma_2_list), max(self.sigma_2_list))
        toolbox.register("t_1_attr", random.uniform, min(self.tmax_1_list), max(self.tmax_1_list))
        toolbox.register("t_2_attr", random.uniform, min(self.tmax_2_list), max(self.tmax_2_list))
        toolbox.register("infall_1_attr", random.uniform, min(self.infall_timescale_1_list), max(self.infall_timescale_1_list))
        toolbox.register("infall_2_attr", random.uniform, min(self.infall_timescale_2_list), max(self.infall_timescale_2_list))
        toolbox.register("sfe_attr", random.uniform, min(self.sfe_array), max(self.sfe_array))
        toolbox.register("imf_upper_attr", random.uniform, min(self.imf_upper_limits), max(self.imf_upper_limits))
        toolbox.register("mgal_attr", random.uniform, min(self.mgal_values), max(self.mgal_values))
        toolbox.register("nb_attr", random.uniform, min(self.nb_array), max(self.nb_array))

        # Create an individual by combining all attributes
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.comp_attr, toolbox.imf_attr, toolbox.sn1a_attr, 
                          toolbox.sy_attr, toolbox.sn1a_rate_attr,
                          toolbox.sigma_2_attr, toolbox.t_1_attr, toolbox.t_2_attr, 
                          toolbox.infall_1_attr, toolbox.infall_2_attr,
                          toolbox.sfe_attr, toolbox.imf_upper_attr, 
                          toolbox.mgal_attr, toolbox.nb_attr), n=1)

        # Create a population by repeating individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", self.evaluate)

        # Register genetic operations
        toolbox.register("mate", self.crossover, max_bias=0.75)

        # Define different mutation functions based on fancy_mutation parameter
        if self.fancy_mutation.lower() == 'uniform':
            def mutate_with_population(individual):
                return self.uniform_mutate(individual)
            
        elif self.fancy_mutation.lower() == 'gaussian':
            def mutate_with_population(individual):
                return self.gaussian_mutate(individual)
                

        toolbox.register("mutate", mutate_with_population)
        
        toolbox.register("select", self.selTournament, tournsize=self.tournament_size)#, lambda_diversity=self.lambda_diversity)

        # Create the initial population
        population = toolbox.population(n=population_size)
        return population, toolbox


    def crossover(self, ind1, ind2, max_bias=0.75):
        """Crossover that favors the fitter parent up to max_bias%"""
        
        # Determine which parent is fitter (lower fitness = better since we minimize)
        if ind1.fitness.valid and ind2.fitness.valid:
            fit1 = ind1.fitness.values[0]
            fit2 = ind2.fitness.values[0]
            
            # Calculate fitness difference and weight toward better parent
            total_fitness = fit1 + fit2
            if total_fitness > 0:
                # Weight inversely proportional to fitness (lower fit = higher weight)
                weight1 = fit2 / total_fitness  # Better parent gets higher weight
                weight2 = fit1 / total_fitness
                
                # Cap the bias at max_bias
                if weight1 > max_bias:
                    weight1 = max_bias
                    weight2 = 1 - max_bias
                elif weight2 > max_bias:
                    weight2 = max_bias
                    weight1 = 1 - max_bias
            else:
                # Fallback if fitness calculation fails
                weight1 = weight2 = 0.5
        else:
            # If fitness not available, use equal weighting
            weight1 = weight2 = 0.5
        
        # Create copies of parents
        ind1_copy = ind1[:]
        ind2_copy = ind2[:]
        
        # Handle categorical parameters with fitness-weighted selection
        categorical_indices = [0, 1, 2, 3, 4]
        for i in categorical_indices:
            if random.random() < weight1:
                # Child 1 inherits from parent 1, child 2 from parent 1
                ind2_copy[i] = ind1[i]
            else:
                # Child 1 inherits from parent 2, child 2 from parent 2  
                ind1_copy[i] = ind2[i]
        
        # Handle continuous parameters with fitness-weighted blending
        continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13]
        for i in continuous_indices:
            # Fitness-weighted average with small noise
            avg_val = weight1 * ind1[i] + weight2 * ind2[i]
            noise_scale = abs(ind1[i] - ind2[i]) * 0.05  # 5% of parent difference
            
            ind1_copy[i] = avg_val + random.gauss(0, noise_scale)
            ind2_copy[i] = avg_val + random.gauss(0, noise_scale)
            
            # Ensure bounds
            min_bound, max_bound = self.get_param_bounds(i)
            ind1_copy[i] = min(max(ind1_copy[i], min_bound), max_bound)
            ind2_copy[i] = min(max(ind2_copy[i], min_bound), max_bound)
        
        return ind1_copy, ind2_copy



    def selTournament(self, individuals, tournsize=3):
        """
        Tournament selection that heavily prioritizes fitness, with occasional diversity.
        """
        selected = []
        
        while len(selected) < len(individuals):
            tournament = random.sample(individuals, tournsize)
            winner = min(tournament, key=lambda ind: ind.fitness.values[0])
            selected.append(winner)
        
        return selected



    def prevent_duplicates(self, offspring, toolbox, max_attempts=5):
        """Replace duplicate individuals by jittering them, not re-drawing from scratch."""
        unique_keys = set()
        distinct_offspring = []

        for ind in offspring:
            # build a hashable key from the genome
            key = tuple(round(x, 6) if isinstance(x, float) else x for x in ind)

            if key in unique_keys:
                # duplicate: clone the Individual (so we keep .fitness, etc.)
                new_ind = toolbox.clone(ind)
                attempt = 0

                while attempt < max_attempts:
                    toolbox.mutate(new_ind)              # use your GA mutation
                    del new_ind.fitness.values           # force re-evaluation

                    new_key = tuple(round(x, 6) if isinstance(x, float) else x
                                    for x in new_ind)
                    if new_key not in unique_keys:
                        key = new_key
                        break

                    attempt += 1

                distinct_offspring.append(new_ind)
                unique_keys.add(key)

            else:
                # first time: just keep the original individual
                unique_keys.add(key)
                distinct_offspring.append(ind)

        return distinct_offspring






    def update_operator_rates(self, population, generation, num_generations):
        """Dynamically adjust operator rates based on progress and diversity"""
        # Calculate population diversity
        gene_array = np.array([ind for ind in population])
        if len(gene_array) > 1:
            # Calculate average pairwise distance
            distances = []
            for i in range(len(gene_array)):
                for j in range(i+1, len(gene_array)):
                    distances.append(np.linalg.norm(gene_array[i] - gene_array[j]))
            diversity = np.mean(distances) if distances else 0
        else:
            diversity = 0
        
        # Get current progress through generations
        progress = generation / num_generations
        
        # If diversity is low, increase mutation rate to explore more
        if diversity < 0.1 * (self.sigma_2_max - self.sigma_2_min):
            self.mutpb = min(self.mutpb * 1.1, 0.7)  # Increase mutation rate
            self.cxpb = max(self.cxpb * 0.9, 0.3)    # Decrease crossover rate
        
        # If we're in later generations and diversity is still high, favor exploitation
        elif progress > 0.6 and diversity > 0.3 * (self.sigma_2_max - self.sigma_2_min):
            self.mutpb = max(self.mutpb * 0.9, 0.1)  # Decrease mutation rate
            self.cxpb = min(self.cxpb * 1.1, 0.9)    # Increase crossover rate
            
        print(f"Generation {generation}: diversity = {diversity:.4f}, " 
              f"mutpb = {self.mutpb:.2f}, cxpb = {self.cxpb:.2f}")







    def get_param_bounds(self, index):
        """
        Returns the min and max bounds for the given continuous parameter index.
        """
        if index == 5:
            return self.sigma_2_min, self.sigma_2_max
        elif index == 6:
            return min(self.tmax_1_list), max(self.tmax_1_list)
        elif index == 7:
            return self.t_2_min, self.t_2_max
        elif index == 8:
            return min(self.infall_timescale_1_list), max(self.infall_timescale_1_list)
        elif index == 9:
            return self.infall_2_min, self.infall_2_max
        elif index == 10:
            return min(self.sfe_array), max(self.sfe_array)
        elif index == 11:
            return min(self.imf_upper_limits), max(self.imf_upper_limits)
        elif index == 12:
            return min(self.mgal_values), max(self.mgal_values)
        elif index == 13:
            return min(self.nb_array), max(self.nb_array)
        else:
            raise IndexError(f"No bounds defined for parameter index {index}")



    

    def uniform_mutate(self, individual, indpb=0.2):
        """
        Uniform mutation that replaces values with uniform random values 
        within parameter bounds.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                # Handle categorical parameters
                if i in self.categorical_indices:
                    param_name = self.index_to_param_map[i]
                    num_categories = len(getattr(self, param_name))
                    individual[i] = random.randint(0, num_categories - 1)
                # Handle continuous parameters
                else:
                    min_bound, max_bound = self.get_param_bounds(i)
                    individual[i] = random.uniform(min_bound, max_bound)
        
        return individual,




    def gaussian_mutate(self, individual, indpb=0.2, base_sigma_scale=0.2):
            
        """Mutation that creates small, connected steps"""
        for i in range(len(individual)):
            if random.random() < indpb:
                if i in self.categorical_indices:
                    # Only change categorical with low probability
                    if random.random() < 0.1:  # 10% chance
                        param_name = self.index_to_param_map[i]
                        num_categories = len(getattr(self, param_name))
                        individual[i] = random.randint(0, num_categories - 1)
                else:
                    # Very small mutations for continuous parameters
                    min_bound, max_bound = self.get_param_bounds(i)
                    range_size = max_bound - min_bound
                    
                    # Start with 2% of range, decay to 0.5%
                    if hasattr(self, 'gen') and hasattr(self, 'num_generations'):
                        progress = self.gen / self.num_generations
                        sigma_scale = 0.02 * (1 - 0.75 * progress)  # 2% -> 0.5%
                    else:
                        sigma_scale = 0.02
                    
                    sigma = range_size * sigma_scale
                    individual[i] += random.gauss(0, sigma)
                    
                    # Reflect at boundaries instead of clamping
                    if individual[i] < min_bound:
                        individual[i] = min_bound + (min_bound - individual[i])
                    elif individual[i] > max_bound:
                        individual[i] = max_bound - (individual[i] - max_bound)
                    
                    # Final clamp if reflection goes out of bounds
                    individual[i] = min(max(individual[i], min_bound), max_bound)
        
        return individual,




    def evaluate(self, individual):
        # Extract parameters from the individual
        # Categorical parameters (indices)
        comp_idx = int(individual[0])
        imf_idx = int(individual[1])
        sn1a_idx = int(individual[2])
        sy_idx = int(individual[3])
        sn1ar_idx = int(individual[4])
        
        # Continuous parameters
        sigma_2 = individual[5]
        t_1 = individual[6]
        t_2 = individual[7]
        infall_1 = individual[8]
        infall_2 = individual[9]
        sfe_val = individual[10]
        imf_upper = individual[11]
        mgal = individual[12]
        nb = individual[13]
        
        # Look up the actual values for categorical parameters
        comp = self.comp_array[comp_idx]
        imf_val = self.imf_array[imf_idx]
        sn1a = self.sn1a_assumptions[sn1a_idx]
        sy = self.stellar_yield_assumptions[sy_idx]
        sn1ar = self.sn1a_rates[sn1ar_idx]
        
        A1 = self.A1
        A2 = self.A2
        sn1a_header = self.sn1a_header
        iniab_header = self.iniab_header

        # GCE Model kwargs
        kwargs = {
            'special_timesteps': self.timesteps,
            'twoinfall_sigmas': [1300, sigma_2],
            'galradius': 1800,
            'exp_infall':[[A1, t_1*1e9, infall_1*1e9], [A2, t_2*1e9, infall_2*1e9]],            
            'tauup': [0.02e9, 0.02e9],
            'mgal': mgal,
            'iniZ': 0.0,
            'mass_loading': 0.0,
            'table': sn1a_header + sy,
            'sfe': sfe_val,
            'imf_type': imf_val,
            'sn1a_table': sn1a_header + sn1a,
            'imf_yields_range': [1, imf_upper],
            'iniabu_table': iniab_header + comp,
            'nb_1a_per_m': nb,
            'sn1a_rate': sn1ar
        }

        # Run GCE model and compute MDF
        GCE_model = omega_plus.omega_plus(**kwargs)
        x_data, y_data = GCE_model.inner.plot_mdf(axis_mdf='[Fe/H]', sigma_gauss=0.1, norm=True, return_x_y=True)
        x_data = np.array(x_data)
        y_data = np.array(y_data)


        # — now compute the α-distribution —
        elements = ['[Mg/Fe]','[Si/Fe]','[Ca/Fe]','[Ti/Fe]']
        alpha_arrs = []
        for el in elements:
            _, y_el = GCE_model.inner.plot_spectro(xaxis='[Fe/H]', yaxis=el, return_x_y=True)
            alpha_arrs.append(y_el)
        # α = mean over the four element tracks
        alpha_y = np.nanmean(np.vstack(alpha_arrs), axis=0)


        # Evaluate the spline at the same [Fe/H] grid as your data
        cs_MDF = CubicSpline(x_data, y_data)
        fmin, fmax = x_data.min(), x_data.max()
        feh_clamped = np.clip(self.feh, fmin, fmax)
        theory_count_array = cs_MDF(feh_clamped)

        # Compare with the observed distribution
        ks, ensemble, wrmse, mae, mape, huber, cos_similarity, log_cosh = calculate_all_metrics(self, theory_count_array)

        # Use selected loss
        primary_loss_value = self.selected_loss_function(self,theory_count_array)

        # Return the result with a detailed label
        label = (f'comp: {comp}, imf: {imf_val}, sn1a: {sn1a}, sy: {sy}, sn1ar: {sn1ar}, '
                 f'sigma2: {sigma_2:.3f}, t1: {t_1:.3f}, t2: {t_2:.3f}, '
                 f'infall1: {infall_1:.3f}, infall2: {infall_2:.3f}, '
                 f'sfe: {sfe_val:.5f}, imf_upper: {imf_upper:.1f}, '
                 f'mgal: {mgal:.2e}, nb: {nb:.2e}')
                 
        # Create metrics list for results storage
        metrics = [comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx,
                   sigma_2, t_1, t_2, infall_1, infall_2, 
                   sfe_val, imf_upper, mgal, nb,
                   ks, ensemble, wrmse, mae, mape, huber, cos_similarity, log_cosh]
        
        result = {
            'label': label,
            'x_data': x_data,
            'y_data': y_data,
            'alpha_y': alpha_y, 
            'metrics': metrics,
            'cs_MDF': cs_MDF,
            'model_number': self.model_count
        }

        return (primary_loss_value,), result







    def GenAl(self, population_size, num_generations, population, toolbox):
        total_eval_time = 0
        total_eval_steps = 0
        total_start_time = time.time()

        # Define helper function for re-quantization
        def requantize(ind):
            ind[0] = min(self.sigma_2_list, key=lambda x: abs(x - ind[0]))  # Snap sigma_2 to nearest
            ind[1] = min(self.tmax_2_list, key=lambda x: abs(x - ind[1]))   # Snap tmax_2 to nearest
            ind[2] = min(self.infall_timescale_2_list, key=lambda x: abs(x - ind[2]))  # Snap infall_2 to nearest
            return ind

        # Use a context manager for the multiprocessing pool
        if self.PP:        

            with Pool(processes=16) as pool:
            #with ThreadPool(processes=16) as pool:
            #with multiprocessing.Pool() as pool:
                toolbox.register("map", pool.map)
                self._run_genetic_algorithm(population, toolbox, num_generations, requantize)
        else:
            self._run_genetic_algorithm(population, toolbox, num_generations, requantize)

        total_time = time.time() - total_start_time

        # Calculate and print the average evaluation time per individual
        if total_eval_steps > 0:
            eff_avg_eval_time = total_time / total_eval_steps
            overall_avg_eval_time = total_eval_time / total_eval_steps
            print(f"Overall average evaluation time per individual: {overall_avg_eval_time:.4f} seconds.")
            print(f"Effective overall average evaluation time per individual: {eff_avg_eval_time:.4f} seconds.")
        else:
            print("No evaluations were performed.")
        
        gc.collect()  # Final garbage collection


    def _run_genetic_algorithm(self, population, toolbox, num_generations, requantize):
        self.walker_history = {i: [] for i in range(len(population))}  # Track each walker's history
        for gen in range(num_generations):
            print(f"-- Generation {gen + 1}/{num_generations} --")
            self.gen = gen
            # Step 1: Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    fitnesses_and_results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    fitnesses_and_results = [toolbox.evaluate(ind) for ind in invalid_ind]

                for (ind, (fit, result)) in zip(invalid_ind, fitnesses_and_results):
                    ind.fitness.values = fit
                    self.labels.append(result['label'])
                    self.mdf_data.append([result['x_data'], result['y_data']])
                    self.results.append(result['metrics'])
                    self.MDFs.append(result['cs_MDF'])
                    self.model_numbers.append(result['model_number'])
                    self.model_count += 1


            gc.collect()

            # Step 2: Select the next generation
            offspring = toolbox.select(population)#, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Step 3: Apply mutation and crossover
            for mutant in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            if self.quant_individuals:
                offspring = [requantize(ind) for ind in offspring]

            if round(gen % (num_generations / 4)) == 0:
                print_population(self, population, generation=gen)


            # Step 4: Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    fitnesses_and_results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    fitnesses_and_results = [toolbox.evaluate(ind) for ind in invalid_ind]

                for (ind, (fit, result)) in zip(invalid_ind, fitnesses_and_results):
                    ind.fitness.values = fit
                    self.labels.append(result['label'])
                    self.mdf_data.append([result['x_data'], result['y_data']])
                    self.alpha_data.append(result['alpha_y'])
                    self.results.append(result['metrics'])
                    self.MDFs.append(result['cs_MDF'])
                    self.model_numbers.append(result['model_number'])
                    self.model_count += 1

            # *** Here’s where we update the operator rates dynamically ***
            self.update_operator_rates(population, gen, num_generations)
            offspring = self.prevent_duplicates(offspring, toolbox)

            # After evaluations, update population and move on to next generation
            for idx, ind in enumerate(population):
                self.walker_history[idx].append(list(ind))
            population[:] = offspring

            gc.collect()  # clean up
