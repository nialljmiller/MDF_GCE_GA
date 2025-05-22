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
from deap import base, creator, tools
import random

# Function to find the index of the nearest value in an array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# Function to compute WRMSE
def compute_wrmse(predicted, observed, sigma):
    return np.sqrt(np.mean(((predicted - observed) / sigma) ** 2))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()


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
                 sn1a_assumptions, stellar_yield_assumptions, mgal_values, nb_array, sn1a_rates, timesteps,A1, A2, feh, normalized_count, loss_metric='huber', fancy_mutation = 'gaussian', shrink_range = False, tournament_size = 3, lambda_diversity = 0.1, threshold = -1, cxpb=0.5, mutpb=0.5,  PP = False):
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
            'wrmse': self.compute_wrmse,
            'mae': self.compute_mae,
            'mape': self.compute_mape,
            'huber': self.compute_huber,
            'cosine_similarity': self.compute_cosine_similarity,
            'ks': self.compute_ks_distance,
            'ensemble': self.compute_ensemble_metric,
            'log_cosh': self.compute_log_cosh
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





    def selDiversityTournament(self, individuals, tournsize, lambda_diversity=0.1):
        """
        Custom tournament selection that promotes diversity.
        
        Args:
            individuals (list): List of individuals (already evaluated).
            tournsize (int): Size of each tournament.
            lambda_diversity (float): Weight for the diversity bonus.
        
        Returns:
            list: The selected individuals.
        """
        selected = []
        # Continue until we've selected as many individuals as in the population.
        while len(selected) < len(individuals):
            # Randomly pick 'tournsize' individuals for the tournament.
            tournament = random.sample(individuals, tournsize)
            best = None
            best_eff_fitness = float('inf')
            for ind in tournament:
                # Compute diversity as the minimum Euclidean distance between this individual and all already selected individuals.
                if selected:
                    distances = [np.linalg.norm(np.array(ind) - np.array(sel)) for sel in selected]
                    diversity = min(distances)
                else:
                    diversity = 0  # No diversity bonus if nothing's been selected yet.
                # Since we're minimizing fitness, lower is better.
                # We subtract (lambda * diversity) so that a larger distance gives a lower effective fitness.
                eff_fitness = ind.fitness.values[0] - lambda_diversity * diversity
                if eff_fitness < best_eff_fitness:
                    best_eff_fitness = eff_fitness
                    best = ind
            selected.append(best)
        return selected



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
        toolbox.register("mate", self.custom_crossover, alpha=0.5)

        # Define different mutation functions based on fancy_mutation parameter
        if self.fancy_mutation.lower() == 'uniform':
            def mutate_with_population(individual):
                return self.uniform_mutate(individual)
            
        elif self.fancy_mutation.lower() == 'gaussian':
            def mutate_with_population(individual):
                return self.gaussian_mutate(individual)
                
        else:  # Default to covariance-aware mutation
            def mutate_with_population(individual):
                return self.covariance_aware_mutate(individual, population)
        
        toolbox.register("mutate", mutate_with_population)
        
        toolbox.register("mutate", mutate_with_population)

        toolbox.register("select", self.selDiversityTournament, tournsize=self.tournament_size, lambda_diversity=self.lambda_diversity)

        # Create the initial population
        population = toolbox.population(n=population_size)
        return population, toolbox

    def custom_crossover(self, ind1, ind2, alpha=0.5):
        """
        Custom crossover function that correctly handles categorical and continuous parameters.
        """
        # Categorical parameters are at indices 0-4
        categorical_indices = [0, 1, 2, 3, 4]  # comp, imf, sn1a, sy, sn1a_rate
        
        # Continuous parameters are at indices 5-13
        continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13]  # sigma_2, t_1, t_2, etc.
        
        # Create copies of the parents
        ind1_copy = ind1[:]
        ind2_copy = ind2[:]
        
        # Handle categorical parameters with random selection
        for i in categorical_indices:
            if random.random() < 0.5:
                # Swap values
                ind1_copy[i], ind2_copy[i] = ind2_copy[i], ind1_copy[i]
        
        # Handle continuous parameters with blending (BLX-alpha)
        for i in continuous_indices:
            # Blend the continuous values
            gamma = (1. + 2. * alpha) * random.random() - alpha
            ind1_copy[i] = (1. - gamma) * ind1[i] + gamma * ind2[i]
            ind2_copy[i] = gamma * ind1[i] + (1. - gamma) * ind2[i]
        
        # Return the new individuals
        return ind1_copy, ind2_copy


    def compute_ks_distance(self, theory_count_array):
        """
        1D Kolmogorov–Smirnov distance between the model distribution
        and the observed distribution (self.normalized_count).
        Lower is better.
        """
        model_cdf = np.cumsum(theory_count_array)
        model_cdf /= model_cdf[-1]  # normalize

        data_cdf = np.cumsum(self.normalized_count)
        data_cdf /= data_cdf[-1]

        return np.max(np.abs(model_cdf - data_cdf))



    def compute_ensemble_metric(self, theory_count_array):
        """
        Weighted combination of Huber loss and (1 - cosine_similarity).
        """
        alpha = 0.7  # Adjust weighting as you like
        beta = 0.3
        
        # Evaluate the existing metrics
        huber_val = self.compute_huber(theory_count_array)            # lower = better
        cos_val   = self.compute_cosine_similarity(theory_count_array) # higher = better
        
        # Combine them so lower is better overall
        return alpha * huber_val + beta * (1.0 - cos_val)

    def compute_wrmse(self, theory_count_array):
        return compute_wrmse(theory_count_array, self.normalized_count, self.placeholder_sigma_array)

    def compute_mae(self, theory_count_array):
        return np.mean(np.abs(np.array(theory_count_array) - np.array(self.normalized_count)))

    def compute_mape(self, theory_count_array):
        return np.mean(np.abs((np.array(theory_count_array) - np.array(self.normalized_count)) / np.array(self.normalized_count))) * 100

    def compute_huber(self, theory_count_array):
        return np.mean(huber_loss(self.normalized_count, theory_count_array))

    def compute_cosine_similarity(self, theory_count_array):
        return np.dot(self.normalized_count, theory_count_array) / (np.linalg.norm(self.normalized_count) * np.linalg.norm(theory_count_array))

    def compute_log_cosh(self, theory_count_array):
        return np.mean(np.log(np.cosh(theory_count_array - self.normalized_count)))

    def calculate_all_metrics(self, theory_count_array):
        # Calculate all metrics
        wrmse = self.compute_wrmse(theory_count_array)
        mae = self.compute_mae(theory_count_array)
        mape = self.compute_mape(theory_count_array)
        huber = self.compute_huber(theory_count_array)
        cos_similarity = self.compute_cosine_similarity(theory_count_array)
        log_cosh = self.compute_log_cosh(theory_count_array)
        ensemble = self.compute_ensemble_metric(theory_count_array)
        ks = self.compute_ks_distance(theory_count_array)
        return ks, ensemble, wrmse, mae, mape, huber, cos_similarity, log_cosh


    def diversity_tournament_selection(self, individuals, k, tournsize=3):
        """Tournament selection that also rewards diversity with increasing tournament size"""
        # Calculate current tournament size based on generation progress
        if hasattr(self, 'gen') and hasattr(self, 'num_generations'):
            progress = min(1.0, self.gen / self.num_generations)
            max_tournsize = min(len(individuals) // 2, tournsize * 3)  # Cap at half population
            current_tournsize = max(tournsize, int(tournsize + progress * (max_tournsize - tournsize)))
        else:
            current_tournsize = tournsize
        
        selected = []
        for i in range(k):
            # Regular tournament selection with current size
            aspirants = random.sample(individuals, min(current_tournsize, len(individuals)))
            aspirants.sort(key=lambda ind: ind.fitness.values[0])
            winner = aspirants[0]  # Best fitness
            
            # If we've already selected some individuals, consider diversity
            if selected and random.random() < 0.3:  # 30% chance to prioritize diversity
                # Calculate distance to already selected individuals
                distances = []
                for ind in aspirants:
                    min_dist = min(np.linalg.norm(np.array(ind) - np.array(sel)) 
                                  for sel in selected)
                    distances.append(min_dist)
                
                # Find most diverse individual among the tournament participants
                diverse_idx = distances.index(max(distances))
                winner = aspirants[diverse_idx]
            
            selected.append(winner)
        
        if hasattr(self, 'gen'):
            print(f"Generation {self.gen}: Using tournament size {current_tournsize}")
            
        return selected


    def prevent_duplicates(self, offspring, toolbox):
        """Replace duplicate individuals with new mutations"""
        unique_genomes = set()
        distinct_offspring = []
        
        for ind in offspring:
            # Create a hashable representation of key parameters
            genome_key = tuple([round(val, 6) if isinstance(val, float) else val for val in ind])
            
            if genome_key in unique_genomes:
                # Replace duplicate with a completely new individual
                new_ind = toolbox.individual()
                distinct_offspring.append(new_ind)
            else:
                unique_genomes.add(genome_key)
                distinct_offspring.append(ind)
    
        return distinct_offspring





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
            with ThreadPool(processes=16) as pool:
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
                    self.results.append(result['metrics'])
                    self.MDFs.append(result['cs_MDF'])
                    self.model_numbers.append(result['model_number'])
                    self.model_count += 1

            # *** Here’s where we update the operator rates dynamically ***
            self.update_operator_rates(population, gen, num_generations)
            offspring = prevent_duplicates(offspring, toolbox)

            # After evaluations, update population and move on to next generation
            for idx, ind in enumerate(population):
                self.walker_history[idx].append(list(ind))
            population[:] = offspring

            gc.collect()  # clean up

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








    def get_param_bounds(self, param_index):
        """Get min and max bounds for a parameter by its index"""
        if param_index == 5:  # sigma_2
            return self.sigma_2_min, self.sigma_2_max
        elif param_index == 6:  # tmax_1
            return min(self.tmax_1_list), max(self.tmax_1_list)
        elif param_index == 7:  # tmax_2
            return min(self.tmax_2_list), max(self.tmax_2_list)
        elif param_index == 8:  # infall_timescale_1
            return min(self.infall_timescale_1_list), max(self.infall_timescale_1_list)
        elif param_index == 9:  # infall_timescale_2
            return min(self.infall_timescale_2_list), max(self.infall_timescale_2_list)
        elif param_index == 10:  # sfe
            return min(self.sfe_array), max(self.sfe_array)
        elif param_index == 11:  # imf_upper_limits
            return min(self.imf_upper_limits), max(self.imf_upper_limits)
        elif param_index == 12:  # mgal_values
            return min(self.mgal_values), max(self.mgal_values)
        elif param_index == 13:  # nb_array
            return min(self.nb_array), max(self.nb_array)
        else:
            # Default for categorical parameters
            return 0, 10  # Arbitrary range for categorical indices

    def covariance_aware_mutate(self, individual, population, top_fraction=0.3, base_scale=1.0, regularization=1e-6):
            """
            An expanded version of covariance_aware_mutate that handles all parameters.
            For categorical parameters, we use a simpler approach.
            """
            # Sort the population by fitness
            pop_sorted = sorted(population, key=lambda ind: ind.fitness.values[0])
            n_top = max(1, int(top_fraction * len(population)))
            top_inds = pop_sorted[:n_top]
            
            # Get parameter type information
            categorical_indices = self.categorical_indices
            continuous_indices = self.continuous_indices
            
            # Handle continuous parameters with covariance-based mutation
            if len(continuous_indices) > 1:  # Need at least 2 dimensions for covariance
                # Extract continuous parameters from top individuals
                continuous_data = []
                for ind in top_inds:
                    continuous_data.append([ind[i] for i in continuous_indices])
                continuous_array = np.array(continuous_data, dtype=float)
                
                # Compute covariance matrix
                cov_matrix = np.cov(continuous_array, rowvar=False)
                cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
                scaled_cov = base_scale * cov_matrix
                
                # Sample mutation vector
                mutation_vector = np.random.multivariate_normal(np.zeros(len(continuous_indices)), scaled_cov)
                
                # Apply mutation to continuous parameters
                for idx, i in enumerate(continuous_indices):
                    individual[i] += mutation_vector[idx]
                    # Clamp within bounds
                    min_bound, max_bound = self.get_param_bounds(i)
                    individual[i] = min(max(individual[i], min_bound), max_bound)
            else:
                # Fallback for the case with only one continuous parameter
                for i in continuous_indices:
                    min_bound, max_bound = self.get_param_bounds(i)
                    scale = (max_bound - min_bound) * 0.1
                    individual[i] += random.gauss(0, scale)
                    individual[i] = min(max(individual[i], min_bound), max_bound)
            
            # Handle categorical parameters with simple mutation
            for i in categorical_indices:
                if random.random() < 0.2:  # 20% chance to mutate each categorical parameter
                    param_name = self.index_to_param_map[i]
                    num_categories = len(getattr(self, param_name))# + '_list'))
                    
                    # Select a new random category (different from current)
                    current_value = individual[i]
                    new_value = current_value
                    while new_value == current_value and num_categories > 1:
                        new_value = random.randint(0, num_categories - 1)
                    individual[i] = new_value
            
            return individual,

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

    def gaussian_mutate(self, individual, indpb=0.2, sigma_scale=0.1):
        """
        Gaussian mutation that perturbs values using a normal distribution.
        Sigma is scaled relative to the parameter range.
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
                    range_size = max_bound - min_bound
                    sigma = range_size * sigma_scale
                    individual[i] += random.gauss(0, sigma)
                    # Clamp to bounds
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

        # Evaluate the spline at the same [Fe/H] grid as your data
        cs_MDF = CubicSpline(x_data, y_data)
        fmin, fmax = x_data.min(), x_data.max()
        feh_clamped = np.clip(self.feh, fmin, fmax)
        theory_count_array = cs_MDF(feh_clamped)

        # Compare with the observed distribution
        ks, ensemble, wrmse, mae, mape, huber, cos_similarity, log_cosh = self.calculate_all_metrics(theory_count_array)

        # Use selected loss
        primary_loss_value = self.selected_loss_function(theory_count_array)

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
            'metrics': metrics,
            'cs_MDF': cs_MDF,
            'model_number': self.model_count
        }

        return (primary_loss_value,), result

