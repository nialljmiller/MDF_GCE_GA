#!/usr/bin/env python3.8
################################
#
# Author: M Joyce, N Miller
#
################################
import matplotlib
import imp
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import itertools
import sys
#from sklearn import preprocessing
sys.path.append('../')
from NuPyCEE import omega
import ast

from string import printable
from scipy.interpolate import CubicSpline

from matplotlib.lines import *
from matplotlib.patches import *
from NuPyCEE import read_yields
from JINAPyCEE import omega_plus
from NuPyCEE import stellab
from NuPyCEE import sygma
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from deap import base, creator, tools, algorithms
import random
import matplotlib as mpl  # Importing the matplotlib module
import csv

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
    """Parse a parameter file into a dictionary with proper Python types."""
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            lowered = value.lower().strip("'\"")
            if lowered in {'true', 'false'}:
                parsed_value = lowered == 'true'
            else:
                try:
                    parsed_value = ast.literal_eval(value)
                except Exception:
                    parsed_value = value

            params[key] = parsed_value

    return params


class GalacticEvolutionGA:
    def __init__(self, sn1a_header, iniab_header, sigma_2_list, tmax_2_list, infall_timescale_list, comp_array, imf_array, sfe_array, imf_upper_limits, 
                 sn1a_assumptions, stellar_yield_assumptions, mgal_values, nb_array, sn1a_rates, timesteps, A2, feh, normalized_count, loss_metric='huber'):
        # Initialize parameters as instance variables
        self.sn1a_header = sn1a_header
        self.iniab_header = iniab_header
        self.sigma_2_list = sigma_2_list
        self.tmax_2_list = tmax_2_list
        self.infall_timescale_list = infall_timescale_list
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
        self.A2 = A2
        self.feh = feh
        self.normalized_count = normalized_count
        self.placeholder_sigma_array = np.zeros(len(normalized_count)) + 1  # Assume all sigmas are 1
        self.fancy_mutation = True

        self.model_count = 0
        self.mdf_data = []
        self.results = []
        self.labels = []
        self.MDFs = []
        self.model_numbers = []

        # Min and max values for sigma_2, t_2, and infall_2
        self.sigma_2_min, self.sigma_2_max = min(sigma_2_list), max(sigma_2_list)
        self.t_2_min, self.t_2_max = min(tmax_2_list), max(tmax_2_list)
        self.infall_2_min, self.infall_2_max = min(infall_timescale_list), max(infall_timescale_list)
        
        
        
        # Define available loss metrics
        self.loss_functions = {
            'wrmse': self.compute_wrmse,
            'mae': self.compute_mae,
            'mape': self.compute_mape,
            'huber': self.compute_huber,
            'cosine_similarity': self.compute_cosine_similarity,
            'log_cosh': self.compute_log_cosh
        }

        # Select the loss function based on user input
        if loss_metric not in self.loss_functions:
            raise ValueError(f"Invalid loss metric. Available options are: {list(self.loss_functions.keys())}")
        
        self.selected_loss_function = self.loss_functions[loss_metric]

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
        return wrmse, mae, mape, huber, cos_similarity, log_cosh


    def init_GenAl(self, population_size):
        # DEAP framework setup for Genetic Algorithm (GA)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Toolbox to define how individuals (solutions) are created and evolve
        toolbox = base.Toolbox()
        
        # Attribute generators for sigma_2, t_2, and infall_2 using uniform random values
        toolbox.register("sigma_2_attr", random.uniform, self.sigma_2_min, self.sigma_2_max)
        toolbox.register("t_2_attr", random.uniform, self.t_2_min, self.t_2_max)
        toolbox.register("infall_2_attr", random.uniform, self.infall_2_min, self.infall_2_max)

        # Create an individual by combining the three attributes (sigma_2, t_2, infall_2)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.sigma_2_attr, toolbox.t_2_attr, toolbox.infall_2_attr), n=1)

        # Create a population by repeating individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", self.evaluate)

        # Register genetic operations: crossover (blending) and mutation (custom)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", self.custom_mutate, indpb=0.2)

        # Register selection mechanism (tournament selection)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create the initial population
        population = toolbox.population(n=population_size)
        return population, toolbox

    def custom_mutate(self, individual, indpb):
        # Custom mutation function
        for i in range(len(individual)):
            if random.random() < indpb:
            
            #NEED MUCH MORE FANCY LIKE DISTRUBUTION FOR THE BOUNDS
                if i == 0:  # sigma_2
                    individual[i] = random.uniform(self.sigma_2_min, self.sigma_2_max)
                elif i == 1:  # t_2
                    individual[i] = random.uniform(self.t_2_min, self.t_2_max)
                elif i == 2:  # infall_2
                    individual[i] = random.uniform(self.infall_2_min, self.infall_2_max)
        return individual,


    def custom_gaussian_mutate(self, individual, population, fitnesses, base_sigma=1.0, indpb=0.1):
        
        def calculate_sigma(loss, max_loss, min_loss, gene_diversity, max_diversity, min_diversity, base_sigma):
            # Normalize loss and diversity to [0, 1]
            loss_norm = (loss - min_loss) / (max_loss - min_loss) if max_loss != min_loss else 1.0
            diversity_norm = (gene_diversity - min_diversity) / (max_diversity - min_diversity) if max_diversity != min_diversity else 1.0

            # Invert loss_norm so that lower loss leads to smaller sigma
            loss_factor = 1 - loss_norm

            # Combine factors (you can adjust weights)
            sigma = base_sigma * (loss_factor + diversity_norm) / 2

            # Ensure sigma is within reasonable bounds
            sigma = max(sigma, base_sigma * 0.1)  # Minimum sigma
            sigma = min(sigma, base_sigma * 10)   # Maximum sigma

            return sigma

        def calculate_successful_diversity(population, fitnesses, threshold):
            # Get individuals with fitness better than threshold
            successful_inds = [ind for ind, fit in zip(population, fitnesses) if fit <= threshold]
            if not successful_inds:
                return [1.0 for _ in range(len(population[0]))]  # Default diversity if no successful individuals
            # Extract genes
            genes = list(zip(*successful_inds))
            gene_diversity = []
            for gene_values in genes:
                gene_std = statistics.stdev(gene_values)
                gene_diversity.append(gene_std)
            return gene_diversity

            
        # Calculate loss statistics
        losses = [fit[0] for fit in fitnesses]
        max_loss = max(losses)
        min_loss = min(losses)
        # Calculate diversity of successful population
        # Define a threshold, e.g., the median loss
        threshold = statistics.median(losses)
        gene_diversity = calculate_successful_diversity(population, losses, threshold)
        max_diversity = max(gene_diversity)
        min_diversity = min(gene_diversity)

        # For each gene in the individual
        for i in range(len(individual)):
            if random.random() < indpb:
                # Calculate sigma for this gene
                loss = individual.fitness.values[0]
                sigma = calculate_sigma(loss, max_loss, min_loss, gene_diversity[i], max_diversity, min_diversity, base_sigma)

                # Apply Gaussian mutation
                individual[i] += random.gauss(0, sigma)

                # Ensure the mutated gene is within bounds
                if i == 0:  # sigma_2
                    min_bound = self.sigma_2_min
                    max_bound = self.sigma_2_max
                elif i == 1:  # t_2
                    min_bound = self.t_2_min
                    max_bound = self.t_2_max
                elif i == 2:  # infall_2
                    min_bound = self.infall_2_min
                    max_bound = self.infall_2_max
                individual[i] = min(max(individual[i], min_bound), max_bound)
        return individual,


    def GenAl(self, population_size, num_generations, cxpb, mutpb, population, toolbox):
        total_eval_time = 0  # Variable to accumulate total evaluation time
        total_eval_steps = 0  # Variable to count the number of evaluation steps
        total_start_time = time.time()
        
        for gen in range(num_generations):
            print(f"-- Generation {gen+1}/{num_generations} --")

            # Step 1: Evaluate the individuals with invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            num_initial_evals = len(invalid_ind)  # Count how many are invalid (to be evaluated)
            print(f"   Evaluating: {num_initial_evals}")
            step_eval_time = 0  # Track evaluation time for this generation (initial population)
            if invalid_ind:  # Only time evaluation if there are invalid individuals
                eval_start_time = time.time()
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                eval_end_time = time.time()
                
                step_eval_time = eval_end_time - eval_start_time
                average_eval_time = step_eval_time/num_initial_evals
                total_eval_time += step_eval_time
                total_eval_steps += num_initial_evals
            
            #NEED TO ITERATVILY UPDATE THE BOUNDS FOR MUTATION
            
            
            
            print(f"  Evaluation time for generation {gen+1}: {step_eval_time:.2f} seconds for {num_initial_evals} evaluations.")

            # Step 2: Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Step 3: Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            if self.fancy_mutation:
                # Step 4: Apply mutation
                # Need to pass population and fitnesses to mutation function
                # Collect fitnesses of the entire population
                population_fitnesses = [ind.fitness.values for ind in population]
                for mutant in offspring:
                    if random.random() < mutpb:
                        self.custom_gaussian_mutate(mutant, population, population_fitnesses, base_sigma=1.0, indpb=0.1)
                        del mutant.fitness.values
            else:
                # Step 4: Apply mutation
                for mutant in offspring:
                    if random.random() < mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values


            # Step 5: Evaluate the offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            num_offspring_evals = len(invalid_ind)  # Count how many offspring need evaluation

            step_eval_time = 0  # Reset for offspring evaluation
            if invalid_ind:  # Only time evaluation if there are invalid individuals
                eval_start_time = time.time()
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                eval_end_time = time.time()
                
                step_eval_time = eval_end_time - eval_start_time
                total_eval_time += step_eval_time
                total_eval_steps += num_offspring_evals
            
            print(f"  Offspring evaluation time for generation {gen+1}: {step_eval_time:.2f} seconds for {num_offspring_evals} evaluations.")

            # Replace the old population with the offspring
            population[:] = offspring
        # Calculate the overall average evaluation time per individual
        if total_eval_steps > 0:
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            eff_avg_eval_time = total_time / total_eval_steps
            overall_avg_eval_time = total_eval_time / total_eval_steps
            print(f"Overall average evaluation time per individual: {overall_avg_eval_time:.4f} seconds.")
            print(f"Effective overall average evaluation time per individual: {eff_avg_eval_time:.4f} seconds.")
        else:
            print("No evaluations were performed.")



    def evaluate(self, individual):
        sigma_2, t_2, infall_2 = individual

        # Fixed parameters from the pcard
        comp = self.comp_array[0]
        imf_val = self.imf_array[0]
        sfe_val = self.sfe_array[0]
        imf_upper = self.imf_upper_limits[0]
        sn1a = self.sn1a_assumptions[0]
        sy = self.stellar_yield_assumptions[0]
        mgal = self.mgal_values[0]
        nb = self.nb_array[0]
        sn1ar = self.sn1a_rates[0]
        A2 = self.A2
        sn1a_header = self.sn1a_header
        iniab_header = self.iniab_header

        # GCE Model kwargs
        kwargs = {
            'special_timesteps': self.timesteps,
            'twoinfall_sigmas': [1300, sigma_2],
            'galradius': 1800,
            'exp_infall': [[-1, 0.1e9, 0.1e9], [A2, t_2 * 1e9, infall_2 * 1e9]],
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

        sample_at = 0.0001
        resampled_x = np.arange(x_data.min(), x_data.max(), sample_at)
        cs_MDF = CubicSpline(x_data, y_data)
        theory_count_array = [cs_MDF(resampled_x)[find_nearest(resampled_x, self.feh[i])[0]] for i in range(len(self.feh))]
        
        # Calculate errors and similarities
        wrmse, mae, mape, huber, cos_similarity, log_cosh = self.calculate_all_metrics(theory_count_array)
        
        # Use the selected loss function for decision-making
        primary_loss_value = self.selected_loss_function(theory_count_array)

        # Store model results
        label = f'sigma2: {sigma_2:.3f}, t2: {t_2:.3f}, infall2: {infall_2:.3f}'

        self.labels.append(label)
        self.mdf_data.append([resampled_x, x_data, y_data])
        self.results.append([sigma_2, t_2, infall_2, wrmse, mae, mape, huber, cos_similarity, log_cosh])
        self.MDFs.append(cs_MDF)
        self.model_numbers.append(self.model_count)
        self.model_count += 1

        return (primary_loss_value,)

