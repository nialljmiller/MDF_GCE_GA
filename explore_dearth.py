#!/usr/bin/env python3.8
"""
Exploration utilities for identifying and targeting sparse regions in parameter space.
Authors: Your Name
"""

import numpy as np
import random
from scipy.spatial import Voronoi
from collections import defaultdict


def voronoi_explore_dearths(GA_instance, population, exploration_fraction=0.2):
    """
    Identify sparse regions using Voronoi analysis and move worst performers there.
    
    Parameters:
    -----------
    GA_instance : GalacticEvolutionGA
        The GA instance with parameter bounds and methods
    population : list
        Current population of individuals
    exploration_fraction : float
        Fraction of worst performers to move to sparse regions (default 0.2 = 20%)
    """
    
    if len(population) < 10:  # Need minimum population for meaningful analysis
        return
    
    # Step 1: Identify sparse regions using Voronoi
    sparse_regions = identify_sparse_regions_voronoi(GA_instance, population)
    
    if not sparse_regions:
        print("No sparse regions identified, skipping exploration")
        return
    
    # Step 2: Identify worst performers
    worst_performers = get_worst_performers(population, exploration_fraction)
    
    # Step 3: Move worst performers to sparse regions
    move_to_sparse_regions(GA_instance, worst_performers, sparse_regions)
    
    print(f"Moved {len(worst_performers)} individuals to {len(sparse_regions)} sparse regions")


def identify_sparse_regions_voronoi(GA_instance, population, n_regions=12):
    """
    Use Voronoi diagrams to identify sparse regions in parameter space.
    Works with 2D projections of most important parameter pairs.
    """

    # Define key parameter pairs for analysis (based on your plots)
    key_param_pairs = [
        (6, 7, 't_1', 't_2'),        # t_1 vs t_2
        (7, 9, 't_2', 'infall_2'),       # t_2 vs infall_2  
        (5, 9, 'sigma_2', 'infall_2'),   # sigma_2 vs infall_2
        (10, 11, 'sfe', 'delta_sfe'),      # sfe vs delta sfe
        (10, 5, 'sfe', 'sigma_2'),      # sfe vs delta sfe
        (13, 14, 'mgal', 'nb'),      # sfe vs delta sfe
    ]
    
    all_sparse_regions = []
    
    # Analyze each parameter pair
    for param1_idx, param2_idx, param1_name, param2_name in key_param_pairs:
        sparse_regions = _analyze_voronoi_2d(
            GA_instance, population, param1_idx, param2_idx, 
            param1_name, param2_name, n_regions_per_pair=2
        )
        all_sparse_regions.extend(sparse_regions)
    
    # Remove duplicates and sort by sparsity
    unique_regions = _deduplicate_regions(all_sparse_regions, threshold=0.1)
    
    return unique_regions[:n_regions]


def _analyze_voronoi_2d(GA_instance, population, param1_idx, param2_idx, 
                       param1_name, param2_name, n_regions_per_pair=2):
    """Analyze a 2D parameter projection using Voronoi diagrams"""
    
    # Extract and normalize the two parameters
    points = []
    for ind in population:
        # Normalize to [0,1] range
        min1, max1 = GA_instance.get_param_bounds(param1_idx)
        min2, max2 = GA_instance.get_param_bounds(param2_idx)
        
        norm1 = (ind[param1_idx] - min1) / (max1 - min1)
        norm2 = (ind[param2_idx] - min2) / (max2 - min2)
        
        points.append([norm1, norm2])
    
    points = np.array(points)
    
    # Handle edge case
    if len(points) < 4:
        return []
    
    try:
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Find largest finite regions (indicating sparse areas)
        large_regions = []
        
        for i, region in enumerate(vor.regions):
            if len(region) > 0 and -1 not in region:  # Valid finite region
                vertices = vor.vertices[region]
                if len(vertices) >= 3:  # Need at least 3 vertices for area calculation
                    # Calculate area
                    area = _polygon_area(vertices)
                    
                    # Calculate centroid
                    centroid = np.mean(vertices, axis=0)
                    
                    # Only consider regions within [0,1] bounds
                    if (0 <= centroid[0] <= 1) and (0 <= centroid[1] <= 1):
                        large_regions.append({
                            'area': area,
                            'centroid_norm': centroid,
                            'param1_name': param1_name,
                            'param2_name': param2_name,
                            'param1_idx': param1_idx,
                            'param2_idx': param2_idx
                        })
        
        # Sort by area and return largest regions
        large_regions.sort(key=lambda x: x['area'], reverse=True)
        
        # Convert back to parameter space
        sparse_regions = []
        for region in large_regions[:n_regions_per_pair]:
            # Denormalize centroid
            min1, max1 = GA_instance.get_param_bounds(param1_idx)
            min2, max2 = GA_instance.get_param_bounds(param2_idx)
            
            param1_val = min1 + region['centroid_norm'][0] * (max1 - min1)
            param2_val = min2 + region['centroid_norm'][1] * (max2 - min2)
            
            sparse_regions.append({
                'target_params': {
                    param1_name: param1_val,
                    param2_name: param2_val
                },
                'area': region['area'],
                'param_indices': {
                    param1_name: param1_idx,
                    param2_name: param2_idx
                }
            })
        
        return sparse_regions
        
    except Exception as e:
        print(f"Voronoi analysis failed for {param1_name}-{param2_name}: {e}")
        return []


def _polygon_area(vertices):
    """Calculate area of polygon using shoelace formula"""
    if len(vertices) < 3:
        return 0
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _deduplicate_regions(regions, threshold=0.1):
    """Remove regions that are too close to each other"""
    if not regions:
        return []
    
    unique_regions = [regions[0]]
    
    for region in regions[1:]:
        is_duplicate = False
        for existing in unique_regions:
            # Check if regions are too close in parameter space
            distance = _region_distance(region, existing)
            if distance < threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_regions.append(region)
    
    return unique_regions


def _region_distance(region1, region2):
    """Calculate distance between two regions in normalized parameter space"""
    # Simple Euclidean distance in parameter space
    dist = 0
    common_params = set(region1['target_params'].keys()) & set(region2['target_params'].keys())
    
    if not common_params:
        return 1.0  # Maximum distance if no common parameters
    
    for param in common_params:
        # Normalize difference (assuming parameters are already in reasonable ranges)
        diff = abs(region1['target_params'][param] - region2['target_params'][param])
        dist += diff ** 2
    
    return np.sqrt(dist / len(common_params))


def get_worst_performers(population, fraction):
    """Get the worst performing fraction of the population"""
    
    # Sort by fitness (assuming lower is better for minimization)
    sorted_pop = sorted(population, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'), reverse=True)
    
    n_worst = int(len(population) * fraction)
    n_worst = max(1, n_worst)  # At least 1 individual
    
    return sorted_pop[:n_worst]


def move_to_sparse_regions(GA_instance, individuals, sparse_regions):
    """Move individuals toward sparse regions with some randomness"""
    
    for ind in individuals:
        if not sparse_regions:
            continue
            
        # Randomly select a sparse region to target
        target_region = random.choice(sparse_regions)
        
        # Move toward the target region with some noise
        _mutate_toward_region(GA_instance, ind, target_region)
        
        # Invalidate fitness since we changed the individual
        del ind.fitness.values


def _mutate_toward_region(GA_instance, individual, target_region):
    """Mutate an individual toward a specific sparse region"""
    
    target_params = target_region['target_params']
    param_indices = target_region['param_indices']
    
    for param_name, target_val in target_params.items():
        param_idx = param_indices[param_name]
        current_val = individual[param_idx]
        
        # Calculate movement toward target
        direction = target_val - current_val
        
        # Move partially toward target with noise
        movement_fraction = 0.9 + 0.1 * random.random()  # 90-100% toward target
        
        min_bound, max_bound = GA_instance.get_param_bounds(param_idx)
        range_size = max_bound - min_bound
        noise_scale = 0.05 * range_size  # 5% noise
        
        new_val = current_val + movement_fraction * direction + random.gauss(0, noise_scale)
        
        # Apply bounds with reflection
        new_val = GA_instance._reflect_at_bounds(new_val, min_bound, max_bound)
        individual[param_idx] = new_val
    
    # Also add some mutation to other parameters to avoid getting stuck
    _add_background_mutation(GA_instance, individual)


def _add_background_mutation(GA_instance, individual, mutation_probability=0.3):
    """Add small mutations to other parameters to maintain exploration"""
    
    continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    for param_idx in continuous_indices:
        if random.random() < mutation_probability:
            min_bound, max_bound = GA_instance.get_param_bounds(param_idx)
            range_size = max_bound - min_bound
            
            # Small mutation
            mutation_size = 0.02 * range_size * random.gauss(0, 1)
            new_val = individual[param_idx] + mutation_size
            new_val = GA_instance._reflect_at_bounds(new_val, min_bound, max_bound)
            individual[param_idx] = new_val