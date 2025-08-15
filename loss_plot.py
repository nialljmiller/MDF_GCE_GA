#!/usr/bin/env python3.8
################################
# Plotting functions for MDF_GA
################################
# Authors: N Miller

"""Plotting utilities for MDF_GA and related bulge diagnostics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors, gridspec
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
import os
from scipy.interpolate import UnivariateSpline
from numpy.polynomial.polynomial import Polynomial
from phys_plot import generate_physics_plots





def plot_walker_loss_history(GalGA, walker_history, results_csv='simulation_results.csv', loss_metric='wrmse'):
    """
    Plot the evolution of loss for all walkers with median and IQR shading.
    Mirrors the style of plot_walker_history.
    """

    save_path = GalGA.output_path + save_path

    os.makedirs(GalGA.output_path + "/loss", exist_ok=True)

    # Load full GA results
    results_df = pd.read_csv(results_csv)

    # Column mapping
    loss_metrics = {
        'ks': 15, 'ensemble': 16, 'wrmse': 17, 'mae': 18, 'mape': 19,
        'huber': 20, 'cosine': 21, 'log_cosh': 22, 'fitness': 23
    }

    if loss_metric not in loss_metrics:
        print(f"Loss metric '{loss_metric}' not found. Falling back to 'wrmse'.")
        loss_metric = 'wrmse'

    loss_column = loss_metrics[loss_metric]

    all_histories = []
    max_gens = 0

    for walker_id, history in walker_history.items():
        if not history:
            continue

        history_array = np.array(history)
        loss_vals = []

        for row in history_array:
            sigma_2, t_2, infall_2 = row[5], row[7], row[9]
            match = results_df[
                (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                (abs(results_df['t_2'] - t_2) < 1e-5) &
                (abs(results_df['infall_2'] - infall_2) < 1e-5)
            ]
            loss_vals.append(match.iloc[0][loss_metric] if not match.empty else np.nan)

        all_histories.append(loss_vals)
        max_gens = max(max_gens, len(loss_vals))

    if not all_histories:
        print("No valid walker loss histories to plot.")
        return None

    # Pad to uniform shape
    for i in range(len(all_histories)):
        if len(all_histories[i]) < max_gens:
            all_histories[i] += [np.nan] * (max_gens - len(all_histories[i]))

    all_histories = np.array(all_histories)
    generations = np.arange(max_gens)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual walkers (faint gray)
    for series in all_histories:
        ax.plot(generations, series, color='gray', alpha=0.01, linewidth=0.75)

    # Median + IQR
    with np.errstate(all='ignore'):
        median = np.nanmedian(all_histories, axis=0)
        lower = np.nanpercentile(all_histories, 25, axis=0)
        upper = np.nanpercentile(all_histories, 75, axis=0)

    ax.plot(generations, median, color='black', label='Median', linewidth=2)
    ax.fill_between(generations, lower, upper, color='blue', alpha=0.2, label='25–75% range')

    ax.set_title(f"Walker Evolution: {loss_metric.upper()}")
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"{loss_metric.upper()}")
    ax.grid(True)
    ax.legend(loc='best')

    fig.tight_layout()
    outpath = f'GA/loss/walker_loss_history_{loss_metric}.png'
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved loss history plot: {outpath}")
    return fig



def plot_walker_success_rate(walker_history, results_csv='simulation_results.csv', 
                             threshold=0.1, loss_metric='wrmse', save_path='loss/walker_success_rate_'):
    """
    Plot the fraction of walkers with loss below threshold over generations.
    
    Parameters:
    -----------
    walker_history : dict
        Dictionary mapping walker IDs to their parameter history
    results_csv : str
        Path to the CSV file containing all evaluation results
    threshold : float
        Loss threshold for success criterion
    loss_metric : str
        Which loss metric to use ('wrmse', 'mae', 'mape', etc.)
    save_path : str
        Where to save the plot
    """
    
    save_path = GalGA.output_path + save_path + str(loss_metric) + '.png'

    if not walker_history:
        print("Walker history data not available. Skipping success rate plot.")
        return None
    
    # Load results containing all evaluations
    import pandas as pd
    results_df = pd.read_csv(results_csv)
    
    # Get maximum number of generations
    max_generations = max(len(history) for history in walker_history.values() if history)
    if max_generations == 0:
        print("No generation data found. Skipping success rate plot.")
        return None
    
    success_fractions = []
    generations = list(range(max_generations))
    
    # For each generation
    for gen in range(max_generations):
        successful_walkers = 0
        total_walkers = 0
        
        # Check each walker
        for walker_id, history in walker_history.items():
            if not history or gen >= len(history):
                continue
                
            total_walkers += 1
            
            # Get parameters for this generation
            params = history[gen]
            
            # Extract key parameters to match with results
            # Assuming indices based on your individual structure
            sigma_2 = params[5]  # sigma_2
            t_2 = params[7]      # t_2  
            infall_2 = params[9] # infall_2
            
            # Find matching result in dataframe
            matches = results_df[
                (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                (abs(results_df['t_2'] - t_2) < 1e-5) &
                (abs(results_df['infall_2'] - infall_2) < 1e-5)
            ]
            
            if not matches.empty:
                loss_value = matches.iloc[0][loss_metric]
                if loss_value < threshold:
                    successful_walkers += 1
        
        # Calculate success fraction
        if total_walkers > 0:
            success_fraction = successful_walkers / total_walkers
        else:
            success_fraction = 0.0
            
        success_fractions.append(success_fraction)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(generations, success_fractions, 'o-', linewidth=2, markersize=4, 
            color='steelblue', label=f'Success Rate (< {threshold})')
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% Success')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Success')
    
    # Formatting
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'Fraction of Walkers with {loss_metric.upper()} < {threshold}')
    ax.set_title(f'Walker Success Rate Over Generations\n({loss_metric.upper()} threshold = {threshold})')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final success rate annotation
    if success_fractions:
        final_rate = success_fractions[-1]
        ax.annotate(f'Final: {final_rate:.1%}', 
                   xy=(len(generations)-1, final_rate),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Walker success rate plot saved to {save_path}")
    print(f"Final success rate: {success_fractions[-1]:.1%} of walkers below {threshold}")
    
    return fig


def plot_multiple_success_thresholds(GalGA, walker_history, results_csv='simulation_results.csv', 
                                   thresholds=[0.01, 0.1, 0.001], loss_metric='wrmse', 
                                   save_path='loss/walker_success_rates_multiple_'):
    """
    Plot success rates for multiple thresholds on the same plot.
    """

    save_path =GalGA.output_path +  save_path + str(loss_metric) + '.png'
    
    if not walker_history:
        print("Walker history data not available.")
        return None
    
    import pandas as pd
    results_df = pd.read_csv(results_csv)
    
    max_generations = max(len(history) for history in walker_history.values() if history)
    if max_generations == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
        
    colors = [
        '#E60026',  # Mondrian red
        '#0047AB',  # Mondrian blue
        '#F7D842',  # Mondrian yellow
        '#000000',  # black
        '#1A1A1A'   # very dark gray/near-black accent
        '#FFD300',  # strong yellow variant
    ]
    
    for i, threshold in enumerate(thresholds):
        success_fractions = []
        generations = list(range(max_generations))
        
        for gen in range(max_generations):
            successful_walkers = 0
            total_walkers = 0
            
            for walker_id, history in walker_history.items():
                if not history or gen >= len(history):
                    continue
                    
                total_walkers += 1
                params = history[gen]
                
                sigma_2 = params[5]
                t_2 = params[7]
                infall_2 = params[9]
                
                matches = results_df[
                    (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                    (abs(results_df['t_2'] - t_2) < 1e-5) &
                    (abs(results_df['infall_2'] - infall_2) < 1e-5)
                ]
                
                if not matches.empty:
                    loss_value = matches.iloc[0][loss_metric]
                    if loss_value < threshold:
                        successful_walkers += 1
            
            if total_walkers > 0:
                success_fraction = successful_walkers / total_walkers
            else:
                success_fraction = 0.0
                
            success_fractions.append(success_fraction)
        
        # Plot this threshold
        color = colors[i % len(colors)]
        ax.plot(generations, success_fractions, 'o-', linewidth=2, markersize=3, 
                color=color, label=f'< {threshold}', alpha=0.8)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'Fraction of Walkers Below Threshold ({loss_metric.upper()})')
    ax.set_title(f'Walker Success Rates for Multiple Thresholds')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Threshold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multiple threshold success rate plot saved to {save_path}")
    return fig




def plot_3d_scatter(GalGA, x, y, z, color_metric, label, xlabel='sigma_2', ylabel='t_2', zlabel='infall_2'):
    """Plot 3D scatter plot with color indicating a specific metric.
    Two plots:
      - All data, color scaled [0, 1]
      - Only points with loss < 0.1, color scaled [0, 0.1]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x, y, z, color_metric = map(np.array, (x, y, z, color_metric))

    def make_plot(GalGA, x_data, y_data, z_data, color_data, vmin, vmax, suffix):
        # Sort to plot best points on top
        idx = np.argsort(color_data)[::-1]
        x_sorted, y_sorted, z_sorted, color_sorted = x_data[idx], y_data[idx], z_data[idx], color_data[idx]

        total = len(color_sorted)
        top_n = min(max(1, int(0.01 * total)), 100)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(x_sorted, y_sorted, z_sorted, c=color_sorted, cmap='nipy_spectral',
                        vmin=vmin, vmax=vmax, s=30, alpha=0.8)

        if top_n > 0:
            top_x = x_sorted[-top_n:]
            top_y = y_sorted[-top_n:]
            top_z = z_sorted[-top_n:]
            top_colors = color_sorted[-top_n:]
            ax.scatter(top_x, top_y, top_z, c=top_colors, cmap='nipy_spectral',
                       vmin=vmin, vmax=vmax, s=50, edgecolors='white', linewidths=2, alpha=1.0)


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.colorbar(sc, label=label)
        plt.savefig(GalGA.output_path + f'/loss/{label}_loss_3d{suffix}.png', bbox_inches='tight')
        plt.close()

    # Full plot
    make_plot(GalGA, x, y, z, color_metric, 0, 1, '')

    # Filtered plot for low loss
    mask = color_metric < 0.1
    if np.any(mask):
        make_plot(GalGA, x[mask], y[mask], z[mask], color_metric[mask], 0, 0.1, '_lowloss')



def plot_2d_scatter(GalGA, x, y, color_metric, label, xlabel='t_2', ylabel='infall_2'):
    """Plot 2D scatter plot with color indicating a specific metric.
    Two plots:
      - All data, color scaled [0, 1]
      - Only points with loss < 0.1, color scaled [0, 0.1]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x, y, color_metric = map(np.array, (x, y, color_metric))

    def make_plot(GalGA, x_data, y_data, color_data, vmin, vmax, suffix):
        idx = np.argsort(color_data)[::-1]
        x_sorted, y_sorted, color_sorted = x_data[idx], y_data[idx], color_data[idx]

        total = len(color_sorted)
        top_n = min(max(1, int(0.01 * total)), 100)

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(x_sorted, y_sorted, c=color_sorted, cmap='nipy_spectral',
                         vmin=vmin, vmax=vmax, s=30, alpha=0.8)

        if top_n > 0:
            top_x = x_sorted[-top_n:]
            top_y = y_sorted[-top_n:]
            top_colors = color_sorted[-top_n:]
            plt.scatter(top_x, top_y, c=top_colors, cmap='nipy_spectral',
                        vmin=vmin, vmax=vmax, s=50, edgecolors='white', linewidths=2, alpha=1.0)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(sc, label=label)
        plt.savefig(GalGA.output_path + f'/loss/{label}_loss_2d{suffix}.png', bbox_inches='tight')
        plt.close()

    # Full plot
    make_plot(GalGA, x, y, color_metric, 0, 1, '')

    # Filtered plot for low loss
    mask = color_metric < 0.1
    if np.any(mask):
        make_plot(GalGA, x[mask], y[mask], color_metric[mask], 0, 0.1, '_lowloss')





def plot_walker_history(GalGA, walker_history, param_names, param_indices):
    """
    Plot the evolution of parameters for all walkers with median + spread.
    """
    if not walker_history:
        print("Walker history data not available. Skipping walker evolution plots.")
        return None

    os.makedirs(GalGA.output_path + "/loss", exist_ok=True)

    figs = []

    for idx, param_name in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(12, 6))
        figs.append(fig)

        all_histories = []
        for walker_idx, history in walker_history.items():
            if not history:
                continue

            history = np.array(history)
            param_idx = param_indices[idx]

            if param_idx >= history.shape[1]:
                continue

            all_histories.append(history[:, param_idx])

        if not all_histories:
            continue

        all_histories = np.array(all_histories)  # shape: (n_walkers, n_generations)
        generations = np.arange(all_histories.shape[1])

        # Plot faint lines for individual walkers
        for walker_series in all_histories:
            ax.plot(generations, walker_series, color='gray', alpha=0.01, linewidth=0.75)

        # Overlay median and shaded quantiles
        median = np.median(all_histories, axis=0)
        lower = np.percentile(all_histories, 25, axis=0)
        upper = np.percentile(all_histories, 75, axis=0)

        ax.plot(generations, median, color='black', label='Median', linewidth=2)
        ax.fill_between(generations, lower, upper, color='blue', alpha=0.2, label='25–75% range')

        ax.set_xlabel("Generation")
        ax.set_ylabel(f"{param_name}")
        ax.set_title(f"Walker Evolution: {param_name}")
        ax.legend(loc='best')
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(GalGA.output_path + f'/loss/walker_evolution_{param_name}.png', bbox_inches='tight')
        plt.close(fig)

    print("Generated walker evolution plots with clarity enhancements")
    return figs







def plot_walker_loss_history(GalGA, walker_history, results_csv='simulation_results.csv', loss_metric='wrmse'):
    """
    Plot the evolution of loss for all walkers with median and IQR shading.
    Now with logarithmic y-axis for better visualization of loss ranges.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(GalGA.output_path + "/loss", exist_ok=True)

    # Load full GA results
    results_df = pd.read_csv(results_csv)

    # Column mapping
    loss_metrics = {
        'ks': 15, 'ensemble': 16, 'wrmse': 17, 'mae': 18, 'mape': 19,
        'huber': 20, 'cosine': 21, 'log_cosh': 22, 'fitness': 23, 'age_meta_fitness': 24, 'physics_penalty': 25
    }

    if loss_metric not in loss_metrics:
        print(f"Loss metric '{loss_metric}' not found. Falling back to 'wrmse'.")
        loss_metric = 'wrmse'

    loss_column = loss_metrics[loss_metric]

    all_histories = []
    max_gens = 0

    for walker_id, history in walker_history.items():
        if not history:
            continue

        history_array = np.array(history)
        loss_vals = []

        for row in history_array:
            sigma_2, t_2, infall_2 = row[5], row[7], row[9]
            match = results_df[
                (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                (abs(results_df['t_2'] - t_2) < 1e-5) &
                (abs(results_df['infall_2'] - infall_2) < 1e-5)
            ]
            loss_vals.append(match.iloc[0][loss_metric] if not match.empty else np.nan)

        all_histories.append(loss_vals)
        max_gens = max(max_gens, len(loss_vals))

    if not all_histories:
        print("No valid walker loss histories to plot.")
        return None

    # Pad to uniform shape
    for i in range(len(all_histories)):
        if len(all_histories[i]) < max_gens:
            all_histories[i] += [np.nan] * (max_gens - len(all_histories[i]))

    all_histories = np.array(all_histories)
    generations = np.arange(max_gens)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual walkers (faint gray)
    for series in all_histories:
        # Only plot valid (non-NaN, positive) values for log scale
        valid_mask = np.isfinite(series) & (series > 0)
        if np.any(valid_mask):
            ax.plot(generations[valid_mask], series[valid_mask], color='gray', alpha=0.01, linewidth=0.75)

    # Median + IQR (only for positive, finite values)
    with np.errstate(all='ignore'):
        # Filter out non-positive values for log scale
        positive_histories = np.where((all_histories > 0) & np.isfinite(all_histories), 
                                    all_histories, np.nan)
        
        median = np.nanmedian(positive_histories, axis=0)
        lower = np.nanpercentile(positive_histories, 25, axis=0)
        upper = np.nanpercentile(positive_histories, 75, axis=0)

    # Only plot where we have valid data
    valid_median = np.isfinite(median) & (median > 0)
    if np.any(valid_median):
        ax.plot(generations[valid_median], median[valid_median], color='black', label='Median', linewidth=2)
        
        # Fill between only where both bounds are valid and positive
        valid_fill = (np.isfinite(lower) & np.isfinite(upper) & 
                     (lower > 0) & (upper > 0) & valid_median)
        if np.any(valid_fill):
            ax.fill_between(generations[valid_fill], lower[valid_fill], upper[valid_fill], 
                           color='blue', alpha=0.2, label='25–75% range')

    ax.set_title(f"Walker Evolution: {loss_metric.upper()} (Log Scale)")
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"{loss_metric.upper()}")
    
    # Set logarithmic y-axis
    ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add some statistics in the plot
    if len(all_histories) > 0:
        final_losses = []
        for series in all_histories:
            valid_final = series[np.isfinite(series) & (series > 0)]
            if len(valid_final) > 0:
                final_losses.append(valid_final[-1])
        
        if final_losses:
            min_final = min(final_losses)
            median_final = np.median(final_losses)
            ax.annotate(f'Final median: {median_final:.4f}\nBest final: {min_final:.4f}', 
                       xy=(0.02, 0.98), xycoords='axes fraction',
                       verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    outpath = GalGA.output_path + f'loss/walker_loss_history_{loss_metric}.png'
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Saved loss history plot with log scale: {outpath}")
    return fig


def plot_multiple_loss_metrics_evolution(GalGA, walker_history, results_csv='simulation_results.csv', 
                                       metrics=['wrmse', 'huber', 'ks', 'fitness'], 
                                       save_path='loss/multiple_loss_evolution.png'):
    """
    Plot evolution of multiple loss metrics on the same figure with subplots.
    All use logarithmic y-axis for better comparison.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    save_path = GalGA.output_path + save_path

    os.makedirs(GalGA.output_path + "/loss", exist_ok=True)

    # Load full GA results
    results_df = pd.read_csv(results_csv)

    # Column mapping
    loss_metrics = {
        'ks': 15, 'ensemble': 16, 'wrmse': 17, 'mae': 18, 'mape': 19,
        'huber': 20, 'cosine': 21, 'log_cosh': 22, 'fitness': 23
    }

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in loss_metrics]
    if not available_metrics:
        print("No valid metrics found for plotting.")
        return None

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        all_histories = []
        max_gens = 0

        for walker_id, history in walker_history.items():
            if not history:
                continue

            history_array = np.array(history)
            loss_vals = []

            for row in history_array:
                sigma_2, t_2, infall_2 = row[5], row[7], row[9]
                match = results_df[
                    (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                    (abs(results_df['t_2'] - t_2) < 1e-5) &
                    (abs(results_df['infall_2'] - infall_2) < 1e-5)
                ]
                loss_vals.append(match.iloc[0][metric] if not match.empty else np.nan)

            all_histories.append(loss_vals)
            max_gens = max(max_gens, len(loss_vals))

        if not all_histories:
            continue

        # Pad to uniform shape
        for i in range(len(all_histories)):
            if len(all_histories[i]) < max_gens:
                all_histories[i] += [np.nan] * (max_gens - len(all_histories[i]))

        all_histories = np.array(all_histories)
        generations = np.arange(max_gens)

        # Individual walkers (very faint)
        for series in all_histories:
            valid_mask = np.isfinite(series) & (series > 0)
            if np.any(valid_mask):
                ax.plot(generations[valid_mask], series[valid_mask], 
                       color='gray', alpha=0.005, linewidth=0.5)

        # Median + IQR
        with np.errstate(all='ignore'):
            positive_histories = np.where((all_histories > 0) & np.isfinite(all_histories), 
                                        all_histories, np.nan)
            
            median = np.nanmedian(positive_histories, axis=0)
            lower = np.nanpercentile(positive_histories, 25, axis=0)
            upper = np.nanpercentile(positive_histories, 75, axis=0)

        # Plot median and IQR
        valid_median = np.isfinite(median) & (median > 0)
        if np.any(valid_median):
            ax.plot(generations[valid_median], median[valid_median], 
                   color='black', label=f'{metric.upper()} Median', linewidth=2)
            
            valid_fill = (np.isfinite(lower) & np.isfinite(upper) & 
                         (lower > 0) & (upper > 0) & valid_median)
            if np.any(valid_fill):
                ax.fill_between(generations[valid_fill], lower[valid_fill], upper[valid_fill], 
                               color='blue', alpha=0.2, label='IQR')

        ax.set_ylabel(f"{metric.upper()}")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add final value annotation
        if len(all_histories) > 0:
            final_losses = []
            for series in all_histories:
                valid_final = series[np.isfinite(series) & (series > 0)]
                if len(valid_final) > 0:
                    final_losses.append(valid_final[-1])
            
            if final_losses:
                min_final = min(final_losses)
                ax.annotate(f'Best: {min_final:.4f}', 
                           xy=(0.98, 0.02), xycoords='axes fraction',
                           horizontalalignment='right', verticalalignment='bottom',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    axes[-1].set_xlabel("Generation")
    fig.suptitle("Loss Metric Evolution (Log Scale)", fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Saved multiple loss metrics plot: {save_path}")
    return fig






def plot_loss_convergence_analysis(GalGA, walker_history, results_csv='simulation_results.csv',
                                 loss_metric='wrmse', save_path='loss/convergence_analysis.png'):
    """
    Analyze and plot convergence characteristics of the loss evolution.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    save_path = GalGA.output_path + save_path

    os.makedirs(GalGA.output_path + "/loss", exist_ok=True)

    # Load results and extract loss histories (same as before)
    results_df = pd.read_csv(results_csv)
    loss_metrics = {
        'ks': 15, 'ensemble': 16, 'wrmse': 17, 'mae': 18, 'mape': 19,
        'huber': 20, 'cosine': 21, 'log_cosh': 22, 'fitness': 23
    }

    if loss_metric not in loss_metrics:
        loss_metric = 'wrmse'

    all_histories = []
    for walker_id, history in walker_history.items():
        if not history:
            continue

        history_array = np.array(history)
        loss_vals = []

        for row in history_array:
            sigma_2, t_2, infall_2 = row[5], row[7], row[9]
            match = results_df[
                (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                (abs(results_df['t_2'] - t_2) < 1e-5) &
                (abs(results_df['infall_2'] - infall_2) < 1e-5)
            ]
            loss_vals.append(match.iloc[0][loss_metric] if not match.empty else np.nan)

        all_histories.append(loss_vals)

    if not all_histories:
        return None

    # Pad histories
    max_gens = max(len(h) for h in all_histories)
    for i in range(len(all_histories)):
        if len(all_histories[i]) < max_gens:
            all_histories[i] += [np.nan] * (max_gens - len(all_histories[i]))

    all_histories = np.array(all_histories)
    generations = np.arange(max_gens)

    # Create comprehensive convergence analysis plot
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Main loss evolution with log scale
    ax1 = fig.add_subplot(gs[0, :2])
    
    positive_histories = np.where((all_histories > 0) & np.isfinite(all_histories), 
                                all_histories, np.nan)
    
    # Individual walkers
    for series in positive_histories:
        valid_mask = np.isfinite(series)
        if np.any(valid_mask):
            ax1.plot(generations[valid_mask], series[valid_mask], 
                    color='gray', alpha=0.01, linewidth=0.5)

    # Statistics
    median = np.nanmedian(positive_histories, axis=0)
    p10 = np.nanpercentile(positive_histories, 10, axis=0)
    p90 = np.nanpercentile(positive_histories, 90, axis=0)
    minimum = np.nanmin(positive_histories, axis=0)

    valid_stats = np.isfinite(median) & (median > 0)
    ax1.plot(generations[valid_stats], median[valid_stats], 'b-', linewidth=2, label='Median')
    ax1.plot(generations[valid_stats], minimum[valid_stats], 'r-', linewidth=2, label='Minimum')
    ax1.fill_between(generations[valid_stats], p10[valid_stats], p90[valid_stats], 
                    alpha=0.2, color='blue', label='10-90% Range')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel(f'{loss_metric.upper()}')
    ax1.set_yscale('log')
    ax1.set_title(f'Loss Evolution with Convergence Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Convergence rate analysis
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Calculate improvement rate (how much loss decreases per generation)
    if np.any(valid_stats) and np.sum(valid_stats) > 10:
        valid_gen = generations[valid_stats]
        valid_median = median[valid_stats]
        
        # Calculate rolling improvement rate
        window = min(10, len(valid_median) // 4)
        if window > 2:
            improvement_rate = []
            improvement_gen = []
            for i in range(window, len(valid_median)):
                start_loss = np.mean(valid_median[i-window:i-window//2])
                end_loss = np.mean(valid_median[i-window//2:i])
                if start_loss > 0 and end_loss > 0:
                    rate = (np.log(start_loss) - np.log(end_loss)) / (window // 2)
                    improvement_rate.append(rate)
                    improvement_gen.append(valid_gen[i])
            
            if improvement_rate:
                ax2.plot(improvement_gen, improvement_rate, 'g-', linewidth=2)
                ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Log Improvement Rate')
                ax2.set_title('Convergence Rate')
                ax2.grid(True, alpha=0.3)

    # 3. Final distribution analysis
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get final losses from each walker
    final_losses = []
    for series in positive_histories:
        valid_final = series[np.isfinite(series)]
        if len(valid_final) > 0:
            final_losses.append(valid_final[-1])
    
    if final_losses:
        ax3.hist(final_losses, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(np.median(final_losses), color='red', linestyle='--', 
                   label=f'Median: {np.median(final_losses):.4f}')
        ax3.axvline(np.min(final_losses), color='green', linestyle='--', 
                   label=f'Best: {np.min(final_losses):.4f}')
        ax3.set_xlabel(f'Final {loss_metric.upper()}')
        ax3.set_ylabel('Walker Count')
        ax3.set_title('Final Loss Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Diversity analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate diversity (standard deviation) over generations
    diversity = np.nanstd(positive_histories, axis=0)
    valid_div = np.isfinite(diversity) & (diversity > 0)
    
    if np.any(valid_div):
        ax4.plot(generations[valid_div], diversity[valid_div], 'purple', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Loss Diversity (Std Dev)')
        ax4.set_yscale('log')
        ax4.set_title('Population Diversity')
        ax4.grid(True, alpha=0.3)

    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    if final_losses:
        summary_text = f"""CONVERGENCE SUMMARY
        
Final Statistics:
• Best loss: {np.min(final_losses):.6f}
• Median loss: {np.median(final_losses):.6f}
• Worst loss: {np.max(final_losses):.6f}
• Std deviation: {np.std(final_losses):.6f}

Convergence Analysis:
• Total generations: {max_gens}
• Walkers analyzed: {len(final_losses)}
• Dynamic range: {np.max(final_losses)/np.min(final_losses):.2f}x

Performance:
• <0.1 threshold: {np.sum(np.array(final_losses) < 0.1)}/{len(final_losses)}
• <0.05 threshold: {np.sum(np.array(final_losses) < 0.05)}/{len(final_losses)}
• <0.01 threshold: {np.sum(np.array(final_losses) < 0.01)}/{len(final_losses)}"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Saved convergence analysis plot: {save_path}")
    return fig










# --- Improved binned maps & deltas (drop-in) ---
from scipy.stats import binned_statistic_2d, iqr
from scipy.ndimage import gaussian_filter

def _auto_bins_1d(x, nmax=60, nmin=8):
    """Freedman–Diaconis with sane guards."""
    x = np.asarray(x)
    n = len(x)
    if n < 50:
        return max(nmin, int(np.sqrt(n)))  # tiny samples
    h = 2 * iqr(x, nan_policy='omit') * n ** (-1/3)
    if not np.isfinite(h) or h <= 0:
        return max(nmin, min(nmax, int(np.sqrt(n))))
    bins = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / h))
    return max(nmin, min(nmax, bins))

def _choose_bins(x, y, bins):
    if isinstance(bins, tuple):
        return bins
    if bins == 'auto':
            return (_auto_bins_1d(x), _auto_bins_1d(y))
    if isinstance(bins, int):
        return (bins, bins)
    return (_auto_bins_1d(x), _auto_bins_1d(y))

def plot_binned_loss(
    df, xcol, ycol, losscol='fitness',
    bins='auto', agg='median', min_per_bin=6, smooth_sigma=1.0,
    cmap='viridis', hatch_lowN='///', overlay_samples=True,
    overlay_contours=True, contour_levels=7, scatter_alpha=0.25,
    save_prefix='GA/analysis/binned'
):
    """
    Creates three figures:
      1) Aggregated loss heatmap (median/mean/min) with hatching where N<min_per_bin
      2) Δ-loss (relative to global min over valid bins)
      3) |∇loss| magnitude + quiver
    Also overlays sample locations and optional contours.

    Returns: (Z, xedges, yedges, N)
    """
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    # data
    x = np.asarray(df[xcol].values, dtype=float)
    y = np.asarray(df[ycol].values, dtype=float)
    z = np.asarray(df[losscol].values, dtype=float)

    bx, by = _choose_bins(x, y, bins)

    # aggregate
    stat_name = {'median':'median', 'mean':'mean'}.get(agg, 'median')
    if agg in ('median','mean'):
        Z, xedges, yedges, _ = binned_statistic_2d(
            x, y, z, statistic=stat_name, bins=(bx, by)
        )
    elif agg == 'min':
        Z, xedges, yedges, _ = binned_statistic_2d(
            x, y, z, statistic=np.min, bins=(bx, by)
        )
    else:
        raise ValueError("agg must be 'median', 'mean', or 'min'")

    # counts (for hatching & masking)
    N, _, _, _ = binned_statistic_2d(x, y, None, statistic='count', bins=(bx, by))
    Z = Z.astype(float)
    Z[N < min_per_bin] = np.nan  # mask sparse bins

    # optional smoothing (keep mask)
    if smooth_sigma and smooth_sigma > 0:
        Zfilled = np.nanmedian(Z) if np.isfinite(np.nanmedian(Z)) else 0.0
        Zs = gaussian_filter(np.nan_to_num(Z, nan=Zfilled), smooth_sigma)
        Z = np.where(np.isnan(Z), np.nan, Zs)

    # convenient centers for overlays
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])

    # common draw helper
    def _draw_base(fig, ax, data, cbar_label, title):
        im = ax.pcolormesh(xedges, yedges, data.T, shading='auto', cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xlabel(xcol.replace('_',' '))
        ax.set_ylabel(ycol.replace('_',' '))
        ax.set_title(title)

        # hatch low-N bins
        low = (N < min_per_bin).T
        if np.any(low) and hatch_lowN:
            ax.contourf(
                Xc, Yc, low, levels=[0.5, 1.5],
                colors='none', hatches=[hatch_lowN], alpha=0
            )
            ax.text(0.98, 0.02, f'hatched: N<{min_per_bin}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9)

        # overlay samples
        if overlay_samples:
            ax.scatter(x, y, s=6, c='k', alpha=scatter_alpha, linewidths=0, zorder=3)

        # overlay contours on valid region
        if overlay_contours:
            try:
                valid = np.isfinite(Z)
                if np.count_nonzero(valid) > 5:
                    ax.contour(Xc, Yc, Z.T, levels=contour_levels, colors='k',
                               alpha=0.35, linewidths=0.8)
            except Exception:
                pass

        ax.margins(x=0.02, y=0.02)
        ax.tick_params(direction='out', length=4)

    # 1) aggregated map
    fig1, ax1 = plt.subplots(figsize=(8.2, 6.2))
    _draw_base(fig1, ax1, Z, f'{agg.capitalize()} {losscol}',
               f'{agg.capitalize()} {losscol} in {xcol}–{ycol} space')
    fig1.savefig(f'{save_prefix}_{xcol}_{ycol}.png', bbox_inches='tight', dpi=200)
    plt.close(fig1)

    # 2) delta loss relative to best valid bin
    Zm = np.ma.masked_invalid(Z)
    if not Zm.mask.all():
        Zmin = Zm.min()
        dL = Zm - Zmin
        fig2, ax2 = plt.subplots(figsize=(8.2, 6.2))
        _draw_base(fig2, ax2, dL, 'Δ loss (relative to global min)',
                   f'Delta loss surface ({xcol}–{ycol})')
        fig2.savefig(f'{save_prefix}_{xcol}_{ycol}_delta.png', bbox_inches='tight', dpi=200)
        plt.close(fig2)

        # 3) gradient magnitude + quiver
        Zfill = Zm.filled(np.nanmedian(Zm))
        gY, gX = np.gradient(Zfill)  # note order (rows, cols)
        grad_mag = np.sqrt(gX**2 + gY**2)
        fig3, ax3 = plt.subplots(figsize=(8.2, 6.2))
        _draw_base(fig3, ax3, grad_mag, '|∇ loss|',
                   f'Gradient magnitude and direction ({xcol}–{ycol})')

        # quiver on coarse grid
        step = max(2, int(np.ceil(max(bx, by) / 20)))
        ax3.quiver(
            Xc[::step], Yc[::step],
            gX[::step, ::step].T, gY[::step, ::step].T,
            color='k', alpha=0.55, pivot='mid', linewidth=0.5, scale_units='xy', scale=1
        )
        fig3.savefig(f'{save_prefix}_{xcol}_{ycol}_grad.png', bbox_inches='tight', dpi=200)
        plt.close(fig3)

    return Z, xedges, yedges, N


# ---------- DELTA LOSS & GRADIENT VISUALS ----------
def plot_delta_and_gradient(Z, xedges, yedges,
                            save_prefix='GA/analysis/binned_loss',
                            quiver_step=3):
    """
    From a binned loss surface Z, plot:
      (a) ΔL = Z - Z_min (relative to global min over valid bins)
      (b) |∇L| magnitude and quiver of gradient (∂L/∂x, ∂L/∂y)
    """
    # Build centers
    Xc = 0.5*(xedges[:-1] + xedges[1:])
    Yc = 0.5*(yedges[:-1] + yedges[1:])

    # mask invalid
    Zm = np.ma.masked_invalid(Z)
    if Zm.mask.all():
        print("All bins invalid; nothing to plot.")
        return

    Zmin = Zm.min()
    dL = Zm - Zmin

    # (a) Delta loss map
    fig1, ax1 = plt.subplots(figsize=(8,6))
    im1 = ax1.pcolormesh(xedges, yedges, dL.T, shading='auto', cmap='magma')
    c1 = fig1.colorbar(im1, ax=ax1)
    c1.set_label('Δ loss (relative to global min)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Delta loss surface')
    plt.tight_layout()
    fig1.savefig(f'{save_prefix}_delta.png', bbox_inches='tight')
    plt.close(fig1)

    # (b) Gradient field (finite difference on masked array)
    # Fill NaNs for gradient calc but keep mask for display
    Zfill = Zm.filled(np.nanmedian(Zm))
    # central differences on grid of centers
    dZdx, dZdy = np.gradient(Zfill)
    grad_mag = np.sqrt(dZdx**2 + dZdy**2)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    im2 = ax2.pcolormesh(xedges, yedges, grad_mag.T, shading='auto', cmap='cubehelix')
    c2 = fig2.colorbar(im2, ax=ax2)
    c2.set_label('|∇ loss|')

    # quiver on a subsampled grid
    xs = Xc[::quiver_step]
    ys = Yc[::quiver_step]
    U = dZdx[::quiver_step, ::quiver_step].T
    V = dZdy[::quiver_step, ::quiver_step].T
    # scale arrows to be readable
    ax2.quiver(xs, ys, U, V, color='k', alpha=0.6, pivot='mid')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Gradient magnitude and direction')
    plt.tight_layout()
    fig2.savefig(f'{save_prefix}_grad.png', bbox_inches='tight')
    plt.close(fig2)

def plot_marginal_loss(df, param, losscol='fitness', bins=60,
                       agg='median', save_path='GA/analysis/marginal_loss.png'):
    """
    1D marginal: aggregated loss vs a single parameter, with sample counts.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = df[param].values
    z = df[losscol].values

    if agg == 'mean':
        stat = 'mean'
    elif agg == 'median':
        stat = 'median'
    elif agg == 'min':
        stat = np.min
    else:
        raise ValueError("agg must be 'mean' | 'median' | 'min'")

    # Bin centers
    counts, edges = np.histogram(x, bins=bins)
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, bins-1)

    # Aggregate per bin
    vals = [[] for _ in range(bins)]
    for ii, zz in zip(idx, z):
        vals[ii].append(zz)
    agg_vals = np.array([stat(v) if len(v) else np.nan for v in vals])
    centers = 0.5*(edges[:-1] + edges[1:])

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(centers, agg_vals, '-', lw=2)
    ax1.set_xlabel(param)
    ax1.set_ylabel(f'{agg} {losscol}')

    ax2 = ax1.twinx()
    ax2.bar(centers, counts[:-1], width=np.diff(edges), alpha=0.2, edgecolor='none')
    ax2.set_ylabel('samples / bin')

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
