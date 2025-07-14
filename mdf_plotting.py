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

# ---------------------------------------------------
# Global style for paper-quality figures
# ---------------------------------------------------

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
})

# ---------------------------------------------------
def ensure_dirs():
    """Ensure necessary directories exist."""
    os.makedirs('GA/loss', exist_ok=True)

# ---------------------------------------------------
def plot_sfr_history(bulge_dict,save_path='GA/SFR_history.png'):
    """
    Plot star formation rate (SFR) history vs Age for bulge models.
    bulge_dict: mapping of label -> model with inner.history.age and .sfr_abs
    """
    fig, ax = plt.subplots(figsize=(6,5))
    for label, model in bulge_dict.items():
        age_gyr = np.array(model.inner.history.age) / 1e9
        sfr = np.array(model.inner.history.sfr_abs)
        ax.plot(age_gyr, sfr, label=label)

    ax.set_xlabel('Age (Gyr)')
    ax.set_ylabel(r'SFR [$M_\odot\ \mathrm{yr}^{-1}$]')
    ax.set_xlim(0, np.max([np.max(np.array(m.inner.history.age)/1e9) for m in bulge_dict.values()]))
    ax.legend(frameon=False, fontsize='small')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig



def plot_walker_success_rate(walker_history, results_csv='GA/simulation_results.csv', 
                             threshold=0.1, loss_metric='wrmse', save_path='GA/walker_success_rate.png'):
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


def plot_multiple_success_thresholds(walker_history, results_csv='GA/simulation_results.csv', 
                                   thresholds=[0.05, 0.1, 0.2, 0.5], loss_metric='wrmse', 
                                   save_path='GA/walker_success_rates_multiple.png'):
    """
    Plot success rates for multiple thresholds on the same plot.
    """
    
    if not walker_history:
        print("Walker history data not available.")
        return None
    
    import pandas as pd
    results_df = pd.read_csv(results_csv)
    
    max_generations = max(len(history) for history in walker_history.values() if history)
    if max_generations == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
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


# ---------------------------------------------------
def plot_mass_evolution(bulge_dict,save_path='GA/Mass_age.png'):
    """
    Plot bulge mass (locked + gas) evolution vs Age.
    bulge_dict: mapping of label -> model with inner.history.m_locked, .m_gas_exp (or .m_gas)
    """
    fig, ax = plt.subplots(figsize=(6,5))
    for label, model in bulge_dict.items():
        age_gyr = np.array(model.inner.history.age) / 1e9
        m_locked = np.array(getattr(model.inner.history, 'm_locked', []))
        # fallback to m_gas_exp or m_gas
        m_gas = np.array(getattr(model.inner.history, 'm_gas_exp', getattr(model.inner.history, 'm_gas', [])))
        mass = m_locked + m_gas
        ax.plot(age_gyr, mass, label=label)

    ax.set_xlabel('Age (Gyr)')
    ax.set_ylabel(r'Bulge Mass [$M_\odot$]')
    ax.axhline(2e10, ls='--', color='k', label='Reference 2e10 $M_\odot$')
    ax.legend(frameon=False, fontsize='small')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig

# ---------------------------------------------------
def plot_alpha_histograms(obs_dict, model_dict, bins=25, save_path='GA/alpha_histograms.png'):
    """
    Plot histograms of alpha-element distributions for observation and models.
    obs_dict: {'[Mg/Fe]': array, ...}
    model_dict: {'label': [array_Mg, array_Si, array_Ca, array_Ti], ...}
    """
    elts = list(obs_dict.keys())
    n = len(elts)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6*ncols, 4*nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.3, hspace=0.4)

    for idx, elt in enumerate(elts):
        ax = fig.add_subplot(gs[idx])
        # observational distribution
        ax.hist(obs_dict[elt], bins=bins,
                histtype='stepfilled', alpha=0.3,
                color='C0', label='Obs')
        # model average
        Ys = [np.asarray(arr[idx], float) for arr in model_dict.values()]
        alpha_mod = np.nanmean(np.vstack(Ys), axis=0)
        ax.hist(alpha_mod, bins=bins,
                histtype='step', lw=2,
                color='C1', label='Model')
        ax.set_title(f'{elt} Distribution')
        ax.set_xlabel(elt)
        ax.set_ylabel('Count')
        ax.legend(frameon=False, fontsize='small')

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig

# ---------------------------------------------------
# Existing MDF and GA plotting functions (slightly tweaked for style)
# ---------------------------------------------------
def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None, save_path='GA/MDF_multiple_results.png'):
    """
    Plot all model MDFs, highlight the best model, and overlay data.
    """
    fig, ax = plt.subplots(figsize=(9,6))
    # Determine best model parameters
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    best_flag = False
    # plot curves
    for (x, y), label, res in zip(GalGA.mdf_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            best_x = x
            best_y = y

            ax.plot(x, y, color='C3', linewidth=2.5,
                    label='Best Model' if not best_flag else None)
            best_flag = True
        else:
            ax.plot(x, y, color='gray', alpha=0.01)

    # observational data
    ax.plot(feh, normalized_count, 'x', ms=8, color='k', label='Observational Data')
    ax.plot(best_x, best_y, color='C3', linewidth=2.5)
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Normalized Number Density')
    ax.set_xlim(-2, 1)
    ax.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_3d_scatter(x, y, z, color_metric, label, xlabel='sigma_2', ylabel='t_2', zlabel='infall_2'):
    """Plot 3D scatter plot with color indicating a specific metric.
    Two plots:
      - All data, color scaled [0, 1]
      - Only points with loss < 0.1, color scaled [0, 0.1]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x, y, z, color_metric = map(np.array, (x, y, z, color_metric))

    def make_plot(x_data, y_data, z_data, color_data, vmin, vmax, suffix):
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
        plt.savefig(f'GA/loss/{label}_loss_3d{suffix}.png', bbox_inches='tight')
        plt.close()

    # Full plot
    make_plot(x, y, z, color_metric, 0, 1, '')

    # Filtered plot for low loss
    mask = color_metric < 0.1
    if np.any(mask):
        make_plot(x[mask], y[mask], z[mask], color_metric[mask], 0, 0.1, '_lowloss')



def plot_2d_scatter(x, y, color_metric, label, xlabel='t_2', ylabel='infall_2'):
    """Plot 2D scatter plot with color indicating a specific metric.
    Two plots:
      - All data, color scaled [0, 1]
      - Only points with loss < 0.1, color scaled [0, 0.1]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x, y, color_metric = map(np.array, (x, y, color_metric))

    def make_plot(x_data, y_data, color_data, vmin, vmax, suffix):
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
        plt.savefig(f'GA/loss/{label}_loss_2d{suffix}.png', bbox_inches='tight')
        plt.close()

    # Full plot
    make_plot(x, y, color_metric, 0, 1, '')

    # Filtered plot for low loss
    mask = color_metric < 0.1
    if np.any(mask):
        make_plot(x[mask], y[mask], color_metric[mask], 0, 0.1, '_lowloss')





def plot_walker_history(walker_history, param_names, param_indices):
    """
    Plot the evolution of parameters for all walkers with median + spread.
    """
    if not walker_history:
        print("Walker history data not available. Skipping walker evolution plots.")
        return None

    os.makedirs("GA/loss", exist_ok=True)
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
        fig.savefig(f'GA/loss/walker_evolution_{param_name}.png', bbox_inches='tight')
        plt.close(fig)

    print("Generated walker evolution plots with clarity enhancements")
    return figs





def create_3d_animation(walker_history):
    """Create an animated 3D visualization of walker evolution"""
    if not walker_history:
        print("Walker history data not available. Skipping 3D animation.")
        return None
        
    # Get maximum number of generations
    num_generations = max(len(v) for v in walker_history.values()) if walker_history else 0
    if num_generations == 0:
        print("No generation data found. Skipping 3D animation.")
        return None
    
    # Initialize figure for 3D animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for walkers
    colors = plt.cm.viridis(np.linspace(0, 1, len(walker_history)))
    
    # Animation function
    def update(num):
        ax.clear()
        ax.set_xlabel("Generation")
        ax.set_ylabel("tmax_2")
        ax.set_zlabel("infall_2")
        ax.view_init(elev=20, azim=num)  # Rotate by 1 degree per frame
    
        for i, (walker_id, history) in enumerate(walker_history.items()):
            if not history:
                continue
            history = np.array(history)
            generations = np.arange(len(history))
            
            # Use correct indices for t_2 (7) and infall_2 (9)
            if num < num_generations:
                # During first rotation, show progressive evolution
                plot_up_to = min(num+1, len(history))
                ax.plot(generations[:plot_up_to], history[:plot_up_to, 7], history[:plot_up_to, 9], 
                        color=colors[i], alpha=0.7, label=f"Walker {i}")
            else:
                # Second rotation shows complete paths
                ax.plot(generations, history[:, 7], history[:, 9], 
                        color=colors[i], alpha=0.7, label=f"Walker {i}")
    
        ax.legend(loc="upper right", fontsize="small")
    
    # Create animation with two full rotations
    total_frames = 360 * 2  # Two full rotations at 1 degree per frame
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)
    
    # Save as GIF with lower frame rate
    gif_path = "GA/loss/walker_evolution_3D.gif"
    ani.save(gif_path, writer="pillow", fps=6)  # Lower fps for slower rotation
    plt.close()
    
    print(f"Generated 3D animation: {gif_path}")
    return ani


def plot_walker_loss_history(walker_history, results_csv='GA/simulation_results.csv', loss_metric='wrmse'):
    """
    Plot the evolution of loss for all walkers with median and IQR shading.
    Mirrors the style of plot_walker_history.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("GA/loss", exist_ok=True)

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







def plot_mutation_info_3D(GA, population, fitnesses, base_sigma=1.0, mutation_type='gaussian'):
    #print('Starting plot...')

    # Calculate losses
    losses = [fit[0] for fit in fitnesses]
    max_loss = max(losses)
    min_loss = min(losses)

    # Update global min and max loss
    if GA.global_min_loss is None or min_loss < GA.global_min_loss:
        GA.global_min_loss = min_loss
    if GA.global_max_loss is None or max_loss > GA.global_max_loss:
        GA.global_max_loss = max_loss

    threshold = np.median(losses)

    # Identify successful and unsuccessful individuals
    successful_inds = []
    unsuccessful_inds = []
    for ind, fit in zip(population, fitnesses):
        if fit[0] <= threshold:
            successful_inds.append((ind, fit[0]))
        else:
            unsuccessful_inds.append((ind, fit[0]))

    # Number of genes
    gene_names = ['sigma_2', 't_2', 'infall_2']
    num_genes = len(gene_names)

    # Collect data for accumulation
    # Successful individuals
    gene_values_successful = []
    losses_successful = []
    for ind, loss in successful_inds:
        genes = ind[:num_genes]
        gene_values_successful.append(genes)
        losses_successful.append(loss)
    GA.all_gene_values_successful.extend(gene_values_successful)
    GA.all_losses_successful.extend(losses_successful)

    # Unsuccessful individuals
    gene_values_unsuccessful = []
    losses_unsuccessful = []
    for ind, loss in unsuccessful_inds:
        genes = ind[:num_genes]
        gene_values_unsuccessful.append(genes)
        losses_unsuccessful.append(loss)
    GA.all_gene_values_unsuccessful.extend(gene_values_unsuccessful)
    GA.all_losses_unsuccessful.extend(losses_unsuccessful)

    # Store gene bounds
    current_gene_bounds = {
        'xmin': GA.sigma_2_min,
        'xmax': GA.sigma_2_max,
        'ymin': GA.t_2_min,
        'ymax': GA.t_2_max,
        'zmin': GA.infall_2_min,
        'zmax': GA.infall_2_max
    }
    GA.gene_bounds.append(current_gene_bounds)

    # At the end of all generations, plot the accumulated data
    if GA.gen + 1 == GA.num_generations:
        # Prepare the colormap for losses in log space for better contrast
        all_losses = GA.all_losses_successful + GA.all_losses_unsuccessful
        log_all = [np.log10(max(l, 1e-10)) for l in all_losses]
        min_loss = min(log_all)
        max_loss = max(log_all)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

        losses_successful_norm = [
            (np.log10(max(l, 1e-10)) - min_loss) / loss_range for l in GA.all_losses_successful
        ]
        losses_unsuccessful_norm = [
            (np.log10(max(l, 1e-10)) - min_loss) / loss_range for l in GA.all_losses_unsuccessful
        ]

        # Create colormap (darker color for lower loss)
        succmap = cm.get_cmap('YlGn')  # Reverse Greys for darker color at lower values
        unsuccmap = cm.get_cmap('Reds_r')  # Reverse Greys for darker color at lower values
        
        colors_successful = [succmap(loss_norm) for loss_norm in losses_successful_norm]
        colors_unsuccessful = [unsuccmap(loss_norm) for loss_norm in losses_unsuccessful_norm]


        # Prepare the colormap for bounding boxes
        num_generations = GA.num_generations
        bbox_cmap = cm.get_cmap('Greys')
        colors_bounding_boxes = [bbox_cmap(i / (num_generations - 1)) for i in range(num_generations)]

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot successful individuals
        if len(GA.all_gene_values_successful) > 0:
            gene_values_successful = np.array(GA.all_gene_values_successful)
            ax.scatter(
                gene_values_successful[:, 0],
                gene_values_successful[:, 1],
                gene_values_successful[:, 2],
                color=colors_successful,
                label='Successful',
                alpha=0.6,
                marker='o'
            )

        # Plot unsuccessful individuals
        if len(GA.all_gene_values_unsuccessful) > 0:
            gene_values_unsuccessful = np.array(GA.all_gene_values_unsuccessful)
            ax.scatter(
                gene_values_unsuccessful[:, 0],
                gene_values_unsuccessful[:, 1],
                gene_values_unsuccessful[:, 2],
                color=colors_unsuccessful,
                label='Unsuccessful',
                alpha=0.6,
                marker='^'
            )

        # Define the edges of the bounding box
        edges = [
            [0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5],
            [2, 3], [2, 6],
            [3, 7],
            [4, 5], [4, 6],
            [5, 7],
            [6, 7]
        ]

        # Plot the bounding boxes
        for i, gene_bound in enumerate(GA.gene_bounds):
            color = colors_bounding_boxes[i]
            # Extract bounds
            xmin = gene_bound['xmin']
            xmax = gene_bound['xmax']
            ymin = gene_bound['ymin']
            ymax = gene_bound['ymax']
            zmin = gene_bound['zmin']
            zmax = gene_bound['zmax']

            # Define the corners of the bounding box
            corners = np.array([
                [xmin, ymin, zmin],
                [xmin, ymin, zmax],
                [xmin, ymax, zmin],
                [xmin, ymax, zmax],
                [xmax, ymin, zmin],
                [xmax, ymin, zmax],
                [xmax, ymax, zmin],
                [xmax, ymax, zmax]
            ])

            # Plot the edges of the bounding box
            for edge in edges:
                x = [corners[edge[0], 0], corners[edge[1], 0]]
                y = [corners[edge[0], 1], corners[edge[1], 1]]
                z = [corners[edge[0], 2], corners[edge[1], 2]]
                ax.plot(x, y, z, color=color, linestyle='--', alpha=0.5)

        # Customize plot
        ax.set_xlabel(gene_names[0])
        ax.set_ylabel(gene_names[1])
        ax.set_zlabel(gene_names[2])
        ax.legend()

        # Adjust the viewing angle for better visualization
        ax.view_init(elev=20., azim=-35)

        plt.tight_layout()
        plt.savefig('GA/MDF_individuals_3D.png', bbox_inches='tight')
        ##plt.show()
        print('...plot made!')



def plot_mutation_info_2d(GA, population, fitnesses, base_sigma=1.0, mutation_type='gaussian'):
    # Calculate losses
    losses = [fit[0] for fit in fitnesses]
    max_loss = max(losses)
    min_loss = min(losses)

    # Update global min and max loss
    if GA.global_min_loss is None or min_loss < GA.global_min_loss:
        GA.global_min_loss = min_loss
    if GA.global_max_loss is None or max_loss > GA.global_max_loss:
        GA.global_max_loss = max_loss

    threshold = np.median(losses)

    # Identify successful and unsuccessful individuals
    successful_inds = []
    unsuccessful_inds = []
    for ind, fit in zip(population, fitnesses):
        if fit[0] <= threshold:
            successful_inds.append((ind, fit[0]))
        else:
            unsuccessful_inds.append((ind, fit[0]))

    # Number of genes (excluding sigma)
    gene_names = ['t_2', 'infall_2']
    num_genes = len(gene_names)

    # Collect data for accumulation
    # Successful individuals
    gene_values_successful = []
    losses_successful = []
    for ind, loss in successful_inds:
        genes = ind[1:num_genes+1]  # Only take `t_2` and `infall_2`
        gene_values_successful.append(genes)
        losses_successful.append(loss)
    GA.all_gene_values_successful.extend(gene_values_successful)
    GA.all_losses_successful.extend(losses_successful)

    # Unsuccessful individuals
    gene_values_unsuccessful = []
    losses_unsuccessful = []
    for ind, loss in unsuccessful_inds:
        genes = ind[1:num_genes+1]  # Only take `t_2` and `infall_2`
        gene_values_unsuccessful.append(genes)
        losses_unsuccessful.append(loss)
    GA.all_gene_values_unsuccessful.extend(gene_values_unsuccessful)
    GA.all_losses_unsuccessful.extend(losses_unsuccessful)

    # Store gene bounds
    current_gene_bounds = {
        'xmin': GA.t_2_min,
        'xmax': GA.t_2_max,
        'ymin': GA.infall_2_min,
        'ymax': GA.infall_2_max
    }
    GA.gene_bounds.append(current_gene_bounds)

    # At the end of all generations, plot the accumulated data
    if GA.gen + 1 == GA.num_generations:
        # Prepare the colormap for losses using log scale
        all_losses = GA.all_losses_successful + GA.all_losses_unsuccessful
        log_all = [np.log10(max(l, 1e-10)) for l in all_losses]
        min_loss = min(log_all)
        max_loss = max(log_all)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

        losses_successful_norm = [
            (np.log10(max(l, 1e-10)) - min_loss) / loss_range for l in GA.all_losses_successful
        ]
        losses_unsuccessful_norm = [
            (np.log10(max(l, 1e-10)) - min_loss) / loss_range for l in GA.all_losses_unsuccessful
        ]

        # Create colormaps
        succmap = cm.get_cmap('YlGn')
        unsuccmap = cm.get_cmap('Reds_r')

        colors_successful = [succmap(loss_norm) for loss_norm in losses_successful_norm]
        colors_unsuccessful = [unsuccmap(loss_norm) for loss_norm in losses_unsuccessful_norm]

        # Prepare the colormap for bounding boxes
        num_generations = GA.num_generations
        bbox_cmap = cm.get_cmap('Greys')
        colors_bounding_boxes = [bbox_cmap(i / (num_generations - 1)) for i in range(num_generations)]

        # Create a 2D scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot successful individuals
        if len(GA.all_gene_values_successful) > 0:
            gene_values_successful = np.array(GA.all_gene_values_successful)
            ax.scatter(
                gene_values_successful[:, 0],  # t_2
                gene_values_successful[:, 1],  # infall_2
                color=colors_successful,
                label='Successful',
                alpha=0.6,
                marker='o'
            )

        # Plot unsuccessful individuals
        if len(GA.all_gene_values_unsuccessful) > 0:
            gene_values_unsuccessful = np.array(GA.all_gene_values_unsuccessful)
            ax.scatter(
                gene_values_unsuccessful[:, 0],  # t_2
                gene_values_unsuccessful[:, 1],  # infall_2
                color=colors_unsuccessful,
                label='Unsuccessful',
                alpha=0.6,
                marker='^'
            )

        # Plot the bounding boxes
        for i, gene_bound in enumerate(GA.gene_bounds):
            color = colors_bounding_boxes[i]
            xmin, xmax = gene_bound['xmin'], gene_bound['xmax']
            ymin, ymax = gene_bound['ymin'], gene_bound['ymax']

            # Plot the bounding box as a rectangle
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       edgecolor=color, fill=False, linestyle='--', alpha=0.5))

        # Customize plot
        ax.set_xlabel(gene_names[0])
        ax.set_ylabel(gene_names[1])
        #ax.legend()
        plt.tight_layout()
        plt.savefig('GA/MDF_individuals_2D.png', bbox_inches='tight')
        print('...2D plot made!')


def plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=None, save_path='GA/Four_Panel_Alpha.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    import os

    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    observational_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.4, left=0.1)  # Add left margin for y-axis labels

    for idx, (element, obs_data) in enumerate(zip(element_names, observational_data)):
        row, col = divmod(idx, 2)
        ax_main = axes[row, col]

        # Position for marginal KDE plot (DO NOT use sharey=ax_main)
        rect = ax_main.get_position()
        ax_kde = fig.add_axes([rect.x1 + 0.0001, rect.y0, 0.1, rect.height])

        # Ensure y-axis is visible on main plots
        ax_main.tick_params(axis='y', which='both', left=True, labelleft=True, right=False, labelright=False)
        ax_main.yaxis.set_ticks_position('left')
        ax_main.yaxis.set_label_position('left')
        ax_main.spines['left'].set_visible(True)  # Force left spine visible

        # Draw all model curves
        for alpha_arrs, _, res in zip(GalGA.alpha_data, GalGA.labels, GalGA.results):
            params = (res[5], res[7], res[9])
            is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
            if idx < len(alpha_arrs):
                x_data, y_data = np.array(alpha_arrs[idx][0]), np.array(alpha_arrs[idx][1])
                if is_best:
                    ax_main.plot(x_data, y_data, color="red", linewidth=2.5, zorder=3)
                else:
                    ax_main.plot(x_data, y_data, color='gray', alpha=0.01, linewidth=1)

        # Clean obs data
        obs_data = np.where((obs_data >= -2.0) & (obs_data <= 2.0), obs_data, np.nan)

        # Color by average of all four elements
        color_array = (Mg_Fe + Si_Fe + Ca_Fe + Ti_Fe) / 4
        mask = np.isfinite(Fe_H) & np.isfinite(obs_data) & np.isfinite(color_array)
        if np.sum(mask) > 10:
            x = Fe_H[mask]
            y = obs_data[mask]
            z = color_array[mask]
            idxs = np.argsort(z)
            ax_main.scatter(x[idxs], y[idxs], c=z[idxs], cmap='viridis', s=20, zorder=2, edgecolor='none')

        # KDEs
        joint_mask = np.isfinite(obs_data) & np.isfinite(Fe_H)
        y_vals = np.linspace(-0.8, 1.0, 200)

        if np.sum(joint_mask) > 2:
            kde_obs_y = gaussian_kde(obs_data[joint_mask])
            kde_y = kde_obs_y(y_vals)
            kde_y /= np.max(kde_y)
            ax_kde.plot(kde_y, y_vals, color='darkblue')
            ax_kde.fill_betweenx(y_vals, 0, kde_y, color='blue', alpha=0.3)

        best_y_model = None
        for alpha_arrs, _, res in zip(GalGA.alpha_data, GalGA.labels, GalGA.results):
            params = (res[5], res[7], res[9])
            if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)) and idx < len(alpha_arrs):
                best_y_model = np.array(alpha_arrs[idx][1])
                break

        if best_y_model is not None:
            kde_model_y = gaussian_kde(best_y_model[np.isfinite(best_y_model)])
            kde_model = kde_model_y(y_vals)
            kde_model /= np.max(kde_model)
            ax_kde.plot(kde_model, y_vals, color='darkred', linestyle='--')
            ax_kde.fill_betweenx(y_vals, 0, kde_model, color='red', alpha=0.2)

        # Set limits and labels for main plot
        ax_main.set_xlim(-2, 1)
        ax_main.set_ylim(-0.8, 0.8)
        ax_main.set_xlabel("[Fe/H]")
        ax_main.set_ylabel(f"[{element}/Fe]")
        ax_main.text(-1.8, 0.85, element, fontsize=14, weight='bold')

        # Move top row x-axis to top
        if row == 0:
            ax_main.xaxis.set_ticks_position('top')
            ax_main.xaxis.set_label_position('top')

        # Clean KDE axis - no numbers on histogram
        ax_kde.set_xticks([])
        ax_kde.set_yticks([])
        ax_kde.set_xlabel('')
        ax_kde.set_ylabel('')
        ax_kde.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_kde.set_ylim(-0.8, 1.0)  # Match main plot y-limits
        ax_kde.set_xlim(0.0, 1.0)  # Match main plot y-limits

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Density-enhanced four-panel alpha plot saved to {save_path}")








def plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=None, save_path='GA/Age_FeH_detailed_results.png', n_bins=10):
    """
    Enhanced Age vs [Fe/H] plot with:
    - Model lines (gray + red for best)
    - Polynomial fits to Joyce & Bensby
    - Faint fillbetween for difference
    - Marginal KDE histogram on the right
    - Binned observational data as lines
    - Axes: Age on X, [Fe/H] on Y
    - Improved fit metrics that better reflect visual agreement
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import UnivariateSpline, interp1d
    from scipy.stats import gaussian_kde, binned_statistic, ks_2samp
    import os
    
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age data available for plotting")
        return None
    
    # Create main figure with proper layout
    fig = plt.figure(figsize=(18, 10))
    
    # Create gridspec for proper alignment
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.0, 
                          left=0.08, right=0.95, top=0.95, bottom=0.1)
    
    ax_main = fig.add_subplot(gs[0])
    ax_kde = fig.add_subplot(gs[1], sharey=ax_main)
    
    # Ensure array typing for safe masking
    Fe_H = np.asarray(Fe_H)
    age_Joyce = np.asarray(age_Joyce)
    age_Bensby = np.asarray(age_Bensby)
    
    # Determine best model
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
    
    best_plotted = False
    best_model_feh = None
    best_model_age_gyr = None
    
    # Calculate average spacing of real data for model interpolation
    all_real_ages = np.concatenate([age_Joyce[np.isfinite(age_Joyce)], 
                                   age_Bensby[np.isfinite(age_Bensby)]])
    if len(all_real_ages) > 1:
        sorted_ages = np.sort(all_real_ages)
        avg_spacing = np.mean(np.diff(sorted_ages))
        # Create interpolation grid with similar spacing
        age_interp_grid = np.arange(0, 14 + avg_spacing, avg_spacing)
    else:
        age_interp_grid = np.linspace(0, 14, 100)
    
    # Plot model lines
    for age_data, label, res in zip(GalGA.age_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        
        x_age_raw, y_feh = age_data
        age_gyr = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
        
        if is_best and not best_plotted:
            # Interpolate best model to match real data spacing
            if len(age_gyr) > 1 and len(y_feh) > 1:
                f_best = interp1d(age_gyr, y_feh, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                best_model_age_gyr = age_interp_grid
                best_model_feh = f_best(age_interp_grid)
                
                label_lines = [f"• {p.strip()}" for p in label.split(',')]
                pretty_label = "Best model"
                ax_main.plot(best_model_age_gyr, best_model_feh, color="red", linewidth=2, zorder=3, label=pretty_label)
            else:
                ax_main.plot(age_gyr, y_feh, color="red", linewidth=2, zorder=3)
                best_model_feh = np.array(y_feh)
                best_model_age_gyr = age_gyr
            best_plotted = True
        else:
            ax_main.plot(age_gyr, y_feh, color='gray', alpha=0.01, linewidth=1, zorder=1)
    
    # Create age bins for observational data
    age_bins = np.linspace(0, 14, n_bins + 1)
    bin_centers = (age_bins[:-1] + age_bins[1:]) / 2
    



    # Scatter real data (raw points)
    ax_main.scatter(age_Joyce, Fe_H, marker='*', s=50, color='blue', 
                   alpha=0.6, label='Joyce et al. (raw)', zorder=2)
    ax_main.scatter(age_Bensby, Fe_H, marker='^', s=50, color='orange', 
                   alpha=0.6, label='Bensby et al. (raw)', zorder=2)
    




    
    # Bin Bensby data
    mask_bensby = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    bin_means_bensby = None
    bin_stds_bensby = None
    bin_counts_bensby = None

    # Bin Joyce data
    mask_joyce = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    bin_means_joyce = None
    bin_stds_joyce = None
    bin_counts_joyce = None
   
    bin_means_joyce, _, _ = binned_statistic(age_Joyce[mask_joyce], Fe_H[mask_joyce], 
                                           statistic='mean', bins=age_bins)
    bin_stds_joyce, _, _ = binned_statistic(age_Joyce[mask_joyce], Fe_H[mask_joyce], 
                                          statistic='std', bins=age_bins)
    bin_counts_joyce, _, _ = binned_statistic(age_Joyce[mask_joyce], Fe_H[mask_joyce], 
                                            statistic='count', bins=age_bins)


    bin_means_bensby, _, _ = binned_statistic(age_Bensby[mask_bensby], Fe_H[mask_bensby], 
                                            statistic='mean', bins=age_bins)
    bin_stds_bensby, _, _ = binned_statistic(age_Bensby[mask_bensby], Fe_H[mask_bensby], 
                                           statistic='std', bins=age_bins)
    bin_counts_bensby, _, _ = binned_statistic(age_Bensby[mask_bensby], Fe_H[mask_bensby], 
                                             statistic='count', bins=age_bins)
    





    # Calculate IMPROVED metrics between best model and binned data
    metrics_text = ""

    # Find common age range for fair comparison
    min_age = max(0, np.min(best_model_age_gyr))
    max_age = min(13, np.max(best_model_age_gyr))
    common_age_mask = (bin_centers >= min_age) & (bin_centers <= max_age)

    # Interpolate best model to bin centers
    f_model = interp1d(best_model_age_gyr, best_model_feh, kind='linear',
                      bounds_error=False, fill_value=np.nan)
    model_at_bins = f_model(bin_centers)
    
    # Calculate multiple metrics for Joyce data
    valid_joyce_bins = (np.isfinite(bin_means_joyce) & 
                      np.isfinite(model_at_bins) & 
                      common_age_mask & 
                      (bin_counts_joyce > 2))  # Require at least 3 points per bin
    
    joyce_model_vals = model_at_bins[valid_joyce_bins]
    joyce_obs_vals = bin_means_joyce[valid_joyce_bins]
    joyce_weights = np.sqrt(bin_counts_joyce[valid_joyce_bins])  # Weight by sqrt(N)
    joyce_uncertainties = bin_stds_joyce[valid_joyce_bins] / np.sqrt(bin_counts_joyce[valid_joyce_bins])
    
    # Weighted RMSE (accounts for uncertainties)
    joyce_weighted_rmse = np.sqrt(np.average((joyce_model_vals - joyce_obs_vals)**2, 
                                           weights=joyce_weights))
    
    # MAE (less sensitive to outliers)
    joyce_mae = np.mean(np.abs(joyce_model_vals - joyce_obs_vals))
    
    # Weighted MAE
    joyce_weighted_mae = np.average(np.abs(joyce_model_vals - joyce_obs_vals), 
                                  weights=joyce_weights)
    
    # Chi-squared like metric
    joyce_chi2 = np.mean(((joyce_model_vals - joyce_obs_vals) / 
                        np.maximum(joyce_uncertainties, 0.1))**2)
    
    metrics_text += f"Joyce Weighted RMSE: {joyce_weighted_rmse:.3f}\n"
    metrics_text += f"Joyce MAE: {joyce_mae:.3f}\n"
    metrics_text += f"Joyce χ²/dof: {joyce_chi2:.3f}\n"

    # Calculate same metrics for Bensby data
    valid_bensby_bins = (np.isfinite(bin_means_bensby) & 
                       np.isfinite(model_at_bins) & 
                       common_age_mask & 
                       (bin_counts_bensby > 2))
    
    bensby_model_vals = model_at_bins[valid_bensby_bins]
    bensby_obs_vals = bin_means_bensby[valid_bensby_bins]
    bensby_weights = np.sqrt(bin_counts_bensby[valid_bensby_bins])
    bensby_uncertainties = bin_stds_bensby[valid_bensby_bins] / np.sqrt(bin_counts_bensby[valid_bensby_bins])
    
    # Same metrics for Bensby
    bensby_weighted_rmse = np.sqrt(np.average((bensby_model_vals - bensby_obs_vals)**2, 
                                            weights=bensby_weights))
    bensby_mae = np.mean(np.abs(bensby_model_vals - bensby_obs_vals))
    bensby_weighted_mae = np.average(np.abs(bensby_model_vals - bensby_obs_vals), 
                                   weights=bensby_weights)
    bensby_chi2 = np.mean(((bensby_model_vals - bensby_obs_vals) / 
                         np.maximum(bensby_uncertainties, 0.1))**2)
    
    metrics_text += f"Bensby Weighted RMSE: {bensby_weighted_rmse:.3f}\n"
    metrics_text += f"Bensby MAE: {bensby_mae:.3f}\n"
    metrics_text += f"Bensby χ²/dof: {bensby_chi2:.3f}\n"

    # Add debugging info
    metrics_text += f"\nValid Joyce bins: {np.sum(valid_joyce_bins)}\n"
    metrics_text += f"Valid Bensby bins: {np.sum(valid_bensby_bins)}\n"
    metrics_text += f"Age range: {min_age:.1f}-{max_age:.1f} Gyr"







    # Plot binned Bensby data as line with error bars
    valid_bensby = np.isfinite(bin_means_bensby) & (bin_counts_bensby > 0)
    ax_main.plot(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                color='orange', linewidth=3, linestyle='-', 
                label=f"Bensby (χ²/dof: {bensby_chi2:.3f})", zorder=5)
    ax_main.errorbar(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                    yerr=bin_stds_bensby[valid_bensby], 
                    color='orange', alpha=0.3, capsize=3, zorder=4)
    
    
    # Plot binned Joyce data as line with error bars
    valid_joyce = np.isfinite(bin_means_joyce) & (bin_counts_joyce > 0)
    ax_main.plot(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                color='blue', linewidth=3, linestyle='-', 
                label=f"Joyce (χ²/dof: {joyce_chi2:.3f})", zorder=5)
    ax_main.errorbar(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                    yerr=bin_stds_joyce[valid_joyce], 
                    color='blue', alpha=0.3, capsize=3, zorder=4)







    # Polynomial fits (degree=3) - keeping your original spline fits
    if np.sum(mask_joyce) > 3 and np.sum(mask_bensby) > 3:
        sort_J = np.argsort(age_Joyce[mask_joyce])
        sort_B = np.argsort(age_Bensby[mask_bensby])
        
        s_joyce = UnivariateSpline(age_Joyce[mask_joyce][sort_J], Fe_H[mask_joyce][sort_J], k=3, s=0.5)
        s_bensby = UnivariateSpline(age_Bensby[mask_bensby][sort_B], Fe_H[mask_bensby][sort_B], k=3, s=0.5)
        
        x_vals = np.linspace(0, 14, 100)
        y_joyce = s_joyce(x_vals)
        y_bensby = s_bensby(x_vals)
        
        ax_main.plot(x_vals, y_joyce, color='blue', linestyle='--', lw=2, zorder=4, alpha=0.7)
        ax_main.plot(x_vals, y_bensby, color='orange', linestyle='--', lw=2, zorder=4, alpha=0.7)
        
        if best_model_feh is not None:
            # Interpolate best model to match x_vals length
            f_model = interp1d(best_model_age_gyr, best_model_feh, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
            y_model_interp = f_model(x_vals)
            
            ax_main.fill_between(x_vals, y_joyce, y_model_interp, color='purple', alpha=0.1, zorder=0)
            ax_main.fill_between(x_vals, y_model_interp, y_bensby, color='purple', alpha=0.1, zorder=0)
    
    # Create KDE plots on the right margin
    feh_vals = np.linspace(-2, 1, 200)
    
    # KDE for Joyce data
    mask_joyce_kde = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    joyce_feh_data = None
    if np.sum(mask_joyce_kde) > 2:
        joyce_feh_data = Fe_H[mask_joyce_kde]
        kde_joyce = gaussian_kde(joyce_feh_data)
        kde_j = kde_joyce(feh_vals)
        kde_j_norm = kde_j / np.max(kde_j) if np.max(kde_j) > 0 else kde_j
        ax_kde.plot(kde_j_norm, feh_vals, color='darkblue', linewidth=4, label='Joyce')
        ax_kde.fill_betweenx(feh_vals, 0, kde_j_norm, color='blue', alpha=0.3)
    

    # KDE for best model - use MDF data instead of age-metallicity track
    model_feh_data = None
    if hasattr(GalGA, 'mdf_data') and len(GalGA.mdf_data) > 0:
        # Find the MDF data for the best model
        for mdf_data, res in zip(GalGA.mdf_data, GalGA.results):
            params = (res[5], res[7], res[9])  # sigma_2, t_2, infall_2
            is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
            if is_best:
                mdf_x, mdf_y = mdf_data
                # Convert MDF to sample points for KDE (sample according to MDF weights)
                mdf_x = np.array(mdf_x)
                mdf_y = np.array(mdf_y)
                valid_mdf = np.isfinite(mdf_x) & np.isfinite(mdf_y) & (mdf_y > 0)
                if np.sum(valid_mdf) > 0:
                    mdf_x_valid = mdf_x[valid_mdf]
                    mdf_y_valid = mdf_y[valid_mdf]
                    # Sample from MDF to create synthetic data points for KDE
                    n_samples = min(1000, int(np.sum(mdf_y_valid) * 1000))
                    if n_samples > 10:
                        # Create samples weighted by MDF
                        samples = np.random.choice(mdf_x_valid, size=n_samples, 
                                                 p=mdf_y_valid/np.sum(mdf_y_valid))
                        model_feh_data = samples
                break
    
    if model_feh_data is not None and len(model_feh_data) > 2:
        kde_model = gaussian_kde(model_feh_data)
        kde_m = kde_model(feh_vals)
        kde_m_norm = kde_m / np.max(kde_m) if np.max(kde_m) > 0 else kde_m
        ax_kde.plot(kde_m_norm, feh_vals, color='darkred', linestyle='--', linewidth=4, label='Best Model')
        ax_kde.fill_betweenx(feh_vals, 0, kde_m_norm, color='red', alpha=0.3)


    # Calculate KS statistics and add to histogram plot
    ks_text = ""
    if joyce_feh_data is not None and model_feh_data is not None:
        ks_stat_joyce, _ = ks_2samp(joyce_feh_data, model_feh_data)
        ks_text += f"KS - Best fit vs. Observational: {ks_stat_joyce:.3f}\n"

    # Add KS text to histogram plot (top left)
    if ks_text:
        ax_kde.text(0.2, 0.95, ks_text.strip(), transform=ax_kde.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set reasonable limits
    ax_kde.set_xlim(0, 1.2)
    
    # Axis control for main plot
    ax_main.set_xlim(0, 14)
    ax_main.set_ylim(-2, 1)
    ax_main.set_xlabel("Age (Gyr)", fontsize=16)
    ax_main.set_ylabel("[Fe/H]", fontsize=16)
    
    # Create legend with metrics info in lower left with faint white background
    #if metrics_text:
    #    # Add metrics as separate text
    #    ax_main.text(0.02, 0.02, metrics_text.strip(), transform=ax_main.transAxes,
    #                fontsize=8, verticalalignment='bottom', fontfamily='monospace',
    #                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Main legend
    legend = ax_main.legend(loc="lower left", bbox_to_anchor=(0., 0.), frameon=True, 
                          fontsize=10, facecolor='white', edgecolor='gray')
    legend.get_frame().set_alpha(0.8)
    
    # Clean up KDE axis
    ax_kde.set_xticks([])
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('')
    ax_kde.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, 
                       right=False, labelright=False)
    ax_kde.spines['left'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Enhanced age-metallicity plot with improved metrics saved to {save_path}")
    return fig






def plot_omni_info_figure(GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, 
                          feh_mdf, normalized_count_mdf, results_df=None, 
                          save_path='GA/Omni_Info_Figure.png'):
    """
    Create a comprehensive dashboard showing the best-fit model parameters and performance
    across all key observational diagnostics.
    
    Parameters:
    -----------
    GalGA : Galactic Evolution GA object
    Fe_H, age_Joyce, age_Bensby : observational age-metallicity data
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe : observational alpha element data
    feh_mdf, normalized_count_mdf : observational MDF data
    results_df : DataFrame with model results
    save_path : output file path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline, interp1d
    from scipy.stats import gaussian_kde, binned_statistic
    from matplotlib.gridspec import GridSpec
    import os
    
    # Ensure we have the required data
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age data available for plotting")
        return None
        
    if not hasattr(GalGA, 'mdf_data') or len(GalGA.mdf_data) == 0:
        print("No MDF data available for plotting")
        return None
        
    if not hasattr(GalGA, 'alpha_data') or len(GalGA.alpha_data) == 0:
        print("No alpha data available for plotting")
        return None
    
    # Determine best model parameters
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
        best_row = bm
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
        # Create a mock row for parameter display
        col_names = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
            'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber',
            'cosine', 'log_cosh', 'fitness'
        ]
        best_row = dict(zip(col_names, r))
    
    # Create comprehensive figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3,
                  left=0.05, right=0.98, top=0.95, bottom=0.05)
    
    # Add main title
    fig.suptitle('Best-Fit Galactic Chemical Evolution Model Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # =====================================================
    # PANEL 1: MODEL PARAMETERS (Top Left)
    # =====================================================
    ax_params = fig.add_subplot(gs[0, :2])
    ax_params.axis('off')
    
    # Create parameter text
    param_text = "BEST-FIT MODEL PARAMETERS\n" + "="*35 + "\n"
    param_text += f"σ₂ (second infall dispersion): {best_row['sigma_2']:.1f} pc\n"
    param_text += f"t₁ (first infall time): {best_row['t_1']:.3f} Gyr\n"
    param_text += f"t₂ (second infall time): {best_row['t_2']:.3f} Gyr\n"
    param_text += f"τ₁ (first infall timescale): {best_row['infall_1']:.3f} Gyr\n"
    param_text += f"τ₂ (second infall timescale): {best_row['infall_2']:.3f} Gyr\n"
    param_text += f"SFE (star formation efficiency): {best_row['sfe']:.5f}\n"
    param_text += f"ΔSFE (SFE change at t₂): {best_row['delta_sfe']:.3f}\n"
    param_text += f"IMF upper limit: {best_row['imf_upper']:.1f} M☉\n"
    param_text += f"Galaxy mass: {best_row['mgal']:.2e} M☉\n"
    param_text += f"SN Ia rate: {best_row['nb']:.2e} per M☉\n"
    
    ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # =====================================================
    # PANEL 2: FIT QUALITY METRICS (Top Middle)
    # =====================================================
    ax_metrics = fig.add_subplot(gs[0, 2:4])
    ax_metrics.axis('off')
    
    # Create metrics text
    metrics_text = "FIT QUALITY METRICS\n" + "="*25 + "\n"
    metrics_text += f"Primary Loss (Fitness): {best_row['fitness']:.4f}\n"
    metrics_text += f"WRMSE: {best_row['wrmse']:.4f}\n"
    metrics_text += f"MAE: {best_row['mae']:.4f}\n"
    metrics_text += f"Huber Loss: {best_row['huber']:.4f}\n"
    metrics_text += f"Cosine Similarity: {best_row['cosine']:.4f}\n"
    metrics_text += f"KS Distance: {best_row['ks']:.4f}\n"
    metrics_text += f"Ensemble Metric: {best_row['ensemble']:.4f}\n"
    
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # =====================================================
    # PANEL 3: MODEL SUMMARY (Top Right)
    # =====================================================
    ax_summary = fig.add_subplot(gs[0, 4:])
    ax_summary.axis('off')
    
    # Create model summary
    summary_text = "MODEL INTERPRETATION\n" + "="*25 + "\n"
    
    # Interpret the parameters
    if best_row['t_2'] < 2.0:
        infall_interp = "Early second infall"
    elif best_row['t_2'] < 8.0:
        infall_interp = "Mid-age second infall"
    else:
        infall_interp = "Late second infall"
        
    if best_row['delta_sfe'] > 0:
        sfe_interp = "SFE increases at second infall"
    elif best_row['delta_sfe'] < -0.01:
        sfe_interp = "SFE decreases at second infall"
    else:
        sfe_interp = "SFE unchanged at second infall"
        
    summary_text += f"• {infall_interp}\n"
    summary_text += f"• {sfe_interp}\n"
    summary_text += f"• First infall: τ = {best_row['infall_1']:.2f} Gyr\n"
    summary_text += f"• Second infall: τ = {best_row['infall_2']:.2f} Gyr\n"
    
    if best_row['infall_2'] < best_row['infall_1']:
        summary_text += "• Faster second infall\n"
    else:
        summary_text += "• Slower second infall\n"
        
    summary_text += f"• Total models evaluated: {len(GalGA.results)}\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # =====================================================
    # PANEL 4: METALLICITY DISTRIBUTION FUNCTION
    # =====================================================
    ax_mdf = fig.add_subplot(gs[1, :3])
    
    # Find best MDF model
    best_mdf_x = None
    best_mdf_y = None
    for mdf_data, res in zip(GalGA.mdf_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            best_mdf_x, best_mdf_y = mdf_data
            break
    
    if best_mdf_x is not None:
        ax_mdf.plot(best_mdf_x, best_mdf_y, 'r-', linewidth=3, label='Best Model', zorder=3)
    ax_mdf.plot(feh_mdf, normalized_count_mdf, 'ko', markersize=6, label='Observed', zorder=2)
    
    ax_mdf.set_xlabel('[Fe/H]', fontsize=14)
    ax_mdf.set_ylabel('Normalized Number Density', fontsize=14)
    ax_mdf.set_title('Metallicity Distribution Function', fontsize=16, fontweight='bold')
    ax_mdf.set_xlim(-2, 1)
    ax_mdf.legend(fontsize=12)
    ax_mdf.grid(True, alpha=0.3)
    
    # =====================================================
    # PANEL 5: AGE-METALLICITY RELATION
    # =====================================================
    ax_age = fig.add_subplot(gs[1, 3:])
    
    # Find best age-metallicity model
    best_age_x = None
    best_age_y = None
    for age_data, res in zip(GalGA.age_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            x_age_raw, y_feh = age_data
            best_age_x = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
            best_age_y = np.array(y_feh)
            break
    
    # Plot observational data
    ax_age.scatter(age_Joyce, Fe_H, marker='*', s=40, color='blue', alpha=0.6, label='Joyce et al.')
    ax_age.scatter(age_Bensby, Fe_H, marker='^', s=40, color='orange', alpha=0.6, label='Bensby et al.')
    
    # Plot best model
    if best_age_x is not None:
        ax_age.plot(best_age_x, best_age_y, 'r-', linewidth=3, label='Best Model', zorder=3)
    
    ax_age.set_xlabel('Age (Gyr)', fontsize=14)
    ax_age.set_ylabel('[Fe/H]', fontsize=14)
    ax_age.set_title('Age-Metallicity Relation', fontsize=16, fontweight='bold')
    ax_age.set_xlim(0, 14)
    ax_age.set_ylim(-2, 1)
    ax_age.legend(fontsize=11)
    ax_age.grid(True, alpha=0.3)
    
    # =====================================================
    # PANEL 6-9: ALPHA ELEMENT ABUNDANCES (2x2 grid)
    # =====================================================
    alpha_elements = ['Mg', 'Si', 'Ca', 'Ti']
    alpha_obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]
    
    for idx, (element, obs_data) in enumerate(zip(alpha_elements, alpha_obs_data)):
        row = 2 + idx // 2
        col = (idx % 2) * 3
        ax_alpha = fig.add_subplot(gs[row, col:col+3])
        
        # Find best alpha model for this element
        best_alpha_x = None
        best_alpha_y = None
        for alpha_arrs, res in zip(GalGA.alpha_data, GalGA.results):
            params = (res[5], res[7], res[9])
            is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
            if is_best and idx < len(alpha_arrs):
                best_alpha_x, best_alpha_y = alpha_arrs[idx]
                break
        
        # Clean observational data
        obs_clean = np.where((obs_data >= -2.0) & (obs_data <= 2.0), obs_data, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_clean)
        
        # Plot observational data
        if np.sum(mask) > 10:
            ax_alpha.scatter(Fe_H[mask], obs_clean[mask], s=20, alpha=0.6, 
                           color='gray', label='Observed', zorder=1)
        
        # Plot best model
        if best_alpha_x is not None:
            ax_alpha.plot(best_alpha_x, best_alpha_y, 'r-', linewidth=3, 
                         label='Best Model', zorder=3)
        
        ax_alpha.set_xlabel('[Fe/H]', fontsize=12)
        ax_alpha.set_ylabel(f'[{element}/Fe]', fontsize=12)
        ax_alpha.set_title(f'{element} Abundance', fontsize=14, fontweight='bold')
        ax_alpha.set_xlim(-2, 1)
        ax_alpha.set_ylim(-0.6, 0.8)
        ax_alpha.legend(fontsize=10, loc='upper right')
        ax_alpha.grid(True, alpha=0.3)
        
        # Add element label
        ax_alpha.text(0.05, 0.9, element, transform=ax_alpha.transAxes, 
                     fontsize=16, fontweight='bold', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # =====================================================
    # FINAL TOUCHES
    # =====================================================
    
    # Add a subtle background color to distinguish sections
    fig.patch.set_facecolor('white')
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Comprehensive dashboard saved to {save_path}")
    print(f"Best-fit parameters:")
    print(f"  σ₂ = {best_row['sigma_2']:.1f} pc")
    print(f"  t₂ = {best_row['t_2']:.3f} Gyr") 
    print(f"  τ₂ = {best_row['infall_2']:.3f} Gyr")
    print(f"  SFE = {best_row['sfe']:.5f}")
    print(f"  Fitness = {best_row['fitness']:.4f}")
    
    return fig


# Add this function to the generate_all_plots function
def generate_all_plots_with_omni(GalGA, feh, normalized_count, results_file='GA/simulation_results.csv'):
    """Enhanced version that includes the omni info figure"""
    
    # Load observational alpha element data (same as before)
    f = open('data/Bensby_Data.tsv')
    lines = f.readlines()
    Fe_H = []
    age_Joyce = []
    age_Bensby = []
    Si_Fe = []
    Ca_Fe = []
    Mg_Fe = []
    Ti_Fe = []
    
    for line in lines[1::]:
        line = line.split()
        
        Fe_H_ind = lines[0].split().index('[Fe/H]')
        Si_Fe_ind = lines[0].split().index('[Si/Fe]')
        Ca_Fe_ind = lines[0].split().index('[Ca/Fe]')
        Mg_Fe_ind = lines[0].split().index('[Mg/Fe]')
        Ti_Fe_ind = lines[0].split().index('[Ti/Fe]')
        age_Joyce_ind = lines[0].split().index('Joyce_age')
        age_Bensby_ind = lines[0].split().index('Bensby')
        
        age_Joyce.append(float(line[age_Joyce_ind]))
        age_Bensby.append(float(line[age_Bensby_ind]))
        Fe_H.append(float(line[Fe_H_ind]))
        Si_Fe.append(float(line[Si_Fe_ind]))
        Ca_Fe.append(float(line[Ca_Fe_ind]))
        Mg_Fe.append(float(line[Mg_Fe_ind]))
        Ti_Fe.append(float(line[Ti_Fe_ind]))
    
    f.close()
    
    # Convert to numpy arrays
    Fe_H = np.array(Fe_H)
    age_Joyce = np.array(age_Joyce)
    age_Bensby = np.array(age_Bensby)
    Si_Fe = np.array(Si_Fe)
    Ca_Fe = np.array(Ca_Fe) 
    Mg_Fe = np.array(Mg_Fe)
    Ti_Fe = np.array(Ti_Fe)
    
    # Load results DataFrame
    import pandas as pd
    df = pd.read_csv(results_file)
    
    # Generate the comprehensive omni info figure
    print("Generating comprehensive dashboard figure...")
    plot_omni_info_figure(GalGA, Fe_H, age_Joyce, age_Bensby, 
                          Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
                          feh, normalized_count, df)
    
    print("Omni info figure generated!")
    return df









def extract_metrics(results_file):
    """Extract metrics from CSV file for plotting"""
    # Load the dataframe directly
    df = pd.read_csv(results_file)
    
    comp_idx_vals    = df['comp_idx'].values
    imf_idx_vals     = df['imf_idx'].values
    sn1a_idx_vals    = df['sn1a_idx'].values
    sy_idx_vals      = df['sy_idx'].values
    sn1ar_idx_vals   = df['sn1ar_idx'].values
    sigma_2_vals     = df['sigma_2'].values
    t_1_vals         = df['t_1'].values
    t_2_vals         = df['t_2'].values
    infall_1_vals    = df['infall_1'].values
    infall_2_vals    = df['infall_2'].values
    sfe_vals         = df['sfe'].values
    delta_sfe_vals   = df['delta_sfe'].values
    imf_upper_vals   = df['imf_upper'].values
    mgal_vals        = df['mgal'].values
    nb_vals          = df['nb'].values

    # Extract metrics
    metrics_dict = {}
    #for metric in ['wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh', 'ks', 'ensemble', 'fitness']:
    for metric in ['fitness']:
        if metric in df.columns:
            metrics_dict[metric] = df[metric].values
    
    return sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals, sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals, metrics_dict, df



def plot_pca_degeneracy_analysis(GalGA, results_file='GA/simulation_results.csv', save_path='GA/pca_degeneracy_analysis.png'):
    """
    Perform PCA analysis on the fittest 10% of the population to reveal parameter degeneracies.
    Shows how the best models spread along degenerate manifolds vs constrained directions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Load results and extract continuous parameters
    df = pd.read_csv(results_file)
    
    # Sort by fitness (assuming lower is better) and take top 10%
    if 'fitness' in df.columns:
        fitness_col = 'fitness'
    elif 'wrmse' in df.columns:
        fitness_col = 'wrmse'
    else:
        # Fallback to first loss metric available
        possible_metrics = ['ks', 'ensemble', 'mae', 'mape', 'huber', 'cosine', 'log_cosh']
        fitness_col = next((col for col in possible_metrics if col in df.columns), df.columns[-1])
    
    df_sorted = df.sort_values(fitness_col, ascending=True)
    top_10_percent = int(len(df_sorted) * 0.1)
    df_top = df_sorted.head(top_10_percent)
    
    print(f"Analyzing top {top_10_percent} individuals ({10:.0f}%) out of {len(df)} total")
    print(f"Using '{fitness_col}' as fitness metric")
    print(f"Fitness range in top 10%: {df_top[fitness_col].min():.4f} to {df_top[fitness_col].max():.4f}")
    
    # Define continuous parameter names and extract values
    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                        'sfe', 'delta_sfe', 'imf_upper', 'nb']
    
    # Extract parameter matrix from top 10%
    param_matrix = df_top[continuous_params].values
    
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    param_matrix_scaled = scaler.fit_transform(param_matrix)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(param_matrix_scaled)
    
    # Get principal components and explained variance
    components = pca.components_
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add title indicating this is top 10% analysis
    fig.suptitle(f'PCA Degeneracy Analysis - Top 10% Fittest Models (n={top_10_percent})', fontsize=16, y=0.98)
    
    # 1. Eigenvalue/Singular value plot
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(explained_variance)), explained_variance, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue (Variance)')
    ax1.set_yscale('log')
    ax1.set_title('Eigenvalues (Top 10%)')
    
    # Highlight small eigenvalues (degeneracies)
    threshold = np.max(explained_variance) * 0.01  # 1% of maximum
    for i, (bar, val) in enumerate(zip(bars, explained_variance)):
        if val < threshold:
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    # 2. Explained variance ratio
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_var = np.cumsum(explained_variance_ratio)
    ax2.plot(range(len(cumulative_var)), cumulative_var, 'o-', color='darkgreen', linewidth=2)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance')
    ax2.set_title('Cumulative Variance')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95%')
    ax2.legend()
    
    # 3. Principal component loadings heatmap
    ax3 = fig.add_subplot(gs[0, 2:])
    im = ax3.imshow(components[:6], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(continuous_params)))
    ax3.set_xticklabels(continuous_params, rotation=45, ha='right')
    ax3.set_yticks(range(6))
    ax3.set_yticklabels([f'PC{i+1}' for i in range(6)])
    ax3.set_title('Principal Component Loadings')
    plt.colorbar(im, ax=ax3, fraction=0.02)
    
    # Add text annotations for strong loadings
    for i in range(6):
        for j in range(len(continuous_params)):
            if abs(components[i, j]) > 0.5:
                ax3.text(j, i, f'{components[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white' if abs(components[i, j]) > 0.7 else 'black')
    
    # 4. 2D projections onto first few PCs
    # Color points by fitness; use log scale for clarity
    if fitness_col in df_top.columns:
        colors = np.log10(np.clip(df_top[fitness_col].values, 1e-10, None))
    else:
        colors = np.zeros(len(df_top))
    
    # PC1 vs PC2
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax4.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    ax4.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    ax4.set_title('PC1 vs PC2')
    plt.colorbar(scatter, ax=ax4, label=f'log10 {fitness_col}')
    
    # PC2 vs PC3
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(pca_result[:, 1], pca_result[:, 2], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax5.set_xlabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    ax5.set_ylabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)')
    ax5.set_title('PC2 vs PC3')
    
    # PC3 vs PC4
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(pca_result[:, 2], pca_result[:, 3], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax6.set_xlabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)')
    ax6.set_ylabel(f'PC4 ({explained_variance_ratio[3]:.1%} variance)')
    ax6.set_title('PC3 vs PC4')
    
    # 5. Example parameter pair showing degeneracy
    ax7 = fig.add_subplot(gs[1, 3])
    # Find the most correlated parameter pair in top 10%
    param_corr = np.corrcoef(param_matrix_scaled.T)
    np.fill_diagonal(param_corr, 0)  # Remove self-correlation
    max_corr_idx = np.unravel_index(np.argmax(np.abs(param_corr)), param_corr.shape)
    
    param1_idx, param2_idx = max_corr_idx
    param1_name = continuous_params[param1_idx]
    param2_name = continuous_params[param2_idx]
    
    ax7.scatter(df_top[param1_name], df_top[param2_name], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax7.set_xlabel(param1_name)
    ax7.set_ylabel(param2_name)
    ax7.set_title(f'Most Correlated Pair\nr = {param_corr[max_corr_idx]:.3f}')
    
    # 6. Parameter distributions along degenerate vs constrained directions
    ax8 = fig.add_subplot(gs[2, :2])
    
    # Project onto most and least constrained directions
    most_constrained = pca_result[:, 0]  # Highest variance PC
    least_constrained = pca_result[:, -1]  # Lowest variance PC
    
    ax8.hist(most_constrained, bins=30, alpha=0.7, label=f'Most constrained (PC1, λ={explained_variance[0]:.3f})', color='blue')
    ax8.hist(least_constrained, bins=30, alpha=0.7, label=f'Least constrained (PC{len(explained_variance)}, λ={explained_variance[-1]:.6f})', color='red')
    ax8.set_xlabel('Projection Value')
    ax8.set_ylabel('Count')
    ax8.set_title('Parameter Distributions Along PC Directions')
    ax8.legend()
    
    # 7. Degeneracy identification table
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')
    
    # Identify degenerate parameter combinations
    degeneracy_threshold = np.max(explained_variance) * 0.05  # 5% threshold
    degenerate_pcs = np.where(explained_variance < degeneracy_threshold)[0]
    
    table_data = []
    table_data.append(['Principal Component', 'Eigenvalue', 'Status', 'Dominant Parameters'])
    
    for i in range(min(8, len(explained_variance))):
        eigenval = explained_variance[i]
        status = 'DEGENERATE' if eigenval < degeneracy_threshold else 'Constrained'
        
        # Find parameters with highest loadings
        loadings = np.abs(components[i])
        top_params_idx = np.argsort(loadings)[-3:]  # Top 3
        top_params = [continuous_params[idx] for idx in top_params_idx]
        param_str = ', '.join(top_params)
        
        table_data.append([f'PC{i+1}', f'{eigenval:.4f}', status, param_str])
    
    # Create table
    table = ax9.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color degenerate rows
    for i, row in enumerate(table_data[1:], 1):
        if 'DEGENERATE' in row[2]:
            for j in range(len(row)):
                table[(i, j)].set_facecolor('#ffcccc')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"PCA Degeneracy Analysis saved to {save_path}")
    print(f"Found {len(degenerate_pcs)} degenerate directions (eigenvalue < {degeneracy_threshold:.4f})")
    print(f"Top 3 eigenvalues: {explained_variance[:3]}")
    print(f"Bottom 3 eigenvalues: {explained_variance[-3:]}")
    
    return fig


def plot_parameter_correlation_matrix(results_file='GA/simulation_results.csv', save_path='GA/parameter_correlations.png'):
    """
    Create a correlation matrix heatmap showing parameter relationships for the fittest 10% of individuals.
    Complements the PCA analysis by showing direct pairwise correlations.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    df = pd.read_csv(results_file)
    
    # Sort by fitness (assuming lower is better) and take top 10%
    if 'fitness' in df.columns:
        fitness_col = 'fitness'
    elif 'wrmse' in df.columns:
        fitness_col = 'wrmse'
    else:
        # Fallback to first loss metric available
        possible_metrics = ['ks', 'ensemble', 'mae', 'mape', 'huber', 'cosine', 'log_cosh']
        fitness_col = next((col for col in possible_metrics if col in df.columns), df.columns[-1])
    
    df_sorted = df.sort_values(fitness_col, ascending=True)
    top_10_percent = int(len(df_sorted) * 0.1)
    df_top = df_sorted.head(top_10_percent)
    
    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                        'sfe', 'delta_sfe', 'imf_upper', 'nb']
    
    # Calculate correlation matrix for top 10%
    corr_matrix = df_top[continuous_params].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Add title indicating this is top 10% analysis
    fig.suptitle(f'Parameter Correlation Matrix - Top 10% Fittest Models (n={top_10_percent})', fontsize=14, y=0.95)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax,
                fmt='.3f')  # Show 3 decimal places
    
    # Add subtitle with fitness range
    plt.figtext(0.5, 0.91, f'Fitness range: {df_top[fitness_col].min():.4f} to {df_top[fitness_col].max():.4f}', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find strongest correlations in the top 10%
    corr_values = corr_matrix.values
    np.fill_diagonal(corr_values, 0)  # Remove self-correlations
    
    # Get indices of strongest positive and negative correlations
    max_pos_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
    min_neg_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
    
    print(f"Correlation analysis for top {top_10_percent} individuals:")
    print(f"Strongest positive correlation: {continuous_params[max_pos_idx[0]]} - {continuous_params[max_pos_idx[1]]} (r = {corr_values[max_pos_idx]:.3f})")
    print(f"Strongest negative correlation: {continuous_params[min_neg_idx[0]]} - {continuous_params[min_neg_idx[1]]} (r = {corr_values[min_neg_idx]:.3f})")
    
    return fig





def plot_walker_evolution_combined(walker_history, param_names, param_indices, save_path='figures/walker_evolution_combined.png'):
    """
    Generate a combined plot showing the evolution of multiple key parameters across generations.
    Each subplot shows one parameter's evolution with median trajectory and interquartile range.
    
    Parameters:
    -----------
    walker_history : dict
        Dictionary mapping walker IDs to their parameter history arrays
    param_names : list of str
        Names of the parameters to plot
    param_indices : list of int
        Indices of the parameters in the history arrays
    save_path : str
        Path to save the figure
    """
    if not walker_history:
        print("Walker history data not available. Skipping combined walker evolution plot.")
        return None

    n_params = len(param_names)
    fig, axes = plt.subplots(nrows=n_params, ncols=1, figsize=(10, 4 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]  # Ensure iterable

    for ax, param_name, param_idx in zip(axes, param_names, param_indices):
        all_histories = []
        for history in walker_history.values():
            if not history:
                continue
            history = np.array(history)
            if param_idx >= history.shape[1]:
                continue
            all_histories.append(history[:, param_idx])

        if not all_histories:
            continue

        all_histories = np.array(all_histories)  # (n_walkers, n_generations)
        generations = np.arange(all_histories.shape[1])

        # Plot individual walkers faintly
        for walker_series in all_histories:
            ax.plot(generations, walker_series, color='gray', alpha=0.02, linewidth=0.5)

        # Compute and plot median and IQR
        median = np.median(all_histories, axis=0)
        lower = np.percentile(all_histories, 25, axis=0)
        upper = np.percentile(all_histories, 75, axis=0)

        ax.plot(generations, median, color='black', label='Median', linewidth=2)
        ax.fill_between(generations, lower, upper, color='blue', alpha=0.3, label='IQR (25-75%)')

        ax.set_ylabel(param_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Generation')
    fig.suptitle('Parameter Evolution Across Generations', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Combined walker evolution plot saved to {save_path}")
    return fig




# Replace the plotting section in generate_all_plots function with this expanded version:

def generate_all_plots(GalGA, feh, normalized_count, results_file='GA/simulation_results.csv'):
    """Generate all plots from GalGA results including comprehensive parameter combinations"""
    
    # Load observational alpha element data
    f = open('data/Bensby_Data.tsv')
    lines = f.readlines()
    Fe_H = []
    Fe_H_err = []
    age_Joyce = []
    age_Bensby = []
    Si_Fe = []
    Si_Fe_err = []
    Ca_Fe = []
    Ca_Fe_err = []
    Mg_Fe = []
    Mg_Fe_err = []
    Ti_Fe = []
    Ti_Fe_err = []
    
    for line in lines[1::]:
        line = line.split()
        
        Fe_H_ind = lines[0].split().index('[Fe/H]')
        Fe_H_err_ind = lines[0].split().index('error_[Fe/H]')
        
        Si_Fe_ind = lines[0].split().index('[Si/Fe]')
        Si_Fe_err_ind = lines[0].split().index('error_[Si/Fe]')
        
        Ca_Fe_ind = lines[0].split().index('[Ca/Fe]')
        Ca_Fe_err_ind = lines[0].split().index('error_[Ca/Fe]')
        
        Mg_Fe_ind = lines[0].split().index('[Mg/Fe]')
        Mg_Fe_err_ind = lines[0].split().index('error_[Mg/Fe]')
        
        Ti_Fe_ind = lines[0].split().index('[Ti/Fe]')
        Ti_Fe_err_ind = lines[0].split().index('error_[Ti/Fe]')
        
        age_Joyce_ind = lines[0].split().index('Joyce_age')
        age_Bensby_ind = lines[0].split().index('Bensby')
        
        age_Joyce.append(float(line[age_Joyce_ind]))
        age_Bensby.append(float(line[age_Bensby_ind]))
        Fe_H.append(float(line[Fe_H_ind]))
        Fe_H_err.append(float(line[Fe_H_err_ind]))
        Si_Fe.append(float(line[Si_Fe_ind]))
        Si_Fe_err.append(float(line[Si_Fe_err_ind]))
        Ca_Fe.append(float(line[Ca_Fe_ind]))
        Ca_Fe_err.append(float(line[Ca_Fe_err_ind]))
        Mg_Fe.append(float(line[Mg_Fe_ind]))
        Mg_Fe_err.append(float(line[Mg_Fe_err_ind]))
        Ti_Fe.append(float(line[Ti_Fe_ind]))
        Ti_Fe_err.append(float(line[Ti_Fe_err_ind]))
    
    f.close()
    
    # Convert to numpy arrays
    Fe_H = np.array(Fe_H)
    Si_Fe = np.array(Si_Fe)
    Ca_Fe = np.array(Ca_Fe) 
    Mg_Fe = np.array(Mg_Fe)
    Ti_Fe = np.array(Ti_Fe)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Extract metrics for scatter plots
    sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals, sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals, metrics_dict, df = extract_metrics(results_file)
    
    # 1. Plot MDF curves (existing)
    plot_mdf_curves(GalGA, feh, normalized_count, df)
    
    # 2. Plot Four-Panel Alpha Elements
    print("Generating Four-Panel Alpha Elements plot...")
    plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, df)

    print("Generating Age-Metallicity plot...")
    plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, df, save_path='GA/Age_FeH_detailed_results.png')

    # 3. Comprehensive 2D scatter plots
    print("Generating comprehensive 2D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        
        # ========== INFALL PARAMETERS ==========
        # Second infall episode (most important)
        plot_2d_scatter(t_2_vals, infall_2_vals, metric_vals, metric_name + '_t2_infall2', xlabel='t_2 (Gyr)', ylabel='infall_2 (Gyr)')
        plot_2d_scatter(sigma_2_vals, infall_2_vals, metric_vals, metric_name + '_sigma2_infall2', xlabel='sigma_2', ylabel='infall_2 (Gyr)')
        plot_2d_scatter(sigma_2_vals, t_2_vals, metric_vals, metric_name + '_sigma2_t2', xlabel='sigma_2', ylabel='t_2 (Gyr)')
        
        # First infall episode
        plot_2d_scatter(t_1_vals, infall_1_vals, metric_vals, metric_name + '_t1_infall1', xlabel='t_1 (Gyr)', ylabel='infall_1 (Gyr)')
        plot_2d_scatter(t_1_vals, infall_2_vals, metric_vals, metric_name + '_t1_infall2', xlabel='t_1 (Gyr)', ylabel='infall_2 (Gyr)')
        
        # Cross-infall comparisons
        plot_2d_scatter(t_1_vals, t_2_vals, metric_vals, metric_name + '_t1_t2', xlabel='t_1 (Gyr)', ylabel='t_2 (Gyr)')
        plot_2d_scatter(infall_1_vals, infall_2_vals, metric_vals, metric_name + '_infall1_infall2', xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)')
        
        # ========== STAR FORMATION EFFICIENCY ==========
        plot_2d_scatter(sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_sfe_deltasfe', xlabel='SFE', ylabel='Delta SFE')
        plot_2d_scatter(sfe_vals, t_2_vals, metric_vals, metric_name + '_sfe_t2', xlabel='SFE', ylabel='t_2 (Gyr)')
        plot_2d_scatter(sfe_vals, sigma_2_vals, metric_vals, metric_name + '_sfe_sigma2', xlabel='SFE', ylabel='sigma_2')
        plot_2d_scatter(delta_sfe_vals, t_2_vals, metric_vals, metric_name + '_deltasfe_t2', xlabel='Delta SFE', ylabel='t_2 (Gyr)')
        plot_2d_scatter(delta_sfe_vals, infall_2_vals, metric_vals, metric_name + '_deltasfe_infall2', xlabel='Delta SFE', ylabel='infall_2 (Gyr)')
        
        # ========== GALAXY MASS RELATIONS ==========
        plot_2d_scatter(mgal_vals, sfe_vals, metric_vals, metric_name + '_mgal_sfe', xlabel='M_gal (M_sun)', ylabel='SFE')
        plot_2d_scatter(mgal_vals, sigma_2_vals, metric_vals, metric_name + '_mgal_sigma2', xlabel='M_gal (M_sun)', ylabel='sigma_2')
        plot_2d_scatter(mgal_vals, t_2_vals, metric_vals, metric_name + '_mgal_t2', xlabel='M_gal (M_sun)', ylabel='t_2 (Gyr)')
        plot_2d_scatter(mgal_vals, infall_2_vals, metric_vals, metric_name + '_mgal_infall2', xlabel='M_gal (M_sun)', ylabel='infall_2 (Gyr)')
        
        # ========== IMF AND STELLAR PARAMETERS ==========
        plot_2d_scatter(imf_upper_vals, sfe_vals, metric_vals, metric_name + '_imf_sfe', xlabel='IMF Upper (M_sun)', ylabel='SFE')
        plot_2d_scatter(imf_upper_vals, t_2_vals, metric_vals, metric_name + '_imf_t2', xlabel='IMF Upper (M_sun)', ylabel='t_2 (Gyr)')
        plot_2d_scatter(imf_upper_vals, mgal_vals, metric_vals, metric_name + '_imf_mgal', xlabel='IMF Upper (M_sun)', ylabel='M_gal (M_sun)')
        plot_2d_scatter(nb_vals, imf_upper_vals, metric_vals, metric_name + '_nb_imf', xlabel='SN1a per Solar Mass', ylabel='IMF Upper (M_sun)')
        
        # ========== SN1A PARAMETERS ==========
        plot_2d_scatter(nb_vals, sfe_vals, metric_vals, metric_name + '_nb_sfe', xlabel='SN1a per Solar Mass', ylabel='SFE')
        plot_2d_scatter(nb_vals, t_2_vals, metric_vals, metric_name + '_nb_t2', xlabel='SN1a per Solar Mass', ylabel='t_2 (Gyr)')
        plot_2d_scatter(nb_vals, mgal_vals, metric_vals, metric_name + '_nb_mgal', xlabel='SN1a per Solar Mass', ylabel='M_gal (M_sun)')
        plot_2d_scatter(nb_vals, sigma_2_vals, metric_vals, metric_name + '_nb_sigma2', xlabel='SN1a per Solar Mass', ylabel='sigma_2')

    # 4. Comprehensive 3D scatter plots
    print("Generating comprehensive 3D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        
        # ========== INFALL-FOCUSED 3D PLOTS ==========
        # Primary infall relationships
        plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_infall2_complete', 
                       xlabel='sigma_2', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(t_1_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_timing_comparison', 
                       xlabel='t_1 (Gyr)', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(infall_1_vals, infall_2_vals, sigma_2_vals, metric_vals, metric_name + '_infall_timescales', 
                       xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)', zlabel='sigma_2')
        
        # ========== SFE-FOCUSED 3D PLOTS ==========
        plot_3d_scatter(sfe_vals, delta_sfe_vals, infall_2_vals, metric_vals, metric_name + '_sfe_evolution', 
                       xlabel='SFE', ylabel='Delta SFE', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(sfe_vals, t_1_vals, infall_2_vals, metric_vals, metric_name + '_sfe_timing', 
                       xlabel='SFE', ylabel='t_1 (Gyr)', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(sfe_vals, t_2_vals, sigma_2_vals, metric_vals, metric_name + '_sfe_infall2_params', 
                       xlabel='SFE', ylabel='t_2 (Gyr)', zlabel='sigma_2')
        plot_3d_scatter(delta_sfe_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_deltasfe_timing', 
                       xlabel='Delta SFE', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
        
        # ========== GALAXY MASS-FOCUSED 3D PLOTS ==========
        plot_3d_scatter(mgal_vals, sfe_vals, infall_2_vals, metric_vals, metric_name + '_mgal_sfe_infall', 
                       xlabel='M_gal (M_sun)', ylabel='SFE', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(mgal_vals, t_2_vals, sigma_2_vals, metric_vals, metric_name + '_mgal_infall2_params', 
                       xlabel='M_gal (M_sun)', ylabel='t_2 (Gyr)', zlabel='sigma_2')
        plot_3d_scatter(mgal_vals, sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_mgal_sfe_evolution', 
                       xlabel='M_gal (M_sun)', ylabel='SFE', zlabel='Delta SFE')
        
        # ========== STELLAR/IMF-FOCUSED 3D PLOTS ==========
        plot_3d_scatter(imf_upper_vals, sfe_vals, infall_2_vals, metric_vals, metric_name + '_imf_sfe_infall', 
                       xlabel='IMF Upper (M_sun)', ylabel='SFE', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(nb_vals, imf_upper_vals, infall_2_vals, metric_vals, metric_name + '_stellar_params_infall', 
                       xlabel='SN1a per Solar Mass', ylabel='IMF Upper (M_sun)', zlabel='infall_2 (Gyr)')
        plot_3d_scatter(nb_vals, sfe_vals, t_2_vals, metric_vals, metric_name + '_sn1a_sfe_timing', 
                       xlabel='SN1a per Solar Mass', ylabel='SFE', zlabel='t_2 (Gyr)')
        
        # ========== CROSS-PARAMETER EXPLORATION ==========
        plot_3d_scatter(sigma_2_vals, sfe_vals, mgal_vals, metric_vals, metric_name + '_sigma_sfe_mgal', 
                       xlabel='sigma_2', ylabel='SFE', zlabel='M_gal (M_sun)')
        plot_3d_scatter(t_1_vals, sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_t1_sfe_evolution', 
                       xlabel='t_1 (Gyr)', ylabel='SFE', zlabel='Delta SFE')
        plot_3d_scatter(infall_1_vals, infall_2_vals, sfe_vals, metric_vals, metric_name + '_infall_timescales_sfe', 
                       xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)', zlabel='SFE')

    # 5. Walker evolution plots
    print("Generating walker evolution plots...")
    param_names = ["sigma_2", "t_2", "infall_2", "sfe", "delta_sfe"]
    param_indices = [5, 7, 9, 10, 11]
    plot_walker_history(GalGA.walker_history, param_names, param_indices)
    
    # 6. Plot loss history for each walker
    print("Generating walker loss history plots...")
    for metric in ['wrmse', 'huber', 'ks', 'cosine', 'fitness']:
        plot_walker_loss_history(GalGA.walker_history, results_file, loss_metric=metric)
        
    # 7. Create 3D animation
    print("Generating 3D animation...")
    # create_3d_animation(GalGA.walker_history)
    
    # 8. PCA degeneracy analysis
    print("Generating PCA degeneracy analysis...")
    plot_pca_degeneracy_analysis(GalGA, results_file)
    
    print("Generating parameter correlation matrix...")
    plot_parameter_correlation_matrix(results_file)
    
    print("All plotting complete! Check the GA directory for results.")
    print(f"Generated comprehensive parameter space exploration plots:")
    print(f"- {len(metrics_dict)} metrics × 24 2D plots = {len(metrics_dict) * 24} 2D scatter plots")
    print(f"- {len(metrics_dict)} metrics × 16 3D plots = {len(metrics_dict) * 16} 3D scatter plots")
    print(f"- Plus walker evolution, loss history, PCA analysis, and correlation matrix plots")
    print(f"Loaded {len(Fe_H)} observational data points for individual alpha elements")