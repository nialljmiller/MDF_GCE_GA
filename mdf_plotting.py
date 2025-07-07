#!/usr/bin/env python3.8
################################
# Plotting functions for MDF_GA
################################

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
            ax.plot(x, y, color='gray', alpha=0.4)

    # observational data
    ax.plot(feh, normalized_count, 'x', ms=8, color='k', label='Data')
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
    """Plot 3D scatter plot with color indicating a specific metric"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'{label}')
    plt.savefig(f'GA/loss/{label}_loss_3d.png', bbox_inches='tight')
    plt.close()
    
    return fig




def plot_2d_scatter(x, y, color_metric, label, xlabel='t_2', ylabel='infall_2'):
    """Plot 2D scatter plot with color indicating a specific metric"""
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{label} Loss')
    plt.savefig(f'GA/loss/{label}_loss_2d.png', bbox_inches='tight')
    plt.close()
    
    return plt.gcf()




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
            ax.plot(generations, walker_series, color='gray', alpha=0.1, linewidth=0.75)

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
    Plot the loss values for each walker over generations.
    
    Parameters:
    -----------
    walker_history : dict
        Dictionary mapping walker IDs to their parameter history
    results_csv : str
        Path to the CSV file containing all evaluation results
    loss_metric : str
        Which loss metric to plot ('wrmse', 'mae', 'mape', etc.)
    """
    # Load results containing all evaluations
    results_df = pd.read_csv(results_csv)
    
    # Define column mapping based on your results structure
    loss_metrics = {
        'ks': 14,
        'ensemble': 15, 
        'wrmse': 16, 
        'mae': 17, 
        'mape': 18, 
        'huber': 19, 
        'cosine': 20, 
        'log_cosh': 21
    }
    
    # Make sure the loss metric exists in our mapping
    if loss_metric not in loss_metrics:
        print(f"Warning: Loss metric '{loss_metric}' not found. Using 'wrmse' instead.")
        loss_metric = 'wrmse'
    
    # Get the column index for the selected loss metric
    loss_column = loss_metrics[loss_metric]
    
    # Create figure for plotting
    plt.figure(figsize=(12, 8))
    
    # For each walker, extract parameters at each generation and match to results
    for walker_id, history in walker_history.items():
        if not history:  # Skip empty histories
            continue
        
        # Convert to numpy array for easier manipulation
        history_array = np.array(history)
        num_generations = len(history_array)
        
        # Initialize array to store loss values
        loss_values = np.full(num_generations, np.nan)
        
        # For each generation, find matching result from results_df
        for gen_idx in range(num_generations):
            params = history_array[gen_idx]
            
            # Extract key parameters to match (using continuous params like sigma_2, t_2, infall_2)
            sigma_2 = params[5]  # Assuming this is the index for sigma_2
            t_2 = params[7]      # Assuming this is the index for t_2
            infall_2 = params[9] # Assuming this is the index for infall_2
            
            # Find the closest match in results_df
            matches = results_df[
                (abs(results_df['sigma_2'] - sigma_2) < 1e-5) &
                (abs(results_df['t_2'] - t_2) < 1e-5) &
                (abs(results_df['infall_2'] - infall_2) < 1e-5)
            ]
            
            if not matches.empty:
                # Use the first match's loss value
                loss_values[gen_idx] = matches.iloc[0][loss_metric]
        
        # Plot the loss history for this walker
        generations = np.arange(num_generations)
        valid_indices = ~np.isnan(loss_values)
        
        if np.any(valid_indices):
            # If we have enough valid points, use a spline to smooth the curve
            if np.sum(valid_indices) > 3:
                valid_gens = generations[valid_indices]
                valid_loss = loss_values[valid_indices]
                
                # Create a smooth curve through the valid points
                spl = UnivariateSpline(valid_gens, valid_loss, k=min(3, len(valid_loss)-1))
                smooth_x = np.linspace(valid_gens.min(), valid_gens.max(), 100)
                plt.plot(smooth_x, spl(smooth_x), alpha=0.7, linewidth=1)
                
                # Also plot the actual points
                plt.scatter(valid_gens, valid_loss, s=20, alpha=0.6, 
                           label=f"Walker {walker_id}" if walker_id < 5 else "")
            else:
                # Just connect the dots if too few points
                plt.plot(generations[valid_indices], loss_values[valid_indices], 
                        marker='o', label=f"Walker {walker_id}" if walker_id < 5 else "")
    
    # Add plot details
    plt.title(f"{loss_metric.upper()}")
    plt.xlabel("Generation")
    plt.ylabel(f"{loss_metric.upper()} Loss")
    plt.grid(True, alpha=0.3)
    
    # Only show legend for first few walkers to avoid clutter
    #plt.legend(loc='upper right', fontsize='small')
    
    # Add annotations about convergence
    min_losses = []
    for walker_id, history in walker_history.items():
        if not history:
            continue
        history_array = np.array(history)
        loss_values = np.full(len(history_array), np.nan)
        # (same matching logic as above)
        # ...
        valid_loss = loss_values[~np.isnan(loss_values)]
        if len(valid_loss) > 0:
            min_losses.append(np.min(valid_loss))
    
    if min_losses:
        plt.annotate(f"Best overall loss: {min(min_losses):.4f}", 
                    xy=(0.02, 0.02), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'GA/loss/walker_loss_history_{loss_metric}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print(f"Loss history plot saved to GA/loss/walker_loss_history_{loss_metric}.png")
    return plt.gcf()








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
        # Prepare the colormap for losses
        all_losses = GA.all_losses_successful + GA.all_losses_unsuccessful
        min_loss = GA.global_min_loss
        max_loss = GA.global_max_loss
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

        # Normalize losses
        losses_successful_norm = [(loss - min_loss) / loss_range for loss in GA.all_losses_successful]
        losses_unsuccessful_norm = [(loss - min_loss) / loss_range for loss in GA.all_losses_unsuccessful]

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
        # Prepare the colormap for losses
        all_losses = GA.all_losses_successful + GA.all_losses_unsuccessful
        min_loss = GA.global_min_loss
        max_loss = GA.global_max_loss
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

        # Normalize losses
        losses_successful_norm = [(loss - min_loss) / loss_range for loss in GA.all_losses_successful]
        losses_unsuccessful_norm = [(loss - min_loss) / loss_range for loss in GA.all_losses_unsuccessful]

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
                    ax_main.plot(x_data, y_data, color='gray', alpha=0.4, linewidth=1)

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
            ax_kde.plot(kde_y, y_vals, color='gray')
            ax_kde.fill_betweenx(y_vals, 0, kde_y, color='gray', alpha=0.3)

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
            ax_kde.plot(kde_model, y_vals, color='red', linestyle='--')
            ax_kde.fill_betweenx(y_vals, 0, kde_model, color='red', alpha=0.2)

        # Set limits and labels for main plot
        ax_main.set_xlim(-2, 1)
        ax_main.set_ylim(-0.8, 1)
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
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import gaussian_kde, binned_statistic
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
    
    # Plot model lines
    for age_data, label, res in zip(GalGA.age_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        
        x_age_raw, y_feh = age_data
        age_gyr = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
        
        if is_best and not best_plotted:
            label_lines = [f"• {p.strip()}" for p in label.split(',')]
            pretty_label = "\n".join(label_lines) + "\n• (BEST)"
            ax_main.plot(age_gyr, y_feh, color="red", linewidth=2, zorder=3, label=pretty_label)
            best_model_feh = np.array(y_feh)
            best_plotted = True
        else:
            ax_main.plot(age_gyr, y_feh, color='gray', alpha=0.2, linewidth=1, zorder=1)
    
    # Create age bins for observational data
    age_bins = np.linspace(0, 14, n_bins + 1)
    bin_centers = (age_bins[:-1] + age_bins[1:]) / 2
    
    # Bin Joyce data
    mask_joyce = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    if np.sum(mask_joyce) > 0:
        bin_means_joyce, _, _ = binned_statistic(age_Joyce[mask_joyce], Fe_H[mask_joyce], 
                                               statistic='mean', bins=age_bins)
        bin_stds_joyce, _, _ = binned_statistic(age_Joyce[mask_joyce], Fe_H[mask_joyce], 
                                              statistic='std', bins=age_bins)
        
        # Plot binned Joyce data as line with error bars
        valid_joyce = np.isfinite(bin_means_joyce)
        ax_main.plot(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                    color='blue', linewidth=3, linestyle='-', 
                    label='Joyce (binned)', zorder=5)
        ax_main.errorbar(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                        yerr=bin_stds_joyce[valid_joyce], 
                        color='blue', alpha=0.3, capsize=3, zorder=4)
    
    # Bin Bensby data
    mask_bensby = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    if np.sum(mask_bensby) > 0:
        bin_means_bensby, _, _ = binned_statistic(age_Bensby[mask_bensby], Fe_H[mask_bensby], 
                                                statistic='mean', bins=age_bins)
        bin_stds_bensby, _, _ = binned_statistic(age_Bensby[mask_bensby], Fe_H[mask_bensby], 
                                               statistic='std', bins=age_bins)
        
        # Plot binned Bensby data as line with error bars
        valid_bensby = np.isfinite(bin_means_bensby)
        ax_main.plot(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                    color='orange', linewidth=3, linestyle='-', 
                    label='Bensby (binned)', zorder=5)
        ax_main.errorbar(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                        yerr=bin_stds_bensby[valid_bensby], 
                        color='orange', alpha=0.3, capsize=3, zorder=4)
    
    # Scatter real data (raw points)
    ax_main.scatter(age_Joyce, Fe_H, marker='*', s=50, color='blue', 
                   alpha=0.6, label='Joyce et al. (raw)', zorder=2)
    ax_main.scatter(age_Bensby, Fe_H, marker='^', s=50, color='orange', 
                   alpha=0.6, label='Bensby et al. (raw)', zorder=2)
    


    
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
            from scipy.interpolate import interp1d
            age_gyr_model = np.linspace(0, 14, len(best_model_feh))
            f_model = interp1d(age_gyr_model, best_model_feh, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
            y_model_interp = f_model(x_vals)
            
            ax_main.fill_between(x_vals, y_joyce, y_model_interp, color='purple', alpha=0.1, zorder=0)
            ax_main.fill_between(x_vals, y_model_interp, y_bensby, color='purple', alpha=0.1, zorder=0)
    
    # Create KDE plots on the right margin
    feh_vals = np.linspace(-2, 1, 200)
    
    # KDE for Joyce data
    mask_joyce_kde = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    if np.sum(mask_joyce_kde) > 2:
        joyce_feh_data = Fe_H[mask_joyce_kde]
        kde_joyce = gaussian_kde(joyce_feh_data)
        kde_j = kde_joyce(feh_vals)
        kde_j_norm = kde_j / np.max(kde_j) if np.max(kde_j) > 0 else kde_j
        ax_kde.plot(kde_j_norm, feh_vals, color='blue', linewidth=4, label='Joyce')
        ax_kde.fill_betweenx(feh_vals, 0, kde_j_norm, color='blue', alpha=0.3)
    
    # KDE for Bensby data  
    mask_bensby_kde = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    if np.sum(mask_bensby_kde) > 2:
        bensby_feh_data = Fe_H[mask_bensby_kde]
        kde_bensby = gaussian_kde(bensby_feh_data)
        kde_b = kde_bensby(feh_vals)
        kde_b_norm = kde_b / np.max(kde_b) if np.max(kde_b) > 0 else kde_b
        ax_kde.plot(kde_b_norm, feh_vals, color='orange', linewidth=4, label='Bensby')
        ax_kde.fill_betweenx(feh_vals, 0, kde_b_norm, color='orange', alpha=0.3)
    
    # KDE for best model
    if best_model_feh is not None and len(best_model_feh) > 2:
        finite_model = best_model_feh[np.isfinite(best_model_feh)]
        if len(finite_model) > 2:
            kde_model = gaussian_kde(finite_model)
            kde_m = kde_model(feh_vals)
            kde_m_norm = kde_m / np.max(kde_m) if np.max(kde_m) > 0 else kde_m
            ax_kde.plot(kde_m_norm, feh_vals, color='red', linestyle='--', linewidth=4, label='Best Model')
    
    # Set reasonable limits
    ax_kde.set_xlim(0, 1.2)
    
    # Axis control for main plot
    ax_main.set_xlim(0, 14)
    ax_main.set_ylim(-2, 1)
    ax_main.set_xlabel("Age (Gyr)", fontsize=16)
    ax_main.set_ylabel("[Fe/H]", fontsize=16)
    ax_main.legend(loc="upper left", bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=10)
    
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
    print(f"Enhanced age-metallicity plot with KDE and binning saved to {save_path}")
    return fig



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
    for metric in ['wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh', 'ks', 'ensemble']:
        if metric in df.columns:
            metrics_dict[metric] = df[metric].values
    
    return sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals, sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals, metrics_dict, df



def plot_pca_degeneracy_analysis(GalGA, results_file='GA/simulation_results.csv', save_path='GA/pca_degeneracy_analysis.png'):
    """
    Perform PCA analysis on final population to reveal parameter degeneracies.
    Shows how population spreads along degenerate manifolds vs constrained directions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Load results and extract continuous parameters
    df = pd.read_csv(results_file)
    
    # Define continuous parameter names and extract values
    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                        'sfe', 'delta_sfe', 'imf_upper', 'nb']
    
    # Extract parameter matrix
    param_matrix = df[continuous_params].values
    
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
    
    # 1. Eigenvalue/Singular value plot
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(explained_variance)), explained_variance, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue (Variance)')
    ax1.set_yscale('log')
    
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
    plt.colorbar(im, ax=ax3, fraction=0.02)
    
    # Add text annotations for strong loadings
    for i in range(6):
        for j in range(len(continuous_params)):
            if abs(components[i, j]) > 0.5:
                ax3.text(j, i, f'{components[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white' if abs(components[i, j]) > 0.7 else 'black')
    
    # 4. 2D projections onto first few PCs
    # Color points by fitness
    colors = df['wrmse'].values if 'wrmse' in df.columns else np.ones(len(df))
    
    # PC1 vs PC2
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax4.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    ax4.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    plt.colorbar(scatter, ax=ax4, label='Fitness (WRMSE)')
    
    # PC2 vs PC3
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(pca_result[:, 1], pca_result[:, 2], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax5.set_xlabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    ax5.set_ylabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)')
    
    # PC3 vs PC4
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(pca_result[:, 2], pca_result[:, 3], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax6.set_xlabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)')
    ax6.set_ylabel(f'PC4 ({explained_variance_ratio[3]:.1%} variance)')
    
    # 5. Example parameter pair showing degeneracy
    ax7 = fig.add_subplot(gs[1, 3])
    # Find the most correlated parameter pair
    param_corr = np.corrcoef(param_matrix_scaled.T)
    np.fill_diagonal(param_corr, 0)  # Remove self-correlation
    max_corr_idx = np.unravel_index(np.argmax(np.abs(param_corr)), param_corr.shape)
    
    param1_idx, param2_idx = max_corr_idx
    param1_name = continuous_params[param1_idx]
    param2_name = continuous_params[param2_idx]
    
    ax7.scatter(df[param1_name], df[param2_name], c=colors, cmap='viridis', alpha=0.6, s=20, label=f'r = {param_corr[max_corr_idx]:.3f}')
    ax7.set_xlabel(param1_name)
    ax7.set_ylabel(param2_name)
    ax7.legend()

    
    
    # 6. Parameter distributions along degenerate vs constrained directions
    ax8 = fig.add_subplot(gs[2, :2])
    
    # Project onto most and least constrained directions
    most_constrained = pca_result[:, 0]  # Highest variance PC
    least_constrained = pca_result[:, -1]  # Lowest variance PC
    
    ax8.hist(most_constrained, bins=30, alpha=0.7, label=f'Most constrained (PC1, λ={explained_variance[0]:.3f})', color='blue')
    ax8.hist(least_constrained, bins=30, alpha=0.7, label=f'Least constrained (PC{len(explained_variance)}, λ={explained_variance[-1]:.6f})', color='red')
    ax8.set_xlabel('Projection Value')
    ax8.set_ylabel('Count')
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
    Create a correlation matrix heatmap showing parameter relationships.
    Complements the PCA analysis by showing direct pairwise correlations.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    df = pd.read_csv(results_file)
    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                        'sfe', 'delta_sfe', 'imf_upper', 'nb']
    
    # Calculate correlation matrix
    corr_matrix = df[continuous_params].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find strongest correlations
    corr_values = corr_matrix.values
    np.fill_diagonal(corr_values, 0)  # Remove self-correlations
    
    # Get indices of strongest positive and negative correlations
    max_pos_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
    min_neg_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
    
    print(f"Strongest positive correlation: {continuous_params[max_pos_idx[0]]} - {continuous_params[max_pos_idx[1]]} (r = {corr_values[max_pos_idx]:.3f})")
    print(f"Strongest negative correlation: {continuous_params[min_neg_idx[0]]} - {continuous_params[min_neg_idx[1]]} (r = {corr_values[min_neg_idx]:.3f})")
    
    return fig

# Update the generate_all_plots function to include the four-panel alpha plot
def generate_all_plots(GalGA, feh, normalized_count, results_file='GA/simulation_results.csv'):
    """Generate all plots from GalGA results including four-panel alpha plot"""
    
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
    
    sigma_2_vals,t_1_vals,t_2_vals,infall_1_vals,infall_2_vals,sfe_vals,delta_sfe_vals,imf_upper_vals,mgal_vals,nb_vals, metrics_dict, df = extract_metrics(results_file)
    
    # 1. Plot MDF curves (existing)
    plot_mdf_curves(GalGA, feh, normalized_count, df)
    
    # 2. Plot Four-Panel Alpha Elements (NEW)
    print("Generating Four-Panel Alpha Elements plot...")
    plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, df)

    print("Generating AGE  plot...")
    plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, df, save_path='GA/Age_FeH_detailed_results.png')


    # 4. Plot 2D scatter plots
    print("Generating 2D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        plot_2d_scatter(t_2_vals, infall_2_vals, metric_vals, metric_name + '_t2_in2', xlabel='t_2', ylabel='infall_2')
        plot_2d_scatter(t_1_vals, infall_2_vals, metric_vals, metric_name + '_t1_in2', xlabel='t_1', ylabel='infall_2')
        plot_2d_scatter(t_1_vals, t_2_vals, metric_vals, metric_name + '_sfe_in2', xlabel='sfe', ylabel='infall_2')
        plot_2d_scatter(sfe_vals, sigma_2_vals, metric_vals, metric_name + '_sfe_sigma2', xlabel='sfe', ylabel='sigma_2')
        plot_2d_scatter(sfe_vals, t_2_vals, metric_vals, metric_name + '_sfe_t2', xlabel='sfe', ylabel='t_2')
        plot_2d_scatter(delta_sfe_vals, t_2_vals, metric_vals, metric_name + '_sfe_t2', xlabel='delta sfe', ylabel='t_2')
        plot_2d_scatter(delta_sfe_vals, sfe_vals, metric_vals, metric_name + '_deltasfe_sfe', xlabel='delta sfe', ylabel='sfe')
        plot_2d_scatter(nb_vals, imf_upper_vals, metric_vals, metric_name + '_nb_imf', xlabel='SN1a per Solar Mass', ylabel='IMF')
    

    print("Generating 3D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, metric_vals, metric_name)
        plot_3d_scatter(sfe_vals, t_1_vals, infall_2_vals, metric_vals, metric_name + '_sii', xlabel='sfe', ylabel='t_1', zlabel='infall_2')
        plot_3d_scatter(sfe_vals, delta_sfe_vals, infall_2_vals, metric_vals, metric_name + '_sii', xlabel='sfe', ylabel='delta sfe', zlabel='infall_2')
        plot_3d_scatter(t_1_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_tti', xlabel='t_1', ylabel='t_2', zlabel='infall_2')
        plot_3d_scatter(delta_sfe_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_sti', xlabel='delta sfe', ylabel='t_2', zlabel='infall_2')
        plot_3d_scatter(nb_vals, imf_upper_vals, infall_2_vals, metric_vals, metric_name + '_nti', xlabel='SN1a per Solar Mass', ylabel='IMF', zlabel='infall_2')
    

    
    # 5. Walker evolution plots
    print("Generating walker evolution plots...")

    param_names = ["sigma_2", "t_2", "infall_2", "sfe", "delta_sfe"]
    # Indices of the parameters in the GA individual's gene list
    param_indices = [5, 7, 9, 10, 11]
    plot_walker_history(GalGA.walker_history, param_names, param_indices)
    
    # 6. Plot loss history for each walker
    print("Generating walker loss history plots...")
    for metric in ['wrmse', 'huber', 'ks', 'cosine']:
        plot_walker_loss_history(GalGA.walker_history, results_file, loss_metric=metric)
        
    # 7. Create 3D animation
    print("Generating 3D animation...")
    #create_3d_animation(GalGA.walker_history)
    
    # NEW: PCA degeneracy analysis
    print("Generating PCA degeneracy analysis...")
    plot_pca_degeneracy_analysis(GalGA, results_file)
    
    print("Generating parameter correlation matrix...")
    plot_parameter_correlation_matrix(results_file)
    
    print("All plotting complete! Check the GA directory for results.")
    print(f"Loaded {len(Fe_H)} observational data points for individual alpha elements")
