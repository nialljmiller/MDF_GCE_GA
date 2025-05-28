#!/usr/bin/env python3.8
################################
# Plotting functions for MDF_GA
################################

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import pandas as pd
from scipy.interpolate import UnivariateSpline


def ensure_dirs():
    """Ensure necessary directories exist"""
    os.makedirs('GA/loss', exist_ok=True)


#!/usr/bin/env python3
"""
Plotting functions for MDF_GA and related bulge diagnostics.
Each function is standalone and saves a publication-quality figure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------
# Global style for paper-quality figures
# ---------------------------------------------------
plt.style.use('seaborn-whitegrid')
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
            ax.plot(x, y, color='C3', linewidth=2.5,
                    label='Best Model' if not best_flag else None)
            best_flag = True
        else:
            ax.plot(x, y, color='gray', alpha=0.4)

    # observational data
    ax.plot(feh, normalized_count, 'x', ms=8, color='k', label='Data')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Normalized Number Density')
    ax.set_xlim(-2, 1)
    ax.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig





def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 12))
    ax = plt.gca()

    # determine the best‐model params
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    best_plotted = False
    for (x_data, y_data), label, res in zip(GalGA.mdf_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))

        if is_best:
            if not best_plotted:
                # make a little bullet-list out of your comma-separated features
                pieces = [f"• {p.strip()}" for p in label.split(',')]
                pretty_label = "\n".join(pieces) + "\n• (BEST)"
                ax.plot(x_data, y_data,
                        label=pretty_label,
                        color="red", linewidth=2, zorder=3)
                best_plotted = True
            else:
                ax.plot(x_data, y_data,
                        color="red", linewidth=2, zorder=3)
        else:
            ax.plot(x_data, y_data,
                    alpha=0.5, zorder=1)

    # observational data
    ax.plot(feh, normalized_count,
            label="Observational Data",
            color="black", marker="x", linestyle="-", markersize=8, zorder=2)

    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Normalized Number Density")
    ax.set_xlim(-2, 1)
    ax.set_title("Metallicity Distribution Functions (MDFs)")

    # legend out to the left so multiline shows up
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1), frameon=False)

    plt.tight_layout()
    plt.savefig("GA/MDF_multiple_results.png", bbox_inches="tight")
    return plt.gcf()

def plot_3d_scatter(x, y, z, color_metric, label, xlabel='sigma_2', ylabel='t_2', zlabel='infall_2'):
    """Plot 3D scatter plot with color indicating a specific metric"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'3D Parameter Space Colored by {label}')
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
    """Plot the evolution of parameters for each walker"""
    if not walker_history:
        print("Walker history data not available. Skipping walker evolution plots.")
        return None
        
    figs = []
    for idx, param_name in enumerate(param_names):
        fig = plt.figure(figsize=(12, 8))
        figs.append(fig)
        
        for walker_idx, history in walker_history.items():
            if not history:  # Skip if history is empty
                continue
                
            history = np.array(history)  # Convert to numpy array for easier slicing
            param_idx = param_indices[idx]
            
            if param_idx >= history.shape[1]:  # Skip if parameter index is out of bounds
                continue
                
            generations = np.arange(len(history))
            
            # Plot the parameter value for this walker
            plt.plot(
                generations, 
                history[:, param_idx], 
                label=f"Walker {walker_idx}",
                alpha=0.5  # Adjust transparency for better visualization
            )
        
        plt.xlabel("Generation")
        plt.ylabel(f"{param_name} Value")
        plt.title(f"Evolution of {param_name} Over Generations")
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'GA/loss/walker_evolution_{param_name}.png', bbox_inches='tight')
        plt.close()
    
    print("Generated walker evolution plots")
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
        ax.set_title("Walker Evolution in 3D")
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

def extract_metrics(results_file):
    """Extract metrics from CSV file for plotting"""
    # Load the dataframe directly
    df = pd.read_csv(results_file)
    
    # Extract parameters using column names
    sigma_2_vals = df['sigma_2'].values
    t_2_vals = df['t_2'].values
    infall_2_vals = df['infall_2'].values
    
    # Extract metrics
    metrics_dict = {}
    for metric in ['wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh', 'ks', 'ensemble']:
        if metric in df.columns:
            metrics_dict[metric] = df[metric].values
    
    return sigma_2_vals, t_2_vals, infall_2_vals, metrics_dict, df



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
    plt.title(f"{loss_metric.upper()} Loss Over Generations")
    plt.xlabel("Generation")
    plt.ylabel(f"{loss_metric.upper()} Loss")
    plt.grid(True, alpha=0.3)
    
    # Only show legend for first few walkers to avoid clutter
    plt.legend(loc='upper right', fontsize='small')
    
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
        ax.set_title("3D Scatter Plot of Individuals with Gene Bounds")
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
        ax.set_title("2D Scatter Plot of Individuals with Gene Bounds")
        ax.set_xlabel(gene_names[0])
        ax.set_ylabel(gene_names[1])
        ax.legend()
        plt.tight_layout()
        plt.savefig('GA/MDF_individuals_2D.png', bbox_inches='tight')
        print('...2D plot made!')









def generate_all_plots(GalGA, feh, normalized_count, results_file='GA/simulation_results.csv'):
    """Generate all plots from GalGA results"""
    # Ensure directories exist
    ensure_dirs()
    
    # Extract metrics for scatter plots
    sigma_2_vals, t_2_vals, infall_2_vals, metrics_dict, df = extract_metrics(results_file)
    
    # 1. Plot MDF curves
    plot_mdf_curves(GalGA, feh, normalized_count, df)
    
    # 2. Plot 3D scatter plots for various metrics
    print("Generating 3D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, metric_vals, metric_name)
    
    # 3. Plot 2D scatter plots
    print("Generating 2D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        plot_2d_scatter(t_2_vals, infall_2_vals, metric_vals, metric_name)
    

    print("Generating walker evolution plots...")
    param_names = ["sigma_2", "t_2", "infall_2"]
    param_indices = [0, 1, 5, 7, 9]  # Indices in the individual arrays
    plot_walker_history(GalGA.walker_history, param_names, param_indices)
    
    # NEW: Plot loss history for each walker
    print("Generating walker loss history plots...")
    for metric in ['wrmse', 'huber', 'mae', 'cosine']:
        plot_walker_loss_history(GalGA.walker_history, results_file, loss_metric=metric)

    # 5. Create 3D animation
    print("Generating 3D animation...")
    create_3d_animation(GalGA.walker_history)


    print("All plotting complete! Check the GA directory for results.")