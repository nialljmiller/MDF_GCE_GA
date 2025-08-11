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
from loss_plot import *
from analysis_plot import run_analysis
import age_meta
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
    os.makedirs('GA/analysis', exist_ok=True)


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


def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None, save_path='GA/MDF_multiple_results.png'):
    """
    Plot all model MDFs, highlight the best model, overlay data, and show residuals.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Create figure with subplots - main plot and residuals
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(9, 8), 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    
    # Determine best model parameters
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
    
    best_flag = False
    best_x = None
    best_y = None
    
    # Plot all model curves on main panel
    for (x, y), label, res in zip(GalGA.mdf_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            best_x = np.array(x)
            best_y = np.array(y)
            ax_main.plot(x, y, color='C3', linewidth=2.5,
                        label='Best Model' if not best_flag else None)
            best_flag = True
        else:
            ax_main.plot(x, y, color='gray', alpha=0.01)
    
    # Plot observational data on main panel
    ax_main.plot(feh, normalized_count, 'x', ms=8, color='k', label='Observational Data')
    
    # Calculate and plot residuals
    if best_x is not None and best_y is not None:
        # Interpolate best model to observational data points
        # Only interpolate within the model's [Fe/H] range
        model_min, model_max = np.min(best_x), np.max(best_x)
        
        # Filter observational data to model range
        obs_mask = (feh >= model_min) & (feh <= model_max)
        feh_filtered = feh[obs_mask]
        obs_filtered = normalized_count[obs_mask]
        
        if len(feh_filtered) > 0:
            # Interpolate model to observational points
            interp_func = interp1d(best_x, best_y, kind='linear', 
                                 bounds_error=False, fill_value=np.nan)
            model_interp = interp_func(feh_filtered)
            
            # Calculate residuals (model - observations)
            residuals = model_interp - obs_filtered
            
            # Plot residuals
            ax_res.plot(feh_filtered, residuals, 'rx', ms=6, alpha=0.8, label='Residuals')
            ax_res.axhline(0, color='k', linestyle='--', alpha=0.5)
            
            # Calculate and display RMS residual
            valid_residuals = residuals[~np.isnan(residuals)]
            if len(valid_residuals) > 0:
                rms_residual = np.sqrt(np.mean(valid_residuals**2))
                ax_res.text(0.02, 0.95, f'RMS = {rms_residual:.3f}', 
                           transform=ax_res.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Format main plot
    ax_main.set_ylabel('Normalized Number Density')
    ax_main.set_xlim(-2, 1)
    ax_main.legend(loc='upper left', frameon=False)
    ax_main.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels from main plot
    
    # Format residuals plot
    ax_res.set_xlabel('[Fe/H]')
    ax_res.set_ylabel('Model - Obs')
    ax_res.set_xlim(-2, 1)
    ax_res.grid(True, alpha=0.3)
    
    # Set reasonable y-limits for residuals
    if 'residuals' in locals() and len(valid_residuals) > 0:
        res_std = np.std(valid_residuals)
        ax_res.set_ylim(-3*res_std, 3*res_std)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


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
        ax.view_init(elev=20, azim=num % 360)  # One rotation, loops if more frames
        
        for i, (walker_id, history) in enumerate(walker_history.items()):
            if not history:
                continue
            history = np.array(history)
            generations = np.arange(len(history))
            
            # Plot full path using indices for t_2 (7) and infall_2 (9)
            ax.plot(generations, history[:, 7], history[:, 9],
                    color=colors[i], alpha=0.7)  # No legend to save resources
    
    # Create animation with one full rotation
    total_frames = 360
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
    
    # Save as GIF with lower fps and dpi
    gif_path = "GA/loss/walker_evolution_3D.gif"
    ani.save(gif_path, writer="pillow", fps=5, dpi=72)
    plt.close()
    
    print(f"Generated 3D animation: {gif_path}")
    return ani






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







def plot_omni_info_figure(GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, 
                          feh_mdf, normalized_count_mdf, results_df=None, 
                          save_path='GA/Omni_Info_Figure.png'):
    """
    Create a dashboard showing the best-fit model parameters and performance
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
    
    # Create figure with custom layout
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
    param_text += f"σ₂ (second infall radio): {best_row['sigma_2']:.1f} \n"
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
    
    print(f"dashboard saved to {save_path}")
    print(f"Best-fit parameters:")
    print(f"  σ₂ = {best_row['sigma_2']:.1f}")
    print(f"  t₂ = {best_row['t_2']:.3f} Gyr") 
    print(f"  τ₂ = {best_row['infall_2']:.3f} Gyr")
    print(f"  SFE = {best_row['sfe']:.5f}")
    print(f"  Fitness = {best_row['fitness']:.4f}")
    
    return fig






def generate_all_plots(GalGA, feh, normalized_count, results_file='GA/simulation_results.csv'):
    """Generate all plots from GalGA results including parameter combinations"""
    
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
    
    run_analysis(GalGA, results_file)

    # 8. PCA degeneracy analysis
    print("Generating PCA degeneracy analysis...")
    try:
        run_analysis(GalGA, results_file)
    except:
        print("probably not enough samples yet...")

    # 1. Plot MDF curves (existing)
    plot_mdf_curves(GalGA, feh, normalized_count, df)
    
    # 2. Plot Four-Panel Alpha Elements
    print("Generating Four-Panel Alpha Elements plot...")
    plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, df)

    print("Generating more physics plots...")
    generate_physics_plots(GalGA, results_file=results_file)

    # 3. 2D scatter plots
    print("Generating 2D scatter plots...")
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

    # 4. 3D scatter plots
    print("Generating 3D scatter plots...")
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
    for metric in ['wrmse', 'huber', 'ks', 'cosine', 'ensemble']:
        plot_walker_loss_history(GalGA.walker_history, results_file, loss_metric=metric)
        
        plot_multiple_success_thresholds(GalGA.walker_history, results_csv=results_file, thresholds=[0.01, 0.1, 0.001], loss_metric=metric)



    # 7. Create 3D animation
    #print("Generating 3D animation...")
    #create_3d_animation(GalGA.walker_history)


    # Generate the omni info figure
    print("Generating dashboard figure...")
    plot_omni_info_figure(GalGA, Fe_H, age_Joyce, age_Bensby, 
                          Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
                          feh, normalized_count, df)
    
    print("Omni info figure generated!")


    # FIXED: Import age_meta and pass DataFrame instead of string
    # Pass the DataFrame (df) instead of the file path string (results_file)
    age_meta.plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=df, save_path='GA/Age_FeH_detailed_results.png', n_bins=10)

    print("All plotting complete! Check the GA directory for results.")
    print(f"Generated parameter space exploration plots:")
    print(f"- {len(metrics_dict)} metrics × 24 2D plots = {len(metrics_dict) * 24} 2D scatter plots")
    print(f"- {len(metrics_dict)} metrics × 16 3D plots = {len(metrics_dict) * 16} 3D scatter plots")
    print(f"- Plus walker evolution, loss history, PCA analysis, and correlation matrix plots")
    print(f"Loaded {len(Fe_H)} observational data points for individual alpha elements")