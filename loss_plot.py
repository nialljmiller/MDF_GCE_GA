#!/usr/bin/env python3.8
################################
# Plotting functions for MDF_GA
################################
# Authors: N Miller

"""Plotting utilities for MDF_GA and related bulge diagnostics."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors, gridspec
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
import os
from scipy.interpolate import UnivariateSpline
from numpy.polynomial.polynomial import Polynomial
from phys_plot import generate_physics_plots





def plot_corner_of_top_params(
    GalGA,
    results_file='simulation_results.csv',
    losscol='fitness',
    top_k=8,
    bins_1d=40,
    bins_2d=35,
    save_path=None,
    preferred_params=(
        'sigma_2','t_1','t_2','infall_1','infall_2','sfe',
        'nb','mgal','imf_upper','comp_idx','imf_idx','sn1a_idx','sy_idx','sn1ar_idx'
    ),
    min_unique=20,
    triangle='lower',           # lower triangle only
    label_fs=14,                # axis label fontsize
    tick_fs=12                  # tick label fontsize
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.gridspec import GridSpec
    from scipy.stats import spearmanr

    # ---------- data ----------
    df = pd.read_csv(results_file)
    if losscol not in df.columns:
        raise ValueError(f"loss column '{losscol}' not found in {results_file}")

    def is_continuous(col: pd.Series) -> bool:
        if np.issubdtype(col.dtype, np.integer):
            return False
        v = col.values
        v = v[np.isfinite(v)]
        return (len(np.unique(v)) >= min_unique)

    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != losscol]
    cont = [c for c in num_cols if is_continuous(df[c])]

    chosen = [c for c in preferred_params if c in cont]
    if len(chosen) < top_k:
        y = df[losscol].values
        m_y = np.isfinite(y)
        ranks = []
        for c in [c for c in cont if c not in chosen]:
            x = df[c].values
            m = m_y & np.isfinite(x)
            if np.count_nonzero(m) < 20:
                continue
            rho, _ = spearmanr(x[m], y[m])
            if np.isfinite(rho):
                ranks.append((c, abs(float(rho))))
        ranks.sort(key=lambda t: t[1], reverse=True)
        chosen.extend([c for c, _ in ranks[:max(0, top_k-len(chosen))]])

    top_params = chosen[:top_k]
    if not top_params:
        raise ValueError("No continuous parameters available after filtering.")

    # ---------- figure layout (no wasted space) ----------
    if save_path is None:
        base = getattr(GalGA, "output_path", "GA/")
        save_path = os.path.join(base, "analysis", f"corner_top{len(top_params)}_{losscol}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    K = len(top_params)

    # Build a (K x (K+1)) grid; last column reserved for colorbar
    fig = plt.figure(figsize=(2.35*K, 2.15*K))
    gs = GridSpec(K, K+1, figure=fig, width_ratios=[*(1 for _ in range(K)), 0.05],
                  wspace=0.06, hspace=0.06)

    # loss scale / colormap
    y = df[losscol].values
    finite_y = np.isfinite(y)
    y_valid = y[finite_y]
    if len(y_valid) < 10:
        raise ValueError("Insufficient finite loss values for plotting.")
    vmin, vmax = float(np.nanmin(y_valid)), float(np.nanmax(y_valid))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap('viridis')
    q01 = np.percentile(y_valid, 1)
    q05 = np.percentile(y_valid, 5)

    data = {c: df[c].values.astype(float) for c in top_params}

    used_axes = []
    for i, pi in enumerate(top_params):
        xi = data[pi]
        fi = np.isfinite(xi) & finite_y
        xig = xi[fi]; yg = y[fi]

        for j, pj in enumerate(top_params[:i+1]):   # lower triangle only
            ax = fig.add_subplot(gs[i, j])
            used_axes.append(ax)

            if i == j:
                # diagonal: histogram; NO y ticks/labels
                ax.hist(xig, bins=bins_1d, color='0.78', edgecolor='none')
                best5 = yg <= q05
                if np.any(best5):
                    ax.hist(xig[best5], bins=bins_1d, histtype='step', linewidth=1.8, color='red')
                # axis cosmetics
                ax.tick_params(axis='y', left=False, labelleft=False)   # <- remove y ticks/labels
                if i == K-1:
                    ax.set_xlabel(pi.replace('_',' '), fontsize=label_fs)
                else:
                    ax.tick_params(labelbottom=False)
                # make x ticks readable (scientific if needed)
                ax.tick_params(axis='x', labelsize=tick_fs)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
                # minimal spines
                for s in ('top','right'):
                    ax.spines[s].set_visible(False)

            else:
                xj = data[pj]
                fj = np.isfinite(xj)
                m = fi & fj
                xv = xi[m]; yv = xj[m]; lv = y[m]
                if len(xv) == 0:
                    continue

                hb = ax.hexbin(
                    xv, yv, C=lv, reduce_C_function=np.nanmedian,
                    gridsize=bins_2d, cmap=cmap, norm=norm, mincnt=1
                )
                best1 = lv <= q01
                if np.any(best1):
                    ax.plot(xv[best1], yv[best1], '.', ms=1.2, color='k', alpha=0.6)

                # labels on bottom row / left col only
                if i == K-1:
                    ax.set_xlabel(pj.replace('_',' '), fontsize=label_fs)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0:
                    ax.set_ylabel(pi.replace('_',' '), fontsize=label_fs)
                else:
                    ax.tick_params(labelleft=False)

                ax.tick_params(axis='both', labelsize=tick_fs)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))

    # slim colorbar in dedicated column (no outer whitespace)
    cax = fig.add_subplot(gs[:, -1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.set_label(f'{losscol}', fontsize=label_fs)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[corner] Saved: {save_path}")
    return fig, top_params




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

    ax.set_xlabel(f"Generation ({loss_metric.upper()})")
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








#!/usr/bin/env python3.8
################################
# Fixed plotting functions for MDF_GA
################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, iqr, gaussian_kde
from scipy.ndimage import gaussian_filter

def plot_marginal_loss(df, param, losscol='fitness', bins=60,
                       agg='median', save_path='GA/analysis/marginal_loss.png'):
    """
    1D marginal: aggregated loss vs a single parameter, with sample counts.
    
    Fixed version that properly handles the aggregation function.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = df[param].values
    z = df[losscol].values
    
    # Create bins
    bin_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Bin the data
    idx = np.digitize(x, bin_edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    
    # Aggregate per bin
    vals = [[] for _ in range(bins)]
    for ii, zz in zip(idx, z):
        if 0 <= ii < bins and np.isfinite(zz):
            vals[ii].append(zz)
    
    # Apply aggregation function
    if agg == 'mean':
        agg_vals = np.array([np.mean(v) if len(v) > 0 else np.nan for v in vals])
    elif agg == 'median':
        agg_vals = np.array([np.median(v) if len(v) > 0 else np.nan for v in vals])
    elif agg == 'min':
        agg_vals = np.array([np.min(v) if len(v) > 0 else np.nan for v in vals])
    else:
        raise ValueError("agg must be 'mean', 'median', or 'min'")
    
    # Count samples per bin
    counts = np.array([len(v) for v in vals])
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    # Plot aggregated values
    valid_mask = np.isfinite(agg_vals)
    ax1.plot(bin_centers[valid_mask], agg_vals[valid_mask], 'b-', lw=2, label=f'{agg} {losscol}')
    ax1.set_xlabel(param.replace('_', ' '))
    ax1.set_ylabel(f'{agg} {losscol}')
    ax1.grid(True, alpha=0.3)
    
    # Add histogram on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.2, 
            edgecolor='none', color='gray', label='samples/bin')
    ax2.set_ylabel('samples / bin')
    ax2.set_ylim(0, np.max(counts) * 1.1)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved marginal plot: {save_path}")


def plot_binned_loss(GalGA, df, xcol, ycol, losscol='fitness',
                     bins=(50, 50), agg='median', min_per_bin=6, smooth_sigma=1.0,
                     cmap='viridis', save_path=None):
    """
    Fixed version of plot_binned_loss that properly handles parameters.
    
    Creates a binned heatmap of loss values with proper parameter handling.
    """
    if save_path is None:
        save_path = os.path.join(GalGA.output_path, 'analysis', f'binned_{losscol}_{xcol}_{ycol}.png')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract data
    x = np.asarray(df[xcol].values, dtype=float)
    y = np.asarray(df[ycol].values, dtype=float)
    z = np.asarray(df[losscol].values, dtype=float)
    
    # Remove invalid data
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    
    if len(x) == 0:
        print(f"No valid data for {xcol} vs {ycol}")
        return None, None, None, None
    
    # Ensure bins is a tuple
    if isinstance(bins, int):
        bins = (bins, bins)
    elif not isinstance(bins, tuple):
        bins = (50, 50)
    
    # Aggregate data in bins
    if agg == 'mean':
        Z, xedges, yedges, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=bins)
    elif agg == 'median':
        Z, xedges, yedges, _ = binned_statistic_2d(x, y, z, statistic='median', bins=bins)
    elif agg == 'min':
        Z, xedges, yedges, _ = binned_statistic_2d(x, y, z, statistic=np.min, bins=bins)
    else:
        Z, xedges, yedges, _ = binned_statistic_2d(x, y, z, statistic='median', bins=bins)
    
    # Count samples per bin
    N, _, _, _ = binned_statistic_2d(x, y, None, statistic='count', bins=bins)
    
    # Mask bins with insufficient data
    Z = Z.astype(float)
    Z[N < min_per_bin] = np.nan
    
    # Optional smoothing
    if smooth_sigma and smooth_sigma > 0:
        Zfilled = np.nanmedian(Z) if np.isfinite(np.nanmedian(Z)) else 0.0
        Zs = gaussian_filter(np.nan_to_num(Z, nan=Zfilled), smooth_sigma)
        Z = np.where(np.isnan(Z), np.nan, Zs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.pcolormesh(xedges, yedges, Z.T, shading='auto', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{agg} {losscol}')
    
    # Add contours
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    
    try:
        valid_contour = np.isfinite(Z)
        if np.count_nonzero(valid_contour) > 5:
            ax.contour(Xc, Yc, Z.T, levels=7, colors='k', alpha=0.3, linewidths=0.8)
    except Exception:
        pass
    
    # Overlay sample points
    ax.scatter(x, y, s=6, c='k', alpha=0.1, linewidths=0, zorder=3)
    
    # Mark low-count bins
    low_count = (N < min_per_bin).T
    if np.any(low_count):
        ax.contourf(Xc, Yc, low_count, levels=[0.5, 1.5], 
                    colors='none', hatches=['///'], alpha=0)
        ax.text(0.98, 0.02, f'hatched: N<{min_per_bin}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax.set_xlabel(xcol.replace('_', ' '))
    ax.set_ylabel(ycol.replace('_', ' '))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    
    print(f"Saved binned loss plot: {save_path}")
    return Z, xedges, yedges, N


def plot_delta_and_gradient(xcol, ycol, Z, xedges, yedges, save_prefix='GA/analysis/binned_loss', quiver_step=3):
    """
    Fixed version of delta and gradient plotting.
    
    From a binned loss surface Z, plot:
      (a) ΔL = Z - Z_min (relative to global min over valid bins)
      (b) |∇L| magnitude and quiver of gradient (∂L/∂x, ∂L/∂y)
    """
    # Build centers
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    
    # Mask invalid
    Zm = np.ma.masked_invalid(Z)
    if Zm.mask.all():
        print("All bins invalid; nothing to plot.")
        return
    
    Zmin = Zm.min()
    dL = Zm - Zmin
    
    # (a) Delta loss map
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.pcolormesh(xedges, yedges, dL.T, shading='auto', cmap='brg')
    c1 = fig1.colorbar(im1, ax=ax1)
    c1.set_label('Δ loss (relative to global min)')
    ax1.set_xlabel(xcol)
    ax1.set_ylabel(ycol)
    plt.tight_layout()
    fig1.savefig(f'{save_prefix}_delta.png', bbox_inches='tight', dpi=200)
    plt.close(fig1)
    
    # (b) Gradient field
    Zfill = Zm.filled(np.nanmedian(Zm))
    dZdx, dZdy = np.gradient(Zfill)
    grad_mag = np.sqrt(dZdx**2 + dZdy**2)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.pcolormesh(xedges, yedges, grad_mag.T, shading='auto', cmap='brg')
    c2 = fig2.colorbar(im2, ax=ax2)
    c2.set_label('|∇ loss|')
    
    # Quiver on subsampled grid
    xs = Xc[::quiver_step]
    ys = Yc[::quiver_step]
    U = dZdx[::quiver_step, ::quiver_step].T
    V = dZdy[::quiver_step, ::quiver_step].T
    ax2.quiver(xs, ys, U, V, color='k', alpha=0.6, pivot='mid')
    
    ax2.set_xlabel(xcol)
    ax2.set_ylabel(ycol)
    plt.tight_layout()
    fig2.savefig(f'{save_prefix}_grad.png', bbox_inches='tight', dpi=200)
    plt.close(fig2)
    
    print(f"Saved delta and gradient plots: {save_prefix}_delta.png, {save_prefix}_grad.png")


# Example of how to fix the calling code in generate_all_plots:
def fixed_analysis_section(GalGA, df, analysis_dir):
    """
    Fixed version of the analysis section that was causing errors.
    """
    # Key pairs we care about most
    key_pairs = [
        ('t_2', 'infall_2'),
        ('sigma_2', 't_2'),
        ('sigma_2', 'infall_2'),
    ]

    # 1D marginals (fixed function calls)
    for p in {'t_2', 'infall_2', 'sigma_2'}:
        if p in df.columns and 'fitness' in df.columns:
            try:
                plot_marginal_loss(
                    df, p, losscol='fitness', bins=60, agg='median',
                    save_path=os.path.join(analysis_dir, f'marginal_{p}.png')
                )
            except Exception as e:
                print(f"[marginal {p}] skipped: {e}")

    # 2D binned surfaces + Δ-loss + gradient fields (fixed function calls)
    for xcol, ycol in key_pairs:
        if all(c in df.columns for c in [xcol, ycol, 'fitness']):
            try:
                out_base = os.path.join(analysis_dir, f"binned_fitness_{xcol}_{ycol}")
                Z, xedges, yedges, N = plot_binned_loss(
                    GalGA, df, xcol=xcol, ycol=ycol, losscol='fitness',
                    bins=(50, 50), agg='median', min_per_bin=6, smooth_sigma=1.0,
                    save_path=out_base + ".png"
                )
                if Z is not None:
                    plot_delta_and_gradient(
                        Z, xedges, yedges, save_prefix=out_base, quiver_step=3
                    )
            except Exception as e:
                print(f"[binned {xcol} vs {ycol}] skipped: {e}")
        else:
            missing = [c for c in [xcol, ycol, 'fitness'] if c not in df.columns]
            print(f"[binned {xcol} vs {ycol}] missing columns: {missing}")


def _auto_bins_1d(x, nmax=60, nmin=8):
    """Freedman–Diaconis rule with sane guards."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 50:
        return max(nmin, int(np.sqrt(n)))
    
    try:
        h = 2 * iqr(x, nan_policy='omit') * n ** (-1/3)
        if not np.isfinite(h) or h <= 0:
            return max(nmin, min(nmax, int(np.sqrt(n))))
        bins = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / h))
        return max(nmin, min(nmax, bins))
    except:
        return max(nmin, min(nmax, int(np.sqrt(n))))