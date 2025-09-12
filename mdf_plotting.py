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
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
})

# ---------------------------------------------------
def ensure_dirs(output_path):
    """Ensure necessary directories exist."""
    os.makedirs(output_path + 'loss', exist_ok=True)
    os.makedirs(output_path + 'analysis', exist_ok=True)


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
def plot_sfr_history(bulge_dict, output_path, save_path=None):
    """
    Plot star formation rate (SFR) history vs Age for bulge models.
    bulge_dict: mapping of label -> model with inner.history.age and .sfr_abs
    """
    if save_path is None:
        save_path = output_path + 'SFR_history.png'
    
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig



# ---------------------------------------------------
def plot_mass_evolution(bulge_dict, output_path, save_path=None):
    """
    Plot bulge mass (locked + gas) evolution vs Age.
    bulge_dict: mapping of label -> model with inner.history.m_locked, .m_gas_exp (or .m_gas)
    """
    if save_path is None:
        save_path = output_path + 'Mass_age.png'
    
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig

# ---------------------------------------------------
def plot_alpha_histograms(obs_dict, model_dict, output_path, bins=25, save_path=None):
    """
    Plot histograms of alpha-element distributions for observation and models.
    obs_dict: {'[Mg/Fe]': array, ...}
    model_dict: {'label': [array_Mg, array_Si, array_Ca, array_Ti], ...}
    """
    if save_path is None:
        save_path = output_path + 'alpha_histograms.png'
    
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig

# ---------------------------------------------------
# Existing MDF and GA plotting functions (slightly tweaked for style)
# ---------------------------------------------------
def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None, save_path=None):
    """
    Plot all model MDFs, highlight the best model, overlay data, and show residuals.
    """
    if save_path is None:
        save_path = GalGA.output_path + 'MDF_multiple_results.png'
    
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

    alpha = 10/len(GalGA.results)    
    # Plot all model curves on main panel
    for (x, y), label, res in zip(GalGA.mdf_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            best_x = np.array(x)
            best_y = np.array(y)
            ax_main.plot(x, y, color='C3', linewidth=2.5, zorder=10, label='Best Model' if not best_flag else None)
            best_flag = True
        else:
            ax_main.plot(x, y, linewidth=1, color='gray', alpha=alpha)
    
    # Plot observational data on main panel
    ax_main.plot(feh, normalized_count, 'x', ms=8, color='k', zorder=11, label='Observational Data')
    
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
                ax_res.text(0.02, 0.9, f'RMS = {rms_residual:.3f}', 
                           transform=ax_res.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Format main plot
    ax_main.set_ylabel('Normalized Number Density')
    ax_main.set_xlim(-2, 1)
    ax_main.legend(loc='upper left', frameon=False)
    ax_main.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels from main plot
    
    # Format residuals plot
    ax_res.set_xlabel('[Fe/H]')
    ax_res.set_ylabel('Model - Obs')
    ax_res.set_xlim(-2, 1)
    
    # Set reasonable y-limits for residuals
    if 'residuals' in locals() and len(valid_residuals) > 0:
        res_std = np.std(valid_residuals)
        ax_res.set_ylim(-3*res_std, 3*res_std)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


def create_3d_animation(walker_history, output_path):
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
    gif_path = output_path + "loss/walker_evolution_3D.gif"
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    ani.save(gif_path, writer="pillow", fps=5, dpi=72)
    plt.close()
    
    print(f"Generated 3D animation: {gif_path}")
    return ani

def plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=None, save_path=None):
    if save_path is None:
        save_path = GalGA.output_path + 'Four_Panel_Alpha.png'

    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    observational_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.01, wspace=0.2, left=0.08, right=0.92, top=0.97, bottom=0.08)

    # precompute color array once (unused for color now, but keeps mask logic)
    color_array = (Mg_Fe + Si_Fe + Ca_Fe + Ti_Fe) / 4.0

    for idx, (element, obs_data) in enumerate(zip(element_names, observational_data)):
        row, col = divmod(idx, 2)
        ax_main = axes[row, col]

        # side KDE axis (no sharing)
        rect = ax_main.get_position()
        ax_kde = fig.add_axes([rect.x1 + 0.002, rect.y0, 0.07, rect.height])

        # draw model curves (best in red, rest faint gray)
        for alpha_arrs, _, res in zip(GalGA.alpha_data, GalGA.labels, GalGA.results):
            params = (res[5], res[7], res[9])
            if idx < len(alpha_arrs):
                x_data = np.array(alpha_arrs[idx][0])
                y_data = np.array(alpha_arrs[idx][1])
                if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)):
                    ax_main.plot(x_data, y_data, color="red", linewidth=2.5, zorder=3)
                else:
                    ax_main.plot(x_data, y_data, color='gray', alpha=0.01, linewidth=1, zorder=1)

        # clean obs & scatter (black points)
        obs_clipped = np.where((obs_data >= -2.0) & (obs_data <= 2.0), obs_data, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_clipped) & np.isfinite(color_array)
        if np.sum(mask) > 10:
            x = Fe_H[mask]
            y = obs_clipped[mask]
            ax_main.scatter(x, y, c='k', s=16, zorder=2, edgecolor='none')

        # KDEs (obs vs best model)
        joint_mask = np.isfinite(obs_clipped) & np.isfinite(Fe_H)
        y_vals = np.linspace(-0.8, 1.0, 200)

        if np.sum(joint_mask) > 2:
            kde_obs_y = gaussian_kde(obs_clipped[joint_mask])
            kde_y = kde_obs_y(y_vals)
            kde_y /= np.max(kde_y)
            ax_kde.plot(kde_y, y_vals, linestyle='-', color='darkblue')
            ax_kde.fill_betweenx(y_vals, 0, kde_y, alpha=0.30)

        best_y_model = None
        for alpha_arrs, _, res in zip(GalGA.alpha_data, GalGA.labels, GalGA.results):
            params = (res[5], res[7], res[9])
            if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)) and idx < len(alpha_arrs):
                best_y_model = np.array(alpha_arrs[idx][1])
                break

        if best_y_model is not None:
            finite = np.isfinite(best_y_model)
            if np.sum(finite) > 2:
                kde_model_y = gaussian_kde(best_y_model[finite])
                kde_model = kde_model_y(y_vals)
                kde_model /= np.max(kde_model)
                ax_kde.plot(kde_model, y_vals, linestyle='--', color='red')
                ax_kde.fill_betweenx(y_vals, 0, kde_model, alpha=0.20)

        # limits, labels, axes layout
        ax_main.set_xlim(-2.0, 1.0)
        ax_main.set_ylim(-0.8, 0.8)
        ax_main.set_xlabel("[Fe/H]")

        # y-axis handling per column
        if col == 0:
            ax_main.set_ylabel(r"[$\alpha$/Fe]")   # only left column
            ax_main.yaxis.set_ticks_position('left')
            ax_main.yaxis.set_label_position('left')
            ax_main.tick_params(axis='y', which='both', left=True, right=False)
        else:
            # right column: y-axis on the right, no y-label text
            ax_main.set_ylabel("")
            ax_main.yaxis.set_ticks_position('right')
            ax_main.yaxis.set_label_position('right')
            ax_main.tick_params(axis='y', which='both', left=False, right=False)

        # put x-axis on top for the top row (to match your previous style)
        if row == 0:
            ax_main.xaxis.set_ticks_position('top')
            ax_main.xaxis.set_label_position('top')

        # element tag in top-left inside axes
        ax_main.text(0.1, 0.9, element, transform=ax_main.transAxes,
                     ha='left', va='top', fontsize=22, weight='bold')

        # clean KDE axis
        ax_kde.set_xticks([]); ax_kde.set_yticks([])
        ax_kde.set_xlim(0.0, 1.0)
        ax_kde.set_ylim(-0.8, 1.0)
        for spine in ax_kde.spines.values():
            spine.set_visible(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Density-enhanced four-panel alpha plot saved to {save_path}")


def plot_omni_info_figure(GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, 
                          feh_mdf, normalized_count_mdf, results_df=None, 
                          save_path=None):
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
    
    if save_path is None:
        save_path = GalGA.output_path + 'Omni_Info_Figure.png'
    
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

def plot_omni_figure(
    GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
    feh_mdf, normalized_count_mdf, results_df=None, save_path=None
):
    """
    ApJ-clean figure: MDF (top-left), AMR (top-right), 4×alpha panels (bottom).
    Minimal legends/labels. Tight spacing. Same IO pattern as your code.
    Returns the Matplotlib Figure.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os

    # ------ Minimal, print-safe style ------
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "legend.frameon": False,
    })

    if save_path is None:
        save_path = os.path.join(getattr(GalGA, "output_path", ""), "Omni_Info_Figure_ApJ.png")


    # ------ Select best model tuple ------
    if results_df is not None and hasattr(results_df, "empty") and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm["sigma_2"], bm["t_2"], bm["infall_2"])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    # ------ Figure layout (tight, no wasted whitespace) ------
    fig = plt.figure(figsize=(15, 8))  # ApJ 2-col width
    gs = GridSpec(
        2, 8, figure=fig,
        left=0.065, right=0.995, bottom=0.10, top=0.965,
        wspace=0.16, hspace=0.2  # small gap between rows, as requested
    )

    # Top row
    ax_mdf = fig.add_subplot(gs[0, 0:4])
    ax_amr = fig.add_subplot(gs[0, 4:8])

    # ------ MDF ------
    best_x = best_y = None
    for (x, y), res in zip(GalGA.mdf_data, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            best_x, best_y = np.asarray(x), np.asarray(y)
        else:
            ax_mdf.plot(x, y, color="0.75", alpha=0.001, lw=0.8, zorder=1)

    if best_x is not None:
        ax_mdf.plot(best_x, best_y, color="crimson", lw=1.8, label="Model", zorder=3)

    ax_mdf.plot(feh_mdf, normalized_count_mdf, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    ax_mdf.set_xlim(-2, 1)
    ax_mdf.set_ylabel("Normalized number")

    # x-axis at top only
    ax_mdf.xaxis.set_ticks_position("top")
    ax_mdf.xaxis.set_label_position("top")
    ax_mdf.set_xlabel("[Fe/H]")
    ax_mdf.tick_params(axis="x", bottom=False)

    ax_mdf.legend(loc="upper left", fontsize=9, handlelength=1.6)

    # ------ AMR (y-axis on right) ------
    best_age_x = best_age_y = None
    for (t_arr, feh_arr), res in zip(GalGA.age_data, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            t = np.asarray(t_arr, float)  # years
            age = (t[-1] - t) / 1e9       # Age (Gyr), increasing to the right
            best_age_x, best_age_y = age, np.asarray(feh_arr, float)
        else:
            t = np.asarray(t_arr, float)  # years
            age = (t[-1] - t) / 1e9       # Age (Gyr), increasing to the right
            age_x, age_y = age, np.asarray(feh_arr, float)            
            ax_amr.plot(age_x, age_y, color="0.75", alpha=0.001, lw=0.8, zorder=1)

    ax_amr.scatter(age_Joyce, Fe_H, s=10, facecolor="none", edgecolor="0.35", lw=0.7, label="Joyce")
    ax_amr.scatter(age_Bensby, Fe_H, s=10, marker="^", facecolor="none", edgecolor="0.55", lw=0.7, label="Bensby")
    if best_age_x is not None:
        ax_amr.plot(best_age_x, best_age_y, color="crimson", lw=1.8, label="Model", zorder=3)

    ax_amr.set_xlim(0, 14)
    ax_amr.set_ylim(-2, 1)

    # x-axis at top only
    ax_amr.xaxis.set_ticks_position("top")
    ax_amr.xaxis.set_label_position("top")
    ax_amr.set_xlabel("Age (Gyr)")
    ax_amr.tick_params(axis="x", bottom=False)

    # y-axis on right
    ax_amr.yaxis.tick_right()
    ax_amr.yaxis.set_label_position("right")
    ax_amr.set_ylabel("[Fe/H]")

    ax_amr.legend(loc="lower left", fontsize=9, ncol=3, columnspacing=0.9, handlelength=1.6)

    # ------ Alpha row ------
    alpha_elems = ["Mg", "Si", "Ca", "Ti"]
    alpha_obs   = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]
    axes_alpha  = [fig.add_subplot(gs[1, 2*i:2*i+2]) for i in range(4)]

    # Fetch best alpha arrays once
    best_alpha = None
    for alpha_arrs, res in zip(GalGA.alpha_data, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            best_alpha = alpha_arrs
            break

    xlim = (-2, 1)
    ylim = (-0.6, 0.8)

    for i, (elt, obs, ax) in enumerate(zip(alpha_elems, alpha_obs, axes_alpha)):
        # Observations
        obs_clean = np.where((obs > -2.5) & (obs < 2.5), obs, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_clean)
        if np.count_nonzero(mask) > 5:
            ax.scatter(Fe_H[mask], obs_clean[mask], s=10, color="0.35", alpha=0.9, edgecolor="none", label="Data")

        # Model
        if best_alpha is not None and i < len(best_alpha):
            mx, my = best_alpha[i]
            ax.plot(mx, my, color="crimson", lw=1.6, label="Model")

        # Limits, labels
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("[Fe/H]")

        # Only leftmost has y-label
        if i == 0:
            ax.set_ylabel("[α/Fe]")
        else:
            ax.set_ylabel("")

        # Element tag
        ax.text(0.03, 0.95, elt, transform=ax.transAxes, ha="left", va="top", fontsize=17)

        # Middle two: hide y-numbering (keep ticks for alignment)
        if i in (1, 2):
            ax.set_yticklabels([])

        # Rightmost: y-axis on right (no y-label)
        if i == 3:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    # Single small legend for the alpha set inside the last panel
    h, l = axes_alpha[-1].get_legend_handles_labels()
    if h:
        axes_alpha[-1].legend(loc="lower right", fontsize=9, handlelength=1.6)

    # ------ Save & return ------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    return fig
















def generate_all_plots(GalGA, feh, normalized_count, results_file=None):
    """Generate all plots from GalGA results including parameter combinations"""
    
    if results_file is None:
        results_file = GalGA.output_path + 'simulation_results.csv'
    
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
    ensure_dirs(GalGA.output_path)
    
    # Extract metrics for scatter plots
    sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals, sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals, metrics_dict, df = extract_metrics(results_file)

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

    metric_name = 'fitness'
    metric_vals = metrics_dict['fitness']        


    plot_corner_of_top_params(
        GalGA,
        results_file=results_file,
        losscol='fitness',
        top_k=8,
        preferred_params=('sigma_2','t_1','t_2','infall_1','infall_2','sfe','nb','mgal'),
        min_unique=20,
        triangle='lower'
    )



    # === NEW: binned loss + delta + gradient + 1D marginals for key parameter pairs ===
    analysis_dir = os.path.join(GalGA.output_path, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Key pairs we care about most
    key_pairs = [
        ('t_2', 'infall_2'),
        ('sigma_2', 't_2'),
        ('sigma_2', 'infall_2'),
    ]

    # 1D marginals (quick structure scans)
    for p in {'t_2', 'infall_2', 'sigma_2'}:
        if p in df.columns and 'fitness' in df.columns:
            try:
                plot_marginal_loss(
                    df, p, losscol='fitness', bins=50, agg='median',
                    save_path=os.path.join(analysis_dir, f'marginal_{p}.png')
                )
            except Exception as e:
                print(f"[marginal {p}] skipped: {e}")

    # 2D binned surfaces + Δ-loss + gradient fields
    for xcol, ycol in key_pairs:
        if all(c in df.columns for c in [xcol, ycol, 'fitness']):
            try:
                out_base = os.path.join(analysis_dir, f"binned_fitness_{xcol}_{ycol}")
                Z, xedges, yedges, N = plot_binned_loss(
                    GalGA, df, xcol=xcol, ycol=ycol, losscol='fitness',
                    bins=(50, 50), agg='median', min_per_bin=1, smooth_sigma=1.0,
                    cmap='rainbow', save_path=out_base + ".png"
                )
                plot_delta_and_gradient(xcol, ycol,
                    Z, xedges, yedges, save_prefix=out_base, quiver_step=3
                )
            except Exception as e:
                print(f"[binned {xcol} vs {ycol}] skipped: {e}")
        else:
            missing = [c for c in [xcol, ycol, 'fitness'] if c not in df.columns]
            print(f"[binned {xcol} vs {ycol}] missing columns: {missing}")




    # ========== INFALL PARAMETERS ==========
    # Second infall episode (most important)

    plot_2d_scatter(GalGA, t_2_vals, infall_2_vals, metric_vals, metric_name + '_t2_infall2', xlabel='t_2 (Gyr)', ylabel='infall_2 (Gyr)')
    plot_2d_scatter(GalGA, sigma_2_vals, infall_2_vals, metric_vals, metric_name + '_sigma2_infall2', xlabel='sigma_2', ylabel='infall_2 (Gyr)')
    plot_2d_scatter(GalGA, sigma_2_vals, t_2_vals, metric_vals, metric_name + '_sigma2_t2', xlabel='sigma_2', ylabel='t_2 (Gyr)')
    
    # First infall episode
    plot_2d_scatter(GalGA, t_1_vals, infall_1_vals, metric_vals, metric_name + '_t1_infall1', xlabel='t_1 (Gyr)', ylabel='infall_1 (Gyr)')
    plot_2d_scatter(GalGA, t_1_vals, infall_2_vals, metric_vals, metric_name + '_t1_infall2', xlabel='t_1 (Gyr)', ylabel='infall_2 (Gyr)')
    
    # Cross-infall comparisons
    plot_2d_scatter(GalGA, t_1_vals, t_2_vals, metric_vals, metric_name + '_t1_t2', xlabel='t_1 (Gyr)', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, infall_1_vals, infall_2_vals, metric_vals, metric_name + '_infall1_infall2', xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)')
    
    # ========== STAR FORMATION EFFICIENCY ==========
    plot_2d_scatter(GalGA, sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_sfe_deltasfe', xlabel='SFE', ylabel='Delta SFE')
    plot_2d_scatter(GalGA, sfe_vals, t_2_vals, metric_vals, metric_name + '_sfe_t2', xlabel='SFE', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, sfe_vals, sigma_2_vals, metric_vals, metric_name + '_sfe_sigma2', xlabel='SFE', ylabel='sigma_2')
    plot_2d_scatter(GalGA, delta_sfe_vals, t_2_vals, metric_vals, metric_name + '_deltasfe_t2', xlabel='Delta SFE', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, delta_sfe_vals, infall_2_vals, metric_vals, metric_name + '_deltasfe_infall2', xlabel='Delta SFE', ylabel='infall_2 (Gyr)')
    
    # ========== GALAXY MASS RELATIONS ==========
    plot_2d_scatter(GalGA, mgal_vals, sfe_vals, metric_vals, metric_name + '_mgal_sfe', xlabel='M_gal (M_sun)', ylabel='SFE')
    plot_2d_scatter(GalGA, mgal_vals, sigma_2_vals, metric_vals, metric_name + '_mgal_sigma2', xlabel='M_gal (M_sun)', ylabel='sigma_2')
    plot_2d_scatter(GalGA, mgal_vals, t_2_vals, metric_vals, metric_name + '_mgal_t2', xlabel='M_gal (M_sun)', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, mgal_vals, infall_2_vals, metric_vals, metric_name + '_mgal_infall2', xlabel='M_gal (M_sun)', ylabel='infall_2 (Gyr)')
    
    # ========== IMF AND STELLAR PARAMETERS ==========
    plot_2d_scatter(GalGA, imf_upper_vals, sfe_vals, metric_vals, metric_name + '_imf_sfe', xlabel='IMF Upper (M_sun)', ylabel='SFE')
    plot_2d_scatter(GalGA, imf_upper_vals, t_2_vals, metric_vals, metric_name + '_imf_t2', xlabel='IMF Upper (M_sun)', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, imf_upper_vals, mgal_vals, metric_vals, metric_name + '_imf_mgal', xlabel='IMF Upper (M_sun)', ylabel='M_gal (M_sun)')
    plot_2d_scatter(GalGA, nb_vals, imf_upper_vals, metric_vals, metric_name + '_nb_imf', xlabel='SN1a per Solar Mass', ylabel='IMF Upper (M_sun)')
    
    # ========== SN1A PARAMETERS ==========
    plot_2d_scatter(GalGA, nb_vals, sfe_vals, metric_vals, metric_name + '_nb_sfe', xlabel='SN1a per Solar Mass', ylabel='SFE')
    plot_2d_scatter(GalGA, nb_vals, t_2_vals, metric_vals, metric_name + '_nb_t2', xlabel='SN1a per Solar Mass', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, nb_vals, mgal_vals, metric_vals, metric_name + '_nb_mgal', xlabel='SN1a per Solar Mass', ylabel='M_gal (M_sun)')
    plot_2d_scatter(GalGA, nb_vals, sigma_2_vals, metric_vals, metric_name + '_nb_sigma2', xlabel='SN1a per Solar Mass', ylabel='sigma_2')

    # ========== INFALL-FOCUSED 3D PLOTS ==========
    # Primary infall relationships
    plot_3d_scatter(GalGA, sigma_2_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_infall2_complete',
                   xlabel='sigma_2', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, t_1_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_timing_comparison',
                   xlabel='t_1 (Gyr)', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, infall_1_vals, infall_2_vals, sigma_2_vals, metric_vals, metric_name + '_infall_timescales',
                   xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)', zlabel='sigma_2')
    
    # ========== SFE-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, sfe_vals, delta_sfe_vals, infall_2_vals, metric_vals, metric_name + '_sfe_evolution',
                   xlabel='SFE', ylabel='Delta SFE', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, sfe_vals, t_1_vals, infall_2_vals, metric_vals, metric_name + '_sfe_timing',
                   xlabel='SFE', ylabel='t_1 (Gyr)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, sfe_vals, t_2_vals, sigma_2_vals, metric_vals, metric_name + '_sfe_infall2_params',
                   xlabel='SFE', ylabel='t_2 (Gyr)', zlabel='sigma_2')
    plot_3d_scatter(GalGA, delta_sfe_vals, t_2_vals, infall_2_vals, metric_vals, metric_name + '_deltasfe_timing',
                   xlabel='Delta SFE', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
    
    # ========== GALAXY MASS-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, mgal_vals, sfe_vals, infall_2_vals, metric_vals, metric_name + '_mgal_sfe_infall',
                   xlabel='M_gal (M_sun)', ylabel='SFE', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, mgal_vals, t_2_vals, sigma_2_vals, metric_vals, metric_name + '_mgal_infall2_params',
                   xlabel='M_gal (M_sun)', ylabel='t_2 (Gyr)', zlabel='sigma_2')
    plot_3d_scatter(GalGA, mgal_vals, sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_mgal_sfe_evolution',
                   xlabel='M_gal (M_sun)', ylabel='SFE', zlabel='Delta SFE')
    
    # ========== STELLAR/IMF-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, imf_upper_vals, sfe_vals, infall_2_vals, metric_vals, metric_name + '_imf_sfe_infall',
                   xlabel='IMF Upper (M_sun)', ylabel='SFE', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, nb_vals, imf_upper_vals, infall_2_vals, metric_vals, metric_name + '_stellar_params_infall',
                   xlabel='SN1a per Solar Mass', ylabel='IMF Upper (M_sun)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, nb_vals, sfe_vals, t_2_vals, metric_vals, metric_name + '_sn1a_sfe_timing',
                   xlabel='SN1a per Solar Mass', ylabel='SFE', zlabel='t_2 (Gyr)')
    
    # ========== CROSS-PARAMETER EXPLORATION ==========
    plot_3d_scatter(GalGA, sigma_2_vals, sfe_vals, mgal_vals, metric_vals, metric_name + '_sigma_sfe_mgal',
                   xlabel='sigma_2', ylabel='SFE', zlabel='M_gal (M_sun)')
    plot_3d_scatter(GalGA, t_1_vals, sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_t1_sfe_evolution',
                   xlabel='t_1 (Gyr)', ylabel='SFE', zlabel='Delta SFE')
    plot_3d_scatter(GalGA, infall_1_vals, infall_2_vals, sfe_vals, metric_vals, metric_name + '_infall_timescales_sfe',
                   xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)', zlabel='SFE')

    # 5. Walker evolution plots
    print("Generating walker evolution plots...")
    param_names = ["sigma_2", "t_2", "infall_2", "sfe", "delta_sfe"]
    param_indices = [5, 7, 9, 10, 11]
    plot_walker_history(GalGA, GalGA.walker_history, param_names, param_indices)

    # 6. Plot loss history for each walker
    print("Generating walker loss history plots...")

    for metric in ['ks', 'huber','cosine', 'log_cosh', 'fitness', 'age_meta_fitness', 'physics_penalty']:
        plot_walker_loss_history(GalGA, GalGA.walker_history, results_file, loss_metric=metric)
        
        plot_multiple_success_thresholds(GalGA, GalGA.walker_history, results_csv=results_file, thresholds=[0.01, 0.1, 0.001], loss_metric=metric)

    # 7. Create 3D animation
    #print("Generating 3D animation...")
    #create_3d_animation(GalGA.walker_history, GalGA.output_path)

    # Generate the omni info figure
    print("Generating dashboard figure...")
    plot_omni_info_figure(GalGA, Fe_H, age_Joyce, age_Bensby, 
                          Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
                          feh, normalized_count, df)


    plot_omni_figure(GalGA, Fe_H, age_Joyce, age_Bensby, 
                      Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
                      feh, normalized_count, df)



    
    print("Omni info figure generated!")

    # FIXED: Import age_meta and pass DataFrame instead of string
    # Pass the DataFrame (df) instead of the file path string (results_file)
    age_meta.plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=df, n_bins=10)

    print("Generating Age-Metallicity curves with residuals...")
    age_meta.plot_age_metallicity_curves(GalGA, Fe_H, age_Joyce, age_Bensby, df)
    
    plt.close('all')               # (optional) belt-and-suspenders at the end of an iteration

    print("All plotting complete! Check the output directory for results.")
    print(f"Generated parameter space exploration plots:")
    print(f"- {len(metrics_dict)} metrics × 24 2D plots = {len(metrics_dict) * 24} 2D scatter plots")
    print(f"- {len(metrics_dict)} metrics × 16 3D plots = {len(metrics_dict) * 16} 3D scatter plots")
    print(f"- Plus walker evolution, loss history, PCA analysis, and correlation matrix plots")
    print(f"Loaded {len(Fe_H)} observational data points for individual alpha elements")
