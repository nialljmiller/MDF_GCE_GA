import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import gaussian_kde, binned_statistic, ks_2samp, mannwhitneyu
from scipy import stats
import os

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




def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function (robust to outliers)"""
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean() * 10



def calculate_likelihood_metrics(model_vals, obs_vals, obs_uncertainties):
    """Calculate likelihood-based metrics"""
    # Ensure minimum uncertainty to avoid division by zero
    uncertainties = np.maximum(obs_uncertainties, 0.05)
    
    # Log-likelihood assuming Gaussian errors
    residuals = obs_vals - model_vals
    chi2 = np.sum((residuals / uncertainties)**2)
    log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                             2*np.sum(np.log(uncertainties)))
    
    # AIC (lower is better)
    n_params = 3  # Assume 3 model parameters
    aic = 2 * n_params - 2 * log_likelihood
    
    # BIC (lower is better)
    bic = n_params * np.log(len(obs_vals)) - 2 * log_likelihood
    
    return log_likelihood, aic, bic, chi2/len(obs_vals)

def bootstrap_comparison(model_vals, obs_vals, obs_uncertainties, n_bootstrap=1000):
    """Bootstrap resampling for robust metric estimation"""
    n_points = len(obs_vals)
    mae_scores = []
    rmse_scores = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_points, n_points, replace=True)
        
        # Add noise based on uncertainties
        noisy_obs = obs_vals[idx] + np.random.normal(0, obs_uncertainties[idx])
        
        # Calculate metrics
        mae_scores.append(np.mean(np.abs(model_vals[idx] - noisy_obs)))
        rmse_scores.append(np.sqrt(np.mean((model_vals[idx] - noisy_obs)**2)))
    
    return {
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores), 
        'rmse_std': np.std(rmse_scores)
    }


def robust_regression_metrics(model_vals, obs_vals):
    """Calculate robust regression metrics"""
    residuals = model_vals - obs_vals
    
    metrics = {
        'huber_loss': huber_loss(obs_vals, model_vals),
        'mad': np.median(np.abs(residuals)),  # Median Absolute Deviation
        'p90_error': np.percentile(np.abs(residuals), 90),  # 90th percentile error
        'p95_error': np.percentile(np.abs(residuals), 95),  # 95th percentile error
        'iqr_error': np.percentile(np.abs(residuals), 75) - np.percentile(np.abs(residuals), 25),
        'trimmed_mean_error': stats.trim_mean(np.abs(residuals), 0.1)  # 10% trimmed mean
    }
    
    return metrics

def weighted_ks_test(ages1, feh1, ages2, feh2, weights1=None, weights2=None):
    """Weighted Kolmogorov-Smirnov test between two age-metallicity relations"""
    if weights1 is None:
        weights1 = np.ones(len(ages1))
    if weights2 is None:
        weights2 = np.ones(len(ages2))
    
    # Create common age grid
    min_age = max(np.min(ages1), np.min(ages2))
    max_age = min(np.max(ages1), np.max(ages2))
    age_grid = np.linspace(min_age, max_age, 100)
    
    # Interpolate metallicities to common grid
    feh1_interp = np.interp(age_grid, ages1, feh1)
    feh2_interp = np.interp(age_grid, ages2, feh2)
    
    # Calculate KS statistic
    ks_stat = np.max(np.abs(feh1_interp - feh2_interp))
    
    return ks_stat




def calculate_all_metrics(model_ages, model_feh, obs_ages, obs_feh, obs_uncertainties, dataset_name):
    """Calculate comprehensive metrics for model vs observations"""
    
    # Interpolate model to observation ages
    model_interp = np.interp(obs_ages, model_ages, model_feh)
    
    results = {'dataset': dataset_name}
    
    # Basic metrics
    residuals = model_interp - obs_feh
    results['mae'] = np.mean(np.abs(residuals))
    results['rmse'] = np.sqrt(np.mean(residuals**2))
    results['mape'] = np.mean(np.abs(residuals / np.maximum(np.abs(obs_feh), 0.1))) * 100
    
    # Weighted metrics
    weights = 1.0 / np.maximum(obs_uncertainties, 0.05)
    results['weighted_mae'] = np.average(np.abs(residuals), weights=weights)
    results['weighted_rmse'] = np.sqrt(np.average(residuals**2, weights=weights))
    
    # Likelihood-based metrics
    log_likelihood, aic, bic, chi2_reduced = calculate_likelihood_metrics(
        model_interp, obs_feh, obs_uncertainties)
    results['log_likelihood'] = log_likelihood
    results['aic'] = aic
    results['bic'] = bic
    results['chi2_reduced'] = chi2_reduced
    
    # Bootstrap metrics
    bootstrap_results = bootstrap_comparison(model_interp, obs_feh, obs_uncertainties)
    results.update(bootstrap_results)
    
    # Robust regression metrics
    robust_results = robust_regression_metrics(model_interp, obs_feh)
    results.update(robust_results)
    
    # Correlation metrics
    correlation, p_value = stats.pearsonr(model_interp, obs_feh)
    results['correlation'] = correlation
    results['correlation_p_value'] = p_value
    
    # Spearman rank correlation (robust to outliers)
    spearman_corr, spearman_p = stats.spearmanr(model_interp, obs_feh)
    results['spearman_correlation'] = spearman_corr
    results['spearman_p_value'] = spearman_p
    
    return results

def find_best_metric_for_joyce(joyce_metrics, bensby_metrics):
    """Find the metric where Joyce performs best relative to Bensby"""
    
    # Metrics where LOWER is better
    lower_is_better = ['mae', 'rmse', 'mape', 'weighted_mae', 'weighted_rmse', 
                      'aic', 'bic', 'chi2_reduced', 'mae_mean', 'rmse_mean',
                      'huber_loss', 'mad', 'p90_error', 'p95_error', 'iqr_error',
                      'trimmed_mean_error']
    
    # Metrics where HIGHER is better  
    higher_is_better = ['log_likelihood', 'correlation', 'spearman_correlation']
    
    best_ratio = 0
    best_metric = 'mae'
    best_joyce_val = 0
    best_bensby_val = 0
    
    for metric in lower_is_better:
        if metric in joyce_metrics and metric in bensby_metrics:
            joyce_val = joyce_metrics[metric]
            bensby_val = bensby_metrics[metric]
            
            # Calculate how much better Joyce is (larger ratio = Joyce much better)
            if bensby_val > 0 and joyce_val > 0:
                ratio = bensby_val / joyce_val  # >1 means Joyce is better
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_metric = metric
                    best_joyce_val = joyce_val
                    best_bensby_val = bensby_val
    
    for metric in higher_is_better:
        if metric in joyce_metrics and metric in bensby_metrics:
            joyce_val = joyce_metrics[metric]
            bensby_val = bensby_metrics[metric]
            
            # Calculate how much better Joyce is
            if joyce_val > 0 and bensby_val > 0:
                ratio = joyce_val / bensby_val  # >1 means Joyce is better
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_metric = metric
                    best_joyce_val = joyce_val
                    best_bensby_val = bensby_val
    
    return best_metric, best_joyce_val, best_bensby_val, best_ratio

def create_metrics_comparison_plot(GalGA, joyce_metrics, bensby_metrics, save_path):
    """Create supplementary plot showing all metrics"""

    # Organize metrics by category
    basic_metrics = ['mae', 'rmse', 'mape', 'weighted_mae', 'weighted_rmse']
    likelihood_metrics = ['log_likelihood', 'aic', 'bic', 'chi2_reduced']
    robust_metrics = ['huber_loss', 'mad', 'p90_error', 'p95_error', 'trimmed_mean_error']
    correlation_metrics = ['correlation', 'spearman_correlation']
    bootstrap_metrics = ['mae_mean', 'rmse_mean']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Statistical Metrics Comparison: Joyce vs Bensby', fontsize=16)
    
    metric_groups = [
        (basic_metrics, 'Basic Metrics', 'lower_better'),
        (likelihood_metrics, 'Likelihood Metrics', 'mixed'),
        (robust_metrics, 'Robust Metrics', 'lower_better'),
        (correlation_metrics, 'Correlation Metrics', 'higher_better'),
        (bootstrap_metrics, 'Bootstrap Metrics', 'lower_better')
    ]
    
    for idx, (metrics, title, direction) in enumerate(metric_groups):
        if idx >= 5:  # Only 5 subplots available
            break
            
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        joyce_vals = [joyce_metrics.get(m, np.nan) for m in metrics]
        bensby_vals = [bensby_metrics.get(m, np.nan) for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, joyce_vals, width, label='Joyce', color='blue', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, bensby_vals, width, label='Bensby', color='orange', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1 + bars2, joyce_vals + bensby_vals):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add summary table in the last subplot
    ax_table = axes[1, 2]
    ax_table.axis('off')
    
    # Calculate ratios for key metrics
    comparison_data = []
    key_metrics = ['mae', 'weighted_rmse', 'huber_loss', 'log_likelihood', 'correlation']
    
    for metric in key_metrics:
        if metric in joyce_metrics and metric in bensby_metrics:
            joyce_val = joyce_metrics[metric]
            bensby_val = bensby_metrics[metric]
            
            if metric in ['log_likelihood', 'correlation']:
                ratio = joyce_val / bensby_val if bensby_val != 0 else np.inf
                better = 'Joyce' if ratio > 1 else 'Bensby'
            else:
                ratio = bensby_val / joyce_val if joyce_val != 0 else np.inf
                better = 'Joyce' if ratio > 1 else 'Bensby'
            
            comparison_data.append([metric, f'{joyce_val:.3f}', f'{bensby_val:.3f}', 
                                  f'{ratio:.3f}', better])
    
    if comparison_data:
        table = ax_table.table(cellText=comparison_data,
                             colLabels=['Metric', 'Joyce', 'Bensby', 'Ratio', 'Better'],
                             cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax_table.set_title('Key Metrics Summary')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics comparison plot saved to {save_path}")





def plot_age_metallicity_curves(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=None, save_path=None):

    """
    Plot all age-metallicity model curves, highlight the best model, overlay data, and show residuals.
    Similar to plot_mdf_curves but for age-metallicity relation.
    
    This function creates a two-panel plot:
    - Top: Age vs [Fe/H] with all models (gray) + best model (red) + observations
    - Bottom: Age vs Residuals (Model - Observations)
    """
    if save_path is None:
        save_path = GalGA.output_path + 'Age_Metallicity_multiple_results.png'
    
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import os
    
    # Check if we have age data
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age data available for plotting")
        return None
    
    # Ensure all inputs are numpy arrays to avoid indexing issues
    Fe_H = np.asarray(Fe_H, dtype=float)
    age_Joyce = np.asarray(age_Joyce, dtype=float)
    age_Bensby = np.asarray(age_Bensby, dtype=float)
    
    # Create figure with subplots - main plot and residuals
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(12, 10), 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    
    # Determine best model parameters
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
    
    best_flag = False
    best_age_x = None
    best_age_y = None
    
    alpha = 10/len(GalGA.results)
    # Plot all model curves on main panel
    for (age_data, label, res) in zip(GalGA.age_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        
        # Extract and transform age data
        x_age_raw, y_feh = age_data
        x_age_gyr = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
        
        if is_best:
            best_age_x = np.array(x_age_gyr)
            best_age_y = np.array(y_feh)
            if not best_flag:
                ax_main.plot(x_age_gyr, y_feh, color='red', linewidth=2.5,
                           label="Best", zorder=3)
                best_flag = True
            else:
                ax_main.plot(x_age_gyr, y_feh, color='red', linewidth=2.5, zorder=3)
        else:

            ax_main.plot(x_age_gyr, y_feh, color='gray', alpha=alpha, linewidth=0.5, zorder=1)
    
    # Plot observational data on main panel
    ax_main.scatter(age_Joyce, Fe_H, marker='*', s=60, color='blue', 
                   alpha=0.7, label='Joyce et al.', zorder=2)
    ax_main.scatter(age_Bensby, Fe_H, marker='^', s=60, color='orange', 
                   alpha=0.7, label='Bensby et al.', zorder=2)
    
    # Calculate and plot residuals
    residuals_calculated = False
    
    if best_age_x is not None and best_age_y is not None:
        # Create interpolation function for the best model
        try:
            # Sort model data by age for proper interpolation
            sort_idx = np.argsort(best_age_x)
            sorted_age_x = best_age_x[sort_idx]
            sorted_age_y = best_age_y[sort_idx]
            
            # Remove any duplicate age values that could cause interpolation issues
            unique_mask = np.concatenate(([True], np.diff(sorted_age_x) > 1e-10))
            unique_age_x = sorted_age_x[unique_mask]
            unique_age_y = sorted_age_y[unique_mask]
            
            if len(unique_age_x) > 1:
                interp_func = interp1d(unique_age_x, unique_age_y, kind='linear', 
                                     bounds_error=False, fill_value=np.nan)
                
                # Model age range for filtering observations
                model_age_min, model_age_max = np.min(unique_age_x), np.max(unique_age_x)
                
                # For Joyce data
                joyce_mask = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
                if np.sum(joyce_mask) > 0:
                    joyce_age_filtered = age_Joyce[joyce_mask]
                    joyce_feh_filtered = Fe_H[joyce_mask]
                    
                    # Only use Joyce data within model age range
                    joyce_in_range = ((joyce_age_filtered >= model_age_min) & 
                                     (joyce_age_filtered <= model_age_max))
                    
                    if np.sum(joyce_in_range) > 0:
                        joyce_ages_valid = joyce_age_filtered[joyce_in_range]
                        joyce_feh_valid = joyce_feh_filtered[joyce_in_range]
                        
                        # Interpolate model to Joyce ages
                        model_interp_joyce = interp_func(joyce_ages_valid)
                        
                        # Calculate residuals (model - observations)
                        residuals_joyce = model_interp_joyce - joyce_feh_valid
                        
                        # Plot residuals for valid points
                        valid_joyce_res = np.isfinite(residuals_joyce)
                        if np.sum(valid_joyce_res) > 0:
                            ax_res.scatter(joyce_ages_valid[valid_joyce_res], 
                                         residuals_joyce[valid_joyce_res], 
                                         marker='*', s=40, color='blue', alpha=0.7, 
                                         label='Joyce residuals')
                            residuals_calculated = True
                
                # For Bensby data
                bensby_mask = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
                if np.sum(bensby_mask) > 0:
                    bensby_age_filtered = age_Bensby[bensby_mask]
                    bensby_feh_filtered = Fe_H[bensby_mask]
                    
                    # Only use Bensby data within model age range
                    bensby_in_range = ((bensby_age_filtered >= model_age_min) & 
                                      (bensby_age_filtered <= model_age_max))
                    
                    if np.sum(bensby_in_range) > 0:
                        bensby_ages_valid = bensby_age_filtered[bensby_in_range]
                        bensby_feh_valid = bensby_feh_filtered[bensby_in_range]
                        
                        # Interpolate model to Bensby ages
                        model_interp_bensby = interp_func(bensby_ages_valid)
                        
                        # Calculate residuals (model - observations)
                        residuals_bensby = model_interp_bensby - bensby_feh_valid
                        
                        # Plot residuals for valid points
                        valid_bensby_res = np.isfinite(residuals_bensby)
                        if np.sum(valid_bensby_res) > 0:
                            ax_res.scatter(bensby_ages_valid[valid_bensby_res], 
                                         residuals_bensby[valid_bensby_res], 
                                         marker='^', s=40, color='orange', alpha=0.7, 
                                         label='Bensby residuals')
                            residuals_calculated = True
                
                # Calculate and display RMS residuals
                rms_text = ""
                all_residuals = []
                
                if 'residuals_joyce' in locals():
                    valid_joyce_residuals = residuals_joyce[np.isfinite(residuals_joyce)]
                    if len(valid_joyce_residuals) > 0:
                        rms_joyce = np.sqrt(np.mean(valid_joyce_residuals**2))
                        rms_text += f'Joyce RMS = {rms_joyce:.3f}\n'
                        all_residuals.extend(valid_joyce_residuals)
                
                if 'residuals_bensby' in locals():
                    valid_bensby_residuals = residuals_bensby[np.isfinite(residuals_bensby)]
                    if len(valid_bensby_residuals) > 0:
                        rms_bensby = np.sqrt(np.mean(valid_bensby_residuals**2))
                        rms_text += f'Bensby RMS = {rms_bensby:.3f}'
                        all_residuals.extend(valid_bensby_residuals)
                
                if rms_text:
                    ax_res.text(0.02, 0.95, rms_text.strip(), 
                               transform=ax_res.transAxes, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               verticalalignment='top')
                
                # Set reasonable y-limits for residuals
                if len(all_residuals) > 0:
                    res_std = np.std(all_residuals)
                    res_range = max(3*res_std, 0.5)  # Ensure minimum visible range
                    ax_res.set_ylim(-res_range, res_range)
                
        except Exception as e:
            print(f"Warning: Could not calculate residuals - {e}")
            residuals_calculated = False
    
    # Add zero line to residuals regardless of whether we calculated residuals
    ax_res.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # Format main plot
    ax_main.set_ylabel('[Fe/H]', fontsize=14)
    ax_main.set_xlim(0, 14.2)
    ax_main.set_ylim(-2, 1)
    
    # Create legend with multi-line label positioned appropriately
    legend = ax_main.legend(loc='upper left', bbox_to_anchor=(0., 1.), frameon=True, 
                          fontsize=9, facecolor='white', edgecolor='gray')
    legend.get_frame().set_alpha(0.9)
    
    ax_main.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels from main plot
    
    # Format residuals plot
    ax_res.set_xlabel('Age (Gyr)', fontsize=14)
    ax_res.set_ylabel('Model - Obs [Fe/H]', fontsize=12)
    ax_res.set_xlim(0, 14.2)

    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Age-metallicity curves with residuals saved to {save_path}")
    return fig


def plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=None, save_path='Age_FeH_detailed_results.png', n_bins=10):
    """
    Enhanced Age vs [Fe/H] plot with comprehensive statistical analysis
    """
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age data available for plotting")
        return None


    save_path = GalGA.output_path + save_path

    
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
        age_interp_grid = np.arange(0, 14 + avg_spacing, avg_spacing)
    else:
        age_interp_grid = np.linspace(0, 14, 100)
    
    # Plot model lines and extract best model
    for age_data, label, res in zip(GalGA.age_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        
        x_age_raw, y_feh = age_data
        age_gyr = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
        
        if is_best and not best_plotted:
            if len(age_gyr) > 1 and len(y_feh) > 1:
                f_best = interp1d(age_gyr, y_feh, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                best_model_age_gyr = age_interp_grid
                best_model_feh = f_best(age_interp_grid)
                
                ax_main.plot(best_model_age_gyr, best_model_feh, color="red", 
                           linewidth=2, zorder=3, label="Best model")
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
    
    # Scatter raw observational data
    ax_main.scatter(age_Joyce, Fe_H, marker='*', s=50, color='blue', 
                   alpha=0.6, label='Joyce et al. (raw)', zorder=2)
    ax_main.scatter(age_Bensby, Fe_H, marker='^', s=50, color='orange', 
                   alpha=0.6, label='Bensby et al. (raw)', zorder=2)
    
    # Calculate binned statistics
    mask_joyce = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mask_bensby = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    
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

    # For Joyce data
    joyce_uncertainties = bin_stds_joyce / np.sqrt(np.maximum(bin_counts_joyce, 1))
    valid_joyce_bins = (np.isfinite(bin_means_joyce) & 
                       np.isfinite(best_model_feh[:len(bin_centers)]) & 
                       (bin_counts_joyce > 2))
    
    if np.sum(valid_joyce_bins) > 0:
        joyce_ages_valid = bin_centers[valid_joyce_bins]
        joyce_feh_valid = bin_means_joyce[valid_joyce_bins]
        joyce_uncert_valid = joyce_uncertainties[valid_joyce_bins]
        
        joyce_metrics = calculate_all_metrics(
            best_model_age_gyr, best_model_feh,
            joyce_ages_valid, joyce_feh_valid, joyce_uncert_valid, 'Joyce')
    else:
        joyce_metrics = {}
    
    # For Bensby data
    bensby_uncertainties = bin_stds_bensby / np.sqrt(np.maximum(bin_counts_bensby, 1))
    valid_bensby_bins = (np.isfinite(bin_means_bensby) & 
                        np.isfinite(best_model_feh[:len(bin_centers)]) & 
                        (bin_counts_bensby > 2))
    
    if np.sum(valid_bensby_bins) > 0:
        bensby_ages_valid = bin_centers[valid_bensby_bins]
        bensby_feh_valid = bin_means_bensby[valid_bensby_bins]
        bensby_uncert_valid = bensby_uncertainties[valid_bensby_bins]
        
        bensby_metrics = calculate_all_metrics(
            best_model_age_gyr, best_model_feh,
            bensby_ages_valid, bensby_feh_valid, bensby_uncert_valid, 'Bensby')
    else:
        bensby_metrics = {}
    
    # Find the best metric for Joyce
    if joyce_metrics and bensby_metrics:
        best_metric, joyce_best_val, bensby_best_val, improvement_ratio = find_best_metric_for_joyce(
            joyce_metrics, bensby_metrics)

        # Create supplementary metrics comparison plot
        metrics_save_path = save_path.replace('.png', '_metrics_comparison.png')
        create_metrics_comparison_plot(GalGA, joyce_metrics, bensby_metrics, metrics_save_path)
    else:
        best_metric = 'mae'  # fallback
        joyce_best_val = 0
        bensby_best_val = 0
    
    # Plot binned data with the best metric in the labels
    valid_joyce = np.isfinite(bin_means_joyce) & (bin_counts_joyce > 0)
    ax_main.plot(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                color='blue', linewidth=3, linestyle='-', 
                label=f"Joyce (Binned)", zorder=5)
                #label=f"Joyce ({best_metric}: {joyce_best_val:.3f})", zorder=5)
    ax_main.errorbar(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                    yerr=bin_stds_joyce[valid_joyce], 
                    color='blue', alpha=0.3, capsize=3, zorder=4)
    
    valid_bensby = np.isfinite(bin_means_bensby) & (bin_counts_bensby > 0)
    ax_main.plot(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                color='orange', linewidth=3, linestyle='-', 
                label=f"Bensby (Binned)", zorder=5)
                #label=f"Bensby ({best_metric}: {bensby_best_val:.3f})", zorder=5)
    ax_main.errorbar(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                    yerr=bin_stds_bensby[valid_bensby], 
                    color='orange', alpha=0.3, capsize=3, zorder=4)
    
    # Rest of the plotting code (spline fits, KDE, etc.)
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
            f_model = interp1d(best_model_age_gyr, best_model_feh, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
            y_model_interp = f_model(x_vals)
            
            ax_main.fill_between(x_vals, y_joyce, y_model_interp, color='purple', alpha=0.1, zorder=0)
            ax_main.fill_between(x_vals, y_model_interp, y_bensby, color='purple', alpha=0.1, zorder=0)
    
    # KDE plots
    feh_vals = np.linspace(-2, 1, 200)
    
    mask_joyce_kde = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    if np.sum(mask_joyce_kde) > 2:
        joyce_feh_data = Fe_H[mask_joyce_kde]
        kde_joyce = gaussian_kde(joyce_feh_data)
        kde_j = kde_joyce(feh_vals)
        kde_j_norm = kde_j / np.max(kde_j) if np.max(kde_j) > 0 else kde_j
        ax_kde.plot(kde_j_norm, feh_vals, color='darkblue', linewidth=4, label='Joyce')
        ax_kde.fill_betweenx(feh_vals, 0, kde_j_norm, color='blue', alpha=0.3)
    
    # KDE for best model
    if hasattr(GalGA, 'mdf_data') and len(GalGA.mdf_data) > 0:
        for mdf_data, res in zip(GalGA.mdf_data, GalGA.results):
            params = (res[5], res[7], res[9])
            is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
            if is_best:
                mdf_x, mdf_y = mdf_data
                mdf_x = np.array(mdf_x)
                mdf_y = np.array(mdf_y)
                valid_mdf = np.isfinite(mdf_x) & np.isfinite(mdf_y) & (mdf_y > 0)
                if np.sum(valid_mdf) > 0:
                    mdf_x_valid = mdf_x[valid_mdf]
                    mdf_y_valid = mdf_y[valid_mdf]
                    n_samples = min(1000, int(np.sum(mdf_y_valid) * 1000))
                    if n_samples > 10:
                        samples = np.random.choice(mdf_x_valid, size=n_samples, 
                                                 p=mdf_y_valid/np.sum(mdf_y_valid))
                        kde_model = gaussian_kde(samples)
                        kde_m = kde_model(feh_vals)
                        kde_m_norm = kde_m / np.max(kde_m) if np.max(kde_m) > 0 else kde_m
                        ax_kde.plot(kde_m_norm, feh_vals, color='darkred', linestyle='--', 
                                   linewidth=4, label='Best Model')
                        ax_kde.fill_betweenx(feh_vals, 0, kde_m_norm, color='red', alpha=0.3)
                break
    
    # Final plot formatting
    ax_kde.set_xlim(0, 1.2)
    ax_main.set_xlim(0, 14)
    ax_main.set_ylim(-2, 1)
    ax_main.set_xlabel("Age (Gyr)", fontsize=16)
    ax_main.set_ylabel("[Fe/H]", fontsize=16)
    
    legend = ax_main.legend(loc="lower left", bbox_to_anchor=(0., 0.), frameon=True, 
                          fontsize=10, facecolor='white', edgecolor='gray')
    legend.get_frame().set_alpha(0.8)
    
    # Clean up KDE axis
    ax_kde.set_xticks([])
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('')
    ax_kde.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, 
                       right=False, labelright=False)
    for spine in ax_kde.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Enhanced age-metallicity plot saved to {save_path}")
    print(f"Supplementary metrics comparison saved to {metrics_save_path}")
    
    return fig




def age_meta_loss(GalGA, model_age_x, model_age_y, obs_age_data, loss_metric, dataset='joyce', 
                  n_bins=10, create_plot=False, save_path=None):
    """
    Calculate age-metallicity relation loss between model and observations using BINNED data.
    This matches the approach used in the plotting function with proper statistical validation.
    
    Parameters:
    -----------
    GalGA : object
        Genetic algorithm object containing output path
    model_age_x : array
        Model ages in years (will be converted to Gyr)
    model_age_y : array  
        Model [Fe/H] values
    obs_age_data : pandas.DataFrame
        Observational data with columns for ages and [Fe/H]
    loss_metric : str
        Loss metric to use for comparison
    dataset : str
        Which dataset to use: 'joyce' or 'bensby'
    n_bins : int
        Number of age bins to use (default 10, matching plot function)
    create_plot : bool
        Whether to create diagnostic plot
    save_path : str
        Path to save diagnostic plot (if None, auto-generated)
        
    Returns:
    --------
    float : Loss value (lower is better for most metrics)
    """
    
    # Convert model ages to Gyr (assuming input is in years)
    if np.max(model_age_x) > 100:  # Likely in years
        model_age_gyr = (model_age_x[-1] / 1e9) - np.array(model_age_x) / 1e9
    else:  # Already in Gyr
        model_age_gyr = np.array(model_age_x)
    
    model_feh = np.array(model_age_y)
    
    # Extract observational data for the specified dataset
    obs_feh = obs_age_data['[Fe/H]'].values
    
    if dataset.lower() == 'joyce':
        obs_ages = obs_age_data['Joyce_age'].values
    elif dataset.lower() == 'bensby':
        obs_ages = obs_age_data['Bensby'].values
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'joyce' or 'bensby'.")
    
    # Clean observational data - remove NaN values
    mask = np.isfinite(obs_ages) & np.isfinite(obs_feh)
    
    if np.sum(mask) < 5:
        return 10.0  # High penalty if insufficient data
    
    # Get clean data
    clean_ages = obs_ages[mask]
    clean_feh = obs_feh[mask]
    
    # Create age bins (matching the plotting function exactly)
    age_bins = np.linspace(0, 14, n_bins + 1)
    bin_centers = (age_bins[:-1] + age_bins[1:]) / 2
    
    # Calculate binned statistics for observational data
    bin_means_obs, _, _ = binned_statistic(clean_ages, clean_feh, 
                                         statistic='mean', bins=age_bins)
    bin_stds_obs, _, _ = binned_statistic(clean_ages, clean_feh, 
                                        statistic='std', bins=age_bins)
    bin_counts_obs, _, _ = binned_statistic(clean_ages, clean_feh, 
                                          statistic='count', bins=age_bins)
    
    # Only use bins with sufficient data (minimum 3 points, matching plot function)
    valid_bins = (np.isfinite(bin_means_obs) & (bin_counts_obs > 2))
    
    if np.sum(valid_bins) < 3:
        return 10.0  # High penalty if insufficient valid bins
    
    # Get valid bin data
    valid_bin_centers = bin_centers[valid_bins]
    valid_bin_means = bin_means_obs[valid_bins]
    valid_bin_stds = bin_stds_obs[valid_bins]
    valid_bin_counts = bin_counts_obs[valid_bins]
    
    # Calculate uncertainties (standard error of the mean)
    valid_bin_uncertainties = valid_bin_stds / np.sqrt(np.maximum(valid_bin_counts, 1))
    
    # CRITICAL FIX: Interpolate model to valid bin centers
    # Ensure model ages are monotonic for interpolation
    sort_idx = np.argsort(model_age_gyr)
    model_age_sorted = model_age_gyr[sort_idx]
    model_feh_sorted = model_feh[sort_idx]
    
    # Remove any duplicates in age that could cause interpolation issues
    unique_mask = np.concatenate(([True], np.diff(model_age_sorted) > 1e-10))
    model_age_unique = model_age_sorted[unique_mask]
    model_feh_unique = model_feh_sorted[unique_mask]
    
    # Interpolate model to observation bin centers
    model_interp = np.interp(valid_bin_centers, model_age_unique, model_feh_unique)
    
    # Calculate loss using binned data
    loss_value = _calculate_single_loss(model_interp, valid_bin_means, loss_metric, 
                                      uncertainties=valid_bin_uncertainties)
    
    create_plot = False
    
    if loss_value < GalGA.best_amr_loss:
        create_plot = True
        GalGA.best_amr_loss = loss_value
        print(GalGA.best_amr_loss)
    # Create diagnostic plot if requested
    if create_plot:

        save_path = f'age_meta_loss_diagnostic_{dataset}_{loss_metric}.png'

        if hasattr(GalGA, 'output_path') and GalGA.output_path is not None:
            save_path = GalGA.output_path + save_path
        
        _create_loss_diagnostic_plot(
            model_age_unique, model_feh_unique,
            clean_ages, clean_feh,
            age_bins, bin_centers, 
            bin_means_obs, bin_stds_obs, bin_counts_obs,
            valid_bins, model_interp, valid_bin_means,
            loss_value, loss_metric, dataset, save_path
        )
    
    return loss_value


def _calculate_single_loss(model_vals, obs_vals, loss_metric, uncertainties=None):
    """Calculate loss for a single dataset comparison with optional uncertainties"""
    
    if loss_metric == 'mae':
        return np.mean(np.abs(model_vals - obs_vals))
        
    elif loss_metric == 'rms' or loss_metric == 'rmse':
        return np.sqrt(np.mean((model_vals - obs_vals)**2))
        
    elif loss_metric == 'weighted_mae':
        if uncertainties is not None:
            # Use inverse uncertainties as weights
            weights = 1.0 / np.maximum(uncertainties, 0.01)
        else:
            # Use inverse of absolute values as weights (higher for low metallicity)
            weights = 1.0 / np.maximum(np.abs(obs_vals), 0.1)
        return np.average(np.abs(model_vals - obs_vals), weights=weights)
        
    elif loss_metric == 'weighted_rmse':
        if uncertainties is not None:
            weights = 1.0 / np.maximum(uncertainties, 0.01)
        else:
            weights = 1.0 / np.maximum(np.abs(obs_vals), 0.1)
        return np.sqrt(np.average((model_vals - obs_vals)**2, weights=weights))
        
    elif loss_metric == 'chi_squared':
        if uncertainties is not None:
            sigma = uncertainties
        else:
            sigma = 0.1  # Assume fixed uncertainty of 0.1 dex
        residuals = model_vals - obs_vals
        chi2 = np.sum((residuals / sigma)**2)
        return chi2 / len(obs_vals)  # Reduced chi-squared
        
    elif loss_metric == 'log_likelihood':
        if uncertainties is not None:
            sigma = uncertainties
        else:
            sigma = 0.1  # Assume fixed uncertainty of 0.1 dex
        residuals = model_vals - obs_vals
        chi2 = np.sum((residuals / sigma)**2)
        log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                                 2*np.sum(np.log(sigma)))
        return -log_likelihood  # Return negative so lower is better
        
    elif loss_metric == 'huber':
        return huber_loss(obs_vals, model_vals, delta=0.1)
        
    elif loss_metric == 'aic':
        if uncertainties is not None:
            sigma = uncertainties
        else:
            sigma = 0.1
        residuals = model_vals - obs_vals  
        chi2 = np.sum((residuals / sigma)**2)
        log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                                 2*np.sum(np.log(sigma)))
        n_params = 3  # Assume 3 model parameters
        aic = 2 * n_params - 2 * log_likelihood
        return aic
        
    elif loss_metric == 'bic':
        if uncertainties is not None:
            sigma = uncertainties
        else:
            sigma = 0.1
        residuals = model_vals - obs_vals
        chi2 = np.sum((residuals / sigma)**2) 
        log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                                 2*np.sum(np.log(sigma)))
        n_params = 3
        bic = n_params * np.log(len(obs_vals)) - 2 * log_likelihood
        return bic
        
    elif loss_metric == 'correlation':
        # Return 1 - correlation so lower is better
        corr = np.corrcoef(model_vals, obs_vals)[0, 1]
        return (1.0 - np.abs(corr))  # Use absolute correlation
        
    elif loss_metric == 'spearman_correlation':
        from scipy import stats
        # Return 1 - spearman correlation so lower is better
        if len(model_vals) > 1:
            spearman_corr, _ = stats.spearmanr(model_vals, obs_vals)
            if np.isfinite(spearman_corr):
                return 1.0 - np.abs(spearman_corr)
            else:
                return 1.0
        else:
            return 1.0
    
    else:
        # Default to MAE if unknown metric
        return np.mean(np.abs(model_vals - obs_vals))


def _create_loss_diagnostic_plot(model_age_gyr, model_feh, clean_ages, clean_feh,
                               age_bins, bin_centers, bin_means_obs, bin_stds_obs, bin_counts_obs,
                               valid_bins, model_interp, valid_bin_means,
                               loss_value, loss_metric, dataset, save_path):
    """Create diagnostic plot showing how the loss calculation works"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: Full comparison
    ax1.scatter(clean_ages, clean_feh, alpha=0.3, s=20, color='gray', 
               label=f'{dataset.title()} raw data')
    
    # Plot all bins (including invalid ones) with different styling
    valid_centers = bin_centers[valid_bins]
    valid_means = bin_means_obs[valid_bins]
    valid_stds = bin_stds_obs[valid_bins]
    
    invalid_bins = ~valid_bins & np.isfinite(bin_means_obs)
    if np.sum(invalid_bins) > 0:
        ax1.errorbar(bin_centers[invalid_bins], bin_means_obs[invalid_bins], 
                    yerr=bin_stds_obs[invalid_bins], 
                    fmt='x', color='red', alpha=0.5, markersize=8,
                    label='Excluded bins (< 3 points)')
    
    # Plot valid bins used in loss calculation
    ax1.errorbar(valid_centers, valid_means, yerr=valid_stds, 
                fmt='o', color='blue', markersize=8, linewidth=2,
                label=f'{dataset.title()} binned (used for loss)')
    
    # Plot model line
    ax1.plot(model_age_gyr, model_feh, 'r-', linewidth=2, label='Model evolution')
    
    # Plot interpolated model points at bin centers
    ax1.plot(valid_centers, model_interp, 'ro', markersize=8, 
            label='Model at bin centers')
    
    # Fill between model and observations to show differences
    ax1.fill_between(valid_centers, valid_means, model_interp, 
                    alpha=0.3, color='purple', 
                    label='Residuals')
    
    ax1.set_xlabel('Age (Gyr)', fontsize=12)
    ax1.set_ylabel('[Fe/H]', fontsize=12)
    ax1.set_title(f'Age-Metallicity Loss Calculation: {dataset.title()} Dataset\n'
                 f'Loss Metric: {loss_metric}, Value: {loss_value:.4f}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(-2, 1)
    
    # Bottom panel: Residuals
    residuals = model_interp - valid_means
    ax2.scatter(valid_centers, residuals, s=100, c='red', marker='o', zorder=3)
    ax2.errorbar(valid_centers, residuals, yerr=valid_stds, 
                fmt='none', color='red', alpha=0.7, zorder=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.fill_between(valid_centers, residuals, 0, alpha=0.3, color='red')
    
    ax2.set_xlabel('Age (Gyr)', fontsize=12)
    ax2.set_ylabel('Model - Observed [Fe/H]', fontsize=12)
    ax2.set_title('Residuals (Model - Observations)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 14)
    
    # Add statistics text
    stats_text = f"""Loss Statistics:
Mean Residual: {np.mean(residuals):.4f}
RMS Residual: {np.sqrt(np.mean(residuals**2)):.4f}
Max |Residual|: {np.max(np.abs(residuals)):.4f}
N bins used: {len(valid_centers)}
N raw points: {len(clean_ages)}
Model age range: {np.min(model_age_gyr):.1f}-{np.max(model_age_gyr):.1f} Gyr
Model [Fe/H] range: {np.min(model_feh):.2f}-{np.max(model_feh):.2f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Age-metallicity loss diagnostic plot saved to {save_path}")


# Test function that maintains exact same I/O
def test_age_meta_loss_function(GalGA):
    """Test function to demonstrate the fixed age_meta_loss function"""
    
    # Create mock data for testing
    import pandas as pd
    
    # Mock observational data
    np.random.seed(42)
    n_obs = 50
    obs_ages_joyce = np.random.uniform(1, 12, n_obs)
    obs_feh = -0.5 + 0.1 * obs_ages_joyce + np.random.normal(0, 0.2, n_obs)
    obs_ages_bensby = obs_ages_joyce + np.random.normal(0, 1, n_obs)
    
    obs_data = pd.DataFrame({
        '[Fe/H]': obs_feh,
        'Joyce_age': obs_ages_joyce,
        'Bensby': obs_ages_bensby
    })
    
    # Mock realistic model data (ages in years, as typically provided)
    model_times = np.linspace(0, 13e9, 100)  # 0 to 13 Gyr in years
    # Create realistic age-metallicity evolution (metallicity increases with time)
    model_feh = -1.5 + 1.0 * (model_times/1e9) / 13.0  # From -1.5 to -0.5 [Fe/H]
    
    # Test the function with exact same I/O as original
    loss_joyce = age_meta_loss(GalGA, model_times, model_feh, obs_data, 'mae', 'joyce', 
                              create_plot=True, save_path='test_joyce_loss.png')
    loss_bensby = age_meta_loss(GalGA, model_times, model_feh, obs_data, 'rmse', 'bensby',
                               create_plot=True, save_path='test_bensby_loss.png')
    
    print(f"Joyce MAE loss: {loss_joyce:.4f}")
    print(f"Bensby RMSE loss: {loss_bensby:.4f}")
    
    return loss_joyce, loss_bensby