import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import gaussian_kde, binned_statistic, ks_2samp, mannwhitneyu
from scipy import stats
import os

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function (robust to outliers)"""
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

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

def create_metrics_comparison_plot(joyce_metrics, bensby_metrics, save_path):
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

def plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=None, save_path='GA/Age_FeH_detailed_results.png', n_bins=10):
    """
    Enhanced Age vs [Fe/H] plot with comprehensive statistical analysis
    """
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
        create_metrics_comparison_plot(joyce_metrics, bensby_metrics, metrics_save_path)
    else:
        best_metric = 'mae'  # fallback
        joyce_best_val = 0
        bensby_best_val = 0
    
    # Plot binned data with the best metric in the labels
    valid_joyce = np.isfinite(bin_means_joyce) & (bin_counts_joyce > 0)
    ax_main.plot(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                color='blue', linewidth=3, linestyle='-', 
                label=f"Joyce ({best_metric}: {joyce_best_val:.3f})", zorder=5)
    ax_main.errorbar(bin_centers[valid_joyce], bin_means_joyce[valid_joyce], 
                    yerr=bin_stds_joyce[valid_joyce], 
                    color='blue', alpha=0.3, capsize=3, zorder=4)
    
    valid_bensby = np.isfinite(bin_means_bensby) & (bin_counts_bensby > 0)
    ax_main.plot(bin_centers[valid_bensby], bin_means_bensby[valid_bensby], 
                color='orange', linewidth=3, linestyle='-', 
                label=f"Bensby ({best_metric}: {bensby_best_val:.3f})", zorder=5)
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Enhanced age-metallicity plot saved to {save_path}")
    print(f"Supplementary metrics comparison saved to {metrics_save_path}")
    
    return fig



def age_meta_loss(model_age_x, model_age_y, obs_age_data, loss_metric, dataset='joyce'):
    """
    Calculate age-metallicity relation loss between model and observations.
    
    Parameters:
    -----------
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
    
    # Clean data - remove NaN values
    mask = np.isfinite(obs_ages) & np.isfinite(obs_feh)
    
    if np.sum(mask) < 5:
        return 10.0  # High penalty if insufficient data
    
    # Get clean data
    clean_ages = obs_ages[mask]
    clean_feh = obs_feh[mask]
    
# Interpolate model to observation ages
    model_interp = np.interp(clean_ages, model_age_gyr, model_feh)
    return _calculate_single_loss(model_interp, clean_feh, loss_metric)



def _calculate_single_loss(model_vals, obs_vals, loss_metric):
    """Calculate loss for a single dataset comparison"""

    if loss_metric == 'mae':
        return np.mean(np.abs(model_vals - obs_vals))
        
    elif loss_metric == 'rmse':
        return np.sqrt(np.mean((model_vals - obs_vals)**2))
        
    elif loss_metric == 'weighted_mae':
        # Use inverse of absolute values as weights (higher for low metallicity)
        weights = 1.0 / np.maximum(np.abs(obs_vals), 0.1)
        return np.average(np.abs(model_vals - obs_vals), weights=weights)
        
    elif loss_metric == 'weighted_rmse':
        weights = 1.0 / np.maximum(np.abs(obs_vals), 0.1)  
        return np.sqrt(np.average((model_vals - obs_vals)**2, weights=weights))
        
    elif loss_metric == 'huber':
        return huber_loss(obs_vals, model_vals, delta=0.2)
        
    elif loss_metric == 'log_likelihood':
        # Assume fixed uncertainty of 0.1 dex for metallicity
        sigma = 0.1
        residuals = model_vals - obs_vals
        chi2 = np.sum((residuals / sigma)**2)
        log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                                 2*len(obs_vals)*np.log(sigma))
        return -log_likelihood  # Return negative so lower is better
        
    elif loss_metric == 'aic':
        sigma = 0.1
        residuals = model_vals - obs_vals  
        chi2 = np.sum((residuals / sigma)**2)
        log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                                 2*len(obs_vals)*np.log(sigma))
        n_params = 3  # Assume 3 model parameters
        aic = 2 * n_params - 2 * log_likelihood
        return aic
        
    elif loss_metric == 'bic':
        sigma = 0.1
        residuals = model_vals - obs_vals
        chi2 = np.sum((residuals / sigma)**2) 
        log_likelihood = -0.5 * (chi2 + len(obs_vals) * np.log(2*np.pi) + 
                                 2*len(obs_vals)*np.log(sigma))
        n_params = 3
        bic = n_params * np.log(len(obs_vals)) - 2 * log_likelihood
        return bic
        
    elif loss_metric == 'correlation':
        # Return 1 - correlation so lower is better
        if len(model_vals) > 1 and np.std(model_vals) > 0 and np.std(obs_vals) > 0:
            corr = np.corrcoef(model_vals, obs_vals)[0, 1]
            return 1.0 - np.abs(corr)  # Use absolute correlation
        else:
            return 1.0  # No correlation
            
    elif loss_metric == 'spearman_correlation':
        # Return 1 - spearman correlation so lower is better
        if len(model_vals) > 1:
            spearman_corr, _ = stats.spearmanr(model_vals, obs_vals)
            if np.isfinite(spearman_corr):
                return 1.0 - np.abs(spearman_corr)
            else:
                return 1.0
        else:
            return 1.0
    

