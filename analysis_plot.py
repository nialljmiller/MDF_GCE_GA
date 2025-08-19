import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score




# Set ApJ-style plotting defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})

def ensure_analysis_dir():
    """Ensure the analysis directory exists"""
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



def plot_pca_degeneracy_analysis(GalGA, results_file='simulation_results.csv', save_path='analysis/pca_degeneracy_analysis.png'):
    """
    Perform PCA analysis on the fittest 10% of the population to reveal parameter degeneracies.
    Shows how the best models spread along degenerate manifolds vs constrained directions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    

    save_path = GalGA.output_path + save_path


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
    
    # Create comprehensive plot with better spacing
    fig = plt.figure(figsize=(24, 16))
    gs = plt.GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.2, 
                      left=0.06, right=0.98, top=0.96, bottom=0.06)
    
    # 1. Eigenvalue/Singular value plot
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(explained_variance)), explained_variance, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Eigenvalue (Variance)', fontsize=12)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
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
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.95, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(0.99, color='orange', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # 3. Principal component loadings heatmap - FIXED LABELS
    ax3 = fig.add_subplot(gs[0, 2:])
    im = ax3.imshow(components, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Short parameter names to avoid overlap
    param_short = ['σ₂', 't₁', 't₂', 'τ₁', 'τ₂', 'SFE', 'ΔSFE', 'M_up', 'N_Ia']
    
    ax3.set_xticks(range(len(param_short)))
    ax3.set_xticklabels(param_short, fontsize=11, ha='center')
    ax3.set_yticks(range(len(explained_variance)))
    ax3.set_yticklabels([f'PC{i+1}' for i in range(len(explained_variance))], fontsize=10)
    
    # Add text annotations for strong loadings
    for i in range(len(explained_variance)):
        for j in range(len(param_short)):
            if abs(components[i, j]) > 0.5:
                ax3.text(j, i, f'{components[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white' if abs(components[i, j]) > 0.7 else 'black',
                        fontsize=8)
    
    # 4. 2D projections onto first few PCs
    # Color points by fitness; use log scale for clarity
    if fitness_col in df_top.columns:
        colors = np.log10(np.clip(df_top[fitness_col].values, 1e-10, None))
    else:
        colors = np.zeros(len(df_top))
    
    # PC1 vs PC2
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax4.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)', fontsize=11)
    ax4.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)', fontsize=11)
    ax4.tick_params(axis='both', which='major', labelsize=9)
    
    # PC2 vs PC3
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(pca_result[:, 1], pca_result[:, 2], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax5.set_xlabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)', fontsize=11)
    ax5.set_ylabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)', fontsize=11)
    ax5.tick_params(axis='both', which='major', labelsize=9)
    
    # PC3 vs PC4
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(pca_result[:, 2], pca_result[:, 3], c=colors, cmap='viridis', alpha=0.6, s=20)
    ax6.set_xlabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)', fontsize=11)
    ax6.set_ylabel(f'PC4 ({explained_variance_ratio[3]:.1%} variance)', fontsize=11)
    ax6.tick_params(axis='both', which='major', labelsize=9)
    
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
    ax7.set_xlabel(param_short[param1_idx], fontsize=11)
    ax7.set_ylabel(param_short[param2_idx], fontsize=11)
    ax7.text(0.05, 0.95, f'r = {param_corr[max_corr_idx]:.3f}', 
             transform=ax7.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax7.tick_params(axis='both', which='major', labelsize=9)
    
    # 6. Parameter distributions along degenerate vs constrained directions
    ax8 = fig.add_subplot(gs[1, 4])
    
    # Project onto most and least constrained directions
    most_constrained = pca_result[:, 0]  # Highest variance PC
    least_constrained = pca_result[:, -1]  # Lowest variance PC
    
    ax8.hist(most_constrained, bins=15, alpha=0.7, 
            label=f'PC1 (λ={explained_variance[0]:.3f})', color='blue')
    ax8.hist(least_constrained, bins=15, alpha=0.7, 
            label=f'PC{len(explained_variance)} (λ={explained_variance[-1]:.6f})', color='red')
    ax8.set_xlabel('Projection Value', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.legend(fontsize=9)
    ax8.tick_params(axis='both', which='major', labelsize=9)
    
    # 7. Degeneracy identification table - IMPROVED FORMATTING
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    # Identify degenerate parameter combinations
    degeneracy_threshold = np.max(explained_variance) * 0.05  # 5% threshold
    
    table_data = []
    headers = ['PC', 'Eigenvalue', 'Variance %', 'Cumulative %', 'Dominant Parameters']
    
    for i in range(len(explained_variance)):
        eigenval = explained_variance[i]
        var_percent = explained_variance_ratio[i] * 100
        cumulative_percent = cumulative_var[i] * 100
        
        # Find parameters with highest loadings
        loadings = np.abs(components[i])
        top_params_idx = np.argsort(loadings)[-3:][::-1]  # Top 3, reversed
        top_params_with_values = []
        for idx in top_params_idx:
            if loadings[idx] > 0.3:  # Only show significant loadings
                sign = '+' if components[i, idx] > 0 else '-'
                top_params_with_values.append(f'{param_short[idx]} ({sign}{abs(components[i, idx]):.2f})')
        param_str = ', '.join(top_params_with_values)
        
        table_data.append([
            f'{i+1}', 
            f'{eigenval:.4f}', 
            f'{var_percent:.1f}', 
            f'{cumulative_percent:.1f}', 
            param_str
        ])
    
    # Create table with better spacing
    table = ax9.table(cellText=table_data, colLabels=headers, 
                     cellLoc='left', loc='center', bbox=[-0.05, 0.05, 1.05, 1.1])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1,2.1)
    
    # Color degenerate rows
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        else:
            if explained_variance[row-1] < degeneracy_threshold:
                cell.set_facecolor('#ffcccc')  # Light red for degenerate
            else:
                cell.set_facecolor('#f0f0f0')  # Light gray
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    # Print summary
    degenerate_pcs = np.where(explained_variance < degeneracy_threshold)[0]
    print(f"PCA Degeneracy Analysis saved to {save_path}")
    print(f"Found {len(degenerate_pcs)} degenerate directions (eigenvalue < {degeneracy_threshold:.4f})")
    print(f"Top 3 eigenvalues: {explained_variance[:3]}")
    print(f"Bottom 3 eigenvalues: {explained_variance[-3:]}")
    
    return fig



def plot_parameter_correlation_matrix(GalGA, results_file='GA/simulation_results.csv', save_path='GA/analysis/parameter_correlations.png'):
    """
    Create a correlation matrix heatmap showing parameter relationships for the fittest 10% of individuals.
    Complements the PCA analysis by showing direct pairwise correlations.
    """

    save_path = GalGA.output_path + save_path

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
    plt.savefig(save_path, bbox_inches='tight')
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




def analyze_best_fit_parameters(GalGA, results_file='GA/simulation_results.csv', save_path='GA/analysis/best_fit_summary.png'):
    """
    Analyze the single best-fit model parameters with proper statistical context.
    """

    save_path = GalGA.output_path + save_path

    df = pd.read_csv(results_file)
    fitness_col = 'fitness' if 'fitness' in df.columns else 'wrmse'
    df_sorted = df.sort_values(fitness_col, ascending=True)
    
    best_model = df_sorted.iloc[0]
    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                        'sfe', 'delta_sfe', 'imf_upper', 'nb']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Parameter values with population context
    param_values = [best_model[param] for param in continuous_params]
    param_labels = ['σ₂', 't₁', 't₂', 'τ₁', 'τ₂', 'SFE', 'ΔSFE', 'M_up', 'N_Ia']
    
    colors_list = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    bars = ax1.barh(range(len(param_values)), param_values, color=colors_list)
    ax1.set_yticks(range(len(param_values)))
    ax1.set_yticklabels(param_labels)
    ax1.set_xlabel('Parameter Value')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, param_values)):
        if val < 1e-3:
            label = f'{val:.2e}'
        elif val < 1:
            label = f'{val:.4f}'
        else:
            label = f'{val:.3f}'
        ax1.text(bar.get_width() + 0.01*max(param_values), bar.get_y() + bar.get_height()/2, 
                label, ha='left', va='center', fontweight='bold')
    
    # 2. Fitness comparison
    fitness_vals = df_sorted[fitness_col].values
    ax2.semilogy(range(len(fitness_vals)), fitness_vals, 'b-', alpha=0.7, linewidth=2)
    ax2.axhline(best_model[fitness_col], color='red', linestyle='--', linewidth=2, label='Best Fit')
    ax2.axhline(np.percentile(fitness_vals, 10), color='orange', linestyle='--', alpha=0.7, label='90th Percentile')
    ax2.set_xlabel('Model Rank')
    ax2.set_ylabel(f'{fitness_col.upper()} Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Discrete parameter choices
    discrete_params = ['comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx']
    discrete_labels = ['Comp', 'IMF', 'SNIa', 'Yields', 'SNIa Rate']
    discrete_values = [best_model[param] for param in discrete_params]
    
    bars = ax3.bar(range(len(discrete_values)), discrete_values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax3.set_xticks(range(len(discrete_values)))
    ax3.set_xticklabels(discrete_labels, rotation=45, ha='right')
    ax3.set_ylabel('Index Choice')
    
    # Add value labels
    for bar, val in zip(bars, discrete_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                str(int(val)), ha='center', va='bottom', fontweight='bold')
    
    # 4. Loss metrics summary
    loss_metrics = ['wrmse', 'mae', 'huber', 'ks', 'cosine', 'ensemble']
    available_metrics = [m for m in loss_metrics if m in best_model.index]
    metric_values = [best_model[m] for m in available_metrics]
    
    if metric_values:
        bars = ax4.bar(range(len(metric_values)), metric_values, 
                      color=plt.cm.plasma(np.linspace(0, 1, len(metric_values))))
        ax4.set_xticks(range(len(metric_values)))
        ax4.set_xticklabels([m.upper() for m in available_metrics], rotation=45, ha='right')
        ax4.set_ylabel('Loss Value')
        ax4.set_yscale('log')
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{val:.3e}', ha='center', va='bottom', fontweight='bold', 
                    rotation=45, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"Best-fit model analysis saved to {save_path}")
    print(f"Best fitness: {best_model[fitness_col]:.6f}")
    print("Best-fit parameters:")
    for param, label in zip(continuous_params, param_labels):
        print(f"  {label}: {best_model[param]:.4f}")



def analyze_top_percentile_parameters(GalGA, results_file='GA/simulation_results.csv', percentile=10, 
                                      save_path='GA/analysis/top_percentile_analysis.png'):
    """
    Comprehensive analysis of top N% models with proper uncertainty quantification.
    Normalizes violin and comparison plots to avoid skew from large parameters like M_up.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    save_path = GalGA.output_path + save_path

    df = pd.read_csv(results_file)
    fitness_col = 'fitness' if 'fitness' in df.columns else 'wrmse'
    df_sorted = df.sort_values(fitness_col, ascending=True)

    n_top = int(len(df_sorted) * percentile / 100)
    df_top = df_sorted.head(n_top)

    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                         'sfe', 'delta_sfe', 'imf_upper', 'nb']
    param_labels = ['σ₂', 't₁', 't₂', 'τ₁', 'τ₂', 'SFE', 'ΔSFE', 'M_up', 'N_Ia']

    fig = plt.figure(figsize=(20, 20))
    gs = plt.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ------------------------------------------------------------------
    # 1. Normalized Parameter Distributions (Violin)
    ax1 = fig.add_subplot(gs[0, :])
    param_matrix = np.array([df_top[param].values for param in continuous_params])
    param_means = np.mean(param_matrix, axis=1)
    param_stds = np.std(param_matrix, axis=1)
    param_zscores = [(param_matrix[i] - param_means[i]) / param_stds[i] for i in range(len(continuous_params))]

    parts = ax1.violinplot(param_zscores, positions=np.arange(len(continuous_params)),
                           widths=0.7, showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#8dd3c7')
        pc.set_alpha(0.7)

    ax1.set_xticks(np.arange(len(param_labels)))
    ax1.set_xticklabels(param_labels, rotation=45, ha='right')
    ax1.set_ylabel(f'Z-score (n={n_top})')
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 2. Uncertainty Quantification Table (Raw)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')

    best_model = df_sorted.iloc[0]
    table_data = []
    headers = ['Parameter', 'Best Fit', 'Median', 'Mean', 'Std Dev', '16th %ile', '84th %ile', 'IQR', 'CV']

    for param, label in zip(continuous_params, param_labels):
        values = df_top[param].values
        best_val = best_model[param]
        median_val = np.median(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        p16 = np.percentile(values, 16)
        p84 = np.percentile(values, 84)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        cv = std_val / mean_val if mean_val != 0 else np.inf

        table_data.append([
            label,
            f'{best_val:.4f}',
            f'{median_val:.4f}',
            f'{mean_val:.4f}',
            f'{std_val:.4f}',
            f'{p16:.4f}',
            f'{p84:.4f}',
            f'{iqr:.4f}',
            f'{cv:.3f}'
        ])

    table = ax2.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    cvs = [float(row[8]) for row in table_data]
    cv_norm = plt.Normalize(vmin=min(cvs), vmax=max(cvs))
    cmap = plt.cm.RdYlBu_r
    for i, cv in enumerate(cvs):
        color = cmap(cv_norm(cv))
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            if cv > np.median(cvs):
                table[(i+1, j)].set_text_props(weight='bold')

    # ------------------------------------------------------------------
    # 3. Parameter Correlation Matrix
    ax3 = fig.add_subplot(gs[2, 0])
    corr_matrix = df_top[continuous_params].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_values = corr_matrix.values
    corr_values[mask] = np.nan

    im = ax3.imshow(corr_values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(range(len(param_labels)))
    ax3.set_yticks(range(len(param_labels)))
    ax3.set_xticklabels(param_labels, rotation=45, ha='right')
    ax3.set_yticklabels(param_labels)

    for i in range(len(param_labels)):
        for j in range(len(param_labels)):
            if not np.isnan(corr_values[i, j]) and abs(corr_values[i, j]) > 0.3:
                ax3.text(j, i, f'{corr_values[i, j]:.2f}', ha='center', va='center',
                         color='white' if abs(corr_values[i, j]) > 0.7 else 'black', fontweight='bold')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # ------------------------------------------------------------------
    # 4. Parameter Constraint Levels (CV)
    ax4 = fig.add_subplot(gs[2, 1])
    constraint_levels = []
    for param in continuous_params:
        values = df_top[param].values
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
        constraint_levels.append(cv)

    colors_constraint = ['red' if cv > 0.5 else 'orange' if cv > 0.2 else 'green' for cv in constraint_levels]

    ax4.bar(range(len(constraint_levels)), constraint_levels, color=colors_constraint)
    ax4.set_xticks(range(len(constraint_levels)))
    ax4.set_xticklabels(param_labels, rotation=45, ha='right')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='Moderate')
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Poor')
    ax4.set_yscale('log')
    ax4.legend()

    # ------------------------------------------------------------------
    # 5. Fitness Distribution
    ax5 = fig.add_subplot(gs[2, 2])
    fitness_top = df_top[fitness_col].values
    ax5.hist(fitness_top, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(fitness_top[0], color='red', linestyle='--', linewidth=2, label='Best')
    ax5.axvline(np.median(fitness_top), color='orange', linestyle='--', linewidth=2, label='Median')
    ax5.set_xlabel(f'{fitness_col.upper()} Loss')
    ax5.set_ylabel(f'Count (Top {percentile}%)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Top {percentile}% analysis saved to {save_path}")
    print(f"Number of models analyzed: {n_top}")
    print("Most constrained parameters (lowest CV):")
    sorted_params = sorted(zip(param_labels, constraint_levels), key=lambda x: x[1])
    for label, cv in sorted_params[:3]:
        print(f"  {label}: CV = {cv:.3f}")
    print("Least constrained parameters (highest CV):")
    for label, cv in sorted_params[-3:]:
        print(f"  {label}: CV = {cv:.3f}")





def identify_solution_islands(GalGA, results_file='simulation_results.csv', percentile_threshold=90,
                               save_path='analysis/solution_islands.png'):
    """
    Identify and analyze distinct clusters/islands of solutions in parameter space.
    """

    save_path = GalGA.output_path + save_path


    df = pd.read_csv(results_file)
    fitness_col = 'fitness' if 'fitness' in df.columns else 'wrmse'
  
    # Filter to top percentile
    threshold_fitness = np.percentile(df[fitness_col], percentile_threshold)
    df_good = df[df[fitness_col] <= threshold_fitness].copy()
  
    continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
                         'sfe', 'delta_sfe', 'imf_upper', 'nb']
  
    # Standardize parameters for clustering
    X = df_good[continuous_params].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
  
    # Try multiple clustering methods
    fig = plt.figure(figsize=(24, 20)) # Increased height for new subplot
    gs = plt.GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3) # Added extra row
  
    fig.suptitle(f'Solution Islands Analysis (Top {100-percentile_threshold}%)',
                 fontsize=18, fontweight='bold')
  
    # 1. DBSCAN clustering
    ax1 = fig.add_subplot(gs[0, 0])
  
    # Optimize DBSCAN parameters
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1])
  
    # Use elbow method for eps
    knee_point = int(0.95 * len(distances))
    eps = distances[knee_point]
  
    dbscan = DBSCAN(eps=eps, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_scaled)
  
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
  
    # Plot first two PC components colored by cluster
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
  
    unique_labels = set(cluster_labels)
    colors_map = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
  
    for k, col in zip(unique_labels, colors_map):
        if k == -1:
            # Noise points
            ax1.scatter(X_pca[cluster_labels == k, 0], X_pca[cluster_labels == k, 1],
                        c='black', marker='x', s=20, alpha=0.5, label='Noise')
        else:
            ax1.scatter(X_pca[cluster_labels == k, 0], X_pca[cluster_labels == k, 1],
                        c=[col], s=30, alpha=0.7, label=f'Cluster {k}')
  
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title(f'DBSCAN Clustering\n{n_clusters} clusters, {n_noise} noise points')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  
    # 2. K-means clustering comparison
    ax2 = fig.add_subplot(gs[0, 1])
  
    # Determine optimal k using silhouette score
    k_range = range(2, min(8, len(df_good)//10))
    silhouette_scores = []
  
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels_k = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels_k)
        silhouette_scores.append(score)
  
    optimal_k = k_range[np.argmax(silhouette_scores)]
  
    kmeans_best = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_best.fit_predict(X_scaled)
  
    colors_kmeans = plt.cm.Set2(np.linspace(0, 1, optimal_k))
    for k in range(optimal_k):
        mask = kmeans_labels == k
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors_kmeans[k]],
                    s=30, alpha=0.7, label=f'Cluster {k}')
  
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax2.set_title(f'K-means Clustering (k={optimal_k})')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  
    # 3. Silhouette analysis
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax3.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Silhouette Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
  
    # 4. Distance-based island identification
    ax4 = fig.add_subplot(gs[0, 3])
  
    # Calculate pairwise distances in parameter space
    distances_matrix = squareform(pdist(X_scaled, metric='euclidean'))
  
    # Find isolated regions using distance threshold
    distance_threshold = np.percentile(distances_matrix.flatten(), 90)
  
    ax4.hist(distances_matrix.flatten(), bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax4.axvline(distance_threshold, color='red', linestyle='--', linewidth=2,
                label=f'90th %ile = {distance_threshold:.2f}')
    ax4.set_xlabel('Pairwise Distance')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.set_yscale('log')
  
    # 5-8. Show clusters in different 2D projections
    param_pairs = [('sigma_2', 't_2'), ('infall_1', 'infall_2'), ('sfe', 'delta_sfe'), ('t_1', 't_2')]
    axes_2d = [fig.add_subplot(gs[1, i]) for i in range(4)]
  
    for ax, (param1, param2) in zip(axes_2d, param_pairs):
        # Use DBSCAN clustering results
        for k in unique_labels:
            if k == -1:
                continue
            mask = cluster_labels == k
            if np.sum(mask) > 0:
                ax.scatter(df_good[param1].values[mask], df_good[param2].values[mask],
                           c=[colors_map[list(unique_labels).index(k)]], s=30, alpha=0.7,
                           label=f'Cluster {k}')
      
        # Mark noise points
        if n_noise > 0:
            noise_mask = cluster_labels == -1
            ax.scatter(df_good[param1].values[noise_mask], df_good[param2].values[noise_mask],
                       c='black', marker='x', s=20, alpha=0.5)
      
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        if ax == axes_2d[0]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  
    # 9. 3D visualization of main clusters
    ax_3d = fig.add_subplot(gs[2, :2], projection='3d')
  
    # Use three most important parameters from PCA
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
  
    for k in unique_labels:
        if k == -1:
            continue
        mask = cluster_labels == k
        if np.sum(mask) > 0:
            ax_3d.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                          c=[colors_map[list(unique_labels).index(k)]], s=30, alpha=0.7,
                          label=f'Cluster {k}')
  
    ax_3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax_3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax_3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax_3d.legend()
  
    # 10. Cluster statistics table
    ax_table = fig.add_subplot(gs[2, 2:])
    ax_table.axis('off')
  
    # Analyze each cluster and collect stats
    cluster_stats = []
    for k in unique_labels:
        if k == -1:
            continue
      
        mask = cluster_labels == k
        cluster_data = df_good[mask]
      
        size = np.sum(mask)
        avg_fitness = np.mean(cluster_data[fitness_col])
        fitness_range = (np.min(cluster_data[fitness_col]), np.max(cluster_data[fitness_col]))
      
        # Find characteristic parameters (highest variation within cluster)
        param_stds = []
        for param in continuous_params:
            std_val = np.std(cluster_data[param])
            param_stds.append((param, std_val))
      
        param_stds.sort(key=lambda x: x[1], reverse=True)
        main_params = ', '.join([p[0] for p in param_stds[:3]])
      
        # Median parameters for listing
        param_medians = {param: np.median(cluster_data[param]) for param in continuous_params}
      
        cluster_stats.append({
            'cluster_id': k,
            'size': size,
            'avg_fitness': avg_fitness,
            'fitness_range': fitness_range,
            'main_params': main_params,
            'parameters': param_medians
        })
    # Sort clusters by average fitness (lower is better)
    cluster_stats.sort(key=lambda x: x['avg_fitness'])
    # Print ranked clusters
    print("\nRanked Solution Islands (Best to Worst):")
    print("-" * 50)
    for rank, cluster in enumerate(cluster_stats, 1):
        print(f"Rank {rank}: Cluster {cluster['cluster_id']}")
        print(f" Size: {cluster['size']} models")
        print(f" Average Fitness: {cluster['avg_fitness']:.4f}")
        print(f" Fitness Range: {cluster['fitness_range'][0]:.4f} - {cluster['fitness_range'][1]:.4f}")
        print(f" Main Parameters (highest variation): {cluster['main_params']}")
        print(" Median Parameters:")
        for param, value in cluster['parameters'].items():
            print(f" {param}: {value:.4f}")
        print("-" * 50)
    # Table data for plot (basic stats)
    table_data = []
    for cluster in cluster_stats: # Use sorted for consistency
        k = cluster['cluster_id']
        size = cluster['size']
        avg_fitness = f"{cluster['avg_fitness']:.4f}"
        fitness_range = f"{cluster['fitness_range'][0]:.4f}-{cluster['fitness_range'][1]:.4f}"
        main_params = cluster['main_params']
        table_data.append([f'{k}', f'{size}', avg_fitness, fitness_range, main_params])
    headers = ['Cluster', 'Size', 'Avg Fitness', 'Fitness Range', 'Main Parameters (high var)']
  
    if table_data:
        table = ax_table.table(cellText=table_data, colLabels=headers,
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    # New: 10b. Detailed parameter medians table/text
    ax_details = fig.add_subplot(gs[3, 2:])
    ax_details.axis('off')
  
    # Build table for medians (select top params or all if few clusters)
    med_table_data = []
    med_headers = ['Cluster'] + continuous_params[:6] # Limit to first 6 for space
  
    for cluster in cluster_stats:
        row = [f"{cluster['cluster_id']}"]
        for param in med_headers[1:]:
            row.append(f"{cluster['parameters'][param]:.4f}")
        med_table_data.append(row)
  
    if med_table_data:
        med_table = ax_details.table(cellText=med_table_data, colLabels=med_headers,
                                     cellLoc='center', loc='center')
        med_table.auto_set_font_size(False)
        med_table.set_fontsize(9)
        med_table.scale(1.2, 1.5)
  
    # 11. Island separation analysis
    ax_sep = fig.add_subplot(gs[3, :2])
  
    if n_clusters > 1:
        # Calculate inter-cluster vs intra-cluster distances
        inter_distances = []
        intra_distances = []
      
        for i, k1 in enumerate(unique_labels):
            if k1 == -1:
                continue
            mask1 = cluster_labels == k1
          
            # Intra-cluster distances
            if np.sum(mask1) > 1:
                intra_dist = pdist(X_scaled[mask1])
                intra_distances.extend(intra_dist)
          
            # Inter-cluster distances
            for j, k2 in enumerate(unique_labels):
                if k2 == -1 or j <= i:
                    continue
                mask2 = cluster_labels == k2
              
                for point1 in X_scaled[mask1]:
                    for point2 in X_scaled[mask2]:
                        inter_distances.append(np.linalg.norm(point1 - point2))
      
        if inter_distances and intra_distances:
            ax_sep.hist(intra_distances, bins=30, alpha=0.7, label='Intra-cluster', color='blue')
            ax_sep.hist(inter_distances, bins=30, alpha=0.7, label='Inter-cluster', color='red')
            ax_sep.set_xlabel('Distance')
            ax_sep.set_ylabel('Count')
            ax_sep.legend()
            ax_sep.set_yscale('log')
          
            # Calculate separation quality
            sep_ratio = np.mean(inter_distances) / np.mean(intra_distances) if intra_distances else 0
            ax_sep.text(0.7, 0.9, f'Separation ratio: {sep_ratio:.2f}',
                        transform=ax_sep.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
  
    # 12. Parameter space coverage moved to new row
    ax_coverage = fig.add_subplot(gs[4, :])
  
    # Show how clusters cover different regions of parameter space
    coverage_data = []
    for param in continuous_params: # All now, since space
        diff = df[param].max() - df[param].min()
        full_range = diff if diff > 0 else 1
        covered_ranges = []
      
        for k in unique_labels:
            if k == -1:
                continue
            mask = cluster_labels == k
            if np.sum(mask) > 0:
                cluster_range = df_good[param].values[mask].max() - df_good[param].values[mask].min()
                covered_ranges.append(cluster_range / full_range)
      
        coverage_data.append(covered_ranges)
  
    # Create box plot of coverage
    if coverage_data:
        ax_coverage.boxplot(coverage_data, labels=continuous_params)
        ax_coverage.set_ylabel('Fractional Parameter Range Coverage')
        ax_coverage.tick_params(axis='x', rotation=45)
  
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
  
    # Print summary
    print(f"Solution islands analysis saved to {save_path}")
    print(f"DBSCAN found {n_clusters} distinct clusters with {n_noise} noise points")
    print(f"K-means optimal k: {optimal_k}")
    print(f"Average silhouette score: {max(silhouette_scores):.3f}")
  
    return {
        'n_clusters_dbscan': n_clusters,
        'n_noise': n_noise,
        'optimal_k_means': optimal_k,
        'silhouette_score': max(silhouette_scores),
        'cluster_labels': cluster_labels,
        'separation_quality': sep_ratio if 'sep_ratio' in locals() else 0,
        'cluster_stats': cluster_stats
    }





def run_analysis(GalGA, results_file='simulation_results.csv'):
    """
    Run all analysis functions and create a summary report.
    """

    results_file = GalGA.output_path + results_file

    ensure_analysis_dir(GalGA)
    
    # Run all analyses
    analyze_best_fit_parameters(GalGA, results_file)
    
    analyze_top_percentile_parameters(GalGA, results_file, percentile=10)

    island_results = None
    #island_results = identify_solution_islands(results_file, percentile_threshold=90)

    try:    
        plot_pca_degeneracy_analysis(GalGA, results_file)
        
        plot_parameter_correlation_matrix(GalGA, results_file)
        
        
        df = pd.read_csv(results_file)
        fitness_col = 'fitness' if 'fitness' in df.columns else 'wrmse'
        
        print(f"Total models evaluated: {len(df)}")
        print(f"Best fitness: {df[fitness_col].min():.6f}")
        print(f"Fitness range: {df[fitness_col].min():.6f} - {df[fitness_col].max():.6f}")
        
        if island_results:
            print(f"\nSolution structure:")
            print(f"  - DBSCAN clusters: {island_results['n_clusters_dbscan']}")
            print(f"  - Noise points: {island_results['n_noise']}")
            print(f"  - Optimal K-means k: {island_results['optimal_k_means']}")
            print(f"  - Silhouette score: {island_results['silhouette_score']:.3f}")
    except:
        print("Probably not yet enough trials for stats analysis.")    

# Add this to the end of the file to make it runnable
if __name__ == "__main__":
    run_comprehensive_analysis()
