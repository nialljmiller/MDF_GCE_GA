#!/usr/bin/env python3.8
"""
Uncertainty quantification and model framework limitation analysis for GCE GA results.
Addresses all placeholder items in Section 5.2 of the paper.

Authors: N Miller, based on analysis framework
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde, bootstrap, percentileofscore
from scipy import stats
from scipy.interpolate import interp1d
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
})

def ensure_output_dirs(base_path):
    """Create necessary output directories under base_path/uncertainty"""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'uncertainty'), exist_ok=True)


class UncertaintyAnalysis:
    """
    Comprehensive uncertainty quantification for galactic chemical evolution
    genetic algorithm results.
    """
    
    def __init__(self, results_file, output_path='GA/'):
        """
        Initialize uncertainty analysis.
        
        Parameters:
        -----------
        results_file : str
            Path to CSV file containing GA results
        output_path : str
            Base path for output files
        """
        self.results_file = results_file
        self.output_path = output_path
        ensure_output_dirs(output_path)
        
        # Load data
        self.df = pd.read_csv(results_file)
        self.fitness_col = 'fitness' if 'fitness' in self.df.columns else 'wrmse'
        
        # Define parameter sets
        self.continuous_params = [
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
        ]
        self.categorical_params = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx'
        ]
        
        # Filter to available parameters
        self.continuous_params = [p for p in self.continuous_params if p in self.df.columns]
        self.categorical_params = [p for p in self.categorical_params if p in self.df.columns]
        
        # Sort by fitness (lower is better)
        self.df_sorted = self.df.sort_values(self.fitness_col, ascending=True)
        
        print(f"Loaded {len(self.df)} models from {results_file}")
        print(f"Best fitness: {self.df_sorted[self.fitness_col].iloc[0]:.6f}")
        print(f"Available continuous parameters: {self.continuous_params}")


    def plot_marginalized_posteriors_kde_weighted(
            self,
            percentile=10,
            bins=30,
            ncols=3,
            title='Marginalized posteriors (fitness-weighted KDE)'
        ):
        """
        1D marginalized posteriors for continuous parameters using fitness-weighted KDE
        and weighted 16–50–84% markers.

        Saves:
            <output>/uncertainty/marginalized_posteriors.png

        Returns
        -------
        path : str
            Path to saved figure.
        kdes : dict
            {parameter_name: gaussian_kde or None} for each parameter.
        """
        # take top X% by fitness
        df = self.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)

        # inverse-fitness weights (lower fitness -> higher weight)
        fit = top[self.fitness_col].values
        eps = np.min(fit) * 0.001
        w_raw = 1.0 / (fit + eps)
        w = w_raw / np.sum(w_raw)

        # pretty axis labels (only used if the key exists)
        param_labels = {
            'sigma_2': r'$\sigma_2$',
            't_1': r't$_1$ (Gyr)',
            't_2': r't$_2$ (Gyr)',
            'infall_1': r'$\tau_1$ (Gyr)',
            'infall_2': r'$\tau_2$ (Gyr)',
            'sfe': 'SFE',
            'delta_sfe': r'$\Delta$SFE',
            'imf_upper': r'M$_{up}$ (M$_\odot$)',
            'mgal': r'M$_{gal}$ (M$_\odot$)',
            'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
        }

        params = list(self.continuous_params)
        n_params = len(params)
        nrows = int(np.ceil(n_params / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
        if nrows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        def weighted_quantile(values, quantiles, sample_weight):
            values = np.asarray(values)
            quantiles = np.asarray(quantiles)
            sorter = np.argsort(values)
            v = values[sorter]
            sw = np.asarray(sample_weight)[sorter]
            cdf = np.cumsum(sw)
            cdf /= cdf[-1]
            return np.interp(quantiles, cdf, v)

        kdes = {}
        rng = np.random.default_rng(42)

        for i, p in enumerate(params):
            ax = axes[i]
            v = np.asarray(top[p].values, dtype=float)

            # guard against all-equal values (singular covariance)
            if not np.isfinite(v).all():
                v = v[np.isfinite(v)]
            if len(v) == 0:
                # nothing to show
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                kdes[p] = None
                continue

            # KDE (weighted if available; else weighted resample)
            kde = None
            try:
                if np.allclose(np.std(v), 0.0):
                    raise np.linalg.LinAlgError  # force fallback for degenerate variance
                kde = gaussian_kde(v, weights=w)
            except Exception:
                # fallback: weighted discrete resample, then unweighted KDE (if still not degenerate)
                N = min(max(1000, 5*len(v)), 5000)
                resample = rng.choice(v, size=N, replace=True, p=w/np.sum(w))
                if np.allclose(np.std(resample), 0.0):
                    kde = None
                else:
                    kde = gaussian_kde(resample)
            kdes[p] = kde

            # x-grid and density
            x_min, x_max = float(np.min(v)), float(np.max(v))
            xr = x_max - x_min
            if xr <= 0:
                # draw a single vertical line (all values identical)
                ax.axvline(x_min, color='red', ls='--', lw=2, label=f'Median: {x_min:.4f}')
                ax.set_xlim(x_min - 0.5, x_max + 0.5)
                ax.set_yticks([])
                ax.set_xlabel(param_labels.get(p, p))
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8, frameon=False)
                continue

            xs = np.linspace(x_min - 0.1*xr, x_max + 0.1*xr, 256)
            if kde is not None:
                dens = kde(xs)
                ax.plot(xs, dens, lw=2)
                ax.fill_between(xs, dens, alpha=0.25)

            # reference histogram
            ax.hist(v, bins=bins, density=True, alpha=0.50,
                    color='lightsteelblue', edgecolor='black')

            # weighted 16/50/84 markers
            q16, q50, q84 = weighted_quantile(v, [0.16, 0.50, 0.84], w)
            ax.axvline(q50, color='red', ls='--', lw=2, label=f'Median: {q50:.4f}')
            ax.axvline(q16, color='orange', ls=':', alpha=0.85)
            ax.axvline(q84, color='orange', ls=':', alpha=0.85)

            ax.set_xlabel(param_labels.get(p, p))
            ax.set_ylabel('Density')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, frameon=False)

        # remove any extra axes
        for j in range(n_params, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(title + f' — Top {percentile}%', y=0.995, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = os.path.join(self.output_path, 'uncertainty', 'marginalized_posteriors.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path, kdes


    def plot_corner_2d_kde(self,
                           params=None,
                           percentile=10,
                           weight_power=1.0,
                           levels=(0.5, 0.8, 0.95),
                           title='Pairwise parameter posteriors (fitness-weighted 2D KDE)'):
        """
        Triangular corner plot with 2D KDE contours for selected parameters.
        Saves: <output>/uncertainty/posterior_corner.png

        Parameters
        ----------
        params : list[str] or None
            e.g. ['sigma_2','t_2','infall_2','sfe'] (keep 3–6 for readability)
            Defaults to the first 4 continuous params if None.
        percentile : float
            Use top X% by fitness before weighting.
        weight_power : float
            Exponent on inverse fitness for weights.
        levels : tuple
            Credible region contour levels in *probability mass* (not density).

        Returns
        -------
        path : str
            Path to saved figure.
        """
        df = self.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)

        if params is None:
            params = list(self.continuous_params[:4])

        # weights
        fit = top[self.fitness_col].values
        eps = np.min(fit) * 0.001
        w_raw = 1.0 / (fit + eps) ** weight_power
        w = w_raw / np.sum(w_raw)

        # set up figure
        k = len(params)
        fig, axes = plt.subplots(k, k, figsize=(3.2*k, 3.2*k))
        rng = np.random.default_rng(123)

        def kde_1d(vals, weights, xs):
            try:
                return gaussian_kde(vals, weights=weights)(xs)
            except TypeError:
                # fallback to weighted resample
                N = min(max(1000, 5*len(vals)), 5000)
                resample = rng.choice(vals, size=N, replace=True, p=weights/weights.sum())
                return gaussian_kde(resample)(xs)

        def kde_2d(x, y, weights, xg, yg):
            xy = np.vstack([x, y])
            try:
                kde = gaussian_kde(xy, weights=weights)
            except TypeError:
                # fallback to weighted resample
                N = min(max(1000, 5*len(x)), 5000)
                idx = rng.choice(np.arange(len(x)), size=N, replace=True, p=weights/weights.sum())
                xy = np.vstack([x[idx], y[idx]])
                kde = gaussian_kde(xy)

            ZZ = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)

            # --- compute iso-density levels that enclose given probability mass ---
            vals = ZZ.ravel()
            order = np.argsort(vals)[::-1]          # sort dens descending
            vals_sorted = vals[order]
            cdf = np.cumsum(vals_sorted)
            cdf /= cdf[-1]                          # normalize to 1

            # for each target mass L, find density threshold so that integral >= L
            thr = []
            for L in levels:
                idxL = np.searchsorted(cdf, L, side="left")
                idxL = min(idxL, len(vals_sorted) - 1)
                thr.append(vals_sorted[idxL])

            # levels must be strictly increasing
            thr = np.unique(np.sort(np.asarray(thr)))

            return ZZ, thr


        # axes fill
        for i, pi in enumerate(params):
            vi = top[pi].values
            for j, pj in enumerate(params):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off')
                    continue
                if i == j:
                    # 1D posterior
                    xs = np.linspace(vi.min(), vi.max(), 256)
                    dens = kde_1d(vi, w, xs)
                    ax.plot(xs, dens, lw=2)
                    ax.fill_between(xs, dens, alpha=0.25)
                    ax.set_yticks([])
                    ax.grid(alpha=0.2)
                    ax.set_xlabel(pi)
                else:
                    vj = top[pj].values
                    xg = np.linspace(vj.min(), vj.max(), 120)
                    yg = np.linspace(vi.min(), vi.max(), 120)
                    Xg, Yg = np.meshgrid(xg, yg)
                    ZZ, thr = kde_2d(vj, vi, w, Xg, Yg)
                    ax.contour(Xg, Yg, ZZ, levels=thr, colors='k', linewidths=1.2)
                    ax.set_xlabel(pj)
                    ax.set_ylabel(pi)
                    ax.grid(alpha=0.15)

        fig.suptitle(title + f' — Top {percentile}%', y=0.92, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.output_path, 'uncertainty', 'posterior_corner.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path


    def plot_fitness_weighted_param_intervals_facet(
            self,
            params=None,
            weight_power=1.0,
            percentile=None,
            title='Fitness-weighted parameter intervals (per-parameter scales)',
            ncols=2,
            log_range_factor=50.0  # if q84/q16 > this and values > 0 -> log x
        ):
        """
        Draw 16–50–84% intervals with a dedicated x-axis for each parameter.
        Auto-log scaling when the middle 68% span is very large.

        Returns
        -------
        path : str
            Saved figure path.
        table : pd.DataFrame
            Weighted q16/q50/q84 per parameter (same as bar plot).
        """
        # subset by top percentile (optional)
        df = self.df_sorted.copy()
        if percentile is not None:
            n_top = max(1, int(len(df) * percentile / 100))
            df = df.head(n_top)

        # which params
        if params is None:
            params = list(self.continuous_params)

        # weights ~ inverse fitness^power
        fit = df[self.fitness_col].values
        eps = np.min(fit) * 0.001
        w_raw = 1.0 / (fit + eps)**weight_power
        w = w_raw / np.sum(w_raw)

        def weighted_quantile(values, quantiles, sample_weight):
            values = np.asarray(values)
            quantiles = np.asarray(quantiles)
            sorter = np.argsort(values)
            v = values[sorter]
            sw = np.asarray(sample_weight)[sorter]
            cdf = np.cumsum(sw); cdf /= cdf[-1]
            return np.interp(quantiles, cdf, v)

        # pretty labels (optional)
        labels = {
            'sigma_2': r'$\sigma_2$',
            't_1': r't$_1$ (Gyr)',
            't_2': r't$_2$ (Gyr)',
            'infall_1': r'$\tau_1$ (Gyr)',
            'infall_2': r'$\tau_2$ (Gyr)',
            'sfe': 'SFE',
            'delta_sfe': r'$\Delta$SFE',
            'imf_upper': r'M$_{up}$ (M$_\odot$)',
            'mgal': r'M$_{gal}$ (M$_\odot$)',
            'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
        }

        # compute table (weighted 16/50/84)
        rows = []
        for p in params:
            v = df[p].values
            q16, q50, q84 = weighted_quantile(v, [0.16, 0.50, 0.84], w)
            rows.append(dict(param=p, q16=q16, q50=q50, q84=q84, iqr=q84 - q16))
        table = pd.DataFrame(rows)

        # layout
        k = len(params)
        nrows = int(np.ceil(k / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 2.2*nrows))
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        # draw each param on its own axis
        for i, p in enumerate(params):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            q16 = table.loc[table.param == p, 'q16'].values[0]
            q50 = table.loc[table.param == p, 'q50'].values[0]
            q84 = table.loc[table.param == p, 'q84'].values[0]

            # pad limits a bit
            span = max(q84 - q16, 1e-12)
            x_min = min(q16, q50, q84) - 0.15*span
            x_max = max(q16, q50, q84) + 0.15*span

            # line + median dot
            ax.hlines(0, q16, q84, lw=6, alpha=0.7)
            ax.plot(q50, 0, 'o', ms=6, label='Weighted median')

            # axis cosmetics
            ax.set_yticks([])
            ax.set_xlabel(labels.get(p, p))
            ax.grid(axis='x', alpha=0.25)

            # auto log if 68% span is very wide and values are positive
            if (q16 > 0) and (q84/q16 > log_range_factor):
                ax.set_xscale('log')
                # recompute bounds safely for log
                lo = max(min(q16, q50, q84) * 0.85, np.finfo(float).tiny)
                hi = max(q16, q50, q84) * 1.15
                ax.set_xlim(lo, hi)
            else:
                ax.set_xlim(x_min, x_max)

            ax.legend(frameon=False, fontsize=9, loc='upper left')

        # blank any extra axes
        for j in range(k, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis('off')

        fig.suptitle(title + (f' — Top {percentile}%' if percentile else ''), y=0.98, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(self.output_path, 'uncertainty', 'weighted_param_intervals_facet.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path, table



    def plot_bootstrap_loss(self,
                            obs_x, obs_y, obs_sigma,
                            model_x, model_y,
                            loss='wrmse',
                            n_bootstrap=5000,
                            seed=42,
                            label='MDF'):
        """
        Plot the bootstrap distribution of the best-fit model's loss against
        perturbed observations (data-space bootstrap).
        Saves: <output>/uncertainty/bootstrap_<label>_<loss>.png

        Returns
        -------
        stats : dict
            loss distribution stats (mean, std, p68, p95, baseline, p_value)
        path  : str
            path to the saved figure
        """
        # Reuse your implemented routine to compute the distribution
        stats = self.bootstrap_bestfit_loss(
            obs_x=obs_x, obs_y=obs_y, obs_sigma=obs_sigma,
            model_x=model_x, model_y=model_y,
            loss=loss, n_bootstrap=n_bootstrap, seed=seed,
            save_plot=False, label=label
        )

        losses = stats['losses']
        baseline = stats['baseline_loss']
        mean = stats['mean']
        std = stats['std']
        p16, p84 = stats['p68']
        p2p5, p97p5 = stats['p95']
        p_value = float(np.mean(losses >= baseline))

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.hist(losses, bins=50, density=True, alpha=0.75)
        ax.axvline(baseline, color='k', ls='--', lw=2, label=f'Baseline: {baseline:.4f}')
        ax.axvline(mean, color='tab:blue', ls='-', lw=2, label=f'Mean: {mean:.4f} ± {std:.4f}')
        ax.axvspan(p16, p84, color='tab:blue', alpha=0.20, label='68% CI')
        ax.axvspan(p2p5, p97p5, color='tab:orange', alpha=0.15, label='95% CI')
        ax.text(0.98, 0.95, f"p = {p_value:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))
        ax.set_xlabel(f'Bootstrap {loss.upper()} ({label})')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=10)

        save_path = os.path.join(self.output_path, 'uncertainty',
                                 f'bootstrap_{label.lower()}_{loss.lower()}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # include p_value in return for convenience
        stats = dict(stats)
        stats['p_value'] = p_value
        return stats, save_path












    def bootstrap_parameter_uncertainty(self, percentile=10, n_bootstrap=1000, 
                                      confidence_level=95, save_results=True):
        """
        Bootstrap resampling analysis of parameter uncertainties.
        
        Parameters:
        -----------
        percentile : float
            Percentile of top models to analyze
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals (e.g., 95 for 95% CI)
        save_results : bool
            Whether to save results to file
            
        Returns:
        --------
        dict : Bootstrap statistics for each parameter
        """
        print(f"Running bootstrap analysis with {n_bootstrap} samples...")
        
        # Get top percentile of models
        n_top = int(len(self.df_sorted) * percentile / 100)
        top_models = self.df_sorted.head(n_top)
        
        bootstrap_stats = {}
        alpha = (100 - confidence_level) / 2
        
        for param in self.continuous_params:
            data = top_models[param].values
            
            # Bootstrap resampling
            bootstrap_means = []
            bootstrap_stds = []
            bootstrap_medians = []
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(sample))
                bootstrap_stds.append(np.std(sample))
                bootstrap_medians.append(np.median(sample))
            
            # Calculate statistics
            bootstrap_stats[param] = {
                'original_mean': np.mean(data),
                'original_std': np.std(data),
                'original_median': np.median(data),
                'bootstrap_mean': np.mean(bootstrap_means),
                'bootstrap_std_of_mean': np.std(bootstrap_means),
                'bootstrap_median': np.median(bootstrap_medians),
                'ci_lower': np.percentile(bootstrap_means, alpha),
                'ci_upper': np.percentile(bootstrap_means, 100 - alpha),
                'bias': np.mean(bootstrap_means) - np.mean(data),
                'coefficient_of_variation': np.std(bootstrap_means) / np.mean(bootstrap_means)
            }
        
        if save_results:
            # Save to file
            output_file = os.path.join(self.output_path, 'uncertainty', 'bootstrap_results.txt')
            with open(output_file, 'w') as f:
                f.write(f"Bootstrap Parameter Uncertainty Analysis\n")
                f.write(f"{'='*50}\n")
                f.write(f"Number of bootstrap samples: {n_bootstrap}\n")
                f.write(f"Top {percentile}% models analyzed (n={n_top})\n")
                f.write(f"Confidence level: {confidence_level}%\n\n")
                
                for param, stats in bootstrap_stats.items():
                    f.write(f"{param}:\n")
                    f.write(f"  Original: {stats['original_mean']:.6f} ± {stats['original_std']:.6f}\n")
                    f.write(f"  Bootstrap: {stats['bootstrap_mean']:.6f} ± {stats['bootstrap_std_of_mean']:.6f}\n")
                    f.write(f"  {confidence_level}% CI: [{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]\n")
                    f.write(f"  Bias: {stats['bias']:.6e}\n")
                    f.write(f"  CV: {stats['coefficient_of_variation']:.4f}\n\n")
            
            print(f"Bootstrap results saved to {output_file}")
        
        return bootstrap_stats


    def bootstrap_bestfit_loss(self,
                               obs_x, obs_y, obs_sigma,
                               model_x, model_y,
                               loss='wrmse',
                               n_bootstrap=2000,
                               seed=42,
                               save_plot=True,
                               label='MDF'):
        """
        Data-space bootstrap of the best-fit model's loss.

        Parameters
        ----------
        obs_x, obs_y : 1D arrays
            Observed MDF (bin centers and normalized counts).
        obs_sigma : 1D array or float or None
            Per-bin observational uncertainty for obs_y. If None,
            uses sqrt(N) or 0.05 as a fallback (see below).
        model_x, model_y : 1D arrays
            Best-fit model MDF on its own grid.
        loss : {'wrmse','mae','mape','huber','cosine','ks','log_cosh'}
            Loss to recompute per bootstrap (must exist as a column name
            in results CSV or be one of the standard ones you use).
        n_bootstrap : int
            Number of bootstrap resamples.
        seed : int
            RNG seed for reproducibility.
        save_plot : bool
            Whether to save a histogram figure of the bootstrap loss.
        label : str
            For titles/filenames ('MDF' or 'AMR', etc.)

        Returns
        -------
        dict with keys:
            'losses' (np.array), 'mean', 'std', 'p68', 'p95',
            'baseline_loss'
        """
        rng = np.random.default_rng(seed)

        # Interpolate model to obs_x
        fmod = interp1d(model_x, model_y, kind='linear',
                        bounds_error=False, fill_value=np.nan)
        model_on_obs = fmod(obs_x)

        # Mask to the overlap where model exists
        mask = ~np.isnan(model_on_obs)
        x = np.asarray(obs_x)[mask]
        y = np.asarray(obs_y)[mask]
        m = np.asarray(model_on_obs)[mask]

        # Sensible sigma defaults if not provided
        if obs_sigma is None:
            # If obs_y are normalized densities, adopt a small fractional error.
            # If they are counts, you can swap to sqrt(N) here.
            sigma = np.maximum(0.05 * np.maximum(y, 0.0), 1e-4)
        else:
            sig = np.asarray(obs_sigma)
            sigma = (sig[mask] if sig.ndim > 0 else np.full_like(y, float(sig)))
            sigma = np.maximum(sigma, 1e-6)

        # Choose loss function
        def _wrmse(y_true, y_pred, s):
            r = (y_pred - y_true) / s
            return float(np.sqrt(np.mean(r**2)))

        def _mae(y_true, y_pred, s):
            return float(np.mean(np.abs(y_pred - y_true)))

        def _mape(y_true, y_pred, s):
            denom = np.maximum(np.abs(y_true), 1e-8)
            return float(np.mean(np.abs((y_pred - y_true) / denom)))

        def _huber(y_true, y_pred, s, delta=1.0):
            r = y_pred - y_true
            a = np.abs(r)
            quad = 0.5 * np.minimum(a, delta)**2
            lin = delta * (a - delta/2.0)
            return float(np.mean(np.where(a <= delta, quad, lin)))

        def _cosine(y_true, y_pred, s):
            num = np.dot(y_true, y_pred)
            denom = np.linalg.norm(y_true) * np.linalg.norm(y_pred)
            return float(1.0 - num / np.maximum(denom, 1e-12))

        def _ks(y_true, y_pred, s):
            # Treat as CDFs for KS: cum-sum and normalize
            ct = np.cumsum(np.maximum(y_true, 0))
            cp = np.cumsum(np.maximum(y_pred, 0))
            if ct[-1] > 0: ct /= ct[-1]
            if cp[-1] > 0: cp /= cp[-1]
            return float(np.max(np.abs(ct - cp)))

        def _log_cosh(y_true, y_pred, s):
            r = y_pred - y_true
            return float(np.mean(np.log(np.cosh(r))))

        loss_map = {
            'wrmse': _wrmse,
            'mae': _mae,
            'mape': _mape,
            'huber': _huber,
            'cosine': _cosine,
            'ks': _ks,
            'log_cosh': _log_cosh
        }
        if loss.lower() not in loss_map:
            raise ValueError(f"Unknown loss '{loss}'. Choose one of {list(loss_map.keys())}.")

        # Baseline (unperturbed) loss
        baseline = loss_map[loss.lower()](y, m, sigma)

        # Bootstrap
        boot_losses = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            y_pert = y + rng.normal(0.0, sigma)
            boot_losses[i] = loss_map[loss.lower()](y_pert, m, sigma)

        mean = float(np.mean(boot_losses))
        std = float(np.std(boot_losses))
        p16, p84 = np.percentile(boot_losses, [16, 84])
        p2p5, p97p5 = np.percentile(boot_losses, [2.5, 97.5])

        if save_plot:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(boot_losses, bins=40, alpha=0.7, density=True)
            ax.axvline(baseline, color='k', ls='--', lw=2, label=f'Baseline: {baseline:.4f}')
            ax.axvspan(p16, p84, color='tab:blue', alpha=0.2, label='68% CI')
            ax.axvspan(p2p5, p97p5, color='tab:orange', alpha=0.15, label='95% CI')
            ax.set_xlabel(f'Bootstrap {loss.upper()} ({label})')
            ax.set_ylabel('Density')
            ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            output_file = os.path.join(self.output_path, 'uncertainty', 'weighted_stats.txt')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        return {
            'losses': boot_losses,
            'mean': mean,
            'std': std,
            'p68': (float(p16), float(p84)),
            'p95': (float(p2p5), float(p97p5)),
            'baseline_loss': baseline
        }



    def fitness_weighted_statistics(self, weight_power=1.0, save_results=True):
        """
        Calculate fitness-weighted parameter statistics.
        
        Parameters:
        -----------
        weight_power : float
            Power for fitness weighting (higher = more emphasis on best models)
        save_results : bool
            Whether to save results to file
            
        Returns:
        --------
        dict : Fitness-weighted statistics
        """
        print("Calculating fitness-weighted statistics...")
        
        # Convert fitness to weights (lower fitness = higher weight)
        fitness_vals = self.df_sorted[self.fitness_col].values
        
        # Use inverse fitness raised to a power
        raw_weights = 1.0 / (fitness_vals + np.min(fitness_vals) * 0.001)**weight_power
        weights = raw_weights / np.sum(raw_weights)  # Normalize
        
        weighted_stats = {}
        
        for param in self.continuous_params:
            values = self.df_sorted[param].values
            
            # Weighted statistics
            weighted_mean = np.average(values, weights=weights)
            weighted_var = np.average((values - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(weighted_var)
            
            # Weighted percentiles (approximate)
            sorted_idx = np.argsort(values)
            cumsum_weights = np.cumsum(weights[sorted_idx])
            
            # Find weighted quantiles
            q25_idx = np.searchsorted(cumsum_weights, 0.25)
            q50_idx = np.searchsorted(cumsum_weights, 0.50)
            q75_idx = np.searchsorted(cumsum_weights, 0.75)
            
            weighted_q25 = values[sorted_idx[q25_idx]] if q25_idx < len(values) else values[sorted_idx[-1]]
            weighted_median = values[sorted_idx[q50_idx]] if q50_idx < len(values) else values[sorted_idx[-1]]
            weighted_q75 = values[sorted_idx[q75_idx]] if q75_idx < len(values) else values[sorted_idx[-1]]
            
            # Effective sample size
            eff_sample_size = 1.0 / np.sum(weights**2)
            
            weighted_stats[param] = {
                'weighted_mean': weighted_mean,
                'weighted_std': weighted_std,
                'weighted_median': weighted_median,
                'weighted_q25': weighted_q25,
                'weighted_q75': weighted_q75,
                'weighted_iqr': weighted_q75 - weighted_q25,
                'unweighted_mean': np.mean(values),
                'unweighted_std': np.std(values),
                'effective_sample_size': eff_sample_size,
                'weight_concentration': 1.0 / len(weights) / np.max(weights)  # How concentrated are weights
            }
        
        if save_results:
            output_file = os.path.join(self.output_path, 'uncertainty', 'weighted_stats.txt')
            with open(output_file, 'w') as f:
                f.write(f"Fitness-Weighted Parameter Statistics\n")
                f.write(f"{'='*50}\n")
                f.write(f"Weight power: {weight_power}\n")
                f.write(f"Total models: {len(self.df_sorted)}\n\n")
                
                for param, stats in weighted_stats.items():
                    f.write(f"{param}:\n")
                    f.write(f"  Weighted: {stats['weighted_mean']:.6f} ± {stats['weighted_std']:.6f}\n")
                    f.write(f"  Unweighted: {stats['unweighted_mean']:.6f} ± {stats['unweighted_std']:.6f}\n")
                    f.write(f"  Weighted median: {stats['weighted_median']:.6f}\n")
                    f.write(f"  Weighted IQR: {stats['weighted_iqr']:.6f}\n")
                    f.write(f"  Effective N: {stats['effective_sample_size']:.1f}\n\n")
            
            print(f"Weighted statistics saved to {output_file}")
        
        return weighted_stats

    def marginalized_posteriors_kde(self, percentile=10, save_plot=True):
        """
        Generate marginalized posterior distributions via kernel density estimation.
        
        Parameters:
        -----------
        percentile : float
            Percentile of top models to analyze
        save_plot : bool
            Whether to save the plot
            
        Returns:
        --------
        dict : KDE objects for each parameter
        """
        print("Generating marginalized posterior distributions...")
        
        n_top = int(len(self.df_sorted) * percentile / 100)
        top_models = self.df_sorted.head(n_top)
        
        kdes = {}
        n_params = len(self.continuous_params)
        
        # Create subplot grid
        ncols = 3
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        param_labels = {
            'sigma_2': r'$\sigma_2$',
            't_1': r't$_1$ (Gyr)',
            't_2': r't$_2$ (Gyr)', 
            'infall_1': r'$\tau_1$ (Gyr)',
            'infall_2': r'$\tau_2$ (Gyr)',
            'sfe': 'SFE',
            'delta_sfe': r'$\Delta$SFE',
            'imf_upper': r'M$_{up}$ (M$_\odot$)',
            'mgal': r'M$_{gal}$ (M$_\odot$)',
            'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
        }
        
        for i, param in enumerate(self.continuous_params):
            ax = axes[i]
            data = top_models[param].values
            
            # Kernel density estimation
            kde = gaussian_kde(data)
            kdes[param] = kde
            
            # Generate smooth curve
            x_min, x_max = data.min(), data.max()
            x_range = x_max - x_min
            x_eval = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
            density = kde(x_eval)
            
            # Plot KDE
            ax.plot(x_eval, density, 'b-', linewidth=2, label='KDE')
            ax.fill_between(x_eval, density, alpha=0.3, color='blue')
            
            # Add histogram for reference
            ax.hist(data, bins=30, density=True, alpha=0.6, color='lightblue', 
                   edgecolor='black', label='Histogram')
            
            # Mark key statistics
            median_val = np.median(data)
            ax.axvline(median_val, color='red', linestyle='--', 
                      linewidth=2, label=f'Median: {median_val:.4f}')
            
            # Mark confidence intervals
            ci_lower = np.percentile(data, 16)
            ci_upper = np.percentile(data, 84)
            ax.axvline(ci_lower, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(ci_upper, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_xlabel(param_labels.get(param, param))
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
        # Remove empty subplots
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_plot:
            save_path = os.path.join(self.output_path, 'uncertainty', 'marginalized_posteriors.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Marginalized posteriors plot saved to {save_path}")
            
        plt.close()
        return kdes



    def plot_marginalized_posteriors_kde_weighted(
            self,
            percentile=10,
            bins=30,
            ncols=3,
            title='Marginalized posteriors (fitness-weighted)',
            show_kde=True,
            show_gaussian=True
        ):
        """
        For each continuous parameter:
          - plot a weighted histogram (density)
          - optionally overlay a fitness-weighted KDE
          - optionally overlay a Gaussian PDF using weighted mean/std
          - draw mean and ±1σ markers (weighted)

        Saves:
            <output>/uncertainty/marginalized_posteriors.png

        Returns
        -------
        path : str
        kdes : dict  {param: gaussian_kde or None}
        """
        # take top X% by fitness
        df = self.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)

        # inverse-fitness weights
        fit = top[self.fitness_col].values
        eps = np.min(fit) * 0.001
        w_raw = 1.0 / (fit + eps)
        w = w_raw / np.sum(w_raw)

        # labels
        param_labels = {
            'sigma_2': r'$\sigma_2$',
            't_1': r't$_1$ (Gyr)',
            't_2': r't$_2$ (Gyr)',
            'infall_1': r'$\tau_1$ (Gyr)',
            'infall_2': r'$\tau_2$ (Gyr)',
            'sfe': 'SFE',
            'delta_sfe': r'$\Delta$SFE',
            'imf_upper': r'M$_{up}$ (M$_\odot$)',
            'mgal': r'M$_{gal}$ (M$_\odot$)',
            'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
        }

        params = list(self.continuous_params)
        n_params = len(params)
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
        if nrows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        def wmean_wstd(values, weights):
            v = np.asarray(values, dtype=float)
            wts = np.asarray(weights, dtype=float)
            m = np.average(v, weights=wts)
            var = np.average((v - m)**2, weights=wts)
            return float(m), float(np.sqrt(max(var, 0.0)))

        def weighted_quantile(values, quantiles, weights):
            values = np.asarray(values, dtype=float)
            quantiles = np.asarray(quantiles, dtype=float)
            sorter = np.argsort(values)
            v = values[sorter]
            wts = np.asarray(weights)[sorter]
            cdf = np.cumsum(wts); cdf /= cdf[-1]
            return np.interp(quantiles, cdf, v)

        rng = np.random.default_rng(42)
        kdes = {}

        for i, p in enumerate(params):
            ax = axes[i]
            v = np.asarray(top[p].values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                kdes[p] = None
                continue

            # weighted histogram (density=True)
            hist_vals = ax.hist(v, bins=bins, density=True, alpha=0.45,
                                color='lightsteelblue', edgecolor='black', label='Weighted hist')

            # x-grid
            x_min, x_max = float(np.min(v)), float(np.max(v))
            xr = x_max - x_min
            if xr <= 0:
                # degenerate: a vertical line at the single value
                ax.axvline(x_min, color='red', ls='--', lw=2, label=f'Mean: {x_min:.4f}')
                ax.set_xlim(x_min - 0.5, x_max + 0.5)
                ax.set_yticks([])
                ax.set_xlabel(param_labels.get(p, p))
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8, frameon=False)
                kdes[p] = None
                continue

            xs = np.linspace(x_min - 0.1*xr, x_max + 0.1*xr, 512)

            # KDE (fitness-weighted). Fall back to weighted resample if needed.
            kde_obj = None
            if show_kde:
                try:
                    kde_obj = gaussian_kde(v, weights=w)
                except Exception:
                    N = min(max(1000, 5*len(v)), 5000)
                    res = rng.choice(v, size=N, replace=True, p=w/np.sum(w))
                    if np.std(res) > 0:
                        kde_obj = gaussian_kde(res)
                if kde_obj is not None:
                    ax.plot(xs, kde_obj(xs), lw=2, label='Weighted KDE')
            kdes[p] = kde_obj

            # Gaussian fit using weighted mean/std
            mu, sig = wmean_wstd(v, w)
            if show_gaussian and sig > 0 and np.isfinite(sig):
                # normal pdf scaled as a density curve
                gauss = (1.0 / (np.sqrt(2*np.pi) * sig)) * np.exp(-0.5 * ((xs - mu)/sig)**2)
                ax.plot(xs, gauss, lw=1.6, linestyle='--', label='Weighted Gaussian')

            # mean and ±1σ markers
            ax.axvline(mu, color='red', ls='--', lw=2, label=f'Mean: {mu:.4f}')
            if sig > 0 and np.isfinite(sig):
                ax.axvline(mu - sig, color='gray', ls=':', lw=1.5)
                ax.axvline(mu + sig, color='gray', ls=':', lw=1.5)

            # also show 16/84% (often close to ±1σ only if near-Gaussian)
            q16, q84 = weighted_quantile(v, [0.16, 0.84], w)
            ax.axvline(q16, color='orange', ls=':', lw=1.5)
            ax.axvline(q84, color='orange', ls=':', lw=1.5)

            ax.set_xlabel(param_labels.get(p, p))
            ax.set_ylabel('Density')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, frameon=False)

        # remove extra axes
        for j in range(n_params, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(title + f' — Top {percentile}%', y=0.995, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = os.path.join(self.output_path, 'uncertainty', 'marginalised_posteriors.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path, kdes


    # ---- helpers -----------------------------------------------------------
    def _select_top_and_weights(self, percentile=10, weight_power=1.0):
        """Return top subset and normalized inverse-fitness weights."""
        df = self.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)
        fit = np.asarray(top[self.fitness_col].values, dtype=float)
        eps = np.min(fit) * 0.001
        w = 1.0 / np.power(fit + eps, weight_power)
        w = w / np.sum(w)
        return top, w

    def _wmean(self, x, w):
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
        return float(np.sum(w*x) / np.sum(w))

    def _wvar(self, x, w):
        m = self._wmean(x, w)
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sum(w*(x-m)**2) / np.sum(w))

    def _wcov(self, x, y, w):
        mx = self._wmean(x, w); my = self._wmean(y, w)
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sum(w*(x-mx)*(y-my)) / np.sum(w))


    def quantify_pairwise_degeneracy(self,
                                     params=None,
                                     percentile=10,
                                     weight_power=1.0,
                                     mi_bins=24,
                                     save_csv=True):
        """
        Quantify pairwise degeneracy between parameters.

        Metrics per (i,j):
          - weighted Pearson rho_w
          - Spearman rho_s (unweighted ranks)
          - weighted mutual information MI_w (via weighted 2D hist)
          - ellipse axis ratio AR = sqrt(lambda_max / lambda_min) of
            the weighted covariance in z-scored space (degeneracy elongation)

        Returns
        -------
        df_pairs : DataFrame with columns [pi,pj,rho_w,rho_s,MI_w,axis_ratio]
        """
        top, w = self._select_top_and_weights(percentile, weight_power)
        if params is None:
            params = list(self.continuous_params)

        rows = []
        for i in range(len(params)):
            for j in range(i+1, len(params)):
                pi, pj = params[i], params[j]
                xi = np.asarray(top[pi].values, dtype=float)
                xj = np.asarray(top[pj].values, dtype=float)

                # weighted Pearson
                vx = self._wvar(xi, w); vy = self._wvar(xj, w)
                if vx <= 0 or vy <= 0:
                    rho_w = np.nan
                else:
                    rho_w = self._wcov(xi, xj, w) / np.sqrt(vx*vy)

                # Spearman (standard)
                rho_s, _ = stats.spearmanr(xi, xj)

                # weighted mutual information (discretize)
                # weighted histograms with np.histogram{,2}d
                # bounds = data range
                xi_f = xi[np.isfinite(xi)]; xj_f = xj[np.isfinite(xj)]
                if xi_f.size < 5 or xj_f.size < 5:
                    MI_w = np.nan
                else:
                    x_edges = np.linspace(xi_f.min(), xi_f.max(), mi_bins+1)
                    y_edges = np.linspace(xj_f.min(), xj_f.max(), mi_bins+1)
                    Hxy, _, _ = np.histogram2d(xi, xj, bins=[x_edges, y_edges], weights=w)
                    Hx,  _    = np.histogram(xi, bins=x_edges, weights=w)
                    Hy,  _    = np.histogram(xj, bins=y_edges, weights=w)

                    Pxy = Hxy / max(Hxy.sum(), 1e-16)
                    Px  = Hx  / max(Hx.sum(),  1e-16)
                    Py  = Hy  / max(Hy.sum(),  1e-16)

                    # MI = sum p(x,y) log p(x,y)/(p(x)p(y))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        denom = (Px[:,None] * Py[None,:])
                        ratio = np.where((Pxy>0) & (denom>0), Pxy/denom, 1.0)
                        MI_w  = float(np.nansum(Pxy * np.log(ratio)))
                    MI_w = max(MI_w, 0.0)  # numerical safety

                # ellipse axis ratio (weighted, z-scored)
                sx = np.sqrt(max(vx, 1e-30)); sy = np.sqrt(max(vy, 1e-30))
                zx = (xi - self._wmean(xi, w)) / sx
                zy = (xj - self._wmean(xj, w)) / sy
                C11 = self._wcov(zx, zx, w); C22 = self._wcov(zy, zy, w)
                C12 = self._wcov(zx, zy, w)
                C   = np.array([[C11, C12],[C12, C22]], dtype=float)
                evals = np.linalg.eigvalsh(C)
                evals = np.clip(evals, 1e-12, None)
                axis_ratio = float(np.sqrt(evals.max()/evals.min()))

                rows.append(dict(pi=pi, pj=pj,
                                 rho_w=rho_w, rho_s=rho_s,
                                 MI_w=MI_w, axis_ratio=axis_ratio))

        df_pairs = pd.DataFrame(rows).sort_values('MI_w', ascending=False)

        if save_csv:
            out_csv = os.path.join(self.output_path, 'uncertainty', 'degeneracy_pairs.csv')
            df_pairs.to_csv(out_csv, index=False)
            print(f"Pairwise degeneracy CSV: {out_csv}")

        return df_pairs


    def plot_degeneracy_heatmaps(self,
                                 params=None,
                                 percentile=10,
                                 weight_power=1.0,
                                 mi_bins=24):
        """
        Render lower-triangular heatmaps for:
          - weighted Pearson rho_w
          - Spearman rho_s
          - weighted mutual information MI_w
          - ellipse axis ratio (axis_ratio)

        Returns
        -------
        paths : dict of saved PNG paths
        """
        dfp = self.quantify_pairwise_degeneracy(params=params,
                                                percentile=percentile,
                                                weight_power=weight_power,
                                                mi_bins=mi_bins,
                                                save_csv=True)
        if params is None:
            params = list(self.continuous_params)
        n = len(params)

        # helper to fill NxN matrix (lower triangle)
        def mat_from_pairs(key, fill_diag=np.nan):
            M = np.full((n,n), np.nan, dtype=float)
            idx = {p:i for i,p in enumerate(params)}
            for _, row in dfp.iterrows():
                i = idx[row['pi']]; j = idx[row['pj']]
                M[i,j] = row[key]
                M[j,i] = row[key]
            if not np.isnan(fill_diag):
                np.fill_diagonal(M, fill_diag)
            return M

        mats = {
            'rho_w'     : mat_from_pairs('rho_w', fill_diag=1.0),
            'rho_s'     : mat_from_pairs('rho_s', fill_diag=1.0),
            'MI_w'      : mat_from_pairs('MI_w', fill_diag=0.0),
            'axis_ratio': mat_from_pairs('axis_ratio', fill_diag=1.0),
        }

        paths = {}
        for name, M in mats.items():
            fig, ax = plt.subplots(figsize=(1.0+0.5*n, 1.0+0.5*n))
            # mask upper triangle to avoid duplicate info
            mask = np.triu(np.ones_like(M, dtype=bool), k=1)
            if name in ('rho_w', 'rho_s'):
                vmin, vmax, cmap = -1, 1, 'coolwarm'
            elif name == 'MI_w':
                vmin, vmax, cmap = 0, np.nanmax(M), 'mako'
            else:  # axis_ratio
                vmin, vmax, cmap = 1, np.nanmin([np.nanmax(M), 20]), 'viridis'
            sns.heatmap(M, mask=mask, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                        square=True, cbar_kws={'shrink':0.8},
                        xticklabels=params, yticklabels=params, linewidths=0.3, linecolor='w')
            ax.set_title(f'{name} — Top {percentile}%')
            plt.tight_layout()
            out = os.path.join(self.output_path, 'uncertainty', f'degeneracy_{name}.png')
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(fig)
            paths[name] = out
            print(f"{name} heatmap: {out}")

        return paths


    def plot_pca_degeneracy(self,
                            params=None,
                            percentile=10,
                            weight_power=1.0,
                            n_loadings=6):
        """
        Weighted PCA on standardized parameters to expose sloppy/degenerate directions.

        Outputs:
          - Scree plot (explained variance ratio)
          - Loadings heatmap for top components

        Returns
        -------
        paths : dict {'scree':..., 'loadings':...}
        pca_result : dict with eigenvalues, explained_ratio, components (loadings), mean, std
        """
        top, w = self._select_top_and_weights(percentile, weight_power)
        if params is None:
            params = list(self.continuous_params)

        X = np.vstack([np.asarray(top[p].values, dtype=float) for p in params]).T
        # weighted standardization
        mu = np.array([self._wmean(X[:,j], w) for j in range(X.shape[1])])
        sig = np.sqrt(np.array([max(self._wvar(X[:,j], w), 1e-30) for j in range(X.shape[1])]))
        Z = (X - mu) / sig

        # weighted covariance
        W = np.diag(w / np.sum(w))
        C = Z.T @ W @ Z  # (p x p)
        # symmetric -> eigen-decomposition
        evals, evecs = np.linalg.eigh(C)
        # sort descending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        explained = evals / np.sum(evals)

        # scree
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(np.arange(1, len(evals)+1), explained, marker='o')
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        scree_path = os.path.join(self.output_path, 'uncertainty', 'pca_screen.png')
        plt.savefig(scree_path, dpi=300, bbox_inches='tight'); plt.close(fig)

        # loadings heatmap for top k
        k = min(n_loadings, len(params))
        loadings = pd.DataFrame(evecs[:, :k],
                                index=params,
                                columns=[f'PC{i+1}' for i in range(k)])
        fig, ax = plt.subplots(figsize=(1.2*k+4, 0.35*len(params)+3))
        sns.heatmap(loadings, cmap='coolwarm', center=0, ax=ax,
                    cbar_kws={'shrink':0.7}, annot=False, linewidths=0.3, linecolor='w')
        plt.tight_layout()
        loadings_path = os.path.join(self.output_path, 'uncertainty', 'pca_loadings.png')
        plt.savefig(loadings_path, dpi=300, bbox_inches='tight'); plt.close(fig)

        # condition number (sloppiness index)
        cond = float(np.sqrt(np.max(evals)/max(np.min(evals), 1e-30)))
        with open(os.path.join(self.output_path, 'uncertainty', 'pca_summary.txt'), 'w') as f:
            f.write(f"Condition number (sqrt(lambda_max/lambda_min)): {cond:.3e}\n")
            f.write("Explained variance ratios:\n")
            for i, r in enumerate(explained, 1):
                f.write(f"  PC{i}: {r:.4f}\n")

        print(f"PCA scree:    {scree_path}")
        print(f"PCA loadings: {loadings_path}")

        return {'scree':scree_path, 'loadings':loadings_path}, {
            'eigenvalues': evals, 'explained_ratio': explained,
            'components': evecs, 'mean': mu, 'std': sig
        }


    def plot_corner_with_marginals(
            self,
            params=None,
            percentile=10,
            weight_power=1.0,
            bins=40,
            save_path=None,
            assoc_metric='spearman',       # 'spearman' | 'pearson' | 'axis_ratio'
            ink_color='#111111',           # single color; no colormap
            alpha_gamma=0.6                # contrast for continuous alpha (lower = stronger contrast)
        ):
        """
        Corner + marginals, with:
          - Off-diagonal: 2D KDE shown via single-color continuous alpha (no colormap).
          - Diagonal: weighted hist + KDE only (NO HPD, NO Gaussian overlay).
          - Axes limits are FIXED from bulge_pcard.txt ranges.
          - If `params` is None, include every continuous parameter available in pcard & data.
        """
        import os
        import re
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from scipy.stats import gaussian_kde, spearmanr, pearsonr

        # ---------- data ----------
        top, w = self._select_top_and_weights(percentile=percentile, weight_power=weight_power)

        # ---------- load pcard ranges ----------
        # default path; override by setting self.bulge_pcard_path before calling
        pcard_path = getattr(self, 'bulge_pcard_path', 'bulge_pcard.txt')

        # map pcard keys -> run/column names
        p2c = {
            'sigma_2_list': 'sigma_2',
            'tmax_1_list': 't_1',
            'tmax_2_list': 't_2',
            'infall_timescale_1_list': 'infall_1',
            'infall_timescale_2_list': 'infall_2',
            'sfe_array': 'sfe',
            'delta_sfe_array': 'delta_sfe',
            'imf_upper_limits': 'imf_upper',
            'mgal_values': 'mgal',
            'nb_array': 'nb',
        }

        def _parse_ranges_from_pcard(path):
            # returns {colname: (lo, hi)} for numeric 2-length arrays
            txt = open(path, 'r', encoding='utf-8').read()
            ranges = {}
            # capture lines like: key: [a, b]
            for key, col in p2c.items():
                m = re.search(rf'^\s*{re.escape(key)}\s*:\s*\[([^\]]+)\]', txt, flags=re.MULTILINE)
                if not m:
                    continue
                try:
                    nums = [float(x.strip()) for x in m.group(1).split(',')]
                except Exception:
                    continue
                if len(nums) == 2:
                    lo, hi = float(nums[0]), float(nums[1])
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        ranges[col] = (lo, hi)
            return ranges

        param_ranges = _parse_ranges_from_pcard(pcard_path)

        # determine which params to plot
        if params is None:
            # start from pcard continuous keys
            candidates = list(param_ranges.keys())
            # intersect with DataFrame columns we actually have
            candidates = [c for c in candidates if c in top.columns]
            # if class exposes continuous_params, enforce that too
            if hasattr(self, 'continuous_params') and isinstance(self.continuous_params, (list, set, tuple)):
                candidates = [c for c in candidates if c in self.continuous_params]
            params = candidates

        if len(params) == 0:
            raise ValueError("No continuous parameters found in both pcard and data frame.")

        k = len(params)

        # ---------- helpers ----------
        rng = np.random.default_rng(1337)

        def _kde_1d(vals, weights, xs):
            vals = np.asarray(vals, float)
            weights = np.asarray(weights, float)
            good = np.isfinite(vals) & np.isfinite(weights)
            vals = vals[good]; weights = weights[good]
            if len(vals) == 0:
                return np.zeros_like(xs)
            try:
                return gaussian_kde(vals, weights=weights)(xs)
            except Exception:
                # bootstrap if singular
                N = min(max(1000, 5*len(vals)), 5000)
                p = weights / np.sum(weights) if np.sum(weights) > 0 else None
                idx = rng.choice(np.arange(len(vals)), size=N, replace=True, p=p)
                return gaussian_kde(vals[idx])(xs)

        def _kde_2d(x, y, weights, Xg, Yg):
            x = np.asarray(x, float); y = np.asarray(y, float)
            weights = np.asarray(weights, float)
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
            x = x[good]; y = y[good]; weights = weights[good]
            if len(x) == 0:
                return np.zeros_like(Xg)
            xy = np.vstack([x, y])
            try:
                kde = gaussian_kde(xy, weights=weights)
            except Exception:
                N = min(max(1000, 5*len(x)), 5000)
                p = weights / np.sum(weights) if np.sum(weights) > 0 else None
                idx = rng.choice(np.arange(len(x)), size=N, replace=True, p=p)
                kde = gaussian_kde(np.vstack([x[idx], y[idx]]))
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            return Z

        def _assoc(x, y, weights):
            x = np.asarray(x, float); y = np.asarray(y, float)
            if assoc_metric == 'pearson':
                r, _ = pearsonr(x, y); return r, r'$\rho$'
            elif assoc_metric == 'axis_ratio':
                # weighted covariance eigen ratio
                mx = np.average(x, weights=weights); sx = np.sqrt(np.average((x-mx)**2, weights=weights))
                my = np.average(y, weights=weights); sy = np.sqrt(np.average((y-my)**2, weights=weights))
                X = np.vstack([(x-mx)/max(sx,1e-12), (y-my)/max(sy,1e-12)])
                W = np.diag(weights/np.sum(weights))
                C = X @ W @ X.T
                evals, _ = np.linalg.eigh(C); evals = np.sort(evals)
                ar = np.sqrt(max(evals[-1],1e-30)/max(evals[0],1e-30))
                return ar, 'AR'
            else:
                r, _ = spearmanr(x, y); return r, r'$\rho_s$'

        label_map = {
            'sigma_2': r'$\sigma_2$',
            't_1': r't$_1$ (Gyr)', 't_2': r't$_2$ (Gyr)',
            'infall_1': r'$\tau_1$ (Gyr)', 'infall_2': r'$\tau_2$ (Gyr)',
            'sfe': 'SFE', 'delta_sfe': r'$\Delta$SFE',
            'imf_upper': r'M$_{up}$ (M$_\odot$)',
            'mgal': r'M$_{gal}$ (M$_\odot$)', 'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
        }

        # ---------- figure (no gaps) ----------
        fig, axes = plt.subplots(k, k, figsize=(3.0*k, 3.0*k), constrained_layout=False)
        plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.0)

        rgb = np.array(mcolors.to_rgb(ink_color))

        for i, pi in enumerate(params):
            xi = np.asarray(top[pi].values, float)
            xlim_i = param_ranges.get(pi, (np.nanmin(xi), np.nanmax(xi)))

            for j, pj in enumerate(params):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off'); continue

                if i == j:
                    # ----- diagonal: hist + KDE (only) with fixed range -----
                    v = xi[np.isfinite(xi)]
                    if v.size == 0:
                        ax.axis('off'); continue

                    lo, hi = xlim_i
                    xs = np.linspace(lo, hi, 512)

                    # weighted hist within fixed range
                    ax.hist(v, bins=bins, range=(lo, hi), density=True,
                            color='lightgray', edgecolor='black', alpha=0.7, weights=w)

                    # KDE
                    dens = _kde_1d(v, w, xs)
                    ax.plot(xs, dens, lw=2, color='C0')

                    ax.set_xlim(lo, hi)
                    ax.set_yticks([])
                    if i < k-1: ax.set_xticklabels([])
                    ax.set_xlabel(label_map.get(pi, pi))

                else:
                    # ----- lower triangle: single-color alpha KDE + faint points, fixed extents -----
                    x = np.asarray(top[pj].values, float)
                    y = xi
                    xlim_j = param_ranges.get(pj, (np.nanmin(x), np.nanmax(x)))
                    lo_x, hi_x = xlim_j
                    lo_y, hi_y = xlim_i

                    nx = ny = 220
                    xg = np.linspace(lo_x, hi_x, nx)
                    yg = np.linspace(lo_y, hi_y, ny)
                    Xg, Yg = np.meshgrid(xg, yg)
                    Z = _kde_2d(x, y, w, Xg, Yg)
                    if np.all(~np.isfinite(Z)) or np.nanmax(Z) <= 0:
                        Z = np.zeros_like(Xg)

                    # normalize and map to alpha
                    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                    Zn = Z / (Z.max() + 1e-30)
                    alpha = np.power(Zn, alpha_gamma)
                    # cap to avoid fully opaque blocks; keep readable
                    alpha *= 0.95

                    rgba = np.empty((ny, nx, 4), dtype=float)
                    rgba[..., 0] = rgb[0]
                    rgba[..., 1] = rgb[1]
                    rgba[..., 2] = rgb[2]
                    rgba[..., 3] = alpha

                    ax.imshow(rgba, extent=(lo_x, hi_x, lo_y, hi_y),
                              origin='lower', interpolation='bilinear', aspect='auto')

                    # faint, small markers
                    ax.scatter(x, y, s=4, c='k', alpha=0.10, linewidths=0)

                    # association metric (kept; not HPD)
                    try:
                        mval, msym = _assoc(x, y, w)
                        ax.text(0.03, 0.95, f'{msym}={mval:.2f}',
                                transform=ax.transAxes, ha='left', va='top',
                                fontsize=14, bbox=dict(facecolor='white', edgecolor='0.7', boxstyle='round,pad=0.2'))
                    except Exception:
                        pass

                    # labels & limits
                    ax.set_xlim(lo_x, hi_x)
                    ax.set_ylim(lo_y, hi_y)
                    if j == 0:
                        ax.set_ylabel(label_map.get(pi, pi))
                    else:
                        ax.set_yticks([])
                    if i == k-1:
                        ax.set_xlabel(label_map.get(pj, pj))
                    else:
                        ax.set_xticks([])

        if save_path is None:
            save_path = os.path.join(self.output_path, 'uncertainty', 'posterior_corner_combo.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path





    def choose_cutoff_lognorm_mixture(
            self, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6, force_k2=False
        ):
        """
        Simple, reviewer-proof cutoff:
          - Work in y = log(loss).
          - Fit K=1 and K=2 Gaussian mixtures in y by EM; pick K by BIC (unless force_k2=True).
          - If K=2: cutoff = equal-responsibility boundary where pi1*N1(y)=pi2*N2(y).
          - If K=1: no hard cut (use all models).
        Writes two plots and a small audit file; returns cutoff & realized keep fraction.
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde, norm
        from scipy.special import logsumexp

        # ---------- data ----------
        L = np.asarray(self.df_sorted[self.fitness_col].values, float)
        L = L[np.isfinite(L)]
        if L.size == 0:
            raise RuntimeError("No finite losses/fitness values.")
        eps = 1e-12
        y = np.log(L + eps)
        N = y.size

        # ---------- helper: EM for 1D K-component Gaussian mixture ----------
        def em_gmm_1d(y, K, iters=200, tol=1e-6):
            # init by quantiles
            qs = np.linspace(0.2, 0.8, K)
            mu = np.quantile(y, qs) if K > 1 else np.array([float(np.mean(y))])
            s0 = float(np.std(y))
            s0 = s0 if s0 > 1e-6 else 0.1
            sig = np.full(K, s0, float)
            pi = np.full(K, 1.0 / K, float)

            c_norm = -0.5*np.log(2*np.pi)

            def logpdf(y, mu, sig):
                return c_norm - np.log(sig) - 0.5*((y - mu)/sig)**2

            prev_ll = -np.inf
            for _ in range(iters):
                # E-step: responsibilities (log-space)
                log_comp = np.stack([np.log(pi[k]) + logpdf(y, mu[k], sig[k] + 1e-12) for k in range(K)], axis=1)
                log_den = logsumexp(log_comp, axis=1, keepdims=True)
                R = np.exp(log_comp - log_den)  # N x K
                Nk = R.sum(axis=0) + 1e-12

                # M-step
                mu_new = (R * y[:, None]).sum(axis=0) / Nk
                sig_new = np.sqrt((R * (y[:, None] - mu_new[None, :])**2).sum(axis=0) / Nk)
                sig_new = np.maximum(sig_new, 1e-6)
                pi_new = Nk / N

                # log-likelihood
                ll = float(np.sum(log_den))
                if abs(ll - prev_ll) < tol:
                    mu, sig, pi = mu_new, sig_new, pi_new
                    prev_ll = ll
                    break
                mu, sig, pi, prev_ll = mu_new, sig_new, pi_new, ll

            # BIC: p = (K-1) + K (means) + K (stds) = 2K - 1
            bic = -2.0*prev_ll + (2*K - 1)*np.log(N)
            # order by mean
            order = np.argsort(mu)
            return pi[order], mu[order], sig[order], prev_ll, bic

        # ---------- fit K=1 and K=2 ----------
        pi1, mu1, sg1, ll1, bic1 = em_gmm_1d(y, 1, em_max_iter, tol)
        pi2, mu2, sg2, ll2, bic2 = em_gmm_1d(y, 2, em_max_iter, tol)
        choose_K2 = force_k2 or (bic2 < bic1)

        # ---------- cutoff (if K=2), else None ----------
        loss_cutoff = None
        chosen_K = 2 if choose_K2 else 1
        if choose_K2:
            # components already ordered: comp0 is the elite (lower mu)
            pi = pi2; mu = mu2; sig = sg2

            # Solve pi0*N0(y) = pi1*N1(y) analytically
            A = 0.5*(1.0/sig[1]**2 - 1.0/sig[0]**2)
            B = (mu[0]/sig[0]**2 - mu[1]/sig[1]**2)
            D = 0.5*(mu[1]**2/sig[1]**2 - mu[0]**2/sig[0]**2)
            const = np.log((pi[1]/sig[1])/(pi[0]/sig[0]))
            C = D - const

            if abs(A) < 1e-12:
                y_cut = -C / (B + 1e-12)  # equal-variance fallback
            else:
                disc = max(B*B - 4*A*C, 0.0)
                roots = np.sort(( -B + np.array([-1.0, 1.0])*np.sqrt(disc) ) / (2*A))
                # prefer a root between the two means; otherwise, nearest to their midpoint
                mid = 0.5*(mu[0] + mu[1])
                if (mu[0] <= roots[0] <= mu[1]) or (mu[0] <= roots[1] <= mu[1]):
                    y_cut = roots[0] if (mu[0] <= roots[0] <= mu[1]) else roots[1]
                else:
                    y_cut = roots[np.argmin(np.abs(roots - mid))]

            loss_cutoff = float(np.exp(y_cut))
            frac = float(np.mean(L <= loss_cutoff))
        else:
            # no hard selection
            frac = 1.0

        pct = 100.0 * frac

        # ---------- plots ----------
        xs = np.linspace(L.min(), L.max(), max(kde_points, 512))
        kde = gaussian_kde(L)(xs)

        def mixture_pdf_L(x, pi, mu, sig):  # f_L(x) = sum pi_k * phi(log x; mu_k, sig_k)/x
            x = np.maximum(x, eps)
            parts = []
            for k in range(pi.size):
                parts.append(pi[k] * (1.0/(x*(sig[k]*np.sqrt(2*np.pi)))) *
                             np.exp(-0.5*((np.log(x) - mu[k])/sig[k])**2))
            return np.sum(parts, axis=0)

        # distribution panel
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(L, bins=bins, density=True, alpha=0.6, color='lightsteelblue',
                edgecolor='black', label='Histogram')
        ax.plot(xs, kde, color='C0', lw=2, label='KDE')

        if choose_K2:
            ax.plot(xs, mixture_pdf_L(xs, np.array([pi2[0],0]), np.array([mu2[0],0]), np.array([sg2[0],1])),
                    color='C1', ls='--', lw=1.6, label=f'comp 1 (w={pi2[0]:.2f})')
            ax.plot(xs, mixture_pdf_L(xs, np.array([0,pi2[1]]), np.array([0,mu2[1]]), np.array([1,sg2[1]])),
                    color='C3', ls='--', lw=1.6, label=f'comp 2 (w={pi2[1]:.2f})')
            ax.plot(xs, mixture_pdf_L(xs, pi2, mu2, sg2), color='k', lw=1.2, alpha=0.9, label='mixture')
            ax.axvline(loss_cutoff, color='C2', lw=2.2,
                       label=f'cutoff: {loss_cutoff:.4g}  (keep≈{pct:.1f}%)')
        else:
            ax.plot(xs, mixture_pdf_L(xs, pi1, mu1, sg1), color='k', lw=1.2, alpha=0.9,
                    label='single-component fit (no cut)')

        ax.set_xlabel(self.fitness_col.upper()); ax.set_ylabel('Density'); ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=10, ncol=2)
        dist_path = os.path.join(self.output_path, 'uncertainty', 'loss_distribution_analysis.png')
        os.makedirs(os.path.dirname(dist_path), exist_ok=True)
        plt.tight_layout(); plt.savefig(dist_path, dpi=300, bbox_inches='tight'); plt.close(fig)

        # cumulative panel
        Ls = np.sort(L); cum = (np.arange(Ls.size)+1)/Ls.size
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Ls, cum, lw=2, color='C0')
        if choose_K2:
            ax.axvline(loss_cutoff, color='C2', lw=2.2, label='chosen cutoff')
            ax.axhline(frac, color='C2', ls=':', lw=1.2)
            ax.text(loss_cutoff, frac, f'  keep ≈ {pct:.1f}%', va='bottom', ha='left', fontsize=10)
        ax.set_xlabel(self.fitness_col.upper()); ax.set_ylabel('Cumulative population fraction')
        ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(frameon=False)
        cdf_path = os.path.join(self.output_path, 'uncertainty', 'cumulative_population_vs_loss.png')
        plt.tight_layout(); plt.savefig(cdf_path, dpi=300, bbox_inches='tight'); plt.close(fig)

        # ---------- audit ----------
        audit_path = os.path.join(self.output_path, 'uncertainty', 'loss_cutoff.txt')
        with open(audit_path, 'w') as f:
            f.write("method: bic_gaussmix_on_logloss_equal_responsibility\n")
            f.write(f"N: {N}\n")
            f.write(f"BIC_K1: {bic1:.6f}\n")
            f.write(f"BIC_K2: {bic2:.6f}\n")
            f.write(f"chosen_K: {chosen_K}\n")
            if choose_K2:
                f.write(f"pi: {pi2.tolist()}\nmu: {mu2.tolist()}\nsigma: {sg2.tolist()}\n")
                f.write(f"loss_cutoff: {loss_cutoff:.8g}\n")
            else:
                f.write(f"pi: {pi1.tolist()}\nmu: {mu1.tolist()}\nsigma: {sg1.tolist()}\n")
                f.write("loss_cutoff: None\n")
            f.write(f"fraction_kept: {frac:.6f}\npercentile_kept: {pct:.4f}\n")

        return {
            'loss_cutoff': loss_cutoff,
            'fraction': frac,
            'percentile': pct,
            'chosen_K': chosen_K,
            'paths': {'dist': dist_path, 'cdf': cdf_path, 'audit': audit_path}
        }






    def generate_comprehensive_uncertainty_report(self):
        """
        Generate a comprehensive uncertainty analysis report covering all aspects.
        Also triggers the plotting routines for (2) and (3).
        """
        print("Generating comprehensive uncertainty analysis...")


        print("Choosing cutoff via 2-component lognormal mixture...")
        cut = self.choose_cutoff_lognorm_mixture()
        ptile = cut['percentile']        # use everywhere downstream (corner, posteriors, etc.)
        print(f"[cutoff] {cut['loss_cutoff']:.6g}  → keep ≈ {ptile:.2f}%")

        # Run all analyses (text outputs)
        bootstrap_results = self.bootstrap_parameter_uncertainty()
        weighted_results  = self.fitness_weighted_statistics()
        _ = self.marginalized_posteriors_kde()  # legacy (unweighted) 1D plot if you want to keep it


        print("Rendering fitness-weighted parameter intervals...")
        save_path_facet, table = self.plot_fitness_weighted_param_intervals_facet(
            percentile=ptile,   # or your fitness-based cutoff
            weight_power=1.0,
            ncols=2          # tweak layout (2–3 works well)
        )


        path, tabel = self.plot_marginalized_posteriors_kde_weighted(
            percentile=ptile,
            bins=30,
            ncols=3,
            title='Marginalized posteriors (fitness-weighted)',
            show_kde=True,
            show_gaussian=True
        )
        print(path)

        print("Rendering 2D corner (pairwise KDE)...")
        # choose a compact set so the figure is readable; adjust as needed
        corner_path = self.plot_corner_2d_kde(
            params=[p for p in ['sigma_2','t_2','infall_2','sfe'] if p in self.continuous_params],
            percentile=ptile, weight_power=1.0
        )



        # --- Degeneracy quantification ---
        print("Computing pairwise degeneracy metrics...")
        deg_pairs = self.quantify_pairwise_degeneracy(percentile=ptile, weight_power=1.0, mi_bins=24)
        print("Rendering degeneracy heatmaps...")
        deg_paths = self.plot_degeneracy_heatmaps(percentile=ptile, weight_power=1.0, mi_bins=24)
        print("Running weighted PCA (degeneracy)...")
        pca_paths, pca_res = self.plot_pca_degeneracy(percentile=ptile, weight_power=1.0, n_loadings=6)



        combo_path = self.plot_corner_with_marginals(save_path = os.path.join(self.output_path, 'uncertainty', 'posterior_corner_combo.png'),
            params=[p for p in ['sigma_2', 't_2', 'infall_2', 'sfe'] if p in self.continuous_params],
            percentile=ptile, weight_power=1.0, assoc_metric='spearman'
        )


        combo_path = self.plot_corner_with_marginals(save_path = os.path.join(self.output_path, 'uncertainty', 'in1_posterior_corner_combo.png'),
            params=[p for p in ['sigma_2', 't_1', 'infall_1', 'sfe'] if p in self.continuous_params],
            percentile=ptile, weight_power=1.0, assoc_metric='spearman'
        )


        combo_path = self.plot_corner_with_marginals(save_path = os.path.join(self.output_path, 'uncertainty', 'chem_posterior_corner_combo.png'),
            params=[p for p in ['sigma_2', 'mgal', 'delta_sfe', 'sfe'] if p in self.continuous_params],
            percentile=ptile, weight_power=1.0, assoc_metric='spearman'
        )







        # Generate summary report
        report_file = os.path.join(self.output_path, 'uncertainty', 'comprehensive_report.txt')
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE UNCERTAINTY QUANTIFICATION REPORT\n")
            f.write("="*60 + "\n\n")

            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            f.write(f"Total models evaluated: {len(self.df)}\n")
            f.write(f"Best fitness achieved: {self.df_sorted[self.fitness_col].iloc[0]:.6f}\n")
            f.write(f"Fitness range: {self.df_sorted[self.fitness_col].iloc[0]:.6f} - "
                    f"{self.df_sorted[self.fitness_col].iloc[-1]:.6f}\n\n")

            f.write("FIGURES WRITTEN\n")
            f.write("-"*16 + "\n")
            f.write(f"Weighted 1D posteriors:   {path_1d_weighted}\n")
            f.write(f"Weighted CI bars:         {save_path_facet}\n")
            f.write(f"2D corner KDE:            {corner_path}\n\n")

            f.write("PARAMETER CONSTRAINT QUALITY\n")
            f.write("-"*30 + "\n")
            well_constrained, moderate_constrained, poorly_constrained = [], [], []
            for param in self.continuous_params:
                cv = bootstrap_results[param]['coefficient_of_variation']
                if cv < 0.05:   well_constrained.append((param, cv))
                elif cv < 0.2:  moderate_constrained.append((param, cv))
                else:           poorly_constrained.append((param, cv))
            f.write(f"Well-constrained (CV < 5%): {len(well_constrained)} params\n")
            for p, cv in well_constrained: f.write(f"  - {p}: CV = {cv:.4f}\n")
            f.write(f"\nModerately constrained (5% ≤ CV < 20%): {len(moderate_constrained)} params\n")
            for p, cv in moderate_constrained: f.write(f"  - {p}: CV = {cv:.4f}\n")
            f.write(f"\nPoorly constrained (CV ≥ 20%): {len(poorly_constrained)} params\n")
            for p, cv in poorly_constrained: f.write(f"  - {p}: CV = {cv:.4f}\n")

            f.write("\nMETHODOLOGY VALIDATION\n")
            f.write("-"*25 + "\n")
            f.write("Bootstrap vs Weighted Statistics Agreement:\n")
            for param in self.continuous_params:
                boot_mean = bootstrap_results[param]['bootstrap_mean']
                wmean    = weighted_results[param]['weighted_mean']
                percent_diff = 100 * abs(boot_mean - wmean) / max(abs(boot_mean), 1e-12)
                agreement = "Excellent" if percent_diff < 1 else "Good" if percent_diff < 5 else "Poor"
                f.write(f"  {param}: {percent_diff:.2f}% difference ({agreement})\n")


            f.write("DEGENERACY SUMMARY\n")
            f.write("-"*20 + "\n")
            f.write(f"Heatmaps:\n")
            for k, v in deg_paths.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"PCA Scree:    {pca_paths['scree']}\n")
            f.write(f"PCA Loadings: {pca_paths['loadings']}\n\n")

            # Top-5 degenerate pairs by each metric
            f.write("Top pairs by |rho_w|:\n")
            tmp = deg_pairs.copy()
            tmp['abs_rho_w'] = tmp['rho_w'].abs()
            for _, r in tmp.sort_values('abs_rho_w', ascending=False).head(5).iterrows():
                f.write(f"  {r.pi} – {r.pj}: rho_w={r.rho_w:.3f}\n")

            f.write("\nTop pairs by MI_w:\n")
            for _, r in deg_pairs.sort_values('MI_w', ascending=False).head(5).iterrows():
                f.write(f"  {r.pi} – {r.pj}: MI_w={r.MI_w:.3f}\n")

            f.write("\nTop pairs by axis_ratio:\n")
            for _, r in deg_pairs.sort_values('axis_ratio', ascending=False).head(5).iterrows():
                f.write(f"  {r.pi} – {r.pj}: AR={r.axis_ratio:.2f}\n")


        print(f"Comprehensive report saved to {report_file}")

        return {
            'bootstrap': bootstrap_results,
            'weighted': weighted_results,
            'kdes_weighted': _kdes
        }



def main(main_dir):
    """
    Batch uncertainty analysis over all simulation_results*.csv files.

    - Discovers all files matching simulation_results*.csv in main_dir
    - "Primary" file = largest numeric suffix if any (e.g., _gen_512), else largest size
    - Analyzes ALL discovered files
    - Outputs for each file go to: {main_dir}/analysis/{stem}/uncertainty/
      e.g., analysis/simulation_results_gen_80/uncertainty/...
    """
    import os, re, glob

    # Base output root
    base_output = os.path.join(main_dir, "analysis")
    os.makedirs(base_output, exist_ok=True)

    # Find all candidates
    pattern = os.path.join(main_dir, "simulation_results*.csv")
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not candidates:
        print(f"Error: no files matching {pattern}")
        return

    # Helper: parse numeric suffix (prefers _gen_<N>, falls back to _<N>)
    def parse_suffix(path):
        name = os.path.basename(path)
        m = re.search(r'_gen_(\d+)\.csv$', name)
        if m:
            return int(m.group(1))
        m = re.search(r'_(\d+)\.csv$', name)
        if m:
            return int(m.group(1))
        return None  # e.g., simulation_results.csv (final iteration)

    # Collect (path, size, suffix)
    recs = []
    for p in candidates:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        recs.append((p, size, parse_suffix(p)))

    # Choose "primary" file:
    #   prefer max suffix (if any suffix exists), else max size
    if any(suf is not None for _, _, suf in recs):
        primary = max(recs, key=lambda r: ((r[2] if r[2] is not None else -1), r[1]))
    else:
        primary = max(recs, key=lambda r: r[1])

    # Order batch for analysis: by suffix ascending (None last), tiebreak by size
    def sort_key(r):
        path, size, suf = r
        return (suf if suf is not None else 10**9, size)

    batch = sorted(recs, key=sort_key)

    print(f"Found {len(batch)} simulation results file(s) in {main_dir}.")
    print(f"Primary (largest suffix/size): {os.path.basename(primary[0])} "
          f"(suffix={primary[2]}, size={primary[1]} bytes)")

    path = primary[0]
    size = primary[1]
    suf = primary[2]
    stem = os.path.splitext(os.path.basename(path))[0]  # e.g., simulation_results_gen_80
    run_output = os.path.join(base_output, stem) + os.sep

    analyzer = UncertaintyAnalysis(results_file=path, output_path=run_output)
    _ = analyzer.generate_comprehensive_uncertainty_report()

    # Report per-file outputs
    print(f"\nAnalysis complete for {os.path.basename(path)}")
    print(f"- {run_output}uncertainty/bootstrap_results.txt")
    print(f"- {run_output}uncertainty/weighted_stats.txt")
    print(f"- {run_output}uncertainty/marginalized_posteriors.png")
    print(f"- {run_output}uncertainty/comprehensive_report.txt")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Look up all folders in the current dir
        # Create a list of all folders
        # Run main in a loop over all folders
        # Try to do this but accept some folders may fail
        current_dir = os.getcwd()
        folders = [f for f in os.listdir(current_dir) 
                  if os.path.isdir(os.path.join(current_dir, f)) 
                  and not f.startswith('.')]
        
        print(f"Found {len(folders)} folders in current directory: {current_dir}")
        successful_runs = 0
        failed_runs = 0
        
        for folder in folders:
            try:
                print(f"\nProcessing folder: {folder}")
                folder_path = os.path.join(current_dir, folder)
                main(folder_path)
                successful_runs += 1
                print(f"✓ Successfully processed {folder}")
            except Exception as e:
                failed_runs += 1
                print(f"✗ Failed to process folder {folder}: {str(e)}")
                continue
        
        print(f"\n=== Summary ===")
        print(f"Total folders found: {len(folders)}")
        print(f"Successfully processed: {successful_runs}")
        print(f"Failed: {failed_runs}")
        plt.close('all')               # (optional) belt-and-suspenders at the end of an iteration