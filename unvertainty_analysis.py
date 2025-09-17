#!/usr/bin/env python3.8
"""
Uncertainty quantification and model framework limitation analysis for GCE GA results.
Addresses all placeholder items in Section 5.2 of the paper.

Authors: N Miller, based on analysis framework
"""



import numpy as np
from scipy.stats import gaussian_kde

import os
import sys
import numpy as np
import re, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
from scipy.stats import gaussian_kde, bootstrap, percentileofscore
from scipy.special import logsumexp
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
            self, bins=50, kde_points=1024, em_max_iter=200, tol=1e-6, force_k2=False
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

        mk_plts = False
        if mk_plts:
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
            #ax.legend(frameon=False, fontsize=10, ncol=2)
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
    analyzer.generate_comprehensive_uncertainty_report()


















def _discover_result_folders(base_dir):
    """Return [(folder_name, folder_path, [csvs...])] for folders that contain simulation_results*.csv"""
    out = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p) or name.startswith('.'):
            continue
        csvs = sorted(glob.glob(os.path.join(p, "simulation_results*.csv")))
        if csvs:
            out.append((name, p, csvs))
    return out

def _parse_suffix_from_name(fname):
    # prefer _gen_<N>.csv, else _<N>.csv, else None
    m = re.search(r'_gen_(\d+)\.csv$', fname)
    if m: return int(m.group(1))
    m = re.search(r'_(\d+)\.csv$', fname)
    if m: return int(m.group(1))
    return None

def _choose_primary_csv(csv_list):
    """Choose 'primary' results CSV from a list, mirroring your existing logic."""
    recs = []
    for p in csv_list:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        recs.append((p, size, _parse_suffix_from_name(os.path.basename(p))))
    if any(suf is not None for _, _, suf in recs):
        return max(recs, key=lambda r: ((r[2] if r[2] is not None else -1), r[1]))[0]
    return max(recs, key=lambda r: r[1])[0]

def _parse_pcard_ranges(pcard_path):
    """Parse numeric [lo, hi] from bulge_pcard.txt keys -> {col: (lo, hi)}."""
    if not os.path.isfile(pcard_path):
        return {}
    with open(pcard_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    key2col = {
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
    ranges = {}
    for key, col in key2col.items():
        m = re.search(rf'^\s*{re.escape(key)}\s*:\s*\[([^\]]+)\]', txt, flags=re.MULTILINE)
        if not m:
            continue
        try:
            nums = [float(x.strip()) for x in m.group(1).split(',')]
            if len(nums) == 2 and nums[1] > nums[0]:
                ranges[col] = (float(nums[0]), float(nums[1]))
        except Exception:
            pass
    return ranges

def _union_param_ranges(analyzers, params):
    """Union [lo,hi] across analyzers using pcard when available; fallback to data min/max."""
    union = {}
    # gather pcard ranges
    pcard_ranges_list = []
    for a in analyzers:
        pcard_path = getattr(a, 'bulge_pcard_path', 'bulge_pcard.txt')
        # if folder-specific pcard exists, prefer it
        if os.path.isfile(pcard_path):
            pcard_ranges_list.append(_parse_pcard_ranges(pcard_path))
        else:
            p = os.path.join(os.path.dirname(a.results_file), 'bulge_pcard.txt')
            pcard_ranges_list.append(_parse_pcard_ranges(p))

    for p in params:
        los, his = [], []
        # try pcard first
        for R in pcard_ranges_list:
            if p in R:
                lo, hi = R[p]
                los.append(lo); his.append(hi)
        # fallback to data if nothing from pcard
        if not los:
            for a in analyzers:
                if p in a.df.columns:
                    v = a.df[p].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    if v.size:
                        los.append(float(np.min(v)))
                        his.append(float(np.max(v)))
        if los:
            union[p] = (min(los), max(his))
    return union

def _choose_common_params(analyzers, preferred=('sigma_2','t_2','infall_2','sfe')):
    """Intersection of continuous params available across all analyzers; prefer a short, readable set."""
    sets = []
    for a in analyzers:
        cols = set(a.continuous_params) & set(a.df.columns)
        sets.append(cols)
    common = set.intersection(*sets) if sets else set()
    # keep order: prefer given list, then add the rest sorted
    out = [p for p in preferred if p in common]
    for p in sorted(common):
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError("No common continuous parameters across selected folders.")
    return out




# -------- colors ----------
def _colors_for(n, ink_colors=None):
    if ink_colors is not None and len(ink_colors) >= n:
        return list(ink_colors)[:n]
    cyc = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or ['C0','C1','C2','C3','C4','C5','C6']
    out = []
    i = 0
    for _ in range(n):
        out.append(cyc[i % len(cyc)])
        i += 1
    return out

# -------- robust fitness column ----------
def _fitness_col_of(a):
    col = getattr(a, 'fitness_col', None)
    if not col or col not in a.df.columns:
        raise RuntimeError("analyzer.fitness_col missing or not in analyzer.df")
    return col

# -------- EM for 1D GMM on log-loss; return cutoff if K=2 wins ----------
def _em_gmm_1d(y, K, iters=200, tol=1e-6):
    qs = np.linspace(0.2, 0.8, K)
    mu = np.quantile(y, qs) if K > 1 else np.array([float(np.mean(y))])
    s0 = float(np.std(y));  s0 = s0 if s0 > 1e-6 else 0.1
    sig = np.full(K, s0, float)
    pi = np.full(K, 1.0/K, float)
    c_norm = -0.5*np.log(2*np.pi)

    def logpdf(y, mu, sig):
        return c_norm - np.log(sig) - 0.5*((y - mu)/sig)**2

    prev_ll = -np.inf
    for _ in range(iters):
        log_comp = np.stack([np.log(pi[k]) + logpdf(y, mu[k], sig[k] + 1e-12) for k in range(K)], axis=1)
        log_den = logsumexp(log_comp, axis=1, keepdims=True)
        R = np.exp(log_comp - log_den)
        Nk = R.sum(axis=0) + 1e-12

        mu_new = (R * y[:, None]).sum(axis=0) / Nk
        sig_new = np.sqrt((R * (y[:, None] - mu_new[None,:])**2).sum(axis=0) / Nk)
        sig_new = np.maximum(sig_new, 1e-6)
        pi_new = Nk / y.size

        ll = float(np.sum(log_den))
        if abs(ll - prev_ll) < tol:
            mu, sig, pi = mu_new, sig_new, pi_new
            prev_ll = ll
            break
        mu, sig, pi, prev_ll = mu_new, sig_new, pi_new, ll

    # BIC: p = (K-1) + K (means) + K (stds) = 2K - 1
    bic = -2.0*prev_ll + (2*K - 1)*np.log(y.size)
    order = np.argsort(mu)
    return pi[order], mu[order], sig[order], prev_ll, bic

def _mixture_cutoff_from_losses(L, em_max_iter=200, tol=1e-6, force_k2=False):
    """Return (cutoff_or_None, keep_frac, chosen_K, (pi,mu,sig) for chosen K)."""
    L = np.asarray(L, float)
    L = L[np.isfinite(L)]
    if L.size == 0:
        raise RuntimeError("No finite losses/fitness values.")
    eps = 1e-12
    y = np.log(L + eps)

    pi1, mu1, sg1, ll1, bic1 = _em_gmm_1d(y, 1, em_max_iter, tol)
    pi2, mu2, sg2, ll2, bic2 = _em_gmm_1d(y, 2, em_max_iter, tol)
    choose_K2 = force_k2 or (bic2 < bic1)

    if not choose_K2:
        return None, 1.0, 1, (pi1, mu1, sg1)

    # solve pi0*N0(y) = pi1*N1(y)
    pi, mu, sig = pi2, mu2, sg2
    A = 0.5*(1.0/sig[1]**2 - 1.0/sig[0]**2)
    B = (mu[0]/sig[0]**2 - mu[1]/sig[1]**2)
    D = 0.5*(mu[1]**2/sig[1]**2 - mu[0]**2/sig[0]**2)
    const = np.log((pi[1]/sig[1])/(pi[0]/sig[0]))
    C = D - const

    if abs(A) < 1e-12:
        y_cut = -C / (B + 1e-12)
    else:
        disc = max(B*B - 4*A*C, 0.0)
        roots = np.sort(( -B + np.array([-1.0, 1.0])*np.sqrt(disc) ) / (2*A))
        mid = 0.5*(mu[0] + mu[1])
        if (mu[0] <= roots[0] <= mu[1]) or (mu[0] <= roots[1] <= mu[1]):
            y_cut = roots[0] if (mu[0] <= roots[0] <= mu[1]) else roots[1]
        else:
            y_cut = roots[np.argmin(np.abs(roots - mid))]

    cutoff = float(np.exp(y_cut))
    keep_frac = float(np.mean(L <= cutoff))
    return cutoff, keep_frac, 2, (pi2, mu2, sg2)

def _cutoff_for_analyzer(a, fallback=None):
    """Prefer analyzer.choose_cutoff_lognorm_mixture(); else EM; return cutoff or fallback."""
    col = _fitness_col_of(a)
    # try user-defined chooser, flexible return types
    if hasattr(a, 'choose_cutoff_lognorm_mixture'):
        try:
            res = a.choose_cutoff_lognorm_mixture()
            if isinstance(res, (int,float)):   return float(res)
            if isinstance(res, (list,tuple)) and len(res)>=1 and isinstance(res[0], (int,float)):
                return float(res[0])
            if isinstance(res, dict):
                for k in ('cutoff','loss_cutoff','threshold','loss_threshold'):
                    if k in res and isinstance(res[k], (int,float)):
                        return float(res[k])
        except Exception:
            pass
    # fallback: EM on this analyzer's losses
    df = getattr(a, 'df_sorted', None)
    df = df if df is not None else a.df.sort_values(by=col, ascending=True)
    cutoff, _, _, _ = _mixture_cutoff_from_losses(df[col].values)
    if cutoff is not None:
        return cutoff
    return fallback

def _select_by_cutoff_or_percentile(a, cutoff, fallback_percentile=10, weight_power=1.0):
    """Return (df_selected, weights) using cutoff if present; else top percentile."""
    col = _fitness_col_of(a)
    df = getattr(a, 'df_sorted', None)
    df = df if df is not None else a.df.sort_values(by=col, ascending=True)
    if isinstance(cutoff, (int,float)):
        sel = df[df[col] <= cutoff]
    else:
        sel = None
    if sel is None or len(sel)==0:
        n_top = max(1, int(len(df)*fallback_percentile/100))
        sel = df.head(n_top)
    fit = np.asarray(sel[col].values, float)
    eps = np.nanmin(fit) * 1e-3 if np.isfinite(np.nanmin(fit)) else 1e-12
    w = 1.0 / np.power(fit + eps, weight_power)
    w = w / np.sum(w)
    return sel, w



# --- add these helpers inside plot_corner_with_marginals_multi, above the draw-loop ---
def _hpd_threshold(Z, p, dx, dy):
    z = Z.ravel()
    if not np.isfinite(z).any() or z.max() <= 0:
        return None
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    csum = np.cumsum(z_sorted) * dx * dy
    total = z.sum() * dx * dy
    if total <= 0: return None
    idx = np.searchsorted(csum, p * total)
    idx = np.clip(idx, 0, z_sorted.size-1)
    return float(z_sorted[idx])

def _ellipse_from_hpd(Xg, Yg, Z, p=0.68):
    # grid spacing
    dx = float(Xg[0,1] - Xg[0,0]); dy = float(Yg[1,0] - Yg[0,0])
    thr = _hpd_threshold(Z, p, dx, dy)
    if thr is None: return None
    mask = Z >= thr
    xs = Xg[mask]; ys = Yg[mask]
    if xs.size < 8:  # too few points
        return None

    # center & covariance of the filled HPD region (uniform weight)
    mx, my = float(xs.mean()), float(ys.mean())
    Xc = np.vstack([xs - mx, ys - my])
    C = (Xc @ Xc.T) / xs.size  # 2x2

    # principal axes
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    # for a filled ellipse, Var_x = a^2/4, Var_y = b^2/4  →  a=2√λ1, b=2√λ2
    a = 2.0 * np.sqrt(max(evals[0], 0.0))  # semimajor
    b = 2.0 * np.sqrt(max(evals[1], 0.0))  # semiminor
    theta = float(np.arctan2(evecs[1,0], evecs[0,0]))  # orientation of major axis

    return {
        "thr": thr,
        "center": (mx, my),
        "a": float(a),
        "b": float(b),
        "theta": theta,
        "evecs": evecs,  # columns: major, minor
    }

def _ellipse_proj_to_axes(a, b, theta):
    # projection half-widths of rotated ellipse onto x/y axes
    px = np.sqrt((a*np.cos(theta))**2 + (b*np.sin(theta))**2)
    py = np.sqrt((a*np.sin(theta))**2 + (b*np.cos(theta))**2)
    return float(px), float(py)

def _draw_ellipse(ax, center, a, b, theta, color, lw=1.4):
    cx, cy = center
    # contour outline (parametric)
    tt = np.linspace(0, 2*np.pi, 256)
    ct, st = np.cos(theta), np.sin(theta)
    xs = cx + a*np.cos(tt)*ct - b*np.sin(tt)*st
    ys = cy + a*np.cos(tt)*st + b*np.sin(tt)*ct
    ax.plot(xs, ys, color=color, lw=lw, alpha=0.95)

    # major/minor axis sticks
    vmaj = np.array([np.cos(theta), np.sin(theta)])
    vmin = np.array([-np.sin(theta), np.cos(theta)])
    ax.plot([cx - a*vmaj[0], cx + a*vmaj[0]],
            [cy - a*vmaj[1], cy + a*vmaj[1]], color=color, lw=lw)
    ax.plot([cx - b*vmin[0], cx + b*vmin[0]],
            [cy - b*vmin[1], cy + b*vmin[1]], color=color, lw=lw, ls='--')


def _weighted_hpd_1d(x, w, mass=0.68):
    """
    Smallest-width weighted HPD interval on 1D samples.
    Works directly on samples (avoids KDE pathologies).
    Returns (lo, hi). Falls back to weighted central interval if needed.
    """
    import numpy as np
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[good]; w = w[good]
    if x.size == 0:
        return (np.nan, np.nan)
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    ws = ws / np.sum(ws)
    c = np.cumsum(ws)
    # for each start i, find smallest j s.t. c[j]-c[i] >= mass
    best = (np.inf, 0, 0)
    for i in range(xs.size):
        target = c[i] + mass
        j = np.searchsorted(c, target, side='left')
        if j >= xs.size:
            break
        width = xs[j] - xs[i]
        if width < best[0]:
            best = (width, i, j)
    if not np.isfinite(best[0]):
        # fallback to central interval
        lo = np.interp((1-mass)/2, c, xs)
        hi = np.interp(1-(1-mass)/2, c, xs)
        return (float(lo), float(hi))
    _, i, j = best
    return (float(xs[i]), float(xs[j]))


# --- in unvertainty_analysis.py ---

def _top_percentile_only(a, params, percentile=10, weight_power=1.0):
    """Deterministic: top X% by fitness with weights 1/(loss^p)."""
    col = _fitness_col_of(a)
    df = a.df.sort_values(by=col, ascending=True)
    n_top = max(1, int(len(df)*percentile/100.0))
    sel = df.head(n_top)
    w = 1.0 / np.power(sel[col].values + 1e-12, weight_power)
    w = w / np.sum(w)
    keep_cols = [p for p in params if p in sel.columns]
    return sel[keep_cols].copy(), w

# replace use inside _combined_top_selection(...)
def _combined_top_selection(analyzers, params, percentile=10, weight_power=1.0):
    frames, w_all = [], []
    for a in analyzers:
        t, w = _top_percentile_only(a, params, percentile=percentile, weight_power=weight_power)
        if not t.empty:
            frames.append(t); w_all.append(w)
    import numpy as np, pandas as pd
    if not frames: 
        return pd.DataFrame(columns=params), np.array([])
    W = np.concatenate(w_all)
    return pd.concat(frames, axis=0, ignore_index=True), W














# -------- loss overlays (KDE + CDF), color-matched ----------
def plot_loss_overlays_simple(analyzers, ink_colors, legend_labels, cutoffs=None, save_prefix=None, bins=60):
    if cutoffs is None: cutoffs = [None]*len(analyzers)
    # KDE overlay
    fig1, ax1 = plt.subplots(figsize=(8,5))
    xs_lo, xs_hi = np.inf, -np.inf
    all_losses = []
    for a in analyzers:
        col = _fitness_col_of(a)
        L = np.asarray(a.df[col].values, float)
        L = L[np.isfinite(L)]
        all_losses.append(L)
        if L.size:
            xs_lo = min(xs_lo, np.min(L))
            xs_hi = max(xs_hi, np.max(L))
    xs = np.linspace(xs_lo, xs_hi, 1024)
    for L, colr, lab, co in zip(all_losses, ink_colors, legend_labels, cutoffs):
        dens = gaussian_kde(L)(xs)
        ax1.plot(xs, dens, lw=2, color=colr, label=lab)

    ax1.set_xlabel("Loss"); ax1.set_ylabel("Density")
    #ax1.legend(frameon=False, ncol=min(len(legend_labels),4))
    out1 = (save_prefix or os.path.join(os.getcwd(),"analysis","overlay","loss_overlay")) + "_density.png"
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig1.savefig(out1, dpi=300, bbox_inches='tight'); plt.close(fig1)

    # CDF overlay
    fig2, ax2 = plt.subplots(figsize=(8,5))
    for L, colr, lab, co in zip(all_losses, ink_colors, legend_labels, cutoffs):
        Ls = np.sort(L); y = (np.arange(Ls.size)+1)/Ls.size
        ax2.plot(Ls, y, lw=2, color=colr, label=lab)

    ax2.set_xlabel("Loss"); ax2.set_ylabel("Cumulative fraction"); ax2.set_ylim(0,1)
    #ax2.legend(frameon=False, ncol=min(len(legend_labels),4))
    out2 = (save_prefix or os.path.join(os.getcwd(),"analysis","overlay","loss_overlay")) + "_cdf.png"
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    fig2.savefig(out2, dpi=300, bbox_inches='tight'); plt.close(fig2)
    return out1, out2





def cutoff_at_peak(losses, kde_points=2048):
    L = np.asarray(losses, float)
    xs = np.linspace(L.min(), L.max(), int(kde_points))
    dens = gaussian_kde(L)(xs)
    return 1.0#float(xs[np.argmax(dens)])   # cutoff = mode of loss KDE





# --- NEW: multi-overlay corner plot -----------------------------------------
def plot_corner_with_marginals_multi(
        analyzers,
        params=None,
        percentile=10,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        ink_colors=None,
        alpha_gamma=0.8,
        save_path=None,
        legend_labels=None
    ):
    """
    Overplot multiple result folders on a single corner plot.
    Each analyzer contributes its top-`percentile` population with its own color.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from scipy.stats import gaussian_kde, spearmanr, pearsonr

    if not analyzers:
        raise ValueError("No analyzers provided.")

    # choose params
    if params is None:
        params = _choose_common_params(analyzers)

    k = len(params)
    if k == 0:
        raise ValueError("Empty parameter list.")

    # default colors & labels
    if ink_colors is None:
        cyc = cycle(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6']))
        ink_colors = [next(cyc) for _ in analyzers]


    # union of axis limits across folders
    param_ranges = _union_param_ranges(analyzers, params)

    # helpers
    def _top_and_w(a):
        df = a.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)
        fit = np.asarray(top[a.fitness_col].values, dtype=float)
        eps = np.min(fit) * 1e-3
        w = 1.0 / np.power(fit + eps, weight_power)
        w = w / np.sum(w)
        return top, w

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
            rng = np.random.default_rng(1337)
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
            rng = np.random.default_rng(1337)
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

    # figure
    fig, axes = plt.subplots(k, k, figsize=(3.0*k, 3.0*k), constrained_layout=False)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06, wspace=0.0, hspace=0.0)

    # colors (once; reuse for loss overlays too)
    ink_colors = _colors_for(len(analyzers), ink_colors)

    # per-folder cutoffs (prefer user method; else EM)
    cutoffs = [cutoff_at_peak(a.df_sorted[a.fitness_col].to_numpy(float)) for a in analyzers]

    # precompute selections using cutoff or fallback percentile
    tops, weights = [], []
    for a, co in zip(analyzers, cutoffs):
        t, w = _select_by_cutoff_or_percentile(a, cutoff=co, fallback_percentile=percentile, weight_power=weight_power)
        tops.append(t); weights.append(w)


    # draw
    for i, pi in enumerate(params):
        # x-range for diagonal
        lo_i, hi_i = param_ranges.get(pi, (np.nan, np.nan))
        for j, pj in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis('off'); continue

            if i == j:
                # diagonal: KDE lines only (cleaner for overlays)
                if not np.isfinite(lo_i) or not np.isfinite(hi_i) or hi_i <= lo_i:
                    # fallback to union of mins/maxs from data
                    vals_all = []
                    for t in tops:
                        if pi in t.columns:
                            vals_all.append(t[pi].to_numpy(dtype=float))
                    if vals_all:
                        vcat = np.concatenate(vals_all)
                        vcat = vcat[np.isfinite(vcat)]
                        if vcat.size:
                            lo_i, hi_i = float(np.min(vcat)), float(np.max(vcat))
                xs = np.linspace(lo_i, hi_i, 512)
                for t, w, col in zip(tops, weights, ink_colors):
                    if pi not in t.columns:
                        continue
                    v = t[pi].to_numpy(dtype=float)
                    dens = _kde_1d(v, w, xs)
                    ax.plot(xs, dens, lw=2, color=col, alpha=0.95)
                ax.set_xlim(lo_i, hi_i)
                ax.set_yticks([])
                if i < k-1: ax.set_xticklabels([])
                ax.set_xlabel(label_map.get(pi, pi))

            else:
                # lower triangle: per-dataset 2D alpha-KDE overlays
                lo_x, hi_x = param_ranges.get(pj, (np.nan, np.nan))
                lo_y, hi_y = param_ranges.get(pi, (np.nan, np.nan))
                nx = ny = 200
                xg = np.linspace(lo_x, hi_x, nx)
                yg = np.linspace(lo_y, hi_y, ny)
                Xg, Yg = np.meshgrid(xg, yg)

                # draw each dataset
                for t, w, col in zip(tops, weights, ink_colors):
                    if pj not in t.columns or pi not in t.columns:
                        continue
                    x = t[pj].to_numpy(dtype=float)
                    y = t[pi].to_numpy(dtype=float)
                    Z = _kde_2d(x, y, w, Xg, Yg)
                    if not np.isfinite(Z).any() or np.nanmax(Z) <= 0:
                        continue
                    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                    Zn = Z / (Z.max() + 1e-30)
                    alpha = np.power(Zn, alpha_gamma) * 0.9
                    rgb = np.array(mcolors.to_rgb(col))
                    rgba = np.empty((ny, nx, 4), float)
                    rgba[..., 0] = rgb[0]; rgba[..., 1] = rgb[1]; rgba[..., 2] = rgb[2]; rgba[..., 3] = alpha
                    ax.imshow(rgba, extent=(lo_x, hi_x, lo_y, hi_y),
                              origin='lower', interpolation='bilinear', aspect='auto')

                ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
                if j == 0: ax.set_ylabel(label_map.get(pi, pi))
                else: ax.set_yticks([])
                if i == k-1: ax.set_xlabel(label_map.get(pj, pj))
                else: ax.set_xticks([])

    # legend
    if legend_labels is not None:
        handles = [plt.Line2D([0],[0], color=c, lw=1) for c in ink_colors]
        fig.legend(handles, legend_labels, ncol=min(len(legend_labels), 2), frameon=False, loc='center right')

    # save
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "analysis", "overlay", "corner_overlay.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Overlay corner saved to: {save_path}")


    save_prefix = os.path.join(os.getcwd(), "analysis", "overlay", "loss_overlay_selection")
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    plot_loss_overlays_simple(
        analyzers,
        ink_colors=ink_colors,
        legend_labels=labels,      # whatever you already use for the legend
        cutoffs=cutoffs,
        save_prefix=save_prefix
    )

    print(f"Loss plot saved to: {save_path}")

    return save_path




def compute_and_plot_combined_covariant_uncertainties(
        analyzers,
        params=None,
        percentile=10,
        weight_power=1.0,
        p_hpd=0.68,
        grid_n=240,
        alpha_gamma=0.7,
        ink_color='#111111',     # color for combined HPD ellipse/markers
        save_dir=None,
        ink_colors=None,         # OPTIONAL: if you pass the same list used in plot_corner_with_marginals_multi, colors will match 1:1
        legend_labels=None       # OPTIONAL: legend labels (one per analyzer)
    ):
    """
    Combined-only covariant errors (HPD68 ellipses + axis projections),
    while the plot itself still shows each dataset in its own color
    (same palette/order as plot_corner_with_marginals_multi).
    - Uncertainties are computed from the COMBINED cohort (not per-folder).
    - Visuals overlay per-folder KDEs/points in their own colors.
    - Diagonals: per-folder 1D KDE overlays (+ combined 1D HPD marks).
    - Off-diagonals: per-folder 2D alpha-KDE overlays, plus one combined HPD68 ellipse.
    - CSVs remain combined-only: per_param_hpd68.csv, pairwise_hpd68_ellipses.csv, combined_corr.csv
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from scipy.stats import gaussian_kde

    if not analyzers:
        raise ValueError("No analyzers provided.")

    # choose a compact common parameter set if not given
    if params is None:
        params = _choose_common_params(analyzers)
    k = len(params)
    if k == 0:
        raise ValueError("Empty parameter list for combined uncertainties.")

    # ---------------- colors & selections (MATCH multi) ----------------
    # palette identical to plot_corner_with_marginals_multi via _colors_for
    ink_colors = _colors_for(len(analyzers), ink_colors)

    # per-folder cutoffs (prefer analyzer method; else EM), then select
    cutoffs = [cutoff_at_peak(a.df_sorted[a.fitness_col].to_numpy(float)) for a in analyzers]
    tops, weights = [], []
    for a, co in zip(analyzers, cutoffs):
        t, w = _select_by_cutoff_or_percentile(a, cutoff=co, fallback_percentile=percentile, weight_power=weight_power)
        tops.append(t); weights.append(w)

    # ---------------- COMBINED data for uncertainties ----------------
    # concatenate selected cohorts across folders (combined-only truth)
    T, w_comb = _combined_top_selection(analyzers, params, percentile=percentile, weight_power=weight_power)

    # union axis limits across all folders (pcard+data)
    param_ranges = _union_param_ranges(analyzers, params)

    # outputs
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "analysis", "overlay", "combined_uncertainty")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- helpers ----------------
    def _grid_for(pname, N):
        lo, hi = param_ranges.get(pname, (np.nan, np.nan))
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            vals = T[pname].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = float(np.min(vals)), float(np.max(vals))
            else:
                lo, hi = -1.0, 1.0
        return np.linspace(lo, hi, N)

    def _kde_1d(vals, weights, xs):
        vals = np.asarray(vals, float); weights = np.asarray(weights, float)
        good = np.isfinite(vals) & np.isfinite(weights)
        vals = vals[good]; weights = weights[good]
        if len(vals) == 0: return np.zeros_like(xs)
        try:
            return gaussian_kde(vals, weights=weights)(xs)
        except Exception:
            # bootstrap fallback for singular cases
            rng = np.random.default_rng(1337)
            N = min(max(1000, 5*len(vals)), 5000)
            p = weights/np.sum(weights) if np.sum(weights) > 0 else None
            idx = rng.choice(np.arange(len(vals)), size=N, replace=True, p=p)
            return gaussian_kde(vals[idx])(xs)

    def _kde_2d(x, y, weights, Xg, Yg):
        x = np.asarray(x, float); y = np.asarray(y, float)
        weights = np.asarray(weights, float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
        x = x[good]; y = y[good]; weights = weights[good]
        if len(x) == 0: return np.zeros_like(Xg)
        xy = np.vstack([x, y])
        try:
            kde = gaussian_kde(xy, weights=weights)
        except Exception:
            rng = np.random.default_rng(1337)
            N = min(max(1000, 5*len(x)), 5000)
            p = weights/np.sum(weights) if np.sum(weights) > 0 else None
            idx = rng.choice(np.arange(len(x)), size=N, replace=True, p=p)
            kde = gaussian_kde(np.vstack([x[idx], y[idx]]))
        return kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # ---------------- 1D HPD per parameter (combined) ----------------
    per_param = []
    for p in params:
        v = T[p].to_numpy(dtype=float)
        good = np.isfinite(v) & np.isfinite(w_comb)
        v = v[good]; wv = w_comb[good]
        if v.size == 0:
            per_param.append(dict(param=p, mean=np.nan, median=np.nan,
                                  hpd_lo=np.nan, hpd_hi=np.nan, hpd_width=np.nan,
                                  gauss_sigma=np.nan))
            continue
        mean = float(np.average(v, weights=wv))
        if v.size > 1:
            # weighted-median approximation via replication (kept consistent with your code)
            mult = np.maximum((wv / np.max(wv) * 1000).astype(int), 1)
            median = float(np.median(np.repeat(v, mult)))
        else:
            median = float(v[0])
        lo, hi = _weighted_hpd_1d(v, wv, mass=p_hpd)
        hpd_width = float(hi - lo) if (np.isfinite(hi) and np.isfinite(lo)) else np.nan
        mu = mean
        gauss_sigma = float(np.sqrt(np.average((v - mu)**2, weights=wv)))
        per_param.append(dict(param=p, mean=mean, median=median,
                              hpd_lo=lo, hpd_hi=hi, hpd_width=hpd_width,
                              gauss_sigma=gauss_sigma))
    pd.DataFrame(per_param).to_csv(os.path.join(save_dir, "per_param_hpd68.csv"), index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(k, k, figsize=(3.2*k, 3.2*k), constrained_layout=False)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.0)

    label_map = {
        'sigma_2': r'$\sigma_2$',
        't_1': r't$_1$ (Gyr)', 't_2': r't$_2$ (Gyr)',
        'infall_1': r'$\tau_1$ (Gyr)', 'infall_2': r'$\tau_2$ (Gyr)',
        'sfe': 'SFE', 'delta_sfe': r'$\Delta$SFE',
        'imf_upper': r'M$_{up}$ (M$_\odot$)',
        'mgal': r'M$_{gal}$ (M$_\odot$)', 'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
    }

    # for combined HPD ellipse strokes
    rgb_comb = np.array(mcolors.to_rgb(ink_color))

    pair_rows = []
    for i, pi in enumerate(params):
        lo_i, hi_i = param_ranges.get(pi, (np.nan, np.nan))
        if (not np.isfinite(lo_i)) or (not np.isfinite(hi_i)) or (hi_i <= lo_i):
            vals = T[pi].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo_i, hi_i = float(np.min(vals)), float(np.max(vals))

        for j, pj in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis('off'); continue

            if i == j:
                # DIAGONAL: per-folder 1D KDE overlays (colored), plus combined HPD verticals
                xs = np.linspace(lo_i, hi_i, 512)
                for t, w, col in zip(tops, weights, ink_colors):
                    if pi not in t.columns: continue
                    v = t[pi].to_numpy(dtype=float)
                    dens = _kde_1d(v, w, xs)
                    ax.plot(xs, dens, lw=2, color=col, alpha=0.95)
                # combined HPD marks
                v_all = T[pi].to_numpy(dtype=float)
                good = np.isfinite(v_all) & np.isfinite(w_comb)
                v_all = v_all[good]; wv = w_comb[good]
                if v_all.size:
                    lo_h, hi_h = _weighted_hpd_1d(v_all, wv, mass=p_hpd)
                    ax.axvline(lo_h, color='k', lw=1.4, ls='--', alpha=0.9)
                    ax.axvline(hi_h, color='k', lw=1.4, ls='--', alpha=0.9)
                ax.set_xlim(lo_i, hi_i); ax.set_yticks([])
                if i < k-1: ax.set_xticklabels([])
                ax.set_xlabel(label_map.get(pi, pi))

            else:
                # OFF-DIAGONAL: per-folder 2D alpha-KDE overlays in color, plus COMBINED HPD68 ellipse
                lo_x, hi_x = param_ranges.get(pj, (np.nan, np.nan))
                lo_y, hi_y = param_ranges.get(pi, (np.nan, np.nan))
                nx = ny = grid_n
                xg = np.linspace(lo_x, hi_x, nx)
                yg = np.linspace(lo_y, hi_y, ny)
                Xg, Yg = np.meshgrid(xg, yg)

                # per-folder overlays
                for t, w, col in zip(tops, weights, ink_colors):
                    if pj not in t.columns or pi not in t.columns: continue
                    x = t[pj].to_numpy(float)
                    y = t[pi].to_numpy(float)
                    Z = _kde_2d(x, y, w, Xg, Yg)
                    if not np.isfinite(Z).any() or np.nanmax(Z) <= 0: continue
                    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                    Zn = Z / (Z.max() + 1e-30)
                    alpha = np.power(Zn, alpha_gamma) * 0.90
                    rgb = np.array(mcolors.to_rgb(col))
                    rgba = np.empty((ny, nx, 4), float)
                    rgba[..., 0] = rgb[0]; rgba[..., 1] = rgb[1]; rgba[..., 2] = rgb[2]; rgba[..., 3] = alpha
                    ax.imshow(rgba, extent=(lo_x, hi_x, lo_y, hi_y),
                              origin='lower', interpolation='bilinear', aspect='auto')

                # combined KDE (for the HPD ellipse only)
                xC = T[pj].to_numpy(float)
                yC = T[pi].to_numpy(float)
                goodC = np.isfinite(xC) & np.isfinite(yC) & np.isfinite(w_comb)
                xC = xC[goodC]; yC = yC[goodC]; wC = w_comb[goodC]
                if xC.size:
                    try:
                        Zc = gaussian_kde(np.vstack([xC, yC]), weights=wC)(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    except Exception:
                        rng = np.random.default_rng(1337)
                        N = min(max(1000, 5*len(xC)), 5000)
                        p = wC/np.sum(wC)
                        idx = rng.choice(np.arange(len(xC)), size=N, replace=True, p=p)
                        Zc = gaussian_kde(np.vstack([xC[idx], yC[idx]]))(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    Zc = np.nan_to_num(Zc, nan=0.0, posinf=0.0, neginf=0.0)
                    el = _ellipse_from_hpd(Xg, Yg, Zc, p=p_hpd)
                    if el is None:
                        # fallback: Gaussian approx from weighted covariance
                        mx = float(np.average(xC, weights=wC))
                        my = float(np.average(yC, weights=wC))
                        C = np.cov(np.vstack([xC, yC]), aweights=wC, bias=True)
                        evals, evecs = np.linalg.eigh(C)
                        order = np.argsort(evals)[::-1]
                        evals = evals[order]; evecs = evecs[:, order]
                        a = 2.0*np.sqrt(max(evals[0],0.0))
                        b = 2.0*np.sqrt(max(evals[1],0.0))
                        theta = float(np.arctan2(evecs[1,0], evecs[0,0]))
                        cx, cy = mx, my
                    else:
                        a = el["a"]; b = el["b"]; theta = el["theta"]
                        cx, cy = el["center"]

                    # draw combined ellipse + axis sticks in black (or ink_color)
                    _draw_ellipse(ax, (cx, cy), a, b, theta, color=ink_color, lw=1.6)
                    px, py = _ellipse_proj_to_axes(a, b, theta)

                    pair_rows.append(dict(
                        p_row=pi, p_col=pj,
                        center_x=cx, center_y=cy,
                        a=a, b=b, theta_rad=theta, theta_deg=np.degrees(theta),
                        proj_on_col=px,   # uncertainty along pj from combined HPD68
                        proj_on_row=py    # uncertainty along pi from combined HPD68
                    ))

                # cosmetics
                ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
                if j == 0: ax.set_ylabel(label_map.get(pi, pi))
                else: ax.set_yticks([])
                if i == k-1: ax.set_xlabel(label_map.get(pj, pj))
                else: ax.set_xticks([])

    # legend (optional, no syntax change required)
    if legend_labels is not None:
        handles = [plt.Line2D([0],[0], color=c, lw=2) for c in ink_colors]
        fig.legend(handles, legend_labels, ncol=min(len(legend_labels), 2), frameon=False, loc='center right')

    corner_png = os.path.join(save_dir, "combined_corner_ellipses.png")
    fig.savefig(corner_png, dpi=300, bbox_inches='tight'); plt.close(fig)

    # ---------------- pairwise CSV (combined-only) ----------------
    pd.DataFrame(pair_rows).to_csv(os.path.join(save_dir, "pairwise_hpd68_ellipses.csv"), index=False)

    # ---------------- optional Gaussian-approx correlation (combined-only) ----------------
    Td = T[params].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    if Td.shape[0] >= 3:
        ww = w_comb[:len(Td)].copy()
        ww = ww / ww.max()
        mult = np.maximum((ww * 100).astype(int), 1)
        rep = np.repeat(np.arange(Td.shape[0]), mult[:Td.shape[0]])
        C = np.corrcoef(Td.iloc[rep].T)
        pd.DataFrame(C, index=params, columns=params).to_csv(os.path.join(save_dir, "combined_corr.csv"))

    return dict(
        corner_png=corner_png,
        per_param_csv=os.path.join(save_dir, "per_param_hpd68.csv"),
        pairwise_csv=os.path.join(save_dir, "pairwise_hpd68_ellipses.csv")
    )


def export_best_per_folder_csv(chosen, out_csv):
    """
    Write one-row-per-folder CSV with the best (min-loss) model's parameter values.

    Parameters
    ----------
    chosen : list of (folder_name, folder_path, csv_list)
        Exactly what your selection block already builds.
    out_csv : str
        Output CSV path.
    """
    import os
    import pandas as pd

    rows = []
    all_cols = set()

    for (name, path, csvs) in chosen:
        primary_csv = _choose_primary_csv(csvs)  # your helper
        a = UncertaintyAnalysis(               # uses 'fitness' or 'wrmse' automatically
            results_file=primary_csv,
            output_path=os.path.join(path, "analysis", os.path.splitext(os.path.basename(primary_csv))[0]) + os.sep
        )
        best = a.df_sorted.iloc[0]

        # keep simple: union of declared params + fitness column
        keep_cols = list(dict.fromkeys(a.continuous_params + a.categorical_params + [a.fitness_col]))
        row = {
            "folder": name,
            "primary_csv": os.path.basename(primary_csv),
        }
        for c in keep_cols:
            if c in best.index:
                row[c] = best[c]
                all_cols.add(c)
        rows.append(row)

    # stable column order
    cols = ["folder", "primary_csv"]
    # put fitness columns early if they exist, then everything else sorted
    fit_cols = [c for c in ("fitness", "wrmse") if c in all_cols]
    other = sorted([c for c in all_cols if c not in fit_cols])
    cols.extend(fit_cols + other)

    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[best-per-folder] wrote: {out_csv}")






def plot_corner_points_contours(
    runs,
    params=None,                    # if None: common numeric cols across runs
    run_names=None,                 # legend labels
    include_loss_axis=True,         # add fitness/wrmse as a final axis
    alpha_range=(0.05, 0.9),        # min,max point alpha
    alpha_gamma=0.6,                # emphasis on low loss (higher alpha)
    point_size=4,
    bins=40,
    levels=(0.68, 0.95),
    smooth=0.9,                     # KDE smoothing for contours
    colors=None,                    # pass your inkcolrs here
    save_path=None,
    show=False
):
    """
    Overlay corner with ONLY:
      - contours (KDE) per run, in color
      - raw scatter (alpha ∝ 1/loss) per run
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    import corner
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D

    # normalize inputs -> (df, loss_col)
    dfs, loss_cols = [], []
    for r in runs:
        if hasattr(r, "df") and hasattr(r, "fitness_col"):
            df = r.df.copy(); loss_col = r.fitness_col
        elif isinstance(r, str):
            df = pd.read_csv(r)
            loss_col = 'fitness' if 'fitness' in df.columns else ('wrmse' if 'wrmse' in df.columns else None)
        elif hasattr(r, "columns"):
            df = r.copy()
            loss_col = 'fitness' if 'fitness' in df.columns else ('wrmse' if 'wrmse' in df.columns else None)
        else:
            raise TypeError("runs must be UncertaintyAnalysis, DataFrame, or CSV paths")
        if loss_col is None:
            raise ValueError("Need a 'fitness' or 'wrmse' column.")
        dfs.append(df); loss_cols.append(loss_col)

    # choose params (common numeric across runs, excluding loss); append loss if requested
    if params is None:
        common = None
        for df, lc in zip(dfs, loss_cols):
            cols = set(df.select_dtypes(include=[np.number]).columns)
            for drop in ("generation","seed","id","index"): cols.discard(drop)
            cols.discard(lc)
            common = cols if common is None else (common & cols)
        params = sorted(list(common or []))
    if include_loss_axis:
        params = list(params) + [loss_cols[0]]

    k = len(params)
    if k == 0: raise ValueError("No numeric parameters to plot.")

    # global ranges (small pad)
    ranges = []
    for p in params:
        vmin, vmax = np.inf, -np.inf
        for df in dfs:
            if p in df.columns and df[p].notna().any():
                v = df[p].values
                vmin = min(vmin, np.nanmin(v)); vmax = max(vmax, np.nanmax(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        span = (vmax - vmin) or 1.0
        pad = 0.02 * span
        ranges.append((vmin - pad, vmax + pad))

    # colors & labels
    n = len(dfs)
    if colors is None:
        cyc = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6'])
        colors = [cyc[i % len(cyc)] for i in range(n)]
    run_names = run_names or [f'run {i+1}' for i in range(n)]

    # helper: alpha from loss (lower loss -> higher alpha)
    a_min, a_max = alpha_range
    def loss_to_alpha(loss_array):
        L = np.asarray(loss_array, float)
        Lmin, Lmax = np.nanmin(L), np.nanmax(L)
        if not np.isfinite(Lmin) or not np.isfinite(Lmax) or Lmin == Lmax:
            q = np.zeros_like(L)
        else:
            q = (L - Lmin) / (Lmax - Lmin)
        s = (1.0 - q)**float(alpha_gamma)
        return a_min + s * (a_max - a_min)

    # draw: first run initializes grid; others overlay contours; then scatter for each
    fig, axes = None, None
    for idx, (df, lcol, col) in enumerate(zip(dfs, loss_cols, colors)):
        # enforce finite rows for plotted params
        keep = np.ones(len(df), dtype=bool)
        for p in params:
            keep &= (p in df.columns) & np.isfinite(df[p].values)
        D = df.loc[keep, :]
        if D.empty: continue

        X = np.column_stack([D[p].values for p in params])

        if fig is None:
            fig = corner.corner(
                X,
                labels=params,            # swap for TeX if you want
                bins=bins,
                range=ranges,
                color=col,
                smooth=smooth,
                levels=levels,
                plot_datapoints=False,    # we’ll scatter ourselves
                plot_density=True,        # KDE for contours
                plot_contours=True,
                fill_contours=False,
                hist_kwargs=dict(histtype="step", linewidth=1.2),
                contour_kwargs=dict(linewidths=1.5),
            )
            axes = np.array(fig.axes).reshape((k, k))
        else:
            fig = corner.corner(
                X,
                fig=fig,
                bins=bins,
                range=ranges,
                color=col,
                smooth=smooth,
                levels=levels,
                plot_datapoints=False,
                plot_density=True,
                plot_contours=True,
                fill_contours=False,
                hist_kwargs=dict(histtype="step", linewidth=1.2),
                contour_kwargs=dict(linewidths=1.5),
            )

        # scatter (raw data) with per-point alpha on off-diagonals
        alphas = loss_to_alpha(D[lcol].values)
        base_rgba = np.array(mcolors.to_rgba(col))
        rgba = np.repeat(base_rgba[None, :], len(D), axis=0)
        rgba[:, 3] = np.clip(alphas, 0.0, 1.0)

        for i in range(1, k):
            for j in range(i):
                ax = axes[i, j]
                ax.scatter(
                    D[params[j]].values, D[params[i]].values,
                    s=point_size,
                    c=rgba,
                    marker='.',
                    linewidths=0,
                    rasterized=True
                )

    # legend
    handles = [Line2D([0],[0], color=c, lw=2) for c in colors]
    fig.legend(handles, run_names, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig






if __name__ == "__main__":

    # --- NEW: interactive overlay across many folders
    current_dir = os.getcwd()
    found = _discover_result_folders(current_dir)
    if not found:
        print(f"No result folders found under {current_dir}")
        sys.exit(0)

    print(f"\nFound {len(found)} candidate folders:")
    for idx, (name, path, csvs) in enumerate(found, start=1):
        primary = _choose_primary_csv(csvs)
        size = os.path.getsize(primary) if os.path.exists(primary) else -1
        suf = _parse_suffix_from_name(os.path.basename(primary))
        suf_txt = f"_gen_{suf}" if suf is not None else "(no gen suffix)"
        print(f"  [{idx:>2}] {name:30s}  primary={os.path.basename(primary)} {suf_txt}  size={size}")

    sel = input("\nEnter indices to overlay (e.g., 1 3 5 or 1,3,5). Press Enter for ALL: ").strip()

    # default-to-all semantics
    if not sel or sel.lower() in {"all", "a", "*"}:
        idxs = list(range(1, len(found) + 1))
        print(f"Selected ALL {len(found)} folders.")
    else:
        # parse indices
        tokens = re.split(r'[,\s]+', sel)
        try:
            idxs = sorted(set(int(t) for t in tokens if t))
        except ValueError:
            print("Invalid input. Use numbers separated by space/comma or 'all'.")
            sys.exit(1)

    # validate & build selection
    chosen = []
    invalid = []
    for i in idxs:
        if 1 <= i <= len(found):
            chosen.append(found[i-1])
        else:
            invalid.append(i)

    if invalid:
        print(f"Ignoring out-of-range indices: {invalid}")

    if not chosen:
        print("No valid indices chosen. Exiting.")
        sys.exit(0)

    # output dir reused later too
    overlay_dir = os.path.join(current_dir, "analysis", "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    # ---- FIRST: export best row per selected folder
    export_best_per_folder_csv(chosen, os.path.join(overlay_dir, "best_per_folder.csv"))


    # build analyzers for selected
    analyzers = []
    labels = []
    for (name, path, csvs) in chosen:
        primary_csv = _choose_primary_csv(csvs)
        out_root = os.path.join(path, "analysis", os.path.splitext(os.path.basename(primary_csv))[0]) + os.sep
        a = UncertaintyAnalysis(results_file=primary_csv, output_path=out_root)
        if a.df_sorted[a.fitness_col].iloc[0] < 0.03:
            # prefer folder-specific bulge_pcard.txt if present
            pcard_here = os.path.join(path, "bulge_pcard.txt")
            if os.path.isfile(pcard_here):
                a.bulge_pcard_path = pcard_here
            analyzers.append(a)
            labels.append(name + str(a.df_sorted[a.fitness_col].iloc[0]))
        else:
            print("this one shit")

    # choose a compact common parameter set automatically
    params = _choose_common_params(analyzers)

    # output path (overlay)
    overlay_dir = os.path.join(current_dir, "analysis", "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    tag = "_".join([labels[i][:12] for i in range(len(labels))])

    inkcolrs =  ['#F0B800',
                '#004C40',
                '#0099A1',
                '#C20016',
                '#E8DCD8',
                '#97BAAB',
                '#1E6E6C',
                '#99724B',
                '#59454E',]

    params = ['sigma_2', 't_2', 'infall_2', 't_1', 'infall_1', 'sfe', 'mgal', 'delta_sfe', 'nb','imf_upper', 'mae']

    fig = plot_corner_points_contours(
        runs=analyzers,
        params=params,
        run_names=labels,
        include_loss_axis=True,
        colors=inkcolrs,                              # <- use your ink colors
        levels=(0.68,),
        smooth=0.9,
        point_size=4,
        bins=40,
        save_path=os.path.join(overlay_dir, "corner_points_contours.png"),
    )


    save_path = os.path.join(overlay_dir, f"bigger_posterior_corner_combo.png")
    params = ['sigma_2', 't_2', 'infall_2', 't_1', 'infall_1', 'sfe', 'mgal', 'delta_sfe']

    plot_corner_with_marginals_multi(
        analyzers,
        params=params,
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=labels,
        save_path=save_path
    )

    save_path = overlay_dir

    _ = compute_and_plot_combined_covariant_uncertainties(
            analyzers,
            params=params,          # or a shorter list if you prefer
            percentile=100,          # or use 10; selection still respects per-folder mixture cutoff when present
            weight_power=1.0,
            p_hpd=0.68,
            grid_n=240,
            alpha_gamma=0.9,
            ink_colors=inkcolrs,
            save_dir=save_path
    )
    print(f"[combined uncertainties] wrote outputs under: {save_path}")






    save_path = os.path.join(overlay_dir, f"in2_posterior_corner_combo.png")
    params = ['sigma_2', 't_2', 'infall_2', 'sfe']

    plot_corner_with_marginals_multi(
        analyzers,
        params=params,
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=None,
        save_path=save_path
    )


    save_path = os.path.join(overlay_dir, f"in1_posterior_corner_combo.png")
    params = ['sigma_2', 't_1', 'infall_1', 'sfe']

    plot_corner_with_marginals_multi(
        analyzers,
        params=params,
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=None,
        save_path=save_path
    )

    save_path = os.path.join(overlay_dir, f"chem_posterior_corner_combo.png")
    params = ['sigma_2', 'mgal', 'delta_sfe', 'sfe', 'nb','imf_upper']

    plot_corner_with_marginals_multi(
        analyzers,
        params=params,
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=None,
        save_path=save_path
    )

