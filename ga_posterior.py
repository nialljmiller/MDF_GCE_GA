#!/usr/bin/env python3
# ga_posterior.py — GA → pseudo-posterior corners with ESS control

import os, re, glob, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore")#, category=RuntimeWarning)

try:
    import corner
except ImportError as e:
    raise SystemExit("Missing dependency: pip install corner") from e

# ---------------------- discovery ----------------------
def discover_result_folders(base_dir):
    """Return [(name, path, [csvs...])] for subdirs with simulation_results*.csv."""
    out = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p) or name.startswith('.'):
            continue
        csvs = sorted(glob.glob(os.path.join(p, "simulation_results*.csv")))
        if csvs:
            out.append((name, p, csvs))
    return out

def _suffix(fname):
    m = re.search(r'_gen_(\d+)\.csv$', fname)
    if m: return int(m.group(1))
    m = re.search(r'_(\d+)\.csv$', fname)
    return int(m.group(1)) if m else None

def choose_primary_csv(csv_list):
    """Pick most relevant CSV: prefer highest numeric suffix; fallback: largest file."""
    recs = []
    for p in csv_list:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        recs.append((p, size, _suffix(os.path.basename(p))))
    if any(s is not None for _,_,s in recs):
        return max(recs, key=lambda r: ((r[2] if r[2] is not None else -1), r[1]))[0]
    return max(recs, key=lambda r: r[1])[0]

# ---------------------- pcard axis ranges (optional) ----------------------
_PCARD_MAP = {
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
def parse_pcard_ranges(pcard_path):
    if not os.path.isfile(pcard_path):
        return {}
    txt = open(pcard_path, 'r', encoding='utf-8').read()
    ranges = {}
    for key, col in _PCARD_MAP.items():
        m = re.search(rf'^\s*{re.escape(key)}\s*:\s*\[([^\]]+)\]', txt, flags=re.MULTILINE)
        if not m:
            continue
        try:
            nums = [float(x.strip()) for x in m.group(1).split(',')]
            if len(nums) == 2 and np.isfinite(nums[0]) and np.isfinite(nums[1]) and nums[1] > nums[0]:
                ranges[col] = (float(nums[0]), float(nums[1]))
        except Exception:
            pass
    return ranges

# ---------------------- loss → weights + diagnostics ----------------------
def weights_from_loss(loss, mode='exp', temperature=None, power=1.0):
    """
    mode='exp' : w ∝ exp(-(L - Lmin)/T), with T provided or tuned outside
    mode='inv' : w ∝ 1 / (L + eps)^power
    """
    L = np.asarray(loss, float)
    good = np.isfinite(L)
    if good.sum() < 3:
        raise ValueError("Not enough finite loss values.")
    w = np.zeros_like(L, float)
    if mode == 'inv':
        eps = np.nanmin(L[good]) * 1e-3 if np.isfinite(np.nanmin(L[good])) else 1e-12
        w[good] = 1.0 / np.power(L[good] + eps, power)
    else:
        if not (temperature and temperature > 0):
            raise ValueError("temperature must be >0 for exp mode (provide or tune)")
        L0 = np.nanmin(L[good])
        resid = L[good] - L0
        w[good] = np.exp(-resid / temperature)
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    if s > 0:
        w /= s
    return w

def effective_sample_size(w):
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)
    return (w.sum()**2) / (np.sum(w**2) + 1e-300)

def tune_temperature(loss, target_frac=0.30):
    """Pick T so that ESS ≈ target_frac * N for exp weighting."""
    L = np.asarray(loss, float)
    msk = np.isfinite(L)
    L = L[msk]
    if L.size < 3:
        return None
    L0 = np.min(L)
    resid = L - L0
    mad = np.median(np.abs(resid - np.median(resid))) or np.std(resid) or 1.0
    N = L.size
    best_T, best_gap = mad, 1e18
    for s in np.logspace(-2, 2, 33):
        T = mad * s
        w = np.exp(-(resid) / T)
        w /= w.sum()
        ess = (w.sum()**2) / np.sum(w**2)
        gap = abs(ess - target_frac * N)
        if gap < best_gap:
            best_T, best_gap = T, gap
    return best_T

# ---------------------- params & plotting ----------------------
DEFAULT_CONT = ['sigma_2','t_1','t_2','infall_1','infall_2','sfe','delta_sfe','imf_upper','mgal','nb']

def pick_params(df, preferred=None, min_unique=20):
    preferred = preferred or DEFAULT_CONT
    cand = []
    for c in df.columns:
        if c == 'fitness': 
            continue
        if np.issubdtype(df[c].dtype, np.number):
            v = df[c].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size and np.unique(v).size >= min_unique:
                cand.append(c)
    ordered = [c for c in preferred if c in cand]
    rest = [c for c in cand if c not in ordered]
    out = ordered + rest
    if not out:
        raise ValueError("No continuous numeric parameters found.")
    return out

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D

def _colors_for(n, ink_colors=None):
    if ink_colors is not None and len(ink_colors) >= n:
        return list(ink_colors)[:n]
    cyc = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6'])
    out = []
    i = 0
    for _ in range(n):
        out.append(cyc[i % len(cyc)])
        i += 1
    return out

def corner_weighted(
    df, 
    params, 
    weights, 
    out_png, 
    group_by=None,              # e.g. 'source' / 'folder' column; if None, tries 'run' else single group
    colors=None,                # list of hex colors (your ink palette)
    bins=40, 
    point_size=4, 
    title_note=None
):
    # pick group labels
    if group_by is None:
        group_by = 'run' if 'run' in df.columns else None
    if group_by is None:
        df = df.copy()
        df['_group'] = 'all'
        group_by = '_group'

    labels = [c.replace('_',' ') for c in params]
    groups = list(df[group_by].astype(str).unique())
    palette = _colors_for(len(groups), ink_colors=colors)

    fig = None
    for gi, g in enumerate(groups):
        sub = df[df[group_by].astype(str) == g]
        X = sub[params].to_numpy(float)
        w = np.asarray(weights)[sub.index] if hasattr(weights, 'index') else np.asarray(weights)[sub.index] if isinstance(weights, np.ndarray) else np.asarray(weights)[sub.index]
        # overlay: pass fig from previous call
        fig = corner.corner(
            X,
            labels=labels,
            bins=bins,
            weights=w,
            color=palette[gi],
            fig=fig,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            title_fmt=".3g",
            plot_datapoints=True,
            # tune scatter/hist per group so overlays read cleanly
            scatter_kwargs={'s': point_size, 'alpha': 0.5, 'rasterized': True},
            hist_kwargs={'density': True, 'alpha': 0.35, 'linewidth': 1.5}
        )

    # optional title note
    if title_note:
        fig.axes[0].text(0.02, 0.98, title_note, transform=fig.axes[0].transAxes,
                         ha='left', va='top', fontsize=9)

    # make a simple legend in the top-left subplot
    handles = [Line2D([0],[0], color=palette[i], lw=2) for i in range(len(groups))]
    fig.axes[0].legend(handles, groups, frameon=False, fontsize=9, loc='upper right')

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[saved]", out_png)
    return fig


def plot_overlaid_marginals(records, params, bins=60, out_png="analysis/marginals_overlay.png"):
    n = len(params); ncols = 3; nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3*nrows))
    axes = axes.ravel()
    for i,p in enumerate(params):
        ax = axes[i]
        for r in records:
            x = r['df'][p].to_numpy(float); w = r['w']
            hist, edges = np.histogram(x, bins=bins, weights=w, density=True)
            xl = 0.5*(edges[1:]+edges[:-1])
            ax.step(xl, hist, where='mid', alpha=0.9, label=r['name'])
        ax.set_title(p); ax.grid(True, alpha=0.25)
    for j in range(i+1, len(axes)): 
        axes[j].axis('off')
    axes[0].legend(loc='best', fontsize=8)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    print("[saved]", out_png)

# ---------------------- main flow ----------------------
def run(base=".", select="ask", mode="exp", temperature=None, target_ess_frac=0.30,
        power=1.0, bins=40, params="", resample=0, outdir=".", save_json=True):
    folders = discover_result_folders(base)
    if not folders:
        raise SystemExit(f"No subfolders under {base} with simulation_results*.csv")

    print("Discovered result folders:")
    for i,(name,p,csvs) in enumerate(folders):
        primary = choose_primary_csv(csvs)
        try:
            n = sum(1 for _ in open(primary, 'r', encoding='utf-8', errors='ignore')) - 1
        except Exception:
            n = -1
        print(f" [{i}] {name:>3s}  -> {os.path.relpath(primary)}  (rows≈{n})")

    if select == "ask":
        raw = input("Select indices (e.g., 0,2,5) or 'all': ").strip().lower()
        chosen = list(range(len(folders))) if raw == 'all' else [int(x) for x in re.split(r'[,\s]+', raw) if x]
    elif select == "all":
        chosen = list(range(len(folders)))
    else:
        chosen = [int(x) for x in re.split(r'[,\s]+', select) if x]

    if not chosen:
        raise SystemExit("Nothing selected.")

    # Load selections
    records = []
    for i in chosen:
        name, path, csvs = folders[i]
        primary = choose_primary_csv(csvs)
        df = pd.read_csv(primary)

        # choose loss col
        losscol = 'fitness' if 'fitness' in df.columns else ('wrmse' if 'wrmse' in df.columns else None)
        if losscol is None:
            raise ValueError(f"{primary}: cannot find a loss/fitness column")

        # sort by loss, but DO NOT prune; use all rows
        df_sorted = df.sort_values(by=losscol, ascending=True)
        L = df_sorted[losscol].to_numpy(float)

        # temperature (for exp)
        T_used = None
        if mode == "exp":
            if temperature and temperature > 0:
                T_used = float(temperature)
            else:
                T_used = tune_temperature(L, target_frac=target_ess_frac)
        # weights
        w = weights_from_loss(L, mode=mode, temperature=T_used, power=power)
        # per-run normalization
        w = w / (w.sum() + 1e-300)

        # axis ranges from pcard if present
        pranges = parse_pcard_ranges(os.path.join(path, 'bulge_pcard.txt'))

        ess = int(effective_sample_size(w))
        print(f"{name}: N={len(w)}  ESS={ess}  mode={mode}  T={T_used}")

        records.append(dict(name=name, path=path, csv=primary, df=df_sorted, w=w, pr=pranges,
                            losscol=losscol, temperature=T_used, ess=ess))

    # Choose parameters
    if params:
        params = [p.strip() for p in params.split(',') if p.strip()]
    else:
        params = pick_params(records[0]['df'])

    # Union axis ranges across selections (pcard first; fallback to data)
    union_ranges = {}
    for p in params:
        los, his = [], []
        for r in records:
            if p in r['pr']:
                lo, hi = r['pr'][p]; los.append(lo); his.append(hi)
        if not los:
            for r in records:
                if p in r['df'].columns:
                    v = r['df'][p].to_numpy(float)
                    v = v[np.isfinite(v)]
                    if v.size:
                        los.append(float(v.min())); his.append(float(v.max()))
        if los:
            union_ranges[p] = (min(los), max(his))

        # ---------------- Combined weighted corner (color-coded by run) ----------------
        # Build a combined DataFrame with a 'source' column so corner_weighted can group/color
        df_parts, w_parts = [], []
        for r in records:
            sub = r['df'][params].copy()
            sub['source'] = r['name']              # color/group key
            df_parts.append(sub)
            w_parts.append(r['w'])

        df_comb = pd.concat(df_parts, axis=0, ignore_index=True)
        w_comb = np.concatenate(w_parts)
        w_comb /= (w_comb.sum() + 1e-300)
        ess_comb = int(effective_sample_size(w_comb))

        inkcolrs =  ['#F0B800',
                    '#004C40',
                    '#0099A1',
                    '#C20016',
                    '#E8DCD8',
                    '#97BAAB',
                    '#1E6E6C',
                    '#99724B',
                    '#59454E',]

        note = f"weighted by {', '.join(sorted(set(r['losscol'] for r in records)))}; mode={mode}"
        out_comb = os.path.join(outdir, "analysis", "corner_mcmc_like_weighted.png")
        os.makedirs(os.path.dirname(out_comb), exist_ok=True)

        # Use the new corner_weighted (colors + group overlay)
        corner_weighted(
            df=df_comb,
            params=params,
            weights=w_comb,
            out_png=out_comb,
            group_by='source',                # <-- colors per run
            colors=inkcolrs,                  # <-- your palette
            bins=bins,
            point_size=4,
            title_note=note
        )

    print(f"[saved] {out_comb}  | Combined ESS={ess_comb}/{len(w_comb)}")

    # Per-run corners
    #for r in records:
    #    out_one = os.path.join(outdir, "analysis", os.path.basename(r['path']), "corner_weighted.png")
    #    note1 = f"{r['name']}  (weighted by {r['losscol']}; mode={mode}; ESS={r['ess']}/{len(r['w'])})"
    #    corner_weighted(r['df'], params, r['w'], axis_ranges=union_ranges, bins=bins, out_png=out_one, title_note=note1)
    #    print(f"[saved] {out_one}")

    # Per-run 1D marginal overlays (for convergence sanity)
    out_overlay = os.path.join(outdir, "analysis", "marginals_overlay.png")
    plot_overlaid_marginals(records, params, bins=min(80, max(30, bins)), out_png=out_overlay)

    # Optional: write a small JSON with run stats
    if save_json:
        meta = {
            "mode": mode,
            "target_ess_frac": target_ess_frac,
            "temperature_arg": temperature,
            "params": params,
            "runs": [
                dict(name=r['name'], csv=os.path.relpath(r['csv']),
                     N=len(r['w']), ESS=int(r['ess']),
                     losscol=r['losscol'], temperature=r['temperature'])
                for r in records
            ],
            "combined": {"N": int(len(w_comb)), "ESS": int(ess_comb)}
        }
        out_meta = os.path.join(outdir, "analysis", "posterior_meta.json")
        os.makedirs(os.path.dirname(out_meta), exist_ok=True)
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print("[saved]", out_meta)

def main():
    ap = argparse.ArgumentParser(description="Build MCMC-like weighted corner plots from GA outputs.")
    ap.add_argument("--base", default=".", help="Base directory to scan (default: .)")
    ap.add_argument("--select", default="ask", help="'ask', 'all', or comma-list of indices (e.g., 0,2,5)")
    ap.add_argument("--mode", default="exp", choices=["exp","inv"],
                    help="Loss→weight: exp (exp(-ΔL/T)) or inv (1/L^p). Default: exp")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Temperature T for mode=exp; if omitted, auto-tuned for ESS≈30%")
    ap.add_argument("--target-ess-frac", type=float, default=0.30,
                    help="Target ESS fraction for auto-tuning (default 0.30)")
    ap.add_argument("--power", type=float, default=1.0, help="Power p for mode=inv (default 1.0)")
    ap.add_argument("--bins", type=int, default=40, help="Corner bins (default 40)")
    ap.add_argument("--params", type=str, default="sigma_2, t_2, infall_2, t_1, infall_1, sfe, mgal, delta_sfe, nb,imf_upper, mae", help="Comma-separated param names; default=auto")
    ap.add_argument("--resample", type=int, default=0,
                    help="If >0, also output resampled corner with N draws")
    ap.add_argument("--outdir", default=".", help="Output root (default .)")
    ap.add_argument("--no-json", action="store_true", help="Do not write meta JSON")
    args = ap.parse_args()
    run(base=args.base, select=args.select, mode=args.mode, temperature=args.temperature,
        target_ess_frac=args.target_ess_frac, power=args.power, bins=args.bins,
        params=args.params, resample=args.resample, outdir=args.outdir, save_json=(not args.no_json))

if __name__ == "__main__":
    main()
