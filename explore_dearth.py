#!/usr/bin/env python3
"""
Exploration utilities for identifying and targeting sparse regions in parameter space.
This version preserves the original public API (function names) but implements
a robust Voronoi-based "largest empty cell" search with proper polygon clipping.

Public functions kept as-is:
- voronoi_explore_dearths(GA_instance, population, exploration_fraction=0.2)
- identify_sparse_regions_voronoi(GA_instance, population, n_regions=32)
- _analyze_voronoi_2d(GA_instance, population, p1_idx, p2_idx, p1_name, p2_name, n_regions_per_pair=4)
- _mutate_toward_region(GA_instance, individual, target_region)
- _add_background_mutation(GA_instance, individual, mutation_probability=0.3)
"""
import random
import numpy as np
from scipy.spatial import Voronoi

# ---------------------------- helpers ---------------------------------

def _voronoi_finite_polygons_2d(vor, radius=10.0):
    """Make infinite Voronoi regions finite (2D only)."""
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D supported for finite polygon reconstruction.")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}

    for (p, q), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p, []).append((q, v1, v2))
        all_ridges.setdefault(q, []).append((p, v1, v2))

    for p, region_idx in enumerate(vor.point_region):
        verts = vor.regions[region_idx]
        if len(verts) == 0:
            continue
        if all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue

        # Need to close region by extending edges to a "far" point
        ridges = all_ridges.get(p, [])
        new_region = [v for v in verts if v >= 0]
        for q, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[q] - vor.points[p]
            if np.allclose(t, 0):
                continue
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # outward normal
            midpoint = (vor.points[p] + vor.points[q]) * 0.5
            direction = np.sign(np.dot(midpoint - center, n)) * n
            # pick whichever endpoint exists
            base = vor.vertices[v1 if v1 >= 0 else v2]
            far = base + direction * radius
            new_vertices.append(far.tolist())
            new_region.append(len(new_vertices) - 1)

        # order vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _clip_poly_to_unit_square(poly):
    """Sutherland–Hodgman clip of polygon (Nx2) to [0,1]x[0,1]."""
    def clip(poly, inside, intersect):
        out = []
        if len(poly) == 0:
            return out
        A = poly[-1]
        Ain = inside(A)
        for B in poly:
            Bin = inside(B)
            if Ain and Bin:
                out.append(B)
            elif Ain and not Bin:
                out.append(intersect(A, B))
            elif (not Ain) and Bin:
                out.append(intersect(A, B))
                out.append(B)
            A, Ain = B, Bin
        return out

    def inter_x(A, B, xconst):
        Ax, Ay = A; Bx, By = B
        if np.isclose(Bx, Ax):
            t = 0.0
        else:
            t = (xconst - Ax) / (Bx - Ax)
        y = Ay + t * (By - Ay)
        return np.array([xconst, y])

    def inter_y(A, B, yconst):
        Ax, Ay = A; Bx, By = B
        if np.isclose(By, Ay):
            t = 0.0
        else:
            t = (yconst - Ay) / (By - Ay)
        x = Ax + t * (Bx - Ax)
        return np.array([x, yconst])

    poly = clip(list(map(np.asarray, poly)),
                inside=lambda P: P[0] >= 0.0,
                intersect=lambda A, B: inter_x(A, B, 0.0))
    poly = clip(poly,
                inside=lambda P: P[0] <= 1.0,
                intersect=lambda A, B: inter_x(A, B, 1.0))
    poly = clip(poly,
                inside=lambda P: P[1] >= 0.0,
                intersect=lambda A, B: inter_y(A, B, 0.0))
    poly = clip(poly,
                inside=lambda P: P[1] <= 1.0,
                intersect=lambda A, B: inter_y(A, B, 1.0))
    return np.array(poly)


def _poly_area_and_centroid(poly):
    """Return (area, centroid) for a simple polygon (Nx2)."""
    if len(poly) < 3:
        return 0.0, np.array([np.nan, np.nan])
    x = poly[:, 0]; y = poly[:, 1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-15:
        return 0.0, np.array([np.nan, np.nan])
    cx = np.sum((x + x1) * cross) / (6.0 * A)
    cy = np.sum((y + y1) * cross) / (6.0 * A)
    return abs(A), np.array([cx, cy])


def _normalize_pair(GA_instance, population, i, j):
    lo_i, hi_i = GA_instance.get_param_bounds(i)
    lo_j, hi_j = GA_instance.get_param_bounds(j)
    rng_i = hi_i - lo_i
    rng_j = hi_j - lo_j
    pts = np.array([[(ind[i] - lo_i) / rng_i, (ind[j] - lo_j) / rng_j] for ind in population], float)
    # dedupe to avoid Qhull issues from coincident points
    pts = np.unique(np.round(pts, 12), axis=0)
    return pts, (lo_i, hi_i, lo_j, hi_j)

# ---------------------------- core API ---------------------------------

def _analyze_voronoi_2d(GA_instance, population, p1_idx, p2_idx, p1_name, p2_name, n_regions_per_pair=4):
    """
    Build Voronoi on normalized pair (p1_idx, p2_idx), clip cells to [0,1]^2,
    compute area and centroid, return top-N regions as dicts:
      {
        'pair': (p1_idx, p2_idx),
        'param_indices': {p1_name: p1_idx, p2_name: p2_idx},
        'target_params': {p1_name: center_i_denorm, p2_name: center_j_denorm},
        'area_norm': area_in_unit_square,
        'center_norm': (cx, cy),
        'center_denorm': (val_i, val_j),
        'polygon_norm': Nx2 array (for plotting/debug)
      }
    """
    pts, (lo_i, hi_i, lo_j, hi_j) = _normalize_pair(GA_instance, population, p1_idx, p2_idx)
    if len(pts) < 4:
        return []

    vor = Voronoi(pts)
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=10.0)

    results = []
    for region in regions:
        poly = vertices[region]
        poly = _clip_poly_to_unit_square(poly)
        if len(poly) < 3:
            continue
        area, centroid = _poly_area_and_centroid(poly)
        if area <= 0.0 or not np.isfinite(centroid).all():
            continue

        # denormalize centroid to parameter space
        center_i = lo_i + centroid[0] * (hi_i - lo_i)
        center_j = lo_j + centroid[1] * (hi_j - lo_j)

        results.append({
            "pair": (p1_idx, p2_idx),
            "param_indices": {p1_name: p1_idx, p2_name: p2_idx},
            "target_params": {p1_name: float(center_i), p2_name: float(center_j)},
            "area_norm": float(area),
            "center_norm": np.array(centroid, float),
            "center_denorm": np.array([center_i, center_j], float),
            "polygon_norm": np.array(poly, float),
        })

    # largest empty cells first
    results.sort(key=lambda r: r["area_norm"], reverse=True)
    return results[:max(0, int(n_regions_per_pair))]


def identify_sparse_regions_voronoi(GA_instance, population, n_regions=32):
    """
    Use Voronoi diagrams on several 2D parameter pairs; return up to n_regions
    total regions ranked by area.
    """
    # Keep the exact pairs you had in the original file.
    key_param_pairs = [
        (6, 7,  't_1',     't_2'),
        (7, 9,  't_2',     'infall_2'),
        (5, 9,  'sigma_2', 'infall_2'),
        (5, 7,  'sigma_2', 't_2'),
        (5, 14, 'sigma_2', 'nb'),
        (10, 5, 'sfe',     'sigma_2'),
        (10, 11,'sfe',     'delta_sfe'),
        (13, 14,'mgal',    'nb'),
    ]

    per_pair = max(1, int(np.ceil(n_regions / max(1, len(key_param_pairs)))))
    all_regions = []
    for p1_idx, p2_idx, p1_name, p2_name in key_param_pairs:
        regs = _analyze_voronoi_2d(GA_instance, population, p1_idx, p2_idx, p1_name, p2_name,
                                   n_regions_per_pair=per_pair)
        all_regions.extend(regs)

    all_regions.sort(key=lambda r: r["area_norm"], reverse=True)
    return all_regions[:n_regions]


def voronoi_explore_dearths(GA_instance, population, exploration_fraction=0.2):
    """
    Move the worst-performing individuals toward centers of the largest empty regions
    (by Voronoi cell area in normalized space), touching only the two parameters
    that define a given region.
    """
    if not 0.0 < exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in (0,1].")

    # number of individuals to redirect
    n_move = max(1, int(len(population) * exploration_fraction))

    # gather candidate regions
    regions = identify_sparse_regions_voronoi(GA_instance, population, n_regions=n_move)

    if len(regions) == 0:
        return 0

    # worst performers first (assume lower fitness is better; if your GA is the opposite, flip sign)
    def _fitness_value(ind):
        try:
            # DEAP individuals have .fitness.values tuple; lower-is-better assumed
            return ind.fitness.values[0] if getattr(ind.fitness, "valid", False) else float("inf")
        except Exception:
            return float("inf")

    worst = sorted(population, key=_fitness_value, reverse=True)[:n_move]

    moved = 0
    for k, ind in enumerate(worst):
        region = regions[k % len(regions)]
        _mutate_toward_region(GA_instance, ind, region)
        _add_background_mutation(GA_instance, ind, mutation_probability=0.2)
        # invalidate fitness
        try:
            del ind.fitness.values
        except Exception:
            pass
        moved += 1

    return moved

# --------------------- mutation utilities (kept names) ------------------

def _mutate_toward_region(GA_instance, individual, target_region):
    """
    Move an individual toward region center on the two dims only.
    target_region must contain:
      - 'param_indices': {name: idx}
      - 'target_params': {name: value}
    """
    param_indices = target_region["param_indices"]
    target_params = target_region["target_params"]

    # deterministic, strong pull; add small noise for diversity
    for pname, pidx in param_indices.items():
        target_val = float(target_params[pname])
        current_val = float(individual[pidx])
        direction = target_val - current_val

        # move 90–100% of the way + small gaussian noise relative to range
        frac = 0.9 + 0.1 * random.random()
        lo, hi = GA_instance.get_param_bounds(pidx)
        rng = hi - lo
        noise = 0.02 * rng * random.gauss(0.0, 1.0)

        new_val = current_val + frac * direction + noise
        new_val = GA_instance._reflect_at_bounds(new_val, lo, hi)
        individual[pidx] = new_val


def _add_background_mutation(GA_instance, individual, mutation_probability=0.3):
    """
    Light mutations on other continuous parameters to avoid collapsing diversity.
    Adjust the index list if your GA uses different ordering.
    """
    # Keep the exact index set you had before to stay API-compatible
    continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for idx in continuous_indices:
        if random.random() < mutation_probability:
            lo, hi = GA_instance.get_param_bounds(idx)
            rng = hi - lo
            step = 0.02 * rng * random.gauss(0.0, 1.0)
            new_val = GA_instance._reflect_at_bounds(float(individual[idx]) + step, lo, hi)
            individual[idx] = new_val
