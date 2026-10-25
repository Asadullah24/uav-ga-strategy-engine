"""Genetic operators: selection, crossover, mutation and constraint repair."""
import numpy as np
from .chromosome import V, P, S


# ------------------------------------------------------------------ selection
def tournament_selection(fits, k, n, rng):
    """Pick n winners; each winner is the best of k random contestants.
    Larger k = higher selection pressure (k=1 is pure random selection)."""
    cand = rng.integers(0, len(fits), size=(n, k))
    return cand[np.arange(n), np.argmax(fits[cand], axis=1)]


def rank_selection(fits, pressure, n, rng):
    """Linear ranking selection; pressure in [1, 2] (1 = uniform)."""
    N = len(fits)
    ranks = np.empty(N)
    ranks[np.argsort(fits)] = np.arange(N)                 # 0 = worst
    p = (2 - pressure) / N + 2 * ranks * (pressure - 1) / (N * (N - 1))
    return rng.choice(N, size=n, p=p / p.sum())


# ------------------------------------------------------------------ crossover
def hybrid_crossover(g1, g2, rng):
    """Spatial crossover, decided independently for every drone route:
       30%  route swap          - children exchange whole routes (keeps routes intact)
       30%  segment swap        - two-point swap of a block of consecutive waypoints
       40%  geometric blend     - child = a*p1 + (1-a)*p2 on x, y, speed, power
    The binary sensor bits are always inherited by uniform crossover."""
    c1, c2 = g1.copy(), g2.copy()
    D, W, _ = g1.shape
    for d in range(D):
        r = rng.random()
        if r < 0.30:
            c1[d], c2[d] = g2[d].copy(), g1[d].copy()
        elif r < 0.60:
            a, b = sorted(rng.choice(W + 1, size=2, replace=False))
            c1[d, a:b], c2[d, a:b] = g2[d, a:b].copy(), g1[d, a:b].copy()
        else:
            alpha = rng.random()
            c1[d, :, :4] = alpha * g1[d, :, :4] + (1 - alpha) * g2[d, :, :4]
            c2[d, :, :4] = (1 - alpha) * g1[d, :, :4] + alpha * g2[d, :, :4]
            m = rng.random(W) < 0.5
            c1[d, :, S] = np.where(m, g1[d, :, S], g2[d, :, S])
            c2[d, :, S] = np.where(m, g2[d, :, S], g1[d, :, S])
    return c1, c2


# ------------------------------------------------------------------ mutation
def spatial_mutation(g, rng, pm, sigma_scale, cfg, env):
    """Five mutations tailored to spatial routes (g is modified in place):
       1. Gaussian shift of a waypoint (x, y) + small change of speed / power
       2. Relocation  - waypoint jumps to a new (priority-biased) location (exploration)
       3. Sensor flip - toggles the sensor_on bit
       4. Route reversal (2-opt style) of a block of waypoints
       5. Route translation - the whole route is moved by one random vector
    """
    D, W, _ = g.shape
    N = cfg.grid_size
    sig = cfg.sigma_xy * N * sigma_scale

    dd, ww = np.nonzero(rng.random((D, W)) < pm)                       # 1
    k = len(dd)
    if k:
        g[dd, ww, 0:2] += rng.normal(0, sig, (k, 2))
        g[dd, ww, V] += rng.normal(0, cfg.sigma_v * (cfg.v_max - cfg.v_min) * sigma_scale, k)
        g[dd, ww, P] += rng.normal(0, cfg.sigma_p * sigma_scale, k)

    dd, ww = np.nonzero(rng.random((D, W)) < pm * 0.25)                # 2
    k = len(dd)
    if k:
        smart = env.sample_priority_points(rng, k)
        rand = rng.uniform(0, N - 1, (k, 2))
        g[dd, ww, 0:2] = np.where(rng.random((k, 1)) < 0.5, smart, rand)

    dd, ww = np.nonzero(rng.random((D, W)) < pm * 0.5)                 # 3
    g[dd, ww, S] = 1.0 - g[dd, ww, S]

    for d in range(D):
        if rng.random() < min(1.0, pm * 2):                            # 4
            a, b = sorted(rng.choice(W + 1, size=2, replace=False))
            if b - a >= 2:
                g[d, a:b] = g[d, a:b][::-1].copy()
        if rng.random() < pm * 0.5:                                    # 5
            g[d, :, 0:2] += rng.normal(0, sig, 2)
    return g


# ------------------------------------------------------------------ constraint repair
def repair(g, env, cfg):
    """Make a chromosome (mostly) feasible, in place:
       - clip genes to their legal ranges
       - range limit : contract the route towards its launch point if too long
       - no-fly zone : waypoints inside an NFZ snap to the nearest free cell
    Whatever violation is left (e.g. a segment crossing an NFZ, battery, deadline)
    is handled by the dynamic penalty inside the fitness function."""
    N = cfg.grid_size
    g[:, :, 0:2] = np.clip(g[:, :, 0:2], 0, N - 1)
    g[:, :, V] = np.clip(g[:, :, V], cfg.v_min, cfg.v_max)
    g[:, :, P] = np.clip(g[:, :, P], 0.0, 1.0)
    g[:, :, S] = (g[:, :, S] > 0.5).astype(float)

    length = np.linalg.norm(np.diff(g[:, :, :2], axis=1), axis=2).sum(axis=1)
    for d in np.nonzero(length > cfg.max_range)[0]:
        start = g[d, 0, :2].copy()
        g[d, :, :2] = start + (cfg.max_range * 0.98 / length[d]) * (g[d, :, :2] - start)

    xy = g[:, :, 0:2]
    xi, yi = env.cell_index(xy)
    blocked = env.nfz[yi, xi]
    if blocked.any():
        xy[blocked] = env.nearest_free[yi[blocked], xi[blocked]]
    return g
