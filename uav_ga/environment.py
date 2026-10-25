"""The hazard grid: priority zones, hazard pockets, no-fly zones and targets.

All 2-D arrays are indexed [y, x]; a cell centre sits at integer (x, y).
"""
import numpy as np

# (cx, cy, sigma, peak) in *relative* grid units, so the map scales with grid_size
DISASTER_ZONES = [(0.22, 0.72, 0.09, 1.00),
                  (0.72, 0.76, 0.08, 0.85),
                  (0.50, 0.28, 0.10, 0.95),
                  (0.85, 0.22, 0.06, 0.65)]
HAZARD_POCKETS = [(0.47, 0.55, 0.07, 0.95),    # high wind corridor
                  (0.74, 0.55, 0.06, 0.85),    # radiation pocket
                  (0.22, 0.38, 0.08, 0.70),    # turbulence
                  (0.55, 0.20, 0.05, 0.80)]    # radiation, near zone 3
NFZ_RECTS = [(0.36, 0.62, 0.46, 0.84)]         # (x0, y0, x1, y1)
NFZ_CIRCLES = [(0.64, 0.44, 0.07)]             # (cx, cy, r)


class Environment:
    def __init__(self, cfg):
        N = cfg.grid_size
        self.N = N
        yy, xx = np.mgrid[0:N, 0:N]
        self.cell_xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

        def blobs(spec, base):
            m = np.full((N, N), base, dtype=np.float64)
            for cx, cy, s, peak in spec:
                g = peak * np.exp(-((xx - cx * N) ** 2 + (yy - cy * N) ** 2) / (2 * (s * N) ** 2))
                m = np.maximum(m, g)
            return m

        self.priority = blobs(DISASTER_ZONES, 0.05)          # value of observing a cell
        self.hazard = blobs(HAZARD_POCKETS, 0.0)             # danger of flying through a cell

        nfz = np.zeros((N, N), dtype=bool)
        for x0, y0, x1, y1 in NFZ_RECTS:
            nfz |= (xx >= x0 * N) & (xx <= x1 * N) & (yy >= y0 * N) & (yy <= y1 * N)
        for cx, cy, r in NFZ_CIRCLES:
            nfz |= (xx - cx * N) ** 2 + (yy - cy * N) ** 2 <= (r * N) ** 2
        self.nfz = nfz

        # lookup table: nearest free cell for every cell (used by the repair operator)
        free = np.argwhere(~nfz)                              # (y, x)
        self.nearest_free = np.stack([xx, yy], axis=2).astype(np.float64)
        for by, bx in np.argwhere(nfz):
            d = (free[:, 0] - by) ** 2 + (free[:, 1] - bx) ** 2
            fy, fx = free[np.argmin(d)]
            self.nearest_free[by, bx] = (fx, fy)

        # probability of picking a cell when we want "smart" random points
        p = self.priority * (~nfz)
        self.prio_prob = (p / p.sum()).ravel()

        # hidden targets (survivors), more likely inside high-priority zones
        rng = np.random.default_rng(cfg.env_seed)
        idx = rng.choice(N * N, size=cfg.n_targets, replace=False, p=self.prio_prob)
        self.target_xy = self.cell_xy[idx].astype(np.float32)
        self.target_w = self.priority.ravel()[idx].astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def cell_index(self, xy):
        """Round continuous coordinates to integer (xi, yi) cell indices."""
        xi = np.clip(np.rint(xy[..., 0]), 0, self.N - 1).astype(int)
        yi = np.clip(np.rint(xy[..., 1]), 0, self.N - 1).astype(int)
        return xi, yi

    def sample_priority_points(self, rng, k):
        """k random points, biased towards high-priority cells (with sub-cell jitter)."""
        idx = rng.choice(self.N * self.N, size=k, p=self.prio_prob)
        pts = self.cell_xy[idx].astype(np.float64) + rng.uniform(-0.5, 0.5, (k, 2))
        return np.clip(pts, 0, self.N - 1)
