"""Chromosome = the full mission plan of the fleet.

Shape (n_drones, n_waypoints, 5). Every row is one *gene* describing a waypoint:

    [ x , y , speed , sensor_power , sensor_on ]
      real real  real      real          binary        -> hybrid real/binary encoding

* waypoint 0 of each drone is its DEPLOYMENT (launch) position,
* the next waypoints form the trajectory (a multi-segment route),
* speed / power / sensor_on of waypoint i describe the segment i -> i+1.
"""
import numpy as np

X, Y, V, P, S = 0, 1, 2, 3, 4


class Chromosome:
    __slots__ = ("genes", "fitness", "metrics")

    def __init__(self, genes):
        self.genes = genes
        self.fitness = None
        self.metrics = None

    def copy(self):
        c = Chromosome(self.genes.copy())
        c.fitness, c.metrics = self.fitness, self.metrics
        return c

    @classmethod
    def random(cls, cfg, env, rng, heuristic=False):
        """Random individual. heuristic=True seeds waypoints in high-priority areas
        and orders them nearest-neighbour first (short, sensible routes)."""
        D, W, N = cfg.n_drones, cfg.n_waypoints, cfg.grid_size
        g = np.zeros((D, W, 5))
        if heuristic:
            for d in range(D):
                pts = env.sample_priority_points(rng, W)
                order, left = [0], list(range(1, W))
                while left:
                    last = pts[order[-1]]
                    j = min(left, key=lambda i: np.hypot(*(pts[i] - last)))
                    order.append(j)
                    left.remove(j)
                g[d, :, :2] = pts[order]
        else:
            g[:, :, :2] = rng.uniform(0, N - 1, (D, W, 2))
        g[:, :, V] = rng.uniform(cfg.v_min, cfg.v_max, (D, W))
        g[:, :, P] = rng.uniform(0.3, 1.0, (D, W))
        g[:, :, S] = (rng.random((D, W)) < 0.75).astype(float)
        return cls(g)
