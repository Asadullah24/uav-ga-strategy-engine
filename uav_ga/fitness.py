"""Multi-objective fitness function + physical simulation of a mission plan."""
import numpy as np
from .chromosome import V, P, S


class FitnessEvaluator:
    """Turns a chromosome into a trajectory, simulates it and scores it.

        F = w_cov*Coverage + w_det*Detection - w_en*Energy - w_t*Time - w_h*Hazard - Penalty

    Every term is normalised to ~[0, 1] so the weights are directly comparable.
    """

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.cx = env.cell_xy[:, 0]
        self.cy = env.cell_xy[:, 1]
        self.pr = env.priority.ravel().astype(np.float32)
        self.pr_sum = float(self.pr.sum())
        self.tx, self.ty = env.target_xy[:, 0], env.target_xy[:, 1]
        self.tw = env.target_w
        self.n_evals = 0

    # ------------------------------------------------------------- decoding
    def trajectory(self, genes):
        """Sample every segment every `sample_step` cells -> sensing/flight samples."""
        cfg = self.cfg
        D, W, _ = genes.shape
        A, B = genes[:, :-1, :2], genes[:, 1:, :2]
        vec = B - A
        lens = np.linalg.norm(vec, axis=2)                          # (D, W-1)
        n = np.maximum(1, np.ceil(lens / cfg.sample_step)).astype(int).ravel()
        seg = np.repeat(np.arange(n.size), n)
        starts = np.cumsum(n) - n
        local = np.arange(n.sum()) - starts[seg]
        t = ((local + 0.5) / n[seg])[:, None]
        pts = A.reshape(-1, 2)[seg] + t * vec.reshape(-1, 2)[seg]
        return dict(
            pts=pts,
            ds=lens.ravel()[seg] / n[seg],
            spd=genes[:, :-1, V].ravel()[seg],
            pwr=genes[:, :-1, P].ravel()[seg],
            act=genes[:, :-1, S].ravel()[seg] > 0.5,
            drone=seg // (W - 1),
            lens=lens,
        )

    def _quality(self, px, py, r, qx, qy):
        """Detection quality of sensors at (px,py) with radius r on points (qx,qy).
        1.0 inside 60% of the radius, fading linearly to 0 at the radius."""
        d = np.sqrt((qx[None, :] - px[:, None]) ** 2 + (qy[None, :] - py[:, None]) ** 2)
        return np.clip((r[:, None] - d) / (0.4 * r[:, None]), 0.0, 1.0)

    # ------------------------------------------------------------- simulation
    def simulate(self, genes, keep_maps=False):
        cfg, env = self.cfg, self.env
        D = genes.shape[0]
        tr = self.trajectory(genes)
        pts, ds, spd, pwr, act, drone = (tr[k] for k in ("pts", "ds", "spd", "pwr", "act", "drone"))

        dt = ds / spd                                                    # time per sample
        prop = ds * (cfg.c_base + cfg.c_drag * spd ** 2)                 # propulsion energy
        sens = act * cfg.sensor_cost * pwr * dt                          # sensor energy
        E_d = np.bincount(drone, weights=prop + sens, minlength=D)
        T_d = np.bincount(drone, weights=dt, minlength=D)
        L_d = tr["lens"].sum(axis=1)
        makespan = float(T_d.max())

        xi, yi = env.cell_index(pts)
        exposure = float((env.hazard[yi, xi] * dt).sum())                # hazard x time spent
        nfz_frac = float(env.nfz[yi, xi].mean())                         # samples inside NFZ

        cov = np.zeros(self.pr.shape, dtype=np.float32)
        det = np.zeros(len(self.tw), dtype=np.float32)
        if act.any():
            px, py = pts[act, 0].astype(np.float32), pts[act, 1].astype(np.float32)
            r = (cfg.r_min + (cfg.r_max - cfg.r_min) * pwr[act]).astype(np.float32)
            cov = self._quality(px, py, r, self.cx, self.cy).max(axis=0)
            det = self._quality(px, py, r, self.tx, self.ty).max(axis=0)

        out = dict(
            coverage=float((cov * self.pr).sum() / self.pr_sum),
            detection=float((det * self.tw).sum() / self.tw.sum()),
            energy=float(E_d.sum() / (D * cfg.battery)),
            time=makespan / cfg.t_max,
            hazard=exposure / (D * cfg.t_max * cfg.hazard_scale),
            nfz_frac=nfz_frac,
            range_viol=float(np.maximum(0, L_d - cfg.max_range).sum() / cfg.max_range),
            battery_viol=float(np.maximum(0, E_d - cfg.battery).sum() / cfg.battery),
            time_viol=max(0.0, makespan - cfg.t_max) / cfg.t_max,
            path_len=L_d, energy_per_drone=E_d, time_per_drone=T_d,
        )
        if keep_maps:
            out["cov_map"] = cov.reshape(env.N, env.N)
            out["det"] = det
        return out

    # ------------------------------------------------------------- scoring
    def penalty_lambda(self, gen_frac):
        """Dynamic penalty coefficient: lenient early (exploration), strict late."""
        return self.cfg.penalty_base * (1.0 + self.cfg.penalty_growth * gen_frac)

    def score(self, m, gen_frac=1.0):
        c = self.cfg
        raw = (c.w_cov * m["coverage"] + c.w_det * m["detection"]
               - c.w_energy * m["energy"] - c.w_time * m["time"] - c.w_hazard * m["hazard"])
        viol = (c.penalty_nfz * m["nfz_frac"] + m["range_viol"]
                + m["battery_viol"] + m["time_viol"])
        pen = self.penalty_lambda(gen_frac) * viol
        return raw - pen, raw, pen, viol

    def evaluate(self, chrom, gen_frac=1.0):
        """Scores a chromosome in place and returns its fitness."""
        self.n_evals += 1
        m = self.simulate(chrom.genes)
        fit, raw, pen, viol = self.score(m, gen_frac)
        m.update(raw=raw, penalty=pen, violation=viol, feasible=viol < 1e-3)
        chrom.fitness, chrom.metrics = fit, m
        return fit

    # ------------------------------------------------------------- analysis (plots / report)
    def analyze(self, genes, gen_frac=1.0):
        """Full metrics + coverage map + structural statistics of one solution."""
        m = self.simulate(genes, keep_maps=True)
        fit, raw, pen, viol = self.score(m, gen_frac)
        m.update(fitness=fit, raw=raw, penalty=pen, violation=viol, feasible=viol < 1e-3)

        tr = self.trajectory(genes)
        # redundancy: share of covered priority that is observed by >= 2 drones
        counts = np.zeros(self.pr.shape, dtype=np.int32)
        for d in range(genes.shape[0]):
            k = (tr["drone"] == d) & tr["act"]
            if k.any():
                px, py = tr["pts"][k, 0].astype(np.float32), tr["pts"][k, 1].astype(np.float32)
                r = (self.cfg.r_min + (self.cfg.r_max - self.cfg.r_min) * tr["pwr"][k]).astype(np.float32)
                counts += (self._quality(px, py, r, self.cx, self.cy).max(axis=0) >= 0.5)
        covered = (self.pr * (counts >= 1)).sum()
        m["redundancy"] = float((self.pr * (counts >= 2)).sum() / max(covered, 1e-9))
        L = tr["lens"].sum(axis=1)
        m["length_balance_cv"] = float(L.std() / max(L.mean(), 1e-9))
        seg = np.diff(genes[:, :, :2], axis=1)
        a1, a2 = seg[:, :-1], seg[:, 1:]
        cosang = (a1 * a2).sum(-1) / np.maximum(np.linalg.norm(a1, axis=-1) * np.linalg.norm(a2, axis=-1), 1e-9)
        m["mean_turn_deg"] = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean())
        m["sensor_on_ratio"] = float(genes[:, :, S].mean())
        m["mean_speed"] = float(genes[:, :-1, V].mean())
        m["total_length"] = float(L.sum())
        xi, yi = self.env.cell_index(tr["pts"])
        m["path_priority"] = float(self.env.priority[yi, xi].mean())   # how "on target" the flight is
        m["path_hazard"] = float(self.env.hazard[yi, xi].mean())       # how risky the flight is
        return m
