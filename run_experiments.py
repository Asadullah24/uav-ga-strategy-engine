"""Experiments for the technical report.

  1. sensitivity : population size, mutation rate, selection pressure
  2. baseline    : GA vs random search over several seeds (+ solution uniqueness)
  3. weights     : how different fitness weightings change the evolved plan

    python run_experiments.py                 # everything (~10 min on 1 core, faster on many)
    python run_experiments.py --quick         # tiny smoke-test
    python run_experiments.py --only baseline # a single experiment
"""
import argparse, csv, itertools, multiprocessing as mp, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga import Config, Environment, FitnessEvaluator, GeneticAlgorithm, RandomSearch
from uav_ga import visualize as viz

KEYS = ("coverage", "detection", "energy", "time", "hazard")


# ----------------------------------------------------------------- workers
def _ga_job(job):
    tag, kw = job
    cfg = Config(**kw)
    env = Environment(cfg)
    t0 = time.time()
    ga = GeneticAlgorithm(cfg, env, verbose=False)
    best = ga.run()
    h = ga.history
    return dict(tag=tag, seed=cfg.seed, fitness=best.fitness, curve=h["best"], avg_curve=h["avg"],
                genes=best.genes, secs=time.time() - t0, evals=ga.ev.n_evals, gens=h["generations_run"],
                **{k: best.metrics[k] for k in KEYS})


def _rs_job(job):
    tag, kw, n_evals = job
    cfg = Config(**kw)
    env = Environment(cfg)
    rs = RandomSearch(cfg, env, seed=cfg.seed)
    best, _ = rs.run(n_evals)
    return dict(tag=tag, seed=cfg.seed, fitness=best.fitness, genes=best.genes, evals=n_evals,
                **{k: best.metrics[k] for k in KEYS})


def run_pool(fn, jobs):
    n = min(os.cpu_count() or 1, len(jobs))
    if n <= 1:
        return [fn(j) for j in jobs]
    with mp.Pool(n) as p:
        return p.map(fn, jobs, chunksize=1)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def gens_to_95(curve):
    c = np.asarray(curve)
    return int(np.argmax(c >= c[0] + 0.95 * (c[-1] - c[0])))


# ----------------------------------------------------------------- 1. sensitivity
def sensitivity(out, seeds, gens, base_pop):
    grid = {"pop_size": [20, 40, 80, 160],
            "mutation_rate": [0.01, 0.04, 0.08, 0.16, 0.30],
            "tournament_size": [1, 2, 3, 5, 10]}
    jobs = []
    for param, vals in grid.items():
        for v in vals:
            for s in range(seeds):
                kw = dict(pop_size=base_pop, generations=gens, patience=10 ** 9, seed=s)
                kw[param] = v
                jobs.append(((param, v), kw))
    print(f"[sensitivity] {len(jobs)} GA runs ...")
    res = run_pool(_ga_job, jobs)

    rows, curves = [], {}
    for param, vals in grid.items():
        for v in vals:
            r = [x for x in res if x["tag"] == (param, v)]
            f = np.array([x["fitness"] for x in r])
            curves[(param, v)] = np.mean([x["curve"] for x in r], axis=0)
            rows.append(dict(parameter=param, value=v, mean_fitness=round(f.mean(), 4), std_fitness=round(f.std(), 4),
                             best_fitness=round(f.max(), 4),
                             mean_coverage=round(np.mean([x["coverage"] for x in r]), 3),
                             mean_energy=round(np.mean([x["energy"] for x in r]), 3),
                             mean_hazard=round(np.mean([x["hazard"] for x in r]), 3),
                             gens_to_95pct=round(np.mean([gens_to_95(x["curve"]) for x in r]), 1),
                             evaluations=int(np.mean([x["evals"] for x in r])),
                             seconds_per_run=round(np.mean([x["secs"] for x in r]), 1)))
    write_csv(os.path.join(out, "sensitivity_summary.csv"), rows)

    titles = {"pop_size": "Population size", "mutation_rate": "Mutation rate (per waypoint)",
              "tournament_size": "Tournament size (selection pressure)"}
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
    for j, (param, vals) in enumerate(grid.items()):
        m = [r["mean_fitness"] for r in rows if r["parameter"] == param]
        sd = [r["std_fitness"] for r in rows if r["parameter"] == param]
        axs[0, j].errorbar(range(len(vals)), m, yerr=sd, marker="o", capsize=4, lw=2, color="tab:green")
        axs[0, j].set_xticks(range(len(vals))); axs[0, j].set_xticklabels(vals)
        axs[0, j].set_xlabel(titles[param]); axs[0, j].set_ylabel("Final best fitness (mean ± std)")
        axs[0, j].set_title(f"Effect of {titles[param].split(' (')[0].lower()}"); axs[0, j].grid(alpha=0.3)
        for v, col in zip(vals, plt.cm.viridis(np.linspace(0, 0.9, len(vals)))):
            axs[1, j].plot(curves[(param, v)], color=col, lw=1.8, label=f"{v}")
        axs[1, j].set_xlabel("Generation"); axs[1, j].set_ylabel("Mean best fitness")
        axs[1, j].legend(title=titles[param].split(" (")[0], fontsize=8); axs[1, j].grid(alpha=0.3)
        lo = min(c[min(10, len(c) - 1)] for (p_, _), c in curves.items() if p_ == param)
        axs[1, j].set_ylim(lo - 0.02, max(c.max() for (p_, _), c in curves.items() if p_ == param) + 0.02)
    fig.suptitle(f"Hyper-parameter sensitivity ({seeds} seeds per setting, {gens} generations)", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(out, "08_hyperparameter_sensitivity.png"), dpi=140); plt.close(fig)
    print("  saved sensitivity_summary.csv + 08_hyperparameter_sensitivity.png")
    for r in rows:
        print("  ", r["parameter"], r["value"], "->", r["mean_fitness"], "+-", r["std_fitness"],
              "| gens_to_95%:", r["gens_to_95pct"], "| s/run:", r["seconds_per_run"])


# ----------------------------------------------------------------- 2. baseline + uniqueness
def chamfer(A, B):
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def baseline(out, seeds, gens, pop):
    kw = dict(pop_size=pop, generations=gens)
    ga_res = run_pool(_ga_job, [("GA", dict(kw, seed=s)) for s in range(seeds)])
    n_evals = int(np.mean([r["evals"] for r in ga_res]))
    print(f"[baseline] GA done, random search with {n_evals} evaluations per run ...")
    rs_res = run_pool(_rs_job, [("Random", dict(kw, seed=s), r["evals"]) for s, r in enumerate(ga_res)])

    cfg = Config(**kw); env = Environment(cfg); ev = FitnessEvaluator(env, cfg)
    rows = []
    for r in ga_res + rs_res:
        info = ev.analyze(r["genes"], 1.0)
        r["info"] = info
        rows.append(dict(method=r["tag"], seed=r["seed"], fitness=round(r["fitness"], 4),
                         **{k: round(info[k], 3) for k in KEYS},
                         redundancy=round(info["redundancy"], 3), path_priority=round(info["path_priority"], 3),
                         path_hazard=round(info["path_hazard"], 3), total_length=round(info["total_length"], 1),
                         turn_deg=round(info["mean_turn_deg"], 1), feasible=bool(info["feasible"])))
    write_csv(os.path.join(out, "baseline_comparison.csv"), rows)

    summ = []
    for tag in ("GA", "Random"):
        rr = [r for r in rows if r["method"] == tag]
        d = {"method": tag}
        for k in ("fitness",) + KEYS + ("redundancy", "path_priority", "path_hazard", "total_length", "turn_deg"):
            v = np.array([r[k] for r in rr], dtype=float)
            d[k] = f"{v.mean():.3f} +- {v.std():.3f}"
        summ.append(d)
    write_csv(os.path.join(out, "baseline_summary.csv"), summ)
    for d in summ:
        print("  ", d)

    # ---- uniqueness: pairwise Chamfer distance between flight paths
    pts = {(r["tag"], r["seed"]): ev.trajectory(r["genes"])["pts"] for r in ga_res + rs_res}
    def mean_pair(t1, t2):
        v = [chamfer(pts[(t1, i)], pts[(t2, j)]) for i in range(seeds) for j in range(seeds)
             if (t1 != t2 or i < j)]
        return float(np.mean(v)), float(np.std(v))
    uniq = [dict(pair=f"{a}-{b}", mean_chamfer=round(mean_pair(a, b)[0], 2), std=round(mean_pair(a, b)[1], 2))
            for a, b in [("GA", "GA"), ("Random", "Random"), ("GA", "Random")]]
    write_csv(os.path.join(out, "uniqueness_chamfer.csv"), uniq)
    print("   uniqueness (mean Chamfer distance in cells):", uniq)

    # ---- figures
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, k, t in [(axs[0], "fitness", "Final fitness"), (axs[1], "coverage", "Coverage"),
                     (axs[2], "detection", "Detection rate"), (axs[3], "energy", "Energy (lower = better)")]:
        data = [[r[k] for r in rows if r["method"] == m] for m in ("GA", "Random")]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["GA", "Random"])
        for patch, c in zip(bp["boxes"], ["#8fd18f", "#f09a9a"]): patch.set_facecolor(c)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-.08, .08, len(d)), d, c="k", s=14, zorder=5)
        ax.set_title(t); ax.grid(alpha=0.3)
    fig.suptitle(f"GA vs random search - {seeds} independent runs, equal evaluation budget ({n_evals})")
    fig.tight_layout(); fig.savefig(os.path.join(out, "09_ga_vs_random_boxplots.png"), dpi=140); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, tag, rr in [(axs[0], "GA", ga_res), (axs[1], "Random", rs_res)]:
        viz._base_map(ax, env)
        for r in rr:
            c = plt.cm.tab10(r["seed"] % 10)
            for route in r["genes"]:
                ax.plot(route[:, 0], route[:, 1], color=c, lw=1.4, alpha=0.8)
                ax.scatter(*route[0, :2], marker="s", s=28, color=c, edgecolor="k", zorder=5)
        ax.set_title(f"{tag}: best paths of {seeds} independent runs (one colour per run)")
    ax = axs[2]
    names = [u["pair"] for u in uniq]; vals = [u["mean_chamfer"] for u in uniq]; errs = [u["std"] for u in uniq]
    ax.bar(names, vals, yerr=errs, capsize=5, color=["#8fd18f", "#f09a9a", "#bbbbbb"])
    ax.set_ylabel("Mean Chamfer distance between path sets (cells)")
    ax.set_title("How different are the solutions?"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "10_solution_uniqueness.png"), dpi=140); plt.close(fig)
    print("  saved baseline_*.csv, uniqueness_chamfer.csv, 09_*.png, 10_*.png")


# ----------------------------------------------------------------- 3. fitness weight presets
PRESETS = {
    "Balanced (default)": dict(),
    "Coverage-first":  dict(w_cov=0.45, w_det=0.30, w_energy=0.08, w_time=0.07, w_hazard=0.10),
    "Energy-saver":    dict(w_cov=0.25, w_det=0.20, w_energy=0.30, w_time=0.15, w_hazard=0.10),
    "Safety-first":    dict(w_cov=0.25, w_det=0.20, w_energy=0.10, w_time=0.10, w_hazard=0.35),
}


def weights(out, gens, pop):
    jobs = [(n, dict(pop_size=pop, generations=gens, seed=1, **w)) for n, w in PRESETS.items()]
    print(f"[weights] {len(jobs)} GA runs ...")
    res = run_pool(_ga_job, jobs)
    cfg = Config(); env = Environment(cfg); ev = FitnessEvaluator(env, cfg)
    rows = []
    fig = plt.figure(figsize=(19, 10.5)); gs = fig.add_gridspec(2, len(res), height_ratios=[1.3, 1])
    for j, r in enumerate(res):
        info = ev.analyze(r["genes"], 1.0)
        rows.append(dict(preset=r["tag"], **{k: round(info[k], 3) for k in KEYS}, total_length=round(info["total_length"], 1)))
        ax = fig.add_subplot(gs[0, j])
        viz.draw_solution(ax, env, r["genes"], info, f"{r['tag']}\ncoverage {info['coverage']:.0%} | energy {info['energy']:.2f} | hazard {info['hazard']:.2f}")
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(KEYS)); w = 0.8 / len(rows)
    for i, row in enumerate(rows):
        ax.bar(x + i * w - 0.4 + w / 2, [row[k] for k in KEYS], w, label=row["preset"])
    ax.set_xticks(x); ax.set_xticklabels(["Coverage ↑", "Detection ↑", "Energy ↓", "Time ↓", "Hazard ↓"])
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_title("Different weightings -> different (all valid) trade-off solutions")
    fig.tight_layout(); fig.savefig(os.path.join(out, "11_weight_presets.png"), dpi=130); plt.close(fig)
    write_csv(os.path.join(out, "weight_presets.csv"), rows)
    for r in rows: print("  ", r)


# ----------------------------------------------------------------- entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sensitivity", "baseline", "weights"], default=None)
    ap.add_argument("--quick", action="store_true", help="tiny settings, just to test the pipeline")
    ap.add_argument("--out", default="outputs/experiments")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q = a.quick
    t0 = time.time()
    if a.only in (None, "sensitivity"):
        sensitivity(a.out, seeds=2 if q else 3, gens=15 if q else 80, base_pop=20 if q else 60)
    if a.only in (None, "baseline"):
        baseline(a.out, seeds=2 if q else 5, gens=15 if q else 150, pop=20 if q else 80)
    if a.only in (None, "weights"):
        weights(a.out, gens=15 if q else 150, pop=20 if q else 80)
    print(f"\nDone in {time.time() - t0:.0f}s -> {a.out}/")
