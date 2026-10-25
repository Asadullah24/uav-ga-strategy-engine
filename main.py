"""Run the full UAV Genetic-Algorithm simulation and produce every figure.

    python main.py                       # default settings
    python main.py --pop 80 --gens 150   # bigger run
"""
import argparse, json, os, time
import numpy as np
from uav_ga import Config, Environment, FitnessEvaluator, GeneticAlgorithm, RandomSearch
from uav_ga import visualize as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--mut", type=float, default=None, help="mutation rate (per waypoint)")
    ap.add_argument("--tour", type=int, default=None, help="tournament size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    cfg = Config(seed=a.seed)
    if a.pop: cfg.pop_size = a.pop
    if a.gens: cfg.generations = a.gens
    if a.mut is not None: cfg.mutation_rate = a.mut
    if a.tour: cfg.tournament_size = a.tour
    os.makedirs(a.out, exist_ok=True)
    out = lambda f: os.path.join(a.out, f)

    print("=" * 70)
    print(f"UAV GA Strategy Engine | drones={cfg.n_drones} waypoints={cfg.n_waypoints} "
          f"grid={cfg.grid_size}x{cfg.grid_size} pop={cfg.pop_size} gens={cfg.generations} seed={cfg.seed}")
    print("=" * 70)

    env = Environment(cfg)
    viz.plot_environment(env, out("01_environment.png"))

    t0 = time.time()
    ga = GeneticAlgorithm(cfg, env)
    best = ga.run()
    h = ga.history
    ga_time = time.time() - t0
    n_evals = ga.ev.n_evals
    print(f"\nGA finished: {h['generations_run']} generations, {n_evals} evaluations, {ga_time:.1f}s, "
          f"hyper-mutation triggered {h['hyper_events']}x")

    # ---- baseline with the SAME evaluation budget
    ev_rs = FitnessEvaluator(env, cfg)
    rs = RandomSearch(cfg, env, ev_rs, seed=cfg.seed)
    checkpoints = [int(e) for e in h["evals"]]
    rs_best, rs_curve = rs.run(n_evals, checkpoints)

    ga_info = ga.ev.analyze(best.genes, 1.0)
    rs_info = ev_rs.analyze(rs_best.genes, 1.0)

    # ---- figures
    viz.plot_convergence(h, out("02_fitness_convergence.png"))
    viz.plot_diversity(h, cfg, out("03_population_diversity.png"))
    viz.plot_paths(env, best.genes, ga_info, cfg.n_drones, out("04_best_flight_paths.png"))
    viz.plot_coverage(env, ga_info, out("05_coverage_heatmap.png"))
    last = len(h["best"]) - 1
    viz.plot_snapshots(env, ga.ev, h, [0, max(1, last // 8), last // 4, last // 2, last], out("06_evolution_snapshots.png"))
    viz.plot_ga_vs_random(env, cfg.n_drones, best.genes, ga_info, rs_best.genes, rs_info, h, rs_curve,
                          out("07_ga_vs_random_search.png"))

    # ---- console table + json
    keys = ["fitness", "coverage", "detection", "energy", "time", "hazard", "penalty", "feasible",
            "redundancy", "length_balance_cv", "mean_turn_deg", "total_length"]
    print("\n{:<20s}{:>14s}{:>14s}".format("metric", "GA", "Random search"))
    print("-" * 48)
    for k in keys:
        f = lambda v: f"{v:>14}" if isinstance(v, (bool, np.bool_)) else f"{v:>14.4f}"
        print(f"{k:<20s}{f(ga_info[k])}{f(rs_info[k])}")

    to_py = lambda d: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items() if k not in ("cov_map", "det")}
    json.dump(dict(config=cfg.to_dict(), generations_run=h["generations_run"], evaluations=n_evals,
                   ga=to_py(ga_info), random_search=to_py(rs_info), best_genes=best.genes.tolist(),
                   history=dict(best=h["best"], avg=h["avg"], median=h["median"], worst=h["worst"], diversity=h["diversity"])),
              open(out("results.json"), "w"), indent=2)
    print(f"\nAll figures + results.json saved in ./{a.out}/")


if __name__ == "__main__":
    main()
