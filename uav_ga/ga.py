"""The Genetic Algorithm engine."""
import math
import numpy as np
from .chromosome import Chromosome
from .environment import Environment
from .fitness import FitnessEvaluator
from .operators import (tournament_selection, rank_selection, hybrid_crossover,
                        spatial_mutation, repair)


class GeneticAlgorithm:
    def __init__(self, cfg, env=None, evaluator=None, verbose=True):
        self.cfg = cfg
        self.env = env or Environment(cfg)
        self.ev = evaluator or FitnessEvaluator(self.env, cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.verbose = verbose
        self.best = None
        self.history = {}

    # ------------------------------------------------------------ population
    def init_population(self):
        """Half random, half priority-seeded individuals (diverse yet informed)."""
        cfg = self.cfg
        n_heur = int(round(cfg.init_heuristic_ratio * cfg.pop_size))
        pop = []
        for i in range(cfg.pop_size):
            c = Chromosome.random(cfg, self.env, self.rng, heuristic=i < n_heur)
            repair(c.genes, self.env, cfg)
            pop.append(c)
        return pop

    def diversity(self, pop):
        """Mean standard deviation of the (normalised) genes across the population."""
        cfg = self.cfg
        arr = np.stack([c.genes[:, :, :4] for c in pop])
        scale = np.array([cfg.grid_size - 1, cfg.grid_size - 1, cfg.v_max - cfg.v_min, 1.0])
        return float((arr / scale).std(axis=0).mean())

    def _select(self, fits, n):
        cfg = self.cfg
        if cfg.selection == "rank":
            return rank_selection(fits, cfg.rank_pressure, n, self.rng)
        return tournament_selection(fits, cfg.tournament_size, n, self.rng)

    # ------------------------------------------------------------ main loop
    def run(self):
        cfg, rng = self.cfg, self.rng
        G = cfg.generations
        h = {k: [] for k in ("best", "avg", "median", "worst", "diversity", "hyper", "evals")}
        h["best_genes"] = []
        pop = self.init_population()
        best_ref, stall, hyper_events = -np.inf, 0, 0

        for g in range(G + 1):
            frac = g / G
            fits = np.array([self.ev.evaluate(c, frac) for c in pop])
            bi = int(np.argmax(fits))
            div = self.diversity(pop)
            h["best"].append(float(fits[bi]))
            h["avg"].append(float(fits.mean()))
            h["median"].append(float(np.median(fits)))
            h["worst"].append(float(fits.min()))
            h["diversity"].append(div)
            h["evals"].append(self.ev.n_evals)
            h["best_genes"].append(pop[bi].genes.copy())

            if fits[bi] > best_ref + cfg.min_delta:
                best_ref, stall = fits[bi], 0
            else:
                stall += 1

            if self.verbose and (g % 10 == 0 or g == G):
                m = pop[bi].metrics
                print(f"gen {g:3d} | best {fits[bi]:+.4f} avg {fits.mean():+.4f} "
                      f"div {div:.3f} | cov {m['coverage']:.2f} det {m['detection']:.2f} "
                      f"E {m['energy']:.2f} T {m['time']:.2f} H {m['hazard']:.2f} "
                      f"feasible={m['feasible']}")

            if g == G or (g >= cfg.min_generations and stall >= cfg.patience):
                if self.verbose and g < G:
                    print(f"-> early stop at generation {g} (no improvement for {stall} generations)")
                break

            # ---- diversity-triggered hyper-mutation (anti premature convergence)
            pm, sigma_scale = cfg.mutation_rate, 1.0
            h["hyper"].append(False)
            if div < cfg.diversity_threshold:
                pm, sigma_scale = min(0.5, pm * cfg.hyper_factor), 1.5
                h["hyper"][-1] = True
                hyper_events += 1

            # ---- next generation: elitism + offspring + random immigrants
            order = np.argsort(-fits)
            new = [pop[i].copy() for i in order[:cfg.elite_count]]
            n_imm = int(cfg.immigrant_ratio * cfg.pop_size)
            n_off = cfg.pop_size - len(new) - n_imm
            parents = self._select(fits, 2 * math.ceil(n_off / 2))
            for i in range(0, len(parents), 2):
                g1, g2 = pop[parents[i]].genes, pop[parents[i + 1]].genes
                if rng.random() < cfg.crossover_rate:
                    c1, c2 = hybrid_crossover(g1, g2, rng)
                else:
                    c1, c2 = g1.copy(), g2.copy()
                for child in (c1, c2):
                    if len(new) < cfg.pop_size - n_imm:
                        spatial_mutation(child, rng, pm, sigma_scale, cfg, self.env)
                        repair(child, self.env, cfg)
                        new.append(Chromosome(child))
            for _ in range(n_imm):
                c = Chromosome.random(cfg, self.env, rng, heuristic=rng.random() < 0.5)
                repair(c.genes, self.env, cfg)
                new.append(c)
            pop = new

        # ---- hall of fame: re-score every generation's champion with the FINAL penalty
        cands = [Chromosome(x) for x in h["best_genes"]]
        for c in cands:
            self.ev.evaluate(c, 1.0)
        self.best = max(cands, key=lambda c: c.fitness)
        h["hyper_events"] = hyper_events
        h["generations_run"] = len(h["best"]) - 1
        self.history = h
        return self.best
