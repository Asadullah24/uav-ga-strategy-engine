"""Baseline: pure random search with exactly the same evaluation budget."""
import numpy as np
from .chromosome import Chromosome
from .environment import Environment
from .fitness import FitnessEvaluator
from .operators import repair


class RandomSearch:
    def __init__(self, cfg, env=None, evaluator=None, seed=0):
        self.cfg = cfg
        self.env = env or Environment(cfg)
        self.ev = evaluator or FitnessEvaluator(self.env, cfg)
        self.rng = np.random.default_rng(10_000 + seed)

    def run(self, n_evals, checkpoints=None):
        """Samples n_evals random plans (same repair + same final-penalty fitness).
        Returns (best chromosome, best-so-far curve at the given checkpoints)."""
        best, curve = None, []
        marks = set(checkpoints or [])
        for i in range(1, n_evals + 1):
            c = Chromosome.random(self.cfg, self.env, self.rng, heuristic=False)
            repair(c.genes, self.env, self.cfg)
            self.ev.evaluate(c, 1.0)
            if best is None or c.fitness > best.fitness:
                best = c
            if i in marks:
                curve.append(best.fitness)
        return best, curve
