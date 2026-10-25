# UAV Strategy Engine: Genetic Algorithm for Multi-Objective Drone Deployment

Artificial Intelligence (BSCS-5), Bahria University Lahore, Assignment 2 (CLO-4).

A Genetic Algorithm (GA) that plans **deployment positions, waypoint trajectories, speeds and sensor
schedules** for a fleet of UAVs surveying a hazardous 50 x 50 disaster grid. There is no single "correct"
answer: the GA balances coverage, detection, energy, time and hazard exposure while respecting hard
constraints (flight range, battery, deadline, no-fly zones).

## Project structure

```
uav-ga-strategy-engine/
├── main.py                 # one full GA run + all figures (start here)
├── run_experiments.py      # sensitivity analysis, GA vs random search, weight presets
├── requirements.txt
├── uav_ga/
│   ├── config.py           # Config dataclass: every parameter in one place
│   ├── environment.py      # grid: priority zones, hazard pockets, no-fly zones, targets
│   ├── chromosome.py       # hybrid real/binary chromosome + initialisation
│   ├── fitness.py          # trajectory decoding, physics simulation, multi-objective fitness
│   ├── operators.py        # selection, crossover, mutation, constraint repair
│   ├── ga.py               # GeneticAlgorithm engine (loop, elitism, termination)
│   ├── baseline.py         # random-search baseline (same evaluation budget)
│   └── visualize.py        # all plots
├── docs/LEARNING_GUIDE.md  # code walkthrough + viva questions
└── outputs/                # generated figures / csv / json
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.9+ is required (only `numpy` and `matplotlib` are used).

## How to run

```bash
python main.py                          # default run (~20-60 s), writes ./outputs/
python main.py --pop 100 --gens 200     # larger run
python main.py --seed 5                 # different random seed
python run_experiments.py               # all experiments (several minutes)
python run_experiments.py --quick       # fast smoke test
python run_experiments.py --only baseline   # sensitivity | baseline | weights
```

### Generated outputs

| File | Content |
|------|---------|
| `01_environment.png` | priority map, hazard map, no-fly zones, targets |
| `02_fitness_convergence.png` | best / average / median fitness per generation |
| `03_population_diversity.png` | diversity monitor and hyper-mutation threshold |
| `04_best_flight_paths.png` | evolved UAV trajectories on the map |
| `05_coverage_heatmap.png` | required vs achieved coverage and uncovered priority |
| `06_evolution_snapshots.png` | best plan + heatmap at several generations |
| `07_ga_vs_random_search.png` | GA vs random search (paths, curves, objectives) |
| `results.json` | config, metrics, best chromosome, history |
| `experiments/*` | sensitivity plots, boxplots, uniqueness analysis, weight presets, csv tables |

## Design summary

**Chromosome** (`n_drones x n_waypoints x 5`, hybrid real / binary). Each gene is
`[x, y, speed, sensor_power, sensor_on]`. Waypoint 0 of every drone is its *deployment position*; the remaining
waypoints form a multi-segment trajectory. Speed, power and the binary sensor bit of waypoint *i* describe
the segment *i -> i+1*, so the sensor schedule is evolved together with the route.

**Multi-objective fitness** (all terms normalised to about [0, 1]):

```
F = 0.35*Coverage + 0.25*Detection - 0.15*Energy - 0.10*Time - 0.15*Hazard - Penalty
```

* Coverage: priority-weighted fraction of the grid observed (sensor radius grows with power).
* Detection: priority-weighted fraction of hidden targets detected.
* Energy: propulsion (`c_base + c_drag * v^2` per cell) + sensor energy (power x time).
* Time: mission makespan (slowest drone) / deadline. Hazard: hazard intensity integrated over time spent flying.

**Genetic operators**

* Selection: tournament (size *k* = selection pressure) or linear rank, plus elitism.
* Crossover: per-drone choice of route swap / two-point waypoint-segment swap / geometric (blend) crossover.
* Mutation: Gaussian waypoint shift, priority-biased relocation, sensor-bit flip, route reversal (2-opt style),
  whole-route translation. Diversity-triggered hyper-mutation and random immigrants prevent premature convergence.

**Constraint handling**

* No-fly zones: waypoints inside a zone are *repaired* to the nearest free cell; segments crossing a zone are penalised.
* Maximum flight range: *repair* by contracting the route towards its launch point.
* Battery, deadline: *penalty functions*. The penalty coefficient grows with the generation
  (`lambda_g = lambda_0 * (1 + 2 g / G)`), lenient early for exploration and strict late.

**Termination**: maximum generations, or stagnation (no improvement for `patience` generations after `min_generations`).

**Baseline**: random search using the same sampling, repair, penalty and exactly the same number of fitness evaluations.

## Reproducibility

All randomness is seeded (`--seed`, `Config.env_seed`). The disaster map is fixed, so results can be reproduced
exactly with the same library versions. Change any parameter in `uav_ga/config.py`.
