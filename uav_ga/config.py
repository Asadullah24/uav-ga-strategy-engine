"""Central configuration. Every tunable number of the project lives here."""
from dataclasses import dataclass, replace, asdict


@dataclass
class Config:
    # ------------------------------------------------------------ environment
    grid_size: int = 50          # 50 x 50 cell grid
    n_drones: int = 4            # fleet size
    n_waypoints: int = 8         # waypoints per drone (waypoint 0 = deployment point)
    n_targets: int = 30          # hidden targets (survivors) to detect
    env_seed: int = 7            # fixed, so every run sees the same disaster map

    # ------------------------------------------------------------ UAV physics
    v_min: float = 2.0           # cells / time-unit
    v_max: float = 8.0
    c_base: float = 1.0          # propulsion energy per cell (base)
    c_drag: float = 0.03         # extra energy per cell ~ c_drag * v^2
    sensor_cost: float = 6.0     # sensor energy per time-unit at full power
    r_min: float = 2.0           # sensor radius at power 0
    r_max: float = 5.5           # sensor radius at power 1
    max_range: float = 110.0     # HARD: max path length per drone (cells)
    battery: float = 300.0       # HARD: energy budget per drone
    t_max: float = 40.0          # HARD: mission deadline (time-units)
    sample_step: float = 2.0     # spacing of sensing samples along a path
    hazard_scale: float = 0.30   # mean hazard level regarded as "worst case"

    # ------------------------------------------------------------ fitness weights (sum = 1)
    w_cov: float = 0.35          # priority-weighted area coverage      (maximise)
    w_det: float = 0.25          # target detection rate                (maximise)
    w_energy: float = 0.15       # total energy                         (minimise)
    w_time: float = 0.10         # mission time (makespan)              (minimise)
    w_hazard: float = 0.15       # exposure to wind / radiation pockets (minimise)

    # ------------------------------------------------------------ constraint handling
    penalty_base: float = 1.0    # lambda_0
    penalty_growth: float = 2.0  # lambda_g = lambda_0 * (1 + growth * g/G)
    penalty_nfz: float = 5.0     # extra weight of no-fly-zone violations

    # ------------------------------------------------------------ genetic algorithm
    pop_size: int = 80
    generations: int = 150
    selection: str = "tournament"    # "tournament" | "rank"
    tournament_size: int = 5         # selection pressure (k)
    rank_pressure: float = 1.8       # linear-ranking pressure in [1, 2]
    crossover_rate: float = 0.85
    mutation_rate: float = 0.04      # per-waypoint probability
    sigma_xy: float = 0.06           # Gaussian step as a fraction of the grid
    sigma_v: float = 0.15
    sigma_p: float = 0.15
    elite_count: int = 2
    immigrant_ratio: float = 0.05    # fresh random individuals per generation
    init_heuristic_ratio: float = 0.5
    diversity_threshold: float = 0.12
    hyper_factor: float = 2.5        # mutation boost when diversity collapses
    patience: int = 50               # stop if no improvement for this many gens
    min_delta: float = 1e-4
    min_generations: int = 50        # never stop before this
    seed: int = 0

    def copy(self, **changes):
        return replace(self, **changes)

    def to_dict(self):
        return asdict(self)
