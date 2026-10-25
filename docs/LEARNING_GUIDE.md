# Learning guide: how this project works

## 1. The big idea in 5 lines
1. A **chromosome** is a complete mission plan for all drones.
2. The **fitness function** simulates the plan and returns one number (higher = better).
3. The GA keeps a **population** of plans, picks good ones (**selection**), mixes them (**crossover**),
   randomly tweaks them (**mutation**), repairs illegal ones, and repeats for many **generations**.
4. Elitism keeps the best plan; random immigrants + hyper-mutation keep diversity.
5. We compare against **random search** with the same number of evaluations to prove the GA really helps.

## 2. Read the code in this order
| Step | File | What to look for |
|------|------|------------------|
| 1 | `config.py` | every parameter and its meaning |
| 2 | `environment.py` | how the grid, hazards, no-fly zones and targets are built |
| 3 | `chromosome.py` | the gene layout `[x, y, speed, power, on/off]` |
| 4 | `fitness.py` | `trajectory()` -> `simulate()` -> `score()`; this is the heart of the problem |
| 5 | `operators.py` | selection, `hybrid_crossover`, `spatial_mutation`, `repair` |
| 6 | `ga.py` | `run()`: the generation loop |
| 7 | `main.py` | how everything is wired and plotted |

## 3. Key concepts (with where they live)
* **Hybrid encoding**: real numbers (x, y, speed, power) + binary (sensor on/off) -> `chromosome.py`.
* **Why speed is interesting**: propulsion energy grows with v^2 but flight time falls with v -> real trade-off.
* **Normalisation**: each objective is divided by a reference value so the weights are comparable.
* **Weights = preferences**: changing them changes the best plan (see `run_experiments.py --only weights`).
* **Repair vs penalty**: repair edits a bad solution; a penalty only lowers its score. We use both.
* **Dynamic penalty**: `penalty_lambda()` in `fitness.py`; strictness rises as generations go on.
* **Selection pressure**: tournament size k. Large k = greedy and fast but may lose diversity; k = 1 is random.
* **Premature convergence**: population becomes too similar; we measure `diversity()` and boost mutation.

## 4. Likely viva questions
1. *Why a GA and not A\*?* There are many competing objectives and a huge continuous search space with no single optimal path; GA searches it without gradients and handles non-linear objectives and constraints.
2. *Why is the encoding hybrid?* Positions/speeds are continuous, sensor on/off is binary; one chromosome holds both.
3. *What is the role of each weight?* They express mission priorities; the same GA gives different plans under different weights.
4. *How do you handle the no-fly zone?* Waypoint repair + penalty for crossing segments.
5. *What does elitism do?* Guarantees the best solution is never lost.
6. *How do you prevent premature convergence?* Random immigrants, diversity-triggered hyper-mutation, disruptive operators (relocation, route reversal).
7. *What happens with a very high mutation rate?* Search becomes close to random; with a very low rate it stalls (see sensitivity plot).
8. *Why compare with random search?* It shows that the improvement comes from evolution and not from luck or a lenient fitness function.
9. *What is the termination criterion?* Max generations or stagnation.
10. *Why does average fitness fluctuate?* Offspring and immigrants are often infeasible and get penalised; the best fitness (protected by elitism) rises almost monotonically.
