# Benchmark results — VanillaRRT vs KinodynamicRRT

Writeup skeleton. Captures the numbers and the figure findings from the first
full benchmark run while they are fresh. Framing lives in
`docs/research_question.md`; sequencing rationale in `docs/roadmap.md`; this
file is the *results*.

> **One-line claim.** At realistic bevel-tip curvature (R = 1/κ = 50mm) in a
> 150mm workspace, VanillaRRT is faster but its paths are geometrically
> infeasible — discontinuous at every node, and if executed they leave the
> workspace 130–230mm from target — whereas KinodynamicRRT is slower and
> sometimes fails, but produces continuous paths (heading discontinuity
> **exactly 0.000** across all 325 successful runs) that land 1.8–2.3mm from
> target.

---

## Method

**Planners:** `VanillaRRT`, `KinodynamicRRT`. `RRTStar` is **excluded by
design**, not omitted: at R=50mm in this workspace it cannot connect two poses
cheaply enough to rewire (local pose-to-pose connections cost a median far
larger than the scene, because the minimum turning circle is ~314mm around),
so it built ~155 nodes in 10,000 iterations and never reached goal. This is a
recorded finding — see `docs/roadmap.md`, Task 3.6.

**Grid:** 840 runs.
- 4 hand-designed scenarios × 2 planners × 30 seeds = 240
- 30 random scenarios × 2 planners × 10 seeds = 600

**Common conditions (identical for every run):** κ = 1/50, 150×150mm world at
0.5mm resolution, margin 2.0mm, goal_tolerance 3.0mm, step_dt 0.05,
edge_velocity 5.0, max_iterations 20,000. The budget is generous by design so
that *failure means the planner could not solve the scenario*, not that it ran
out of iterations.

**Not equal work.** One VanillaRRT iteration is a straight-line rollout plus a
collision check; one KinodynamicRRT iteration is two arc rollouts plus a
comparison. Equal iteration counts are therefore not equal compute — which is
why wall-clock is reported as its own metric, not inferred from iterations.

**Metrics** (all rolled through the one real model, κ = 1/50 — "execute this
plan on the needle"):
- **path_cost_mm** — executed path length (`n_edges · n_steps · v · dt`).
- **endpoint_error_mm** — roll the returned controls from the start; distance
  of the final pose from goal. The "if you executed this plan, where does the
  needle end up?" number.
- **heading discontinuity** (max / mean / count > 1e-6 rad) — per interior
  node, the wrapped gap between the re-rolled arrival heading and the stored
  departure heading. The measure of geometric infeasibility.

Reported hand-designed and random **separately**: the four are illustrative
(each gets a figure), the 30 are statistical (aggregate only). Mixing them
buries the illustrative cases and biases the aggregate.

Raw per-run records: `experiments/results/benchmark_raw.csv` (regenerable via
`scripts/run_benchmark.py`; gitignored). Tables: `scripts/analyze_benchmark.py`.

---

## Results — hand-designed (mean ± std over successful runs)

| scenario | planner | success | cost_mm | time_s | iters | endpoint_mm | hdisc_max | hdisc_n |
|---|---|---|---|---|---|---|---|---|
| **open** | Vanilla | 30/30 | 138.3 ± 11.3 | 0.04 | 125 | **132.7 ± 9.9** | **1.51** | 26.7 |
| | Kino | 30/30 | 114.0 ± 3.6 | 1.54 | 2370 | **2.08 ± 0.6** | **0.000** | 0 |
| **constrained_passage** | Vanilla | 30/30 | 151.3 | 0.07 | 282 | **168.7** | 1.61 | 29.3 |
| | Kino | **19/30** | 121.1 | 3.26 | 6005 | 2.33 | 0.000 | 0 |
| **target_behind** | Vanilla | 30/30 | 158.8 | 0.05 | 164 | **151.1** | 1.61 | 30.8 |
| | Kino | **0/30** | — | — | — | — | — | — |
| **cluttered** | Vanilla | 30/30 | 190.3 | 0.07 | 219 | **230.2 ± 1.6** | 1.61 | 37.1 |
| | Kino | 29/30 | 177.9 | 2.35 | 4988 | 1.82 | 0.000 | 0 |

Each scenario tests something distinct:
- **open** — efficiency baseline (both solve; the comparison is cost/quality,
  not feasibility).
- **constrained_passage** — a 16mm gap approached off-axis; the curvature
  constraint bites (Kino 19/30, a *rate* scenario) while Vanilla threads
  trivially by teleporting its heading.
- **target_behind** — goal directly behind the obstacle (symmetric); a
  deliberate **PASS/FAIL** scenario (Kino 0/30, Vanilla 30/30), not tuned into
  a middle band because that band is a 2° knife edge (see `scenarios.py`).
- **cluttered** — 6 scattered vessels; the most anatomically representative
  (Kino 29/30).

**Mixed reporting is deliberate:** three scenarios yield success rates /
distributions; `target_behind` yields a binary. Its variance-free row is by
design, not a degenerate measurement.

---

## Results — random (aggregate over the 30-scenario set)

| planner | success | cost_mm | time_s | iters | endpoint_mm | hdisc_max |
|---|---|---|---|---|---|---|
| Vanilla | 300/300 | 146.1 ± 18.0 | 0.05 | 158 | **138.3 ± 12.5** | 1.63 |
| Kino | 247/300 | 112.1 ± 3.0 | 2.03 | 3562 | **1.87 ± 0.7** | 0.000 |

**Solved-by-any-seed breakdown (30 scenarios):** both 26 (87%),
vanilla-only 4 (13%), kino-only 0, **neither 0**.

Caveat for the aggregate: the anticipated dense/large-obstacle unsolvable tail
did **not** materialise at the generator seed used — `neither` is empty and
Vanilla solved all 30. Because Vanilla is so strong, `neither`/`kino-only` can
only fill when Vanilla itself fails, which did not happen here; an empty
`neither` is thus partly a statement about Vanilla's reach, not only scenario
difficulty. The 4 vanilla-only cases are the genuine curvature-limited
discriminators.

---

## Figures and what they reveal (`docs/figures/benchmark_*.png`)

One figure per hand-designed scenario: geometry (obstacles + margin-inflated
boundary + start/goal heading arrows + goal-tolerance circle), Vanilla's
planned polyline, Vanilla's **executed** trace (its controls rolled through the
real model, clipped at the workspace so the excursion shows), and Kino's single
curve (planned = executed). Plotted runs are the **near-median-cost** seed per
(scenario, planner) — representative, not cherry-picked: open v=1/k=0,
constrained_passage v=19/k=0, target_behind v=7/(failed-tree seed 0),
cluttered v=4/k=0. Regenerate with `scripts/plot_benchmark_figures.py`.

Four things the pictures show that the tables do not:

1. **Vanilla's executed trajectory diverges immediately, not gradually.** In
   all four figures the executed curve peels off within the first millimetres
   and exits the *left* edge regardless of where the goal is — because
   Vanilla's first control is b=1 and the planner discards its synthesised
   per-node heading, so under real curvature the needle bends away on the very
   first edge. The large endpoint errors are "wrong from step one," not "drifts
   off near the end."

2. **`target_behind`'s failure has a visible shape.** The faint failed-Kino
   tree fans out **symmetrically around both sides** of the obstacle, sweeping
   the workspace into two dense lobes with a bare cone directly behind where
   the goal sits — the "effort splits between two equally-costly detours,
   neither completing" mechanism, rendered directly. A 0/30 alone hides this.

3. **Two "successful" planned paths, only one feasible.** In
   `constrained_passage` both planners' *planned* paths reach goal, but
   Vanilla's polyline visibly kinks at every node (the 27–37 counted
   discontinuities) while Kino's is smooth. Vanilla's 30/30 is a success *of
   the point-robot fiction*; read alone, that column misleads.

4. **Kino paths are also shorter and more direct.** The Kino curve hugs a
   near-geodesic while Vanilla's planned path wanders (cluttered: 178 vs
   190mm; random: 112 vs 146mm). Vanilla's only real advantage is wall-clock.

---

## Discussion / the claim this supports

The central result is **feasibility, not just optimality**. Vanilla's paths are
not merely too-tightly-curved — they are **discontinuous**, and therefore not
executable by any continuous vehicle. The heading-discontinuity metric makes
this quantitative and unambiguous: 0.000 for Kino everywhere (the stored node
state *is* the rollout endpoint under the model that built it), order-1 radians
at 27–37 nodes for Vanilla. Endpoint error then makes the *consequence*
concrete: execute Vanilla's plan and the needle ends up outside the arena.

The trade the benchmark quantifies: Kino pays ~35–50× in wall-clock (mean
success time per scenario: 34× cluttered, 38× open, 41× random, 47×
constrained) and loses some scenarios to the curvature constraint, and buys
feasibility — continuous, on-target, and (surprisingly) shorter paths.

---

## Honesty notes / limitations

- **No true-vs-model mismatch yet.** All metrics use the *model* needle
  (κ = 1/50). Vanilla's endpoint error is purely its planning cheat, not tissue
  mismatch. The mismatch axis (a distinct `true_needle`) is Phase 3, and is
  where the uncertainty-aware-planning sub-claim gets tested.
- **`neither` = 0 is seed-specific**, see the random-aggregate caveat above.
- **RRT\* absence is a finding, not a gap** — but it means the benchmark
  compares two planners, not three, at this scale. The secondary
  enlarged-workspace RRT\* comparison remains future work.
- **This is integration + benchmarking, not a new algorithm.** The contribution
  is the unified, reproducible testbed and the comparative analysis — pitched
  accordingly (see `README.md`).

---

## Reproduce

```bash
python scripts/generate_random_scenarios.py   # -> experiments/random_scenarios.json (committed)
python scripts/run_benchmark.py               # -> experiments/results/benchmark_raw.csv (gitignored)
python scripts/analyze_benchmark.py           # the tables above
python scripts/plot_benchmark_figures.py      # -> docs/figures/benchmark_*.png
```
