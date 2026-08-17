# Benchmark results — VanillaRRT vs KinodynamicRRT

Writeup skeleton. Captures the numbers and the figure findings from the first
full benchmark run while they are fresh. Framing lives in
`docs/research_question.md`; sequencing rationale in `docs/roadmap.md`; this
file is the *results*.

> **One-line claim.** At realistic bevel-tip curvature (R = 1/κ = 50mm) in a
> 150mm workspace, VanillaRRT is faster but its paths are geometrically
> infeasible — discontinuous at every node (heading discontinuity order-1 rad
> at 27–37 nodes), and even under the *most charitable* feasible execution — an
> open-loop tracker that reinterprets the path as curved controls — they
> **collide with a critical structure in 24/30 to 30/30 runs** and stray 40–101mm
> from the planned polyline. KinodynamicRRT is slower and sometimes fails, but
> produces continuous paths (heading discontinuity **exactly 0.000** across all
> 325 successful runs) that are executable as generated and land 1.8–2.3mm from
> target.

> **Why the vanilla execution metric changed (2026-08-16).** Vanilla's edges
> store one `Control(v, b=+1)` each — all of its steering lives in per-node
> headings it synthesises and discards — so rolling the *raw stored controls*
> just drove a radius-1/κ circle, and the old `endpoint_error_mm` (130–230mm)
> measured where that circle happened to end: a **storage convention, not path
> quality**. It is now replaced, *for VanillaRRT only*, by an open-loop
> segment-following tracker (`benchmark/vanilla_tracker.py`) that reads the
> planned path as a reference polyline and derives feasible curved controls from
> its segment bearings — the fairest reading of what vanilla output. Kinodynamic
> and RRT\* are untouched: their controls are model-generated, so planned and
> executed coincide and `endpoint_error_mm` is already exact (re-deriving *their*
> controls from bearings would manufacture error and degrade an exact metric).
> The endpoint numbers improved substantially under the fairer metric; the
> argument does not rest on that magnitude — it rests on `tracked_collides` and
> the heading discontinuities, neither of which is affected.

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
- **endpoint_error_mm** *(Kino / RRT\* only)* — roll the returned controls from
  the start; distance of the final pose from goal. Because their controls are
  model-generated, this is where the needle actually ends up if you execute the
  plan.
- **VanillaRRT tracked-execution metrics** *(vanilla only)* — the raw stored
  controls are meaningless (see the note above), so vanilla is executed via the
  open-loop tracker instead, giving three numbers:
  - **tracked_endpoint_error_mm** — distance from the tracked trajectory's final
    state to goal.
  - **tracked_max_crosstrack_mm** — max distance from any executed state to the
    reference *polyline* (min over segments, not to nearest node).
  - **tracked_collides** — does any executed state violate `is_free(margin)`
    (including leaving the workspace)? **The headline** — a path that cannot be
    followed without hitting a critical structure is unusable regardless of
    where it ends up.
- **heading discontinuity** (max / mean / count > 1e-6 rad) — per interior
  node, the wrapped gap between the re-rolled arrival heading and the stored
  departure heading. The measure of geometric infeasibility. Unaffected by the
  execution-metric change — it is a property of the *planned* path.

Reported hand-designed and random **separately**: the four are illustrative
(each gets a figure), the 30 are statistical (aggregate only). Mixing them
buries the illustrative cases and biases the aggregate.

Raw per-run records: `experiments/results/benchmark_raw.csv` (regenerable via
`scripts/run_benchmark.py`; gitignored). Tables: `scripts/analyze_benchmark.py`.

---

## Results — hand-designed (mean ± std over successful runs)

| scenario | planner | success | cost_mm | time_s | iters | endpoint_mm | hdisc_max | hdisc_n |
|---|---|---|---|---|---|---|---|---|
| **open** | Vanilla | 30/30 | 138.3 ± 11.3 | 0.04 | 125 | — † | **1.51** | 26.7 |
| | Kino | 30/30 | 114.0 ± 3.6 | 1.57 | 2370 | **2.08 ± 0.6** | **0.000** | 0 |
| **constrained_passage** | Vanilla | 30/30 | 151.3 | 0.08 | 282 | — † | 1.61 | 29.3 |
| | Kino | **19/30** | 121.1 | 3.39 | 6005 | 2.33 | 0.000 | 0 |
| **target_behind** | Vanilla | 30/30 | 158.8 | 0.06 | 164 | — † | 1.61 | 30.8 |
| | Kino | **0/30** | — | — | — | — | — | — |
| **cluttered** | Vanilla | 30/30 | 190.3 | 0.07 | 219 | — † | 1.61 | 37.1 |
| | Kino | 29/30 | 177.9 | 2.67 | 4988 | 1.82 | 0.000 | 0 |

† Vanilla's raw-controls `endpoint_error_mm` is dropped as meaningless; its
execution is measured by the tracker table below.

### VanillaRRT tracked execution (open-loop segment following, vanilla only)

| scenario | success | tracked_endpoint_mm | max_crosstrack_mm | **collides** |
|---|---|---|---|---|
| **open** | 30/30 | 40.8 ± 12.7 | 40.2 ± 11.7 | **24/30** |
| **constrained_passage** | 30/30 | 53.1 ± 17.8 | 53.4 ± 17.6 | **30/30** |
| **target_behind** | 30/30 | 55.0 ± 16.1 | 55.2 ± 16.0 | **30/30** |
| **cluttered** | 30/30 | 101.1 ± 35.7 | 95.7 ± 31.1 | **30/30** |

Even given the most charitable feasible execution, vanilla's path drives into a
critical structure (or off the workspace) in nearly every run — 24/30 on the
permissive `open`, 30/30 on all three harder scenarios — and strays tens of mm
from its own planned polyline en route. `collides` is the load-bearing number:
these paths are unusable irrespective of endpoint.

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
| Vanilla | 300/300 | 146.1 ± 18.0 | 0.06 | 158 | — † | 1.63 |
| Kino | 247/300 | 112.1 ± 3.0 | 2.33 | 3562 | **1.87 ± 0.7** | 0.000 |

† Vanilla tracked execution (aggregate over 300 successful runs): tracked
endpoint **44.1 ± 18.8mm**, max cross-track **43.8 ± 17.7mm**, **collides in
272/300 (91%)**.

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
planned polyline, Vanilla's **executed (open-loop tracked)** trajectory with the
first colliding state marked by an ✗, and Kino's single curve (planned =
executed). Plotted runs are the **near-median-cost** seed per (scenario, planner)
— representative, not cherry-picked: open v=1/k=0, constrained_passage v=19/k=0,
target_behind v=7/(failed-tree seed 0), cluttered v=4/k=0. Regenerate with
`scripts/plot_benchmark_figures.py`.

Four things the pictures show that the tables do not:

1. **Vanilla's tracked trajectory follows the plan but cannot hold the corners,
   and drives into tissue.** The tracked curve (red, dashed) loosely follows the
   orange planned polyline — the tracker is doing its charitable best — but the
   planned path carries 27–37 sharp heading discontinuities that a R=50mm needle
   physically cannot make (a 90° turn needs ~78mm of travel), so it cuts every
   corner and strays tens of mm. The ✗ marks where it first enters a critical
   structure, and the four figures show four distinct failure shapes:
   `target_behind` drives straight *through* the obstacle its plan detoured
   around; `constrained_passage` drifts into the *wall* instead of threading the
   16mm gap; `cluttered` plows into a vessel mid-weave; `open` (the one loose
   enough to nearly work) leaves the arena at the top. This is the visual form of
   `tracked_collides`.

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
at 27–37 nodes for Vanilla. The tracker then makes the *consequence* concrete
under the most charitable feasible execution: even when the path is reinterpreted
as followable curved controls, it **collides in 24/30–30/30 runs** (91% across
the random set). The old raw-controls endpoint (130–230mm) is deliberately gone —
it measured vanilla's storage convention, not its path; the argument now stands
on `tracked_collides` and the heading discontinuities, and is *stronger* for not
leaning on an inflated distance.

The trade the benchmark quantifies: Kino pays ~35–50× in wall-clock (mean
success time per scenario: ~34× cluttered, ~38× open, ~40× random, ~42×
constrained) and loses some scenarios to the curvature constraint, and buys
feasibility — continuous, executable-as-generated, on-target, and (surprisingly)
shorter paths.

### Vanilla's path is not merely infeasible — it is not even *informative*

There is a sharper statement hiding in *why* the tracked trajectory looks like a
near-straight line with slight bows rather than anything resembling the plan.
Vanilla's edges are **5mm long** (`v·dt·n = 5.0 · 0.05 · 20`), and 5mm of arc at
R = 50mm is only **5.7° of turn**. Because vanilla's path zigzags — each
segment's bearing differs from the last, and the sign of the required turn flips
constantly — the tracker emits **alternating short ±5.7° arcs**: b = +1, then −1,
then +1. That is *precisely the duty-cycle pattern* from the Task 1 test
(alternating ±b → κ_eff ≈ 0), so the net heading change is ≈ 0 and the executed
path stays roughly straight. Measured on the real paths: **14–27 sign flips**
per path (roughly every other segment), a **longest same-sign run of only 2–5
edges**, net heading change over the whole path of 0–58° (mostly < 30°), and an
executed trace that bows only **2–16% of its start→end chord**. It is
duty-cycling *by accident*.

The two exclusion reasons compound. The corners demand large heading changes
(reason 1: geometric infeasibility, the 27–37 discontinuities), and the segment
structure *prevents accumulating* them (reason 2: a 90° corner needs ~16
consecutive same-sign 5.7° edges, but the bearing has already flipped by
edge 3–5, giving back what little was gained). So even with **every `b` sign
chosen exactly right**, the tracker recovers a nearly straight path — no run of
same-sign steps is ever long enough to matter.

That is a stronger claim than "the tracker does poorly." It says vanilla's path
contains **almost no recoverable steering information** at this segment length
and turning radius. This closes a loop back to the planner's own design note
(`docs/roadmap.md`, Task 3): vanilla's steering lives *entirely* in the per-node
headings it synthesises and then discards. The tracker tries to recover that
information from node *positions*, and the geometry forbids it — the positions
are too closely spaced relative to the turning radius for the bearing changes
between them to be executable. Vanilla's path is therefore not just unexecutable;
it is uninformative — the needle cannot extract executable steering from it at
realistic curvature. (The edge geometry — 5mm, 5.7°/edge — is fixed by the
harness config; the sign-flip and same-sign-run counts are measured directly
from the tracked hand-designed paths.)

---

## Honesty notes / limitations

- **No true-vs-model mismatch yet.** All metrics use the *model* needle
  (κ = 1/50). Vanilla's tracked collisions are purely its point-robot planning
  cheat (it plans as if it could turn arbitrarily, then cannot execute those
  turns), not tissue mismatch. The mismatch axis (a distinct `true_needle`) is
  Phase 3, and is where the uncertainty-aware-planning sub-claim gets tested.
- **The vanilla tracker is open-loop by design.** It adds no sensing or
  correction — it reads the planned path as an *intent* and executes it blind,
  which is the fairest test of the path itself. A closed-loop pure-pursuit
  tracker would be a test of a *controller's* skill, a different question, and is
  deliberately not built.
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
