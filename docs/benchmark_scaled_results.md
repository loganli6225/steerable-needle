# Secondary benchmark: all three planners at enlarged scale

This is the SECONDARY experiment named in Task 3.6's revised scope. The primary
benchmark (`docs/benchmark_results.md`) excludes `RRTStar` because at realistic
curvature (R = 1/kappa = 50mm) in the 150mm anatomical workspace it cannot
connect poses — 155 nodes in 10,000 iterations, never reaching the goal, a
recorded structural finding. This experiment asks whether that exclusion is a
consequence of **scale** rather than of the algorithm being useless: enlarge the
scene ×10/3 (150 → 500mm) while keeping the needle (kappa, edge length) fixed,
so R = 50mm becomes 10 turning radii across the workspace instead of 3, and see
what all three planners do.

It tests the **strong** version of the claim, not the weak one. Weak: "RRT*
works if you make the workspace bigger" — only shows the implementation isn't
broken. Strong, as originally hypothesised:

> In a regime where all three planners function, RRT*'s optimisation buys
> measurably better paths than KinodynamicRRT's first-found path.

**The strong version is falsified by the data below.** RRT* *functions* at this
scale, but its paths are ~6× longer than kinodynamic's, at ~40× the wall time.
Enlarging the workspace let RRT*'s Dubins loops *fit without colliding* (so it
succeeds) but did nothing to *shorten* them. That is a sharper result than the
one hypothesised: RRT* is unsuited to this problem not only because it can't
connect at anatomical scale, but because even where it can, its path quality is
dominated by the minimum-turning-circle cost floor.

Results here must NEVER be pooled with the primary benchmark — different
workspace, different question. Separate module (`benchmark/scaled_scenarios.py`,
`benchmark/harness_scaled.py`), separate CSV
(`experiments/results/benchmark_scaled_raw.csv`).

---

## Scale

Everything geometric scales ×10/3; the needle does not.

| quantity | primary | scaled | scales? |
|---|---|---|---|
| workspace | 150×150mm | 500×500mm | yes |
| obstacles, start, goal | — | ×10/3 | yes |
| goal_tolerance | 3.0mm | 10.0mm | yes (keeps goal region's relative size) |
| resolution | 0.5mm | 1.667mm | yes (see below) |
| **margin** | 2.0mm | **2.0mm** | **no — needle physical width** |
| **kappa** | 1/50 | **1/50** | **no — same needle** |
| **n_steps_per_extend** (edge = 5mm) | 20 | **20** | **no — see argument** |

**Resolution scaled ×10/3 (0.5 → 1.667mm)** so the grid stays 300×300 cells —
bake time and memory match the primary benchmark exactly rather than exploding
to 1000×1000 at a fixed 0.5mm. Coarser resolution makes the `is_arc_free`
spacing invariant *easier*, not harder: `edge_velocity·step_dt = 0.25 ≤
0.5·1.667 = 0.833`. Collision granularity stays a fixed fraction of scene
features, so fidelity relative to the obstacles is preserved.

**margin NOT scaled.** It represents the needle's physical width, which does not
grow with the workspace. It is the one quantity that must stay fixed.

**n_steps_per_extend NOT scaled.** Edge length = `v·dt·n` = 5mm and turning
radius R = 50mm are both unchanged, so the edge/R = 0.1 design ratio the roadmap
is explicit about is preserved. Scaling `n` would change that ratio and alter
the kinodynamic planner's character mid-experiment. The larger workspace is paid
for in iteration budget only. (In a 500mm workspace kinodynamic needs ~100 5mm
edges just to cross — hence the raised budget below.)

---

## Calibration (the two unknowns, measured before the grid)

Run on scaled `open`. Throwaway calibration script; numbers reproduced here.

### Unknown 1 — KinodynamicRRT budget: 20000 is enough

| seed | success | iters | time |
|---|---|---|---|
| 0 | ✓ | 2479 | 2.2s |
| 1 | ✓ | 4198 | 3.1s |
| 2 | ✓ | 4295 | 2.9s |
| 3 | ✓ | 437 | 0.3s |
| 4 | ✓ | 10482 | 7.4s |

Worst calibrated seed needed 10482 iterations; a 20000 budget leaves headroom.
Kinodynamic stops at first goal contact, so this is a ceiling, not a fixed cost.

### Unknown 2 — RRT* budget + the gamma finding

| gamma | max_radius | budget | success | nodes | best_cost | time |
|---|---|---|---|---|---|---|
| 40 (default) | 40 | 2000 | ✗ | 47 | inf | 1.3s |
| 40 | 40 | 5000 | ✗ | 154 | inf | 5.3s |
| 40 | 40 | 10000 | ✗ | 371 | inf | 14.2s |
| **133 (×10/3)** | 133 | 2000 | ✗ | 594 | inf | 31.6s |
| **133** | 133 | **5000** | **✓** | 1834 | **2515.6** | **128s** |
| **133** | 133 | 10000 | ✓ | 4325 | 2515.6 | 379s |

Two things fall out:

1. **The default gamma collapses RRT* completely** — 0 successes at any budget.
   The rewire radius `min(max_radius, gamma·(log n / n)^(1/3))` is in **mm**.
   Enlarging the workspace ×10/3 spreads the same node count over ~11× the area,
   so inter-node spacing grows ~10/3. At gamma=40 the neighbourhoods shrink
   below what is needed to ever connect near the goal, and RRT* degenerates.
   Scaling gamma/max_radius ×10/3 → 133 is the **scale-consistent** value that
   restores its neighbourhood cardinality — not tuning-for-numbers, the same
   ×10/3 the geometry gets.

2. **Cost plateaus by 5000 iterations** — identical `best_cost` (2515.6) at 5000
   and 10000. So the grid budget is 5000; 10000 is 3× the compute for zero gain.

RRT* path validated (gamma 133, budget 5000, seed 1): endpoint error after
executing the returned controls on the real needle = **0.002mm** (tol 10mm), and
heading discontinuity at interior nodes = **1.2e-13 rad** (~0, as expected — its
nodes are Dubins-path endpoints from one continuous integration). The 2515mm is
real, not a metric bug: an 11-node path whose 10 edges each average ~250mm of
looping Dubins arc.

---

## The grid

4 scaled scenarios × 3 planners × 10 seeds = 120 runs. Same metrics and
raw-CSV-first, resumable, one-row-per-run structure as the primary harness;
RRT-family rows reuse the primary metric code, RRT* rows use three small
adapters (`control_dt_pairs` → `rollout_variable` for endpoint error;
`best_cost` for path cost; node-identity map for heading discontinuity).

Budgets are per-planner and **reported, not equalised**: RRT family 20000, RRT*
5000 (gamma 133). Fixed iterations with wall time reported lets a reader make the
equal-compute comparison themselves and stays reproducible across machines.

### Results (120 runs, `experiments/results/benchmark_scaled_raw.csv`)

Mean ± std over **successful** runs. `endpt_mm` = distance from goal when the
returned controls are executed on the real needle; `hdisc` = max heading
discontinuity at interior nodes (the feasibility measure).

| scenario | planner | succ | cost_mm | time_s | endpt_mm | hdisc_max |
|---|---|---|---|---|---|---|
| open | Vanilla | 10/10 | 460 ± 31 | 0.14 | **368 ± 28** | **1.70** |
| open | Kinodynamic | 10/10 | 413 ± 33 | 3.3 | 8.1 ± 1.4 | 0.000 |
| open | RRTStar | 9/10 | **2242 ± 585** | 133 | 1.3 ± 2.6 | 0.000 |
| constrained_passage | Vanilla | 10/10 | 486 ± 30 | 0.23 | **430 ± 19** | **1.61** |
| constrained_passage | Kinodynamic | 10/10 | 430 ± 55 | 2.9 | 8.2 ± 1.3 | 0.000 |
| constrained_passage | RRTStar | **2/10** | **2871 ± 470** | 133 | 8.6 ± 1.5 | 0.000 |
| target_behind | Vanilla | 10/10 | 490 ± 20 | 0.15 | **404 ± 15** | **1.64** |
| target_behind | Kinodynamic | **10/10** | 520 ± 44 | 0.56 | 8.2 ± 1.2 | 0.000 |
| target_behind | RRTStar | **5/10** | **3132 ± 884** | 148 | 3.2 ± 4.4 | 0.000 |
| cluttered | Vanilla | 10/10 | 622 ± 36 | 0.24 | **498 ± 28** | **1.71** |
| cluttered | Kinodynamic | **10/10** | 620 ± 69 | 0.86 | 7.8 ± 1.5 | 0.000 |
| cluttered | RRTStar | **0/10** | — | — | — | — |

### Headline: RRT* vs Kinodynamic path cost, where BOTH succeed

| scenario | kino cost | RRT* cost | cost diff | kino time | RRT* time | time × |
|---|---|---|---|---|---|---|
| open | 413mm | 2242mm | **+443%** | 3.3s | 133s | 40× |
| constrained_passage | 430mm | 2871mm | **+568%** | 2.9s | 133s | 47× |
| target_behind | 520mm | 3132mm | **+502%** | 0.56s | 148s | 263× |
| cluttered | — | — | (RRT* 0/10) | — | — | — |

**RRT* is 4.4–5.7× longer and 40–263× slower than kinodynamic on every
scenario where both succeed** — the exact inversion of the strong hypothesis.
The mechanism, unchanged by enlargement: RRT* assembles its path from exact
Dubins connections *through sampled intermediate poses*, and at R = 50mm each
such connection loops (minimum turning circle 314mm around). Enlarging the
workspace let those loops *fit without colliding* (RRT* now reaches the goal
with continuous, on-target paths — hdisc 0.000, endpoint error within
tolerance) but did nothing to *shorten* them.

### RRT* reliability degrades with obstacle density, kinodynamic does not

Even at a scale where RRT* functions, its success rate falls off a cliff as the
scene gets harder — 9/10 (open) → 5/10 (target_behind) → 2/10
(constrained_passage) → **0/10 (cluttered)** — because its long looping edges
sweep into obstacles. Kinodynamic is **10/10 on all four** at this scale: its
forward, roughly-heading-aligned extends stay local and thread clutter that
RRT*'s loops cannot.

### target_behind: solved at scale (by BOTH kinodynamic and RRT*)

The task asked specifically whether `target_behind` — kinodynamic 0/30 at the
primary scale (symmetric split-effort between two equal detours) — is solved
here. **Yes, and by kinodynamic too:** kinodynamic 10/10, RRT* 5/10. The
symmetric-split failure was *scale-dependent* — at 500mm the workspace is roomy
enough that kinodynamic commits to a detour and completes within budget rather
than exhausting it between two equal-cost routes. So `target_behind` is no
longer a discriminating PASS/FAIL scenario at this scale; it is one more
scenario kinodynamic solves cheaply (0.56s) and RRT* solves expensively and
unreliably (148s, 5/10).

### Vanilla, unchanged: fast, "successful," geometrically infeasible

Vanilla is 10/10 planning-successes everywhere at ~0.2s, but every path is a
point-robot fiction: executed on the real needle it lands **368–498mm off in a
500mm workspace** (it leaves the arena) with **order-1-radian heading
discontinuities** at every node. This is the same finding as the primary
benchmark, and the enlarged endpoint errors make it starker.

---

## Conclusion

The secondary experiment confirms RRT* is **implementable and correct** — given
a scale-consistent rewire radius (gamma ×10/3) it reaches goals with continuous,
on-target paths — so the primary benchmark's exclusion is genuinely about
**scale**, not a broken planner. But it **falsifies** the strong hypothesis that
RRT*'s optimisation would buy better paths if only it could apply. Where RRT*
functions it is **5–6× longer and 40–263× slower** than kinodynamic, and its
reliability collapses to 0/10 on clutter that kinodynamic threads 10/10.

That sharpens rather than merely defends the primary result. RRT* is unsuited to
this problem on **two** independent grounds: at anatomical scale it cannot
connect poses at all (primary finding), and even at a scale where it can, its
path quality is dominated by the minimum-turning-circle cost floor and its
robustness by the same loops colliding. The right planner for a forward,
monotonic, curvature-constrained insertion is the forward, monotonic
kinodynamic one — at both scales.

*Reproduce:* `python scripts/run_scaled_benchmark.py` (≈90 min, resumable) →
`python scripts/analyze_scaled_benchmark.py`.

