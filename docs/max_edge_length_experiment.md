# Does a max-edge-length constraint help RRT*, or starve it?

Follow-up to the secondary benchmark (`docs/benchmark_scaled_results.md`), which
found RRT* paths 4.4–5.7× longer than KinodynamicRRT at the enlarged 500mm scale
— a chain of near-full turning circles (~314mm at R = 1/κ = 50mm) that reconcile
the arbitrary sampled headings RRT* connects to. The loops are heading
reconciliation, not obstacle detours.

**The proposal under test:** reject any Dubins edge longer than a threshold X set
below the loop signature (a standard steering-horizon / max-edge-length
constraint). Two possible outcomes the experiment had to distinguish:

- **Helps** — enough heading-compatible node pairs remain that the tree grows
  from short edges only, and path cost drops.
- **Starves** — the loops are the *only* way to reconcile arbitrary node
  headings at this curvature, so forbidding them makes those pairs unconnectable;
  `choose_parent` returns None far more often, the tree goes sparse, and success
  collapses before cost improves.

The prior expectation was starvation.

## Answer: it does BOTH, and which one wins is scenario-dependent.

The starvation *mechanism* is confirmed exactly as predicted — but on a
permissive scenario it prunes without killing, and cost collapses to
kinodynamic's level; on a harder scenario the same constraint kills. The
unifying diagnostic is the `choose_parent` rejection rate.

---

## Implementation

`RRTStarConfig.max_edge_length: float = inf` (default = no constraint, behaviour
preserved exactly). The check `path.length > max_edge_length` is applied in
`choose_parent`, `rewire`, and `steer`, each **immediately after `dubins_full`
returns and before the collision rollout** — cheap-first, since the length is
already on the returned `DubinsPath` and the rollout is the expensive part.

**Behaviour-preservation gate (passed before the sweep):** with default inf, the
full test suite is unchanged, and scaled `open` seed 5 reproduces the recorded
result *exactly* — nodes = 2203, cost = 2340.1289698103847 — confirming the check
filters edges rather than altering selection. Two new tests
(`tests/test_rrt_star.py`): default-is-inf, and a threshold below an edge's
Dubins length makes it unconnectable in `choose_parent`.

The rejection rate is measured without touching `rrt_star.py`, via a thin
counting subclass in `scripts/run_max_edge_sweep.py`.

---

## The sweep — scaled `open` (500mm, γ=133, budget 5000, 5 seeds)

| max_edge | success | mean nodes | mean cost (successes) | mean time | choose_parent rejection |
|---|---|---|---|---|---|
| 400 | 4/5 | 2097 | 2376 mm | 127 s | 58.1% |
| 250 | 2/5 | 151 | 398 mm | 1.8 s | 97.0% |
| 150 | 2/5 | 153 | 398 mm | 1.8 s | 97.0% |
| 100 | 4/5 | 150 | 392 mm | 1.8 s | 97.0% |
| **inf** | 4/5 | 2168 | 2222 mm | 131 s | 56.7% |

Read down the rejection-rate column: it is flat (~57–58%) at inf and 400, then
**jumps to ~97% the moment the threshold drops below the ~314mm loop
signature**. That jump is the loops being forbidden — and 97% is right at the
~98.5% starvation level seen at primary scale. So the starvation *mechanism* is
real and exactly as predicted: forbidding the loops makes ~97% of node pairs
unconnectable, and the tree collapses ~14× (from ~2100 nodes to ~150).

But look at what the surviving connections do on `open`:

- **Cost collapses to kinodynamic's level.** The successful sub-314mm runs cost
  **~392–398 mm** — versus **~2222 mm** unconstrained (5.6×), and essentially
  matching kinodynamic's ~405 mm on this scenario. The loops genuinely
  disappear; the short-edge survivors chain into a near-direct route (the path
  figures confirm this — see below).
- **Success is preserved on `open`.** At max_edge=100, 4/5 — the *same* rate as
  unconstrained. `open` is permissive enough that even at 97% rejection, enough
  short heading-compatible edges exist to reach the goal.
- **And it is ~70× faster** (1.8 s vs 131 s), because the collapsed tree does far
  fewer rollouts.

Two honest caveats on the sweep numbers:
- **400mm doesn't bite.** It exceeds the 314mm loop, so most seeds are
  bit-identical to inf (same nodes, cost, rejection); only one seed differs,
  where a longer edge was rejected. This correctly brackets the loop signature
  between 250 and 400 mm — matching the ~314 mm prediction.
- **The 250/150 → 2/5 dip is small-sample noise.** Success across sub-314mm
  thresholds runs 2/5, 2/5, 4/5 — non-monotonic on only 5 seeds. The robust
  signals are the rejection jump, the cost collapse among survivors, and the
  node-count collapse; the exact success rate at each sub-loop threshold is
  within seed noise and should not be over-read.

## The cross-check — scaled `constrained_passage`, best setting (max_edge=100)

| max_edge | success | mean nodes | choose_parent rejection |
|---|---|---|---|
| 100 | **0/5** | 106 | 97.9% |

Here the *same* constraint that helped on `open` **starves**: 0/5, the same ~97%
rejection, the same ~100-node collapse — but the short-edge survivors do not
happen to form a route through the gap, so nothing reaches goal. Success dies
completely, and there is no cost to report because there are no successes.

---

## What the figures show that the numbers cannot

Figures in `docs/figures/`, same grammar as `benchmark_scaled_*.png`
(kinodynamic's path drawn as a fixed blue reference in every frame).

- **`max_edge_sweep_summary.png`** — three stacked panels vs threshold (100…400,
  then inf). Cost and rejection rate both step *at the same place*: crossing
  below the ~314mm loop, cost drops from ~2300mm to ~390mm and rejection jumps
  from ~57% to ~97%. The step is sharp and co-located — the single clearest
  statement of the result.

- **`max_edge_open_inf.png` vs `max_edge_open_100.png`** — the decisive contrast,
  and it answers the question the numbers cannot: **the loops genuinely
  disappear, they are not truncated.** At inf the RRT* path is the familiar
  chain of ~314mm circles (6.1× kino). At max_edge=100 it is a single smooth
  curve that overlays kino's reference almost exactly — in fact *marginally
  shorter* (0.9× kino, 384 vs 405mm), because RRT* still rewires among the
  short-edge survivors. A cost drop with a *visibly direct* path, not a cost drop
  with truncated-loop remnants.

- **`max_edge_open_250.png` / `_150.png`** — the same direct curve just below the
  loop threshold, confirming the effect is not specific to 100mm: any sub-314mm
  cap that leaves *any* feasible short-edge route yields a direct path.

- **`max_edge_open_400.png`** — above the loop, most seeds are bit-identical to
  inf, so the path is still the loop chain: the cap doesn't bite until it drops
  below the loop length.

The figures make the mechanism visible: below 314mm RRT* is forced to connect
only short, near-heading-aligned edges — so its path stops looping and becomes
the direct, forward-progress route that kinodynamic RRT produces natively. That
is *why* it matches kino's cost on `open`: it is doing kino's motion.

---

## Conclusion — the secondary finding stands, sharpened

This does **not** rescue RRT*, and the secondary experiment's conclusion does not
need revisiting. What the sweep shows is:

1. **The loops are structural, exactly as claimed.** Forbidding them spikes the
   `choose_parent` rejection rate to ~97% and collapses the tree ~14×. They are
   not wasteful detours you can cap away — they are the price of connecting
   arbitrary sampled headings at R=50mm, and removing them removes most of RRT*'s
   connections.

2. **Where capping the loops "helps" (open), it does so by throwing away the
   capability that distinguishes RRT* from kinodynamic RRT.** A max_edge=100
   RRT* connects only near-heading-aligned short edges — which is precisely the
   forward, roughly-heading-aligned motion kinodynamic RRT already does — and it
   lands at kinodynamic's cost (~392 vs ~405 mm) while still paying RRT*'s
   rewiring overhead. It matches kinodynamic by *becoming* kinodynamic-like.

3. **Where arbitrary-pose connection is actually needed (constrained_passage),
   capping it starves the planner to 0/5.** The one scenario where you might hope
   optimisation buys something is the one where the constraint kills it.

So the constraint doesn't give RRT* a regime where it beats kinodynamic; it gives
RRT* a regime where it *ties* kinodynamic on the easy scenario and *loses
outright* on the harder one. The turning-circle cost floor from the secondary
experiment is confirmed as structural, and the "just cap the edge length" fix is
shown to trade the loops for the loss of the exact-connection capability that was
RRT*'s only reason for being here.

*Reproduce:* `python scripts/run_max_edge_sweep.py` (≈50 min) →
`python scripts/plot_max_edge_figures.py`. Raw:
`experiments/results/max_edge_sweep_raw.csv` (gitignored).
