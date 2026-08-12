# Roadmap update: full Dubins steering + the RRT* scaling finding

Update `docs/roadmap.md` only. No code changes. Show me a diff of just the
sections you touch before writing anything.

There are three things to record: what was built, why it was built, and the
finding that came out of it. The third is the important one — it changes the
benchmark's scope, so write it as a result, not as a footnote.

## 1. New entry: Task 3.6 — full Dubins (CSC + CCC) steering

`src/needlesim/planning/dubins.py` gained the four CSC words (LSL, RSR, LSR,
RSL) alongside the existing CCC pair (RLR, LRL), plus `dubins_full`, which
returns the shortest over all six. `dubins_ccc` is retained unchanged and
`RRTStarConfig.use_full_dubins` switches between them, so the CCC-only
behaviour stays reproducible rather than being deleted.

Why straight segments were added after Task 3.5 deliberately excluded them:
CCC-only assumes the needle always curves, which looked like the honest model.
At the fictional curvature used for early testing (R=5mm in a 100mm world) it
worked. At realistic curvature it collapsed. The resolution is physical, not a
fudge: the needle can travel effectively straight by duty cycling
(kappa_eff = kappa*(2p-1), so p=0.5 is a straight centerline), a capability
already verified in `tests/test_needle_model.py::test_duty_cycle_scales_curvature`.
Excluding S segments made the planner more restrictive than the hardware.

Modelling assumption to state in any writeup: an S segment represents
duty-cycled motion at p=0.5 in the idealised (infinite cycling frequency)
limit. The true tip path is a zigzag about the straight centerline. Measured
at v=5, dt=0.05, kappa=1/50 over a 100mm segment: max perpendicular deviation
0.0003mm, endpoint shortfall 0.0001mm, heading error exactly 0, arc length
exact. (The naive +1/-1 alternation without half-steps at each end bows
one-sided at 0.25mm — 800x worse; the half-step offsets are what centre the
oscillation on the line.) The 0.0003mm deviation is 0.015% of the 2mm planning
margin, so the idealisation is effectively free at these parameters.

## 2. The finding — write this as a result, it changes the benchmark's scope

Full Dubins fixed reachability but NOT tractability, and the distinction is
the point.

Measured on the physically-grounded scenario (R = 1/kappa = 50mm, 150x150mm
workspace, r=20mm obstacle, margin 2mm — a needle curvature at the optimistic
end of the literature's 40-170mm range, in an anatomically-scaled workspace):

- RRT* built 155 nodes in 10,000 iterations, ~98.5% sample rejection, and
  failed to reach the goal. Identical figures with CCC-only and with full
  Dubins — the word family was never the binding constraint.
- Root cause, measured directly: connecting two poses 10-25mm apart at R=50mm
  costs a median of ~458mm (heading derived from nearest-node-toward-sample,
  which is what the planner actually does) or ~332mm (headings matched).
  Only 3-4% of local connections come in under 60mm. Random headings: 0%.
  The needle's minimum turning circle is ~314mm in circumference, so small
  heading corrections require most of a loop. Those loops sweep the whole
  workspace and collide.
- Heading-derivation strategy was tested and ruled out as the cause: no
  strategy brings local connection lengths into a usable range.
- KinodynamicRRT succeeded on the same scenario (1883 nodes, 3005 iterations).
  VanillaRRT succeeded trivially (103 nodes, 120 iterations).

Interpretation to record: this is a STRUCTURAL mismatch, not an implementation
problem. RRT* requires exact pose-to-pose connection because rewiring is
defined in terms of it — to ask whether node X is cheaper through the new
node, you must connect new-node to X exactly. Kinodynamic RRT never makes that
demand: it advances the needle forward under b=+/-1 and accepts where it
lands. Real needle insertion is monotonic and forward-directed with roughly
aligned heading, so the arbitrary pose-to-pose connections RRT* needs do not
arise clinically. RRT* therefore requires a capability the task does not need
and the hardware cannot provide at anatomical scale.

## 3. Benchmark scope change (record under future work / next task)

The benchmark is now scoped as:
- PRIMARY: VanillaRRT vs KinodynamicRRT on the 150mm anatomically-scaled
  scenario at realistic curvature, with the RRT* limitation above reported as
  a finding.
- SECONDARY: RRT* at an enlarged workspace (roughly 500mm, where the turning
  radius is small relative to scene features) to demonstrate it functions when
  the scaling permits, and to compare path quality against the others there.

Also note as an open question, not a task: the current 150mm scenario is
solved by VanillaRRT in ~120 iterations, so it is too easy to discriminate
planners on efficiency alone. A constrained-passage scenario (the doorway
shape used earlier in Task 3) is likely needed so vanilla's speed advantage
does not dominate the table, and so the path-feasibility metric has something
to bite on.

## 4. General lesson worth one line

Third time curvature-vs-scene scaling has been the hidden cause of a planner
failure (RRT* looping at R=20/50 in Task 4, the 8mm doorway, now this). The
governing ratios: edge length must be small relative to turning radius, and
turning radius small relative to scene features. Violate either and the
planner degrades in a way that looks like a bug.

Keep it concise and in the roadmap's existing voice. Do not touch any other
file.
