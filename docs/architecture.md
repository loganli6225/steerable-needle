# Architecture

The layout enforces the one habit that keeps the benchmark/paper path open:
**everything behind clean interfaces, so any piece can be swapped without
touching the others.**

## Module map

```
src/needlesim/
  models/         Needle kinematics. step(state, control, dt, params) -> state.
                  PURE functions. The planner calls these thousands of times/sec.
  environments/   Obstacle maps + collision checking (Task 2).
  planning/       RRT, RRT*, learned planners (Task 3+). Consume models + envs.
  estimation/     EKF, particle filter (Phase 3). Consume models + sensor models.
  learning/       Learned deflection models, learned sampling (Phase 4). Torch.
  utils/          Plotting, config loading, seeding. No domain logic.
```

## Interface discipline (the five habits)

1. **Modularity from day one.** Swapping a planner must not require editing a
   model or environment file.
2. **Config-driven, seeded experiments.** Every result reproducible from a
   config file + seed. See `configs/example.yaml`.
3. **Log obsessively.** Full state traces, not just success/failure. The
   interesting findings live in the failure cases.
4. **Separate TRUE tissue from MODEL tissue.** `true_needle` and
   `model_needle` are different objects from the start (see NeedleParams).
   Model mismatch is where every interesting result comes from.
5. **Write down the research question first.** See `research_question.md`.

## Data flow (once built)

```
config ─► true_needle ─┐
                       ├─► simulator ─► noisy measurements ─► estimator ─► belief
config ─► model_needle ─┘                                                    │
                       └─► planner ◄──── replan on drift ────────────────────┘
```
