# CLAUDE.md

## Project

`needlesim` — simulation testbed for autonomous bevel-tip steerable-needle
navigation. Integrates planning, estimation, and learned models for
reproducible benchmarking. Personal research project (not coursework).

## Commands

- Install dev env: `pip install -e ".[dev]"`
- Run tests: `pytest -q`
- Run a single test: `pytest tests/test_needle_model.py::test_duty_cycle_halves_curvature -v`
- Generate Task 1 figures: `python scripts/demo_needle_model.py`
- Lint: `ruff check src tests`
- Format: `black src tests`

Run `pytest -q` before every commit.

## Architecture

See `docs/architecture.md` for the full module map. Summary:

- `src/needlesim/models/` — needle kinematics. `step(state, control, dt, params) -> State`.
  MUST be pure: no mutation, no side effects, no plotting. The planner calls
  it thousands of times per second on hypothetical states.
- `src/needlesim/environments/` — obstacle maps + collision checking.
- `src/needlesim/planning/` — RRT, RRT*, learned planners.
- `src/needlesim/estimation/` — EKF, particle filter.
- `src/needlesim/learning/` — learned deflection models (torch).
- `src/needlesim/utils/` — plotting, config, seeding. No domain logic.

## Non-negotiable conventions

1. **Keep `step` pure.** Never make it stateful or give it a plotting side effect.
2. **True vs model separation.** `true_needle` params (simulator ground truth)
   and `model_needle` params (what the planner/filter believe) are ALWAYS
   distinct objects. Never collapse them into one, even when their values are
   identical. Model mismatch is where every interesting result comes from.
3. **Config-driven + seeded.** Every experiment reproducible from a config file
   + seed. No magic numbers in notebooks. See `configs/example.yaml`.
4. **Log full state traces**, not just success/failure. Findings live in the
   failure cases.
5. **Modularity.** Swapping a planner must not require editing a model or
   environment file.

## Working agreement with Claude

This is a portfolio project. I need to be able to explain and defend every
line in an interview or a grad school conversation.

- **Do NOT write the needle kinematics or core algorithms for me.** That
  includes `models/unicycle_needle.py::step`, the planners, and the filters.
  These are the intellectually load-bearing parts and I am writing them myself.
- **Instead:** review my code, run the tests, explain failures, ask leading
  questions, point at the bug's neighborhood without fixing it.
- **Do freely write:** plotting utilities, config loading, test scaffolding,
  benchmark harness, docs, refactors I direct.
- If I ask you to write core logic anyway, push back once and make me confirm.
- Explain the "why," not just the fix. If my integrator is wrong, tell me which
  property is violated and let me find it.

## Current status

Phase 0 complete (scaffolding). Task 1 in progress: implement `step` with RK4
and get the three acceptance tests in `tests/test_needle_model.py` green.

The duty-cycle test is the important one: effective curvature ~kappa/2 must
EMERGE from flipping the bevel direction, not be hardcoded.

## Reference

- Webster et al., "Nonholonomic Modeling of Needle Steering," IJRR 2006.
- Thrun, Burgard, Fox, *Probabilistic Robotics*, MIT Press 2005.
- See `docs/research_question.md` for the framing this project is pointed at.