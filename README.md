# needlesim — autonomous steerable-needle navigation testbed

A simulation testbed for **autonomous bevel-tip needle steering**, integrating
the full **planning → estimation → learning** loop, built for systematic,
reproducible benchmarking of classical and learned methods.

> **What this is:** a rigorous integration and evaluation of known
> needle-steering methods in one consistent framework, with head-to-head
> benchmarking on image-derived environments. It is *not* a novel steering
> algorithm — the core problem has ~20 years of literature (Webster, Cowan,
> Alterovitz, Okamura, and others). The contribution is the unified testbed
> and the comparative analysis it enables.

## Motivation

Flexible bevel-tip needles can curve around critical structures (vessels,
nerves, bone) to reach targets a rigid needle cannot — relevant to biopsy,
brachytherapy, ablation, and neurosurgery. But steering one by hand is
effectively impossible: the trajectory depends on unseen tissue properties and
the tip can't be directly observed. That makes it a robotics problem requiring
planning, estimation, and control together. This repo is a testbed for exactly
that problem.

## Roadmap

- [ ] **Task 1 — needle model.** 2D nonholonomic bevel-tip kinematics + visualizer.
- [ ] **Task 2 — environments.** Obstacle maps (incl. image-derived) + collision checking.
- [ ] **Task 3 — planning.** RRT → kinodynamic RRT → RRT*, benchmarked.
- [ ] **Phase 3 — estimation.** EKF and particle filter for tip tracking; closed-loop replanning.
- [ ] **Phase 4 — learning.** Learned deflection model / learned sampling; uncertainty-aware planning.

## Quickstart

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the package (editable) with dev tools
pip install -e ".[dev]"

# 3. run the tests (Task 1 tests fail until you implement the model — that's expected)
pytest -q

# 4. once the model is implemented, generate the acceptance figures
python scripts/demo_needle_model.py
```

## Layout

See [`docs/architecture.md`](docs/architecture.md) for the module map and the
interface discipline that keeps methods swappable. The current research
framing lives in [`docs/research_question.md`](docs/research_question.md).

## References

- R. J. Webster III et al., "Nonholonomic Modeling of Needle Steering," *IJRR*, 2006.
- N. J. Cowan et al., "Robotic Needle Steering: Design, Modeling, Planning, and Image Guidance," 2011.
- R. Alterovitz et al., motion planning for steerable needles.
- S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.

## License

MIT — see [LICENSE](LICENSE).
