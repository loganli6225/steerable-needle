"""Generate the 30 random benchmark scenarios ONCE and persist them.

The benchmark must be re-runnable against an IDENTICAL scenario set even if the
generator code later changes, so the set is generated from a recorded seed and
written to JSON rather than regenerated at run time. Re-running this script
with the same seed reproduces byte-identical output.

Output: experiments/random_scenarios.json (tracked -- it is small and is the
experimental record, unlike the gitignored regenerable figures under
experiments/results/).

Run:  python scripts/generate_random_scenarios.py
"""

from __future__ import annotations

from pathlib import Path

from needlesim.benchmark.random_scenarios import (
    DEFAULT_SEED,
    N_SCENARIOS,
    generate_scenarios,
    save_scenarios,
)

OUT_PATH = Path("experiments/random_scenarios.json")


def main():
    scenarios = generate_scenarios(seed=DEFAULT_SEED, n=N_SCENARIOS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(OUT_PATH, scenarios, seed=DEFAULT_SEED)

    counts = [len(s.obstacles) for s in scenarios]
    print(f"Generated {len(scenarios)} scenarios (seed {DEFAULT_SEED}) -> {OUT_PATH}")
    print(
        f"obstacle counts: min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.1f}"
    )
    n_circ = sum(
        1 for s in scenarios for o in s.obstacles if type(o).__name__ == "Circle"
    )
    n_rect = sum(
        1 for s in scenarios for o in s.obstacles if type(o).__name__ == "Rect"
    )
    print(f"obstacles total: {n_circ + n_rect}  (circles {n_circ}, rects {n_rect})")


if __name__ == "__main__":
    main()
