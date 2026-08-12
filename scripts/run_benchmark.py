"""Run the full benchmark grid and persist raw per-run records.

840 runs (4 hand-designed x 2 planners x 30 seeds + 30 random x 2 planners x
10 seeds). Rows are appended as each run completes, so this is resumable: re-run
after a crash and it skips the (scenario, planner, seed) combinations already
in the CSV.

RUNTIME: ~30-60 min. Vanilla runs are ~0.05-0.5s; kinodynamic is ~1-4s when it
solves and up to ~10-15s on a full-budget failure (target_behind is 0/10 by
design, so 30 of those are full-budget). Run in the background.

Output: experiments/results/benchmark_raw.csv (gitignored -- regenerable raw
data). Analysis is a separate script (scripts/analyze_benchmark.py).

Run:  python scripts/run_benchmark.py
"""

from __future__ import annotations

from needlesim.benchmark.harness import DEFAULT_CSV_PATH, run_harness


def main():
    run_harness(DEFAULT_CSV_PATH)


if __name__ == "__main__":
    main()
