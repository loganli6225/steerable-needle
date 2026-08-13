"""Run the SECONDARY (enlarged-scale) benchmark grid and persist raw records.

120 runs (4 scaled scenarios x 3 planners x 10 seeds). Unlike the primary grid
this INCLUDES RRTStar -- the whole point is to test whether its primary-scale
exclusion is a consequence of scale (see scaled_scenarios.py). Rows are appended
per run, so this is resumable: re-run after a crash and it skips the
(scenario, planner, seed) combinations already in the CSV.

RUNTIME: ~90 min, dominated ENTIRELY by RRTStar (~130s/run x 40 runs; it never
stops early, running its full 5000-iteration budget by design). Vanilla is
~0.05s and kinodynamic ~3-15s. Run in the background.

Output: experiments/results/benchmark_scaled_raw.csv (gitignored -- regenerable).
This is a SEPARATE file from the primary benchmark_raw.csv on purpose: the two
result sets use different workspaces and answer different questions and must
never be pooled. Analysis is a separate script (scripts/analyze_scaled_benchmark.py).

Run:  python scripts/run_scaled_benchmark.py
"""

from __future__ import annotations

from needlesim.benchmark.harness_scaled import DEFAULT_CSV_PATH, run_harness


def main():
    run_harness(DEFAULT_CSV_PATH)


if __name__ == "__main__":
    main()
