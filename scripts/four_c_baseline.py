"""Phase 4c baseline: the five-way against the field-carrying simulator.

Nothing had been run against Phase 4b's tissue field except its own integration
tests. This is the run that produces the number 4c must beat, plus the two
diagnostics that are meant to shape what 4c fits:

  1. MEASUREMENTS PER LAYER. With ~0.25mm/step and a measurement every 20 steps
     (~5mm of insertion) the 5mm capsule is expected to get ~1 observation --
     which would make the highest-contrast layer close to unobservable at this
     imaging rate. This counts the actual EKF updates that land in each layer
     (via TissueField.layer_at on the true state at each measurement step); it
     does not assume the prediction.

  2. KAPPA-HAT VS THE TRUE FIELD. There is no true constant any more, so the
     question is what the scalar (augmented) estimator settles on -- presumably
     some effective average -- and whether that value is sensible. Plotted
     against depth for the two adaptive conditions.

  3. PROCESS-NOISE SWEEP. With process_noise_std[3]=1e-5 the augmented filter
     is built to CONVERGE to a fixed value, the wrong estimator for a varying
     field. Raising it (1e-4, 1e-3, 1e-2) lets kappa-hat TRACK. Expect a
     lag/variance tradeoff, not a clean winner: low noise -> smooth, lagged,
     washed toward an average; high noise -> responsive but fits measurement
     noise. Reported, not adjudicated.

HEADLINE: how much worse is the scalar-believing stack against a FIELD than
against a constant-kappa mismatch? The constant-world table is printed too, so
the cost of spatial variation reads as a difference rather than an absolute.

Harness/script only -- no planner, filter, model or control logic is touched.
The field reaches only the simulator, through each run's true_params argument.
Everything here is reconstructed from the returned result objects
(executed_states, kappa_estimates), which is why no module needed changing.

Run:  python scripts/four_c_baseline.py
Figures -> docs/figures/four_c_baseline_*.png  (NOT committed)
"""

from __future__ import annotations

import statistics
from collections import Counter

import matplotlib.pyplot as plt
from five_way_comparison import (
    CONSTANT_WORLD,
    FIELD_WORLD,
    adaptive,
    emit_table,
)

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN
from needlesim.control.adaptive_loop import ReplanTrigger
from needlesim.estimation.ekf_augmented import AugmentedEKFConfig
from needlesim.models.tissue_field import TissueField

FIELD = TissueField()
INTERVAL = AugmentedEKFConfig().measurement_interval  # 20
LAYER_ORDER = ("fat", "muscle", "capsule", "gland")
NOISE_LEVELS = (1e-5, 1e-4, 1e-3, 1e-2)
SEEDS = range(1, 6)
FIGDIR = "docs/figures"


# --- instrumentation reconstructed from result objects ----------------------


def measurements_per_layer(result) -> Counter:
    """Count EKF updates (measurement steps) that landed in each layer.

    A measurement fires at global step k iff k % INTERVAL == 0, and
    executed_states[k] is the true state at step k -- so this reproduces the
    loop's measurement schedule exactly from the returned trace."""
    counts = Counter()
    for k, s in enumerate(result.executed_states):
        if k % INTERVAL == 0:
            counts[FIELD.layer_at(s.x, s.y)] += 1
    return counts


def field_tracking_error(result) -> float:
    """Mean absolute RELATIVE error between kappa-hat and the local true field,
    over every step. Relative (dimensionless) so it is comparable across layers
    whose kappa differs 2x. Adaptive results only (needs kappa_estimates)."""
    errs = []
    for s, khat in zip(result.executed_states, result.kappa_estimates):
        ktrue = FIELD.kappa_at(s.x, s.y)
        errs.append(abs(khat - ktrue) / ktrue)
    return statistics.mean(errs) if errs else float("nan")


# --- reports ----------------------------------------------------------------


def report_measurements_per_layer(field_results):
    """Per-layer measurement counts, field world, summed over seeds, for the
    learn/kappa condition (index 3). Trajectories differ slightly by condition
    but the depth traversal -- what sets these counts -- does not."""
    print("\n\n########## MEASUREMENTS PER LAYER (field world, learn/kappa) ##########")
    print("EKF updates landing in each layer; summed over 5 seeds, per-seed range.\n")
    header = f"{'scenario':>20} " + " ".join(f"{name:>9}" for name in LAYER_ORDER)
    print(header)
    print("-" * len(header))
    for scenario in (CONSTRAINED_PASSAGE, OPEN):
        per_seed = [
            measurements_per_layer(field_results[(scenario.name, seed)][3])
            for seed in SEEDS
        ]
        totals = {L: sum(c.get(L, 0) for c in per_seed) for L in LAYER_ORDER}
        ranges = {
            L: (min(c.get(L, 0) for c in per_seed), max(c.get(L, 0) for c in per_seed))
            for L in LAYER_ORDER
        }
        row = f"{scenario.name:>20} " + " ".join(
            f"{totals[L]:>9d}" for L in LAYER_ORDER
        )
        print(row)
        rng = f"{'(per-seed min-max)':>20} " + " ".join(
            f"{ranges[L][0]}-{ranges[L][1]:>7}" for L in LAYER_ORDER
        )
        print(rng)
    print(
        "\nCapsule (65-70mm, 5mm thick) vs a measurement every ~5mm: if the count "
        "is ~1/seed,\nthe highest-contrast layer is effectively unobservable at "
        "this imaging rate."
    )


def plot_kappa_hat_vs_field(field_results, seed=1):
    """kappa-hat vs the true field against depth, both scenarios x {drift,kappa}."""
    conditions = [("drift", 2), ("kappa", 3)]  # result-tuple indices
    scenarios = [CONSTRAINED_PASSAGE, OPEN]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for i, scenario in enumerate(scenarios):
        for j, (cname, idx) in enumerate(conditions):
            ax = axes[i][j]
            r = field_results[(scenario.name, seed)][idx]
            depth = [s.y for s in r.executed_states]
            khat = r.kappa_estimates
            ktrue = [FIELD.kappa_at(s.x, s.y) for s in r.executed_states]
            ax.plot(depth, ktrue, color="black", lw=2, label="true field kappa(y)")
            ax.plot(depth, khat, color="tab:red", lw=1.3, label="kappa-hat (filter)")
            for edge in (45.0, 65.0, 70.0):
                ax.axvline(edge, color="gray", ls=":", lw=0.8)
            ax.set_title(f"{scenario.name} | learn/{cname} | seed {seed}")
            ax.set_ylabel("kappa [1/mm]")
            if i == 1:
                ax.set_xlabel("depth y [mm]")
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Phase 4c baseline: scalar augmented estimator vs the true tissue field",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = f"{FIGDIR}/four_c_baseline_kappa_hat_vs_field.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n[figure] {path}")


def process_noise_sweep():
    """learn/kappa in the field world at rising log-kappa process noise.

    Reports median final error and mean field-tracking error per level, and a
    figure of kappa-hat vs depth at each level (constrained_passage, seed 1)."""
    print("\n\n########## PROCESS-NOISE SWEEP (field world, learn/kappa) ##########")
    print(
        "process_noise_std[3] = log-kappa random-walk std. 1e-5 = the 4a "
        "converge-to-constant\nsetting; higher = more tracking. Final error in "
        "mm (median/5 seeds); field-track = mean\nabsolute relative kappa error "
        "vs the local field (lower = follows the field better).\n"
    )
    header = (
        f"{'scenario':>20} {'noise':>8} {'median_err_mm':>14} " f"{'field_track':>12}"
    )
    print(header)
    print("-" * len(header))
    plot_runs = {}  # (scenario_name, level) -> result for seed 1, for the figure
    for scenario in (CONSTRAINED_PASSAGE, OPEN):
        for level in NOISE_LEVELS:
            errs, tracks = [], []
            for seed in SEEDS:
                cfg = AugmentedEKFConfig(
                    seed=seed, process_noise_std=(0.1, 0.1, 0.01, level)
                )
                r = adaptive(
                    scenario, seed, ReplanTrigger.KAPPA_CHANGE, FIELD_WORLD, cfg
                )
                errs.append(r.final_error_mm)
                tracks.append(field_tracking_error(r))
                if seed == 1:
                    plot_runs[(scenario.name, level)] = r
            print(
                f"{scenario.name:>20} {level:>8.0e} "
                f"{statistics.median(errs):>14.1f} "
                f"{statistics.mean(tracks) * 100:>11.1f}%"
            )
    _plot_sweep(plot_runs)


def _plot_sweep(plot_runs):
    """kappa-hat vs depth at each noise level, both scenarios (seed 1)."""
    scenarios = [CONSTRAINED_PASSAGE, OPEN]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        any_run = plot_runs[(scenario.name, NOISE_LEVELS[0])]
        depth0 = [s.y for s in any_run.executed_states]
        ktrue = [FIELD.kappa_at(s.x, s.y) for s in any_run.executed_states]
        ax.plot(depth0, ktrue, color="black", lw=2.5, label="true field")
        for level in NOISE_LEVELS:
            r = plot_runs[(scenario.name, level)]
            depth = [s.y for s in r.executed_states]
            ax.plot(depth, r.kappa_estimates, lw=1.2, label=f"Q3={level:.0e}")
        for edge in (45.0, 65.0, 70.0):
            ax.axvline(edge, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{scenario.name} | learn/kappa | seed 1")
        ax.set_xlabel("depth y [mm]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("kappa [1/mm]")
    fig.suptitle(
        "Process-noise sweep: lag/variance tradeoff of the scalar tracker",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = f"{FIGDIR}/four_c_baseline_process_noise_sweep.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n[figure] {path}")


def main():
    print("#" * 78)
    print("# FIELD WORLD -- simulator on PROSTATE_PATH, scalar belief 1/29 everywhere")
    print("#" * 78)
    field_results = emit_table(FIELD_WORLD)

    print("\n\n" + "#" * 78)
    print("# CONSTANT WORLD -- true 1/50, belief 1/25 (reference; reproduces history)")
    print("#" * 78)
    emit_table(CONSTANT_WORLD)

    report_measurements_per_layer(field_results)
    plot_kappa_hat_vs_field(field_results)
    process_noise_sweep()


if __name__ == "__main__":
    main()
