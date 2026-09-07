"""Tests for the adaptive closed loop (learned kappa fed back to the planner).

The claim this half of Phase 4a makes: fixing the MODEL recovers accuracy that
merely re-aiming from a better pose does not. Phase 3.5 showed replanning
helps under constraint; these tests pin whether learning kappa helps on top of
that.

WHY MEDIANS OVER SEEDS, not single runs. Seed variation is large -- condition
2 on constrained_passage ranged 2.8mm to 16.1mm across five seeds. A
single-seed assertion would be picking a draw and hoping. The headline tests
therefore run all five seeds and compare medians, which costs ~20 planner runs
each and a couple of minutes. That is the price of testing the claim actually
made rather than a coincidence.

Run:  pytest tests/test_adaptive_loop.py -v
"""

import math
import statistics

import pytest

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN, build_env
from needlesim.control.adaptive_loop import (
    AdaptiveLoopConfig,
    ReplanTrigger,
    run_closed_loop_adaptive,
)
from needlesim.control.closed_loop import (
    ClosedLoopConfig,
    run_closed_loop,
    run_open_loop,
)
from needlesim.estimation.ekf import EKFConfig
from needlesim.estimation.ekf_augmented import AugmentedEKFConfig
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

TRUE_KAPPA = 1.0 / 50.0
WRONG_GUESS = 1.0 / 25.0        # 2x out, the same mismatch Phase 3.5 measured
SEEDS = (1, 2, 3, 4, 5)


def make_planner(env, kappa, seed=1):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, NeedleParams(kappa=kappa), cfg)


# --- condition runners, so the tests read as comparisons -------------------


def _fixed_kappa(scenario, seed):
    """Condition 2: closed loop, drift trigger, planner keeps the wrong kappa."""
    env = build_env(scenario)
    return run_closed_loop(
        planner=make_planner(env, WRONG_GUESS, seed),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=WRONG_GUESS),
        start=scenario.start,
        goal=scenario.goal,
        ekf_config=EKFConfig(seed=seed),
        loop_config=ClosedLoopConfig(seed=seed),
    )


def _learned_kappa(scenario, seed, trigger=ReplanTrigger.DRIFT):
    """Conditions 3 and 4: closed loop, planner rebuilt from the estimate."""
    env = build_env(scenario)
    return run_closed_loop_adaptive(
        planner=make_planner(env, WRONG_GUESS, seed),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        initial_kappa_guess=WRONG_GUESS,
        start=scenario.start,
        goal=scenario.goal,
        ekf_config=AugmentedEKFConfig(seed=seed),
        loop_config=AdaptiveLoopConfig(trigger=trigger, seed=seed),
    )


def _finite_median(values):
    finite = [v for v in values if math.isfinite(v)]
    assert finite, "every run failed to plan; the comparison is meaningless"
    return statistics.median(finite)


# =====================================================================
# 1. MECHANICS -- does the loop run, and does the belief actually move?
# =====================================================================


def test_adaptive_loop_runs_and_records_kappa():
    """Smoke test plus the thing that makes a bad run diagnosable: the kappa
    history must be recorded, or a poor result is ambiguous between bad
    control and a bad estimate."""
    result = _learned_kappa(CONSTRAINED_PASSAGE, seed=3)
    assert result.n_steps > 0
    assert len(result.kappa_estimates) == result.n_steps
    assert all(k > 0 for k in result.kappa_estimates), "kappa went non-positive"


def test_kappa_converges_during_the_run():
    """The estimator must learn while STEERING, not merely in the isolated
    setting of scripts/kappa_convergence.py.

    MEASURED over five seeds on constrained_passage, starting from 1/25
    against a true 1/50: final estimates spanned 1/41.7 to 1/48.3. Note these
    insertions are ~400-600 steps where the convergence script used 1200, so
    the estimate does NOT fully converge -- it gets close enough to change the
    plans, which is what matters here. Do not import the 2-5% figure from that
    script as the expectation.

    The bar below is 40% of the initial error remaining. The worst measured
    seed (1/41.7) left ~28%, so this has headroom while still failing if
    learning stops working.
    """
    finals = []
    for seed in SEEDS:
        r = _learned_kappa(CONSTRAINED_PASSAGE, seed)
        if r.kappa_estimates:
            finals.append(r.kappa_estimates[-1])
    assert finals, "no run produced a kappa history"

    initial_error = abs(WRONG_GUESS - TRUE_KAPPA)
    median_final_error = _finite_median([abs(k - TRUE_KAPPA) for k in finals])

    assert median_final_error < 0.4 * initial_error, (
        f"median kappa error {median_final_error:.5f} is "
        f"{median_final_error/initial_error:.0%} of the initial "
        f"{initial_error:.5f} — the estimate is not converging far enough "
        f"during a real insertion to be worth feeding back"
    )


def test_planner_is_rebuilt_with_the_learned_kappa():
    """The mechanism of this phase, asserted rather than assumed. Each plan
    should be made under the belief current at that moment, so
    kappa_at_replans must TRACK kappa_estimates -- not sit at the initial
    guess.

    A subtle failure this catches: rebuilding the planner but passing the
    original params, which would run the whole experiment as condition 2 with
    extra steps and look plausible throughout.
    """
    result = _learned_kappa(CONSTRAINED_PASSAGE, seed=3)
    if len(result.kappa_at_replans) < 2:
        pytest.skip("run did not replan enough to compare beliefs")
    assert any(
        abs(k - result.kappa_at_replans[0]) > 1e-6
        for k in result.kappa_at_replans[1:]
    ), (
        "every plan used the same kappa — the planner is not being rebuilt "
        "from the estimate, and this is condition 2 in disguise"
    )


# =====================================================================
# 2. THE TRIGGERS -- they must actually differ.
# =====================================================================


def test_kappa_trigger_fires_earlier_than_drift():
    """Kappa converges fast -- most of the movement is in the first ~200 steps
    -- so a kappa-change trigger fires early and then goes quiet, which is a
    qualitatively different pattern from drift-triggered replanning.

    MEASURED on constrained_passage: kappa-triggered replans clustered at
    steps 21/41/61/81, drift-triggered at 101/121/261/281. Almost no overlap,
    which is what justifies treating them as distinct signals rather than two
    names for the same one.

    Asserted on the MEDIAN first-replan step across seeds, because a single
    seed's first replan is noisy.
    """
    first_drift, first_kappa = [], []
    for seed in SEEDS:
        rd = _learned_kappa(CONSTRAINED_PASSAGE, seed, ReplanTrigger.DRIFT)
        rk = _learned_kappa(CONSTRAINED_PASSAGE, seed, ReplanTrigger.KAPPA_CHANGE)
        if rd.replan_steps:
            first_drift.append(rd.replan_steps[0])
        if rk.replan_steps:
            first_kappa.append(rk.replan_steps[0])

    assert first_drift and first_kappa, "one of the triggers never fired at all"
    med_drift = statistics.median(first_drift)
    med_kappa = statistics.median(first_kappa)

    assert med_kappa < med_drift, (
        f"kappa trigger first fired at step {med_kappa} (median), drift at "
        f"{med_drift} — the two triggers are not distinguishable in timing, "
        f"so treating them as separate signals is not justified"
    )


# =====================================================================
# 3. THE CLAIM -- does fixing the model beat merely re-aiming?
# =====================================================================


def test_learned_kappa_beats_fixed_kappa_under_constraint():
    """The Phase 4a payoff, and the reason the four conditions exist.

    MEASURED over five seeds on constrained_passage at 2x kappa mismatch:
        open-loop        median 18.6mm   (Phase 3.5)
        fixed kappa      median  4.4mm   range 2.8-16.1
        learned kappa    median  3.0mm   range 2.8- 6.1

    Two effects, and the second matters more clinically. The median improves
    modestly (4.4 -> 3.0mm, and 3.0 is close to the 3.0mm goal tolerance
    floor). But the TAIL collapses: fixed kappa's worst seed was 16.1mm,
    learned kappa's was 6.1mm. A method that usually works and occasionally
    misses by 16mm is not deployable; one bounded at 6mm is a different
    proposition.

    The bar is on the median with slack, plus a separate bound on the worst
    case, because asserting only the median would miss the effect that
    actually matters.
    """
    fixed, learned = [], []
    for seed in SEEDS:
        fixed.append(_fixed_kappa(CONSTRAINED_PASSAGE, seed).final_error_mm)
        learned.append(_learned_kappa(CONSTRAINED_PASSAGE, seed).final_error_mm)

    med_fixed = _finite_median(fixed)
    med_learned = _finite_median(learned)
    worst_learned = max(v for v in learned if math.isfinite(v))

    assert med_learned <= med_fixed, (
        f"learned kappa median {med_learned:.1f}mm did not improve on fixed "
        f"kappa {med_fixed:.1f}mm — feeding the estimate back is buying "
        f"nothing. Check result.kappa_estimates: if kappa_hat barely moved "
        f"during these insertions, the estimate is not converging far enough "
        f"to change the plans, which is a different problem from the feedback "
        f"being useless."
    )
    assert worst_learned < 12.0, (
        f"learned kappa's worst seed was {worst_learned:.1f}mm — the tail "
        f"collapse (fixed kappa's worst was 16.1mm, learned kappa's 6.1mm) "
        f"is the more valuable half of this result and it is not holding"
    )


def test_learned_kappa_does_not_hurt_in_open_space():
    """The unflattering direction, kept for the same reason Phase 3.5 kept its
    companion test.

    On `open`, replanning of any kind is an intervention where open-loop is
    largely adequate, and Phase 3.5 measured that it degrades results there.
    The question for this phase is only whether LEARNING makes that
    degradation worse.

    MEASURED over five seeds on `open` at 2x mismatch:
        open-loop      2.8, 14.9, 2.8, 2.9, 10.7   median  2.9
        fixed kappa    9.8,  6.0, 2.9, 5.2,  6.5   median  6.0
        learned kappa  4.8,  2.8, 5.5, 6.4,  2.8   median  4.8

    So learning slightly IMPROVES on fixed kappa here (4.8 vs 6.0) while both
    remain worse than not replanning at all. The assertion is therefore weak
    and one-directional: learning must not be worse than fixed. Claiming more
    than that would overstate a five-seed difference.

    Note open-loop's own spread (2.8 to 14.9) is much wider than Phase 3.5
    reported, because those earlier figures came from a run whose TRUE_KAPPA
    had drifted to 1/25 -- a matched-parameter run with no mismatch at all.
    The numbers above are the corrected ones.
    """
    fixed, learned = [], []
    for seed in SEEDS:
        fixed.append(_fixed_kappa(OPEN, seed).final_error_mm)
        learned.append(_learned_kappa(OPEN, seed).final_error_mm)

    med_fixed = _finite_median(fixed)
    med_learned = _finite_median(learned)

    assert med_learned <= med_fixed * 1.2, (
        f"learned kappa median {med_learned:.1f}mm is materially worse than "
        f"fixed kappa {med_fixed:.1f}mm on `open` — feeding the estimate back "
        f"is actively harmful where replanning was already unhelpful, which "
        f"would need explaining before the Phase 4a claim stands"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])