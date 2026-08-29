"""EKF diagnostic plots -- throwaway, for eyeballing the filter.

Runs the filter twice on the same insertion: once with matched true/model
kappa, once with a deliberate mismatch. Four panels each.

What to look for, panel by panel:

1. TRAJECTORY -- the estimate should hug the true path, with small visible
   corrections at each measurement. Under mismatch it should drift between
   measurements and get snapped back.

2. COVARIANCE TRACE -- the sawtooth. Uncertainty grows during the 20
   predict-only steps and collapses at each update. If this does NOT sawtooth,
   something is wrong: either measurements are firing every step, or the
   update is not shrinking P.

3. ERROR -- position and heading error separately. Heading is the interesting
   one: it is never measured, so any heading tracking at all is the filter
   inferring it through the position-heading correlation.

4. INNOVATION -- measurement minus prediction, at each update. With matched
   params this should be zero-mean noise. Under mismatch it should be
   SYSTEMATICALLY BIASED: the filter's predictions are consistently wrong in
   one direction. That bias is what model mismatch looks like from inside the
   filter, and it is the signal a learned model would aim to remove in Phase 4.

Run:  python scripts/eyeball_ekf.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from needlesim.estimation.ekf import EKFConfig, run_estimation
from needlesim.models.unicycle_needle import Control, NeedleParams, State

V = 5.0
DT = 0.05
N_STEPS = 600
START = State(0.0, 0.0, math.pi / 2)

# A gentle S-curve: turn one way, then the other. Exercises the Jacobian at a
# range of headings rather than a single arc.
CONTROLS = [(Control(v=V, b=+1), DT)] * (N_STEPS // 2) + [
    (Control(v=V, b=-1), DT)
] * (N_STEPS // 2)

TRUE_KAPPA = 1.0 / 50.0
MISMATCH_KAPPA = 1.0 / 25.0  # filter believes the needle turns tighter than it does


def panels(ax_traj, ax_cov, ax_err, ax_innov, records, title):
    xs_true = [r.true_state.x for r in records]
    ys_true = [r.true_state.y for r in records]
    xs_est = [r.estimate.x for r in records]
    ys_est = [r.estimate.y for r in records]
    steps = [r.step_index for r in records]

    meas = [(r.step_index, r.measurement) for r in records if r.measurement is not None]

    # 1. trajectory
    ax_traj.plot(xs_true, ys_true, color="black", lw=2, label="true")
    ax_traj.plot(xs_est, ys_est, color="tab:blue", lw=1.5, ls="--", label="estimate")
    if meas:
        ax_traj.scatter(
            [m[1][0] for m in meas],
            [m[1][1] for m in meas],
            s=18,
            color="tab:red",
            zorder=3,
            label="measurements",
        )
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x [mm]")
    ax_traj.set_ylabel("y [mm]")
    ax_traj.set_title(f"{title}: trajectory")
    ax_traj.legend(fontsize=8)

    # 2. covariance trace -- the sawtooth
    ax_cov.plot(steps, [np.trace(r.covariance) for r in records], color="tab:purple")
    ax_cov.set_xlabel("step")
    ax_cov.set_ylabel("trace(P)")
    ax_cov.set_title("covariance (expect sawtooth)")

    # 3. error, position and heading separately
    pos_err = [
        math.hypot(r.estimate.x - r.true_state.x, r.estimate.y - r.true_state.y)
        for r in records
    ]
    head_err = [
        abs((r.estimate.theta - r.true_state.theta + math.pi) % (2 * math.pi) - math.pi)
        for r in records
    ]
    ax_err.plot(steps, pos_err, color="tab:blue", label="position [mm]")
    ax_err.plot(steps, head_err, color="tab:orange", label="heading [rad]")
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("error")
    ax_err.set_title("estimate error vs truth")
    ax_err.legend(fontsize=8)

    # 4. innovation -- zero-mean if the model is right, biased if it is not
    innov = [(r.step_index, r.innovation) for r in records if r.innovation is not None]
    if innov:
        ax_innov.plot(
            [i[0] for i in innov], [i[1][0] for i in innov], "o-", ms=3, label="x"
        )
        ax_innov.plot(
            [i[0] for i in innov], [i[1][1] for i in innov], "o-", ms=3, label="y"
        )
        mean_x = np.mean([i[1][0] for i in innov])
        mean_y = np.mean([i[1][1] for i in innov])
        ax_innov.axhline(0.0, color="grey", lw=0.6)
        ax_innov.set_title(f"innovation (mean x={mean_x:+.3f}, y={mean_y:+.3f})")
    ax_innov.set_xlabel("step")
    ax_innov.set_ylabel("z - prediction [mm]")
    ax_innov.legend(fontsize=8)


def main():
    cfg = EKFConfig(seed=1)

    matched = run_estimation(
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=TRUE_KAPPA),
        controls=CONTROLS,
        start=START,
        config=cfg,
    )
    mismatched = run_estimation(
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=MISMATCH_KAPPA),
        controls=CONTROLS,
        start=START,
        config=cfg,
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    panels(*axes[0], matched, f"matched (kappa=1/{1/TRUE_KAPPA:.0f})")
    panels(*axes[1], mismatched, f"mismatch (true 1/{1/TRUE_KAPPA:.0f}, model 1/{1/MISMATCH_KAPPA:.0f})")
    fig.tight_layout()
    fig.savefig("ekf_eyeball.png", dpi=130)
    print("saved ekf_eyeball.png")

    for name, recs in [("matched", matched), ("mismatch", mismatched)]:
        pos = [
            math.hypot(r.estimate.x - r.true_state.x, r.estimate.y - r.true_state.y)
            for r in recs
        ]
        head = [
            abs(
                (r.estimate.theta - r.true_state.theta + math.pi) % (2 * math.pi)
                - math.pi
            )
            for r in recs
        ]
        innov = [r.innovation for r in recs if r.innovation is not None]
        print(
            f"{name:9s}  final pos err {pos[-1]:6.2f} mm | max {max(pos):6.2f} mm | "
            f"final heading err {head[-1]:.4f} rad | "
            f"innovation mean ({np.mean([i[0] for i in innov]):+.3f}, "
            f"{np.mean([i[1] for i in innov]):+.3f})"
        )


if __name__ == "__main__":
    main()
