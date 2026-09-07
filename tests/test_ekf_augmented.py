"""Tests for the augmented (state + log kappa) EKF.

The first test is the strongest, as in Phase 3: the analytic 4x4 Jacobian
checked against numerical differentiation of the real `step`. It will fail
loudly if the two second-order terms in the log-kappa column are omitted --
their absence costs a factor of 50 in accuracy.

A UNITS RULE that produced three bugs while writing these: the CONSTRUCTOR and
`NeedleParams` take plain kappa; the STATE VECTOR holds log(kappa). Anywhere a
test crosses that boundary it must exponentiate or take the log. In particular
`AugmentedNeedleEKF(cfg, pose, k)` wants k, not log(k) -- it logs it
internally.

Run:  pytest tests/test_ekf_augmented.py -v
"""

import math

import numpy as np
import pytest

from needlesim.estimation.ekf_augmented import (
    AugmentedEKFConfig,
    AugmentedNeedleEKF,
    run_augmented_estimation,
)
from needlesim.models.unicycle_needle import Control, NeedleParams, State, step

TRUE_KAPPA = 1.0 / 50.0
V = 5.0
DT = 0.05


def alternating_controls(n, flip_every=100, v=V):
    """A trajectory that FLIPS b periodically. Kappa is only observable while
    turning in both senses -- see the observability note in ekf_augmented.py --
    so this is the trajectory shape the estimator needs."""
    out = []
    b = 1
    for k in range(n):
        if k and k % flip_every == 0:
            b = -b
        out.append((Control(v=v, b=b), DT))
    return out


# =====================================================================
# 1. THE 4x4 JACOBIAN -- analytic vs finite differences of `step`.
# =====================================================================


@pytest.mark.parametrize("theta", [0.0, 0.7, math.pi / 2, 3.0, -1.2])
def test_augmented_jacobian_matches_finite_differences(theta):
    """The analytic Jacobian must match numerical differentiation of the real
    RK4 step, INCLUDING the log-kappa column.

    TWO assertions, because they guard different things:

    - The WHOLE-MATRIX residual is ~6e-4 (Euler-vs-RK4, the same order as the
      3x3 case in Phase 3), and 5e-3 clears it. But this bar does NOT guard
      the two second-order position terms in the log-kappa column: in log
      space that column carries a factor of kappa (~0.02 here), so those terms
      are ~2e-6 and are dwarfed by the ~6e-4 pose-block residual in the theta
      column. Deleting them leaves the matrix max unchanged, so the
      whole-matrix bar alone would happily pass a Jacobian that omits them.
      (The "3.1e-2 without the terms" figure that used to be quoted here
      belonged to a PLAIN-kappa parameterisation, where d(theta')/d(kappa)=
      v*b*dt=0.25 makes the column dominate the matrix max; under log-kappa it
      no longer applies. See the jacobian docstring in ekf_augmented.py.)

    - The FOURTH-COLUMN residual is where the second-order terms live, so it
      is asserted SEPARATELY at 1e-5. With the terms it is ~2e-6; omitting
      them raises it to ~4-6e-4 (a >200x jump), so this bar -- unlike the
      whole-matrix one -- genuinely fails if they are dropped, or if the
      chain-rule factor of kappa is missing.
    """
    cfg = AugmentedEKFConfig()
    # Constructor takes plain kappa; it stores log(kappa) internally.
    ekf = AugmentedNeedleEKF(cfg, State(10.0, 20.0, theta), TRUE_KAPPA)
    control = Control(v=V, b=1)
    # The STATE vector holds log kappa.
    mean = np.array([10.0, 20.0, theta, math.log(TRUE_KAPPA)])

    analytic = ekf.jacobian(mean, control, DT)
    assert analytic.shape == (4, 4)

    eps = 1e-7
    numeric = np.zeros((4, 4))
    for j in range(4):
        sp, sm = mean.copy(), mean.copy()
        sp[j] += eps
        sm[j] -= eps
        # sp[3] / sm[3] are LOG kappa, so exponentiate for NeedleParams. The
        # fourth row of the numeric column stays in log units, since that is
        # the state being differentiated.
        fp = step(State(*sp[:3]), control, DT, NeedleParams(kappa=math.exp(sp[3])))
        fm = step(State(*sm[:3]), control, DT, NeedleParams(kappa=math.exp(sm[3])))
        numeric[:, j] = (
            np.array([fp.x, fp.y, fp.theta, sp[3]])
            - np.array([fm.x, fm.y, fm.theta, sm[3]])
        ) / (2 * eps)

    err = np.max(np.abs(analytic - numeric))
    assert err < 5e-3, (
        f"augmented Jacobian differs from numeric by {err:.2e} at theta="
        f"{theta} -- the Euler-vs-RK4 residual should be ~6e-4."
    )

    # The whole-matrix bar above is dominated by the pose-block Euler-vs-RK4
    # residual and does NOT see the log-kappa column's second-order terms, so
    # guard them directly: they live in the fourth column. With the terms this
    # is ~2e-6; omitting them raises it to ~4-6e-4.
    col4_err = np.max(np.abs(analytic[:, 3] - numeric[:, 3]))
    assert col4_err < 1e-5, (
        f"log-kappa column differs from numeric by {col4_err:.2e} at theta="
        f"{theta}. If this is around 4e-4, the two second-order terms in the "
        f"log-kappa column (d(x')/d(log k), d(y')/d(log k)) are probably "
        f"missing. If it is off by a factor of ~kappa, the chain-rule factor "
        f"is probably missing."
    )


def test_log_kappa_row_is_identity():
    """Kappa does not evolve -- it is a fixed unknown being estimated, so the
    fourth row is [0, 0, 0, 1]. (In 4c, where kappa varies with POSITION, this
    stops being true and the row gains x and y terms. This test should fail
    then, and that failure is the signal the Jacobian needs its new terms.)"""
    ekf = AugmentedNeedleEKF(AugmentedEKFConfig(), State(0.0, 0.0, 0.0), TRUE_KAPPA)
    F = ekf.jacobian(
        np.array([0.0, 0.0, 0.7, math.log(TRUE_KAPPA)]), Control(v=V, b=1), DT
    )
    assert np.allclose(F[3], [0.0, 0.0, 0.0, 1.0])


# =====================================================================
# 2. THE MECHANISM -- kappa is corrected only through correlation.
# =====================================================================


def test_predict_builds_kappa_position_correlation():
    """H's fourth column is zero, so curvature is never measured directly. It
    can only be corrected through the covariance correlations that `predict`
    builds. If P[3, 0] and P[3, 1] stay zero, the Kalman gain's bottom row is
    zero too and this whole phase does nothing."""
    ekf = AugmentedNeedleEKF(AugmentedEKFConfig(), State(0.0, 0.0, 0.3), TRUE_KAPPA)
    for _ in range(40):
        ekf.predict(Control(v=V, b=1), DT)
    P = ekf.covariance
    assert abs(P[3, 0]) > 1e-12 or abs(P[3, 1]) > 1e-12, (
        "no kappa-position correlation built up; kappa is unobservable and "
        "update() cannot correct it"
    )


def test_kappa_stays_positive_without_clamping():
    """The payoff of the log parameterisation, asserted rather than assumed.

    A filter carrying kappa directly can be driven negative (a needle curving
    the wrong way) or to zero (infinite turning radius, and a division by zero
    in dubins_full, which computes R = 1/kappa) by a large correction. In log
    space kappa = exp(state) is positive however large the update, so no clamp
    is needed -- which also keeps the mean and covariance consistent, since a
    clamp would hold a value the covariance knows nothing about.

    The measurements below are deliberately hostile: far from the prediction
    and in a direction that pushes the estimate hard. That is the case that
    would break the unconstrained kappa-space version.
    """
    ekf = AugmentedNeedleEKF(AugmentedEKFConfig(), State(0.0, 0.0, 0.0), TRUE_KAPPA)
    for k in range(60):
        ekf.predict(Control(v=V, b=1), DT)
        if k % 5 == 0:
            ekf.update((ekf.state.x + 30.0, ekf.state.y - 30.0))
        assert ekf.kappa > 0.0, f"kappa went non-positive: {ekf.kappa}"
        assert math.isfinite(ekf.kappa), f"kappa went non-finite: {ekf.kappa}"


def test_covariance_stays_symmetric_and_psd():
    """Symmetry and positive-semidefiniteness are the invariants a covariance
    must keep. Over many steps floating-point asymmetry accumulates through
    F P F.T; if either breaks, the filter is numerically unsound regardless of
    how good its estimates look."""
    ekf = AugmentedNeedleEKF(AugmentedEKFConfig(), State(0.0, 0.0, 0.0), TRUE_KAPPA)
    for _ in range(500):
        ekf.predict(Control(v=V, b=1), DT)
        P = ekf.covariance
        assert np.allclose(P, P.T, atol=1e-9), "covariance lost symmetry"
        assert np.min(np.linalg.eigvalsh(P)) > -1e-9, "covariance lost PSD"


# =====================================================================
# 3. THE CLAIM -- kappa converges toward truth from a wrong prior.
# =====================================================================


def test_kappa_converges_from_wrong_prior():
    """The point of the phase. Start the filter believing kappa=1/25 while the
    world runs at 1/50 -- a factor of two out, the same mismatch Phase 3.5
    measured -- and the estimate should move substantially toward the truth
    using only noisy position measurements, with curvature never observed
    directly.

    MEASURED at these settings over 1200 steps: the alternating-b trajectory
    left 5.4% of the initial error, the pure arc 2.1%. The bar below is 25%,
    wide headroom over the harder of the two, so it fails only if convergence
    genuinely stops working rather than merely getting slower.
    """
    true_kappa = 1.0 / 50.0
    guess = 1.0 / 25.0

    history = run_augmented_estimation(
        true_params=NeedleParams(kappa=true_kappa),
        initial_kappa_guess=guess,
        controls=alternating_controls(1200),
        start=State(0.0, 0.0, 0.0),
        config=AugmentedEKFConfig(seed=1),
    )

    initial_error = abs(guess - true_kappa)
    final_error = abs(history[-1].kappa_estimate - true_kappa)

    assert final_error < 0.25 * initial_error, (
        f"kappa estimate ended at 1/{1/history[-1].kappa_estimate:.1f}, "
        f"{final_error/initial_error:.1%} of the initial error remaining — "
        f"convergence has stopped working. Check observability (does the "
        f"trajectory turn?), the initial log-kappa covariance (0.7 admits a "
        f"factor of two; too small and the filter refuses to move), and the "
        f"log-kappa process noise (zero makes a parameter converge then freeze)."
    )
