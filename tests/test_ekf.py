"""Tests for the needle EKF.

The strongest test here is the FIRST one: the analytic Jacobian checked
against numerical differentiation of the actual `step`. That is the same
strategy used for the Dubins geometry -- validate hand-derived math against
code you already trust, rather than against a reference you do not have.

Run:  pytest tests/test_ekf.py -v
"""

import math

import numpy as np
import pytest

from needlesim.models.unicycle_needle import Control, NeedleParams, State, step
from needlesim.estimation.ekf import EKFConfig, NeedleEKF, run_estimation

PARAMS = NeedleParams(kappa=1.0 / 50.0)
V = 5.0
DT = 0.05


# =====================================================================
# 1. THE JACOBIAN -- analytic versus finite differences of `step`.
#    Do this first; everything downstream depends on it.
# =====================================================================


@pytest.mark.parametrize("theta", [0.0, 0.7, math.pi / 2, 3.0, -1.2])
def test_jacobian_matches_finite_differences(theta):
    """The analytic Jacobian must match numerical differentiation of the real
    `step`. Note `step` uses RK4 while the analytic form is derived from the
    Euler step, so agreement is approximate -- ~6e-4 at these settings, which
    is the right order for a first-order approximation of a fourth-order
    integrator over dt=0.05, and far below the process noise."""
    ekf = NeedleEKF(PARAMS, EKFConfig(), State(10.0, 20.0, theta))
    control = Control(v=V, b=1)
    mean = np.array([10.0, 20.0, theta])

    analytic = ekf.jacobian(mean, control, DT)

    eps = 1e-7
    numeric = np.zeros((3, 3))
    for j in range(3):
        sp, sm = mean.copy(), mean.copy()
        sp[j] += eps
        sm[j] -= eps
        fp = step(State(*sp), control, DT, PARAMS)
        fm = step(State(*sm), control, DT, PARAMS)
        numeric[:, j] = (
            np.array([fp.x, fp.y, fp.theta]) - np.array([fm.x, fm.y, fm.theta])
        ) / (2 * eps)

    assert np.max(np.abs(analytic - numeric)) < 5e-3, (
        f"analytic Jacobian differs from numeric by "
        f"{np.max(np.abs(analytic - numeric)):.2e} at theta={theta}"
    )


def test_jacobian_structure():
    """Structural facts worth pinning: position does not feed back into
    itself beyond identity, and at constant kappa theta's row is [0, 0, 1]
    because theta_dot is state-independent. When Phase 4 makes kappa
    position-dependent, THIS TEST SHOULD FAIL -- that is the signal the
    Jacobian needs its new terms."""
    ekf = NeedleEKF(PARAMS, EKFConfig(), State(0.0, 0.0, 0.0))
    F = ekf.jacobian(np.array([0.0, 0.0, 0.7]), Control(v=V, b=1), DT)
    assert np.allclose(F[2], [0.0, 0.0, 1.0]), "theta row should be [0,0,1]"
    assert F[0, 0] == pytest.approx(1.0)
    assert F[1, 1] == pytest.approx(1.0)
    assert F[0, 1] == pytest.approx(0.0)
    assert F[1, 0] == pytest.approx(0.0)


# =====================================================================
# 2. PREDICT -- uncertainty must GROW when there is no measurement.
# =====================================================================


def test_predict_grows_covariance():
    ekf = NeedleEKF(PARAMS, EKFConfig(), State(0.0, 0.0, 0.0))
    before = np.trace(ekf.covariance)
    for _ in range(20):
        ekf.predict(Control(v=V, b=1), DT)
    after = np.trace(ekf.covariance)
    assert after > before, "covariance must grow during open-loop prediction"


def test_covariance_stays_symmetric_and_psd():
    """Over many steps floating-point asymmetry creeps in. Symmetry and
    positive-semidefiniteness are the invariants a covariance must keep; if
    either breaks the filter is numerically unsound regardless of its
    estimates."""
    ekf = NeedleEKF(PARAMS, EKFConfig(), State(0.0, 0.0, 0.0))
    for _ in range(500):
        ekf.predict(Control(v=V, b=1), DT)
        P = ekf.covariance
        assert np.allclose(P, P.T, atol=1e-9), "covariance lost symmetry"
        assert np.min(np.linalg.eigvalsh(P)) > -1e-9, "covariance lost PSD"


def test_predict_builds_position_heading_correlation():
    """THE mechanism that makes position-only measurement work: predicting
    forward correlates heading with position, so a later position update can
    correct heading even though heading is never observed. If these stay
    zero, the filter can never learn heading and `update` will do nothing to
    it."""
    ekf = NeedleEKF(PARAMS, EKFConfig(), State(0.0, 0.0, 0.3))
    for _ in range(20):
        ekf.predict(Control(v=V, b=1), DT)
    P = ekf.covariance
    assert abs(P[0, 2]) > 1e-9 or abs(P[1, 2]) > 1e-9, (
        "no position-heading correlation built up; heading is unobservable "
        "and update() cannot correct it"
    )


# =====================================================================
# 3. UPDATE -- a measurement must REDUCE uncertainty and pull the estimate.
# =====================================================================


def test_update_shrinks_covariance():
    ekf = NeedleEKF(PARAMS, EKFConfig(), State(0.0, 0.0, 0.0))
    for _ in range(20):
        ekf.predict(Control(v=V, b=1), DT)
    before = np.trace(ekf.covariance)
    ekf.update((ekf.state.x + 0.5, ekf.state.y - 0.5))
    after = np.trace(ekf.covariance)
    assert after < before, "a measurement must reduce uncertainty"


def test_update_corrects_heading_indirectly():
    """Heading is never measured (H's third column is zero), yet the Kalman
    gain's bottom row is nonzero because of the correlation built in predict.
    Start the filter with a deliberately wrong heading, feed it position
    measurements consistent with the TRUE heading, and its heading estimate
    should improve.

    TODO: construct this. Roll a true trajectory, initialise the filter with
    theta offset by (say) 0.3 rad, run predict/update over several
    measurement intervals, and assert the heading error shrinks. This is the
    test that proves the filter is doing something nontrivial rather than
    just averaging positions.
    """
    predict_state = State(0.0, 0.0, 0.0)
    true_state = State(0.0, 0.0, 0.3)
    initial_error = 0.3
    ekf = NeedleEKF(PARAMS, EKFConfig(), predict_state)
    for k in range(200):
        true_state = step(true_state, Control(v=V, b=1), DT, PARAMS)
        ekf.predict(Control(v=V, b=1), DT)
        if k % 20 == 0:
            ekf.update((true_state.x, true_state.y))
    final_error = abs(
        (ekf.mean[2] - true_state.theta + math.pi) % (2 * math.pi) - math.pi
    )
    assert (
        final_error < 0.5 * initial_error
    ), f"heading error {final_error:.4f} did not shrink from {initial_error}"


# =====================================================================
# 4. MODEL MISMATCH -- the Phase 4 setup.
# =====================================================================


def test_tracks_under_model_mismatch():
    """Simulator uses TRUE kappa, filter believes MODEL kappa. The filter
    should still track -- degraded, but not diverging -- because measurements
    keep pulling it back.

    TODO: pick a mismatch (e.g. true 1/50 vs model 1/45, a 10% error), run
    run_estimation over a few hundred steps, and assert the estimate error
    stays bounded rather than growing without limit. Decide the bound from
    the measurement noise and interval rather than by tuning until green --
    if the filter is doing its job the error should sit within a few
    measurement standard deviations.

    Also worth asserting: the INNOVATION should be systematically biased
    under mismatch (the filter's predictions are consistently wrong in one
    direction), whereas with matched params it should be zero-mean. That bias
    is what model mismatch looks like from inside the filter, and it is the
    signal Phase 4's learned model would aim to remove.
    """
    TRUE_KAPPA = 1.0 / 50.0
    MISMATCH_KAPPA = 1.0 / 25.0

    filter_history = run_estimation(
        true_params=NeedleParams(TRUE_KAPPA),
        model_params=NeedleParams(MISMATCH_KAPPA),
        controls=[(Control(v=V, b=1), DT)] * 200,
        start=State(0.0, 0.0, 0.0),
        config=EKFConfig(),
    )
    position_errors = []
    for record in filter_history:
        position_errors.append(
            math.hypot(
                record.estimate.x - record.true_state.x,
                record.estimate.y - record.true_state.y,
            )
        )
    assert (
        max(position_errors) < 15.0
    ), f"position error reached {max(position_errors):.1f}mm so filter is diverging and not bounded"

    innovations = []
    for record in filter_history:
        if record.innovation is not None:
            innovations.append(record.innovation)
    assert len(innovations) >= 5, "too few measurements to judge bias"
    mean_x_innovation = float(np.mean([i[0] for i in innovations]))
    mean_y_innovation = float(np.mean([i[1] for i in innovations]))
    # With matched params the innovation is zero-mean noise: ~1mm std over
    # ~10 measurements gives a standard error of ~0.3mm, so |mean| should sit
    # under about 0.6mm by chance. A bias comfortably above that is the
    # model error showing through rather than noise.
    assert max(abs(mean_x_innovation), abs(mean_y_innovation)) > 1.0, (
        f"expected innovation bias > 1.0mm under 2x kappa mismatch (true 1/50, "
        f"model 1/25), got mean ({mean_x_innovation:+.3f}, {mean_y_innovation:+.3f}) — "
        f"the filter is not seeing the model error, which would undercut the "
        f"Phase 4 premise"
    )


def test_zero_noise_matched_params_tracks_exactly():
    """Sanity floor: with matched true/model params and zero measurement
    noise, the filter's estimate should follow ground truth almost exactly.
    If this fails, something is wrong with predict or the loop's true/model
    wiring, not with the tuning."""
    kappa = 1.0 / 50.0
    filter_history = run_estimation(
        true_params=NeedleParams(kappa),
        model_params=NeedleParams(kappa),
        controls=[(Control(v=V, b=1), DT)] * 200,
        start=State(0.0, 0.0, 0.0),
        config=EKFConfig(measurement_noise_std=(0.0, 0.0)),
    )
    position_errors = []
    heading_errors = []
    for record in filter_history:
        position_errors.append(
            math.hypot(
                record.estimate.x - record.true_state.x,
                record.estimate.y - record.true_state.y,
            )
        )
        heading_errors.append(
            abs(
                (record.estimate.theta - record.true_state.theta + math.pi)
                % (2 * math.pi)
                - math.pi
            )
        )
    assert max(position_errors) < 0.5, (
        f"position error reached {max(position_errors):.4f}mm with a correct model and a perfect sensor "
        f"- check predict, the Jacobian, and the true/model wiring in run_estimation"
    )
    assert max(heading_errors) < 0.01, (
        f"heading error reached {max(heading_errors):.4f} rad with a correct model and a perfect sensor "
        f"— heading should track essentially exactly here"
    )

    innovations = []
    for record in filter_history:
        if record.innovation is not None:
            innovations.append(record.innovation)
    assert len(innovations) >= 5, "too few measurements to judge bias"
    max_innovation_abs = max(max(abs(i[0]), abs(i[1])) for i in innovations)
    assert max_innovation_abs < 0.5, (
        f"largest single innovation was {max_innovation_abs:.4f}mm with a correct model and a perfect sensor "
        f"— the filter is being surprised by measurements it should be predicting exactly"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
