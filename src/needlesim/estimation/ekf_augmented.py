"""Phase 4a: joint state-parameter estimation -- learning kappa online.

THIS IS YOUR PHASE 4a FILE. The augmented Jacobian is the load-bearing piece
and is yours; the containers and the experiment harness are delegable.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is RECURSIVE PARAMETER ESTIMATION (joint state-parameter estimation), not
machine learning. There is no dataset, no model class, no train/test split --
just one unknown scalar estimated live from a signal the filter already
computes. An EKF augmentation is the right tool for a single well-posed
parameter; reaching for a learned model here would be worse engineering.

The machine learning is 4c, where kappa varies with POSITION and must be
learned from trajectory data. THIS phase is 4c's baseline, and that is why it
comes first: "does learning a spatial field beat optimally estimating a single
number?" is the interesting question, and it cannot be asked without the
number-estimating version to compare against.

THE IDEA
--------
Phase 3 showed the filter's innovation goes systematically biased under a
wrong kappa -- it can TELL something is off. Phase 3.5 showed replanning
recovers some of the lost accuracy but not all, because replanning re-aims
from a better position without fixing the BELIEF that put the needle off
course. Every new plan carried the same wrong kappa.

So: put kappa in the state. Estimate (x, y, theta, kappa) instead of
(x, y, theta), and let the filter correct kappa from the same position
measurements.

WHY IT CAN WORK -- the same mechanism, one level deeper. Heading is never
measured, yet Phase 3's filter recovered it because `predict` builds
correlation between heading and position in the covariance. Kappa is that
chain extended: a wrong kappa produces wrong heading evolution, which produces
wrong position, which shows up in the innovation. The correlation
kappa -> theta -> position is what lets a position measurement correct kappa.

LOG PARAMETERISATION -- the fourth state is log(kappa), not kappa
-----------------------------------------------------------------
`mean[3]` holds log(kappa). Kappa itself is recovered by exponentiating, via
the `kappa` property.

The reason is that kappa must stay physical. An unconstrained filter can drive
it negative (a needle curving the wrong way) or to zero (infinite turning
radius, and a division by zero in `dubins_full`, which computes R = 1/kappa).
Since this phase feeds kappa_hat back to the planner, an unphysical estimate
does not merely make the filter wrong -- it breaks what consumes it.

Clamping kappa after each update would work, but it makes this no longer
strictly a Kalman filter: the covariance is computed by the standard update
and knows nothing about the constraint, so P[3,3] can report high confidence
in a value the clamp is holding. The mean and the covariance stop being
consistent.

Estimating log(kappa) avoids that entirely. Any real number is a valid state,
kappa = exp(state) is positive by construction, and no clamp is needed.

The cost is a chain-rule factor in the Jacobian's fourth column: since
d(kappa)/d(log kappa) = kappa, every entry in that column carries a factor of
kappa. See the jacobian docstring.

A UNITS NOTE THAT MATTERS: uncertainty in log space is RELATIVE. A standard
deviation of sigma means the value could plausibly be anywhere from e^-sigma
to e^+sigma times the current estimate -- so sigma=0.1 is ~10%, sigma=0.7 is a
factor of two either way. That is the natural way to express uncertainty about
a curvature you do not know, but it means the covariance entries for this
state are NOT in units of 1/mm and cannot be read as such.

OBSERVABILITY -- and a hypothesis that measurement contradicted
---------------------------------------------------------------
The classical concern is that on a constant-b arc, a wrong kappa and a wrong
initial heading produce nearly the same position trace, so a filter cannot
separate them -- implying that trajectories which flip b are needed to make
kappa identifiable.

That is NOT what happens here, and the reason is worth knowing. The ambiguity
requires the HEADING to be genuinely uncertain. This filter starts with a
confident heading prior (initial sigma 0.05 rad) and a deliberately bad kappa
prior, so heading is already pinned and every bit of position evidence goes
into kappa.

Measured from a 2x wrong prior (guess 1/25, true 1/50) over 1200 steps:
    alternating b : 5.4% of the initial error remaining
    pure arc      : 2.1% remaining
The pure arc did BETTER. Flipping b partially cancels the accumulating
position error that carries the curvature signal -- turn left too much, then
right too much, and the errors offset.

So if kappa converges slowly, suspect the covariance settings (the initial
log-kappa spread, the process noise) before suspecting the trajectory shape.
The turning-shape concern is real only when the heading prior is weak.

This is recorded rather than tested: it rests on a single seed at one set of
parameters, which is enough to correct an assumption but not enough to assert
as a property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from needlesim.models.unicycle_needle import Control, NeedleParams, State, step


@dataclass
class AugmentedEKFConfig:
    """Tunables for the augmented filter. Mostly mirrors EKFConfig, with the
    kappa terms added.

    NOTE the fourth entry of both covariance tuples is in LOG-KAPPA units, not
    kappa units -- see the log-parameterisation note in the module docstring.
    """

    # Process noise over (x, y, theta, log kappa), as standard deviations.
    # The LOG-KAPPA term answers "how much can kappa change between steps?"
    # In 4a the answer is NONE: kappa is a fixed unknown, and what improves is
    # the filter's knowledge, not the world. So this stays small. It should
    # not be zero, though -- a parameter with zero process noise converges and
    # then FREEZES, becoming so certain it stops listening even when the data
    # disagrees. A small value keeps the filter responsive.
    # (In 4c, where kappa genuinely varies with position, this reasoning
    # changes entirely.)
    process_noise_std: tuple[float, float, float, float] = (0.1, 0.1, 0.01, 1e-5)

    measurement_noise_std: tuple[float, float] = (1.0, 1.0)

    # Initial covariance over (x, y, theta, log kappa). The LOG-KAPPA entry
    # encodes how wrong the prior might be, as a RELATIVE spread: 0.1 is
    # ~10%, 0.3 is ~35%, 0.7 is a factor of two either way. Set it from the
    # range you actually expect. Too small and the filter refuses to move off
    # its prior -- 0.01 would assert 99% confidence that kappa is within 1% of
    # the guess, which would be self-defeating in an experiment that starts
    # the filter deliberately 2x off. 0.7 admits that factor-of-two error.
    initial_covariance_diag: tuple[float, float, float, float] = (1.0, 1.0, 0.05, 0.7)

    measurement_interval: int = 20

    seed: int = 0


@dataclass
class AugmentedEKFRecord:
    """One timestep of history. Adds the kappa estimate and its variance to
    what EKFRecord carried."""

    step_index: int
    true_state: State
    estimate: State
    kappa_estimate: float
    kappa_variance: float  # covariance[3, 3] -- the filter's own
    # confidence, in LOG-kappa units
    covariance: np.ndarray  # 4x4
    measurement: tuple[float, float] | None = None
    innovation: tuple[float, float] | None = None


class AugmentedNeedleEKF:
    """EKF over (x, y, theta, log kappa). Curvature is a state, not a
    parameter, and is carried in log space so it stays positive."""

    def __init__(
        self,
        config: AugmentedEKFConfig,
        initial: State,
        initial_kappa: float,
    ) -> None:
        """
        Args:
            config: AugmentedEKFConfig.
            initial: initial pose estimate.
            initial_kappa: the filter's PRIOR belief about curvature -- its
                starting guess, which is expected to be wrong. Passed as a
                plain kappa and stored as log(kappa). Note there is no
                NeedleParams argument any more: kappa is part of the state, so
                there is no fixed model parameter to pass in. That is the
                structural change this phase makes.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # mean = [x, y, theta, log(kappa)]; covariance from the config
        # diagonal SQUARED (config stores standard deviations, covariance
        # needs variances).
        self.mean = np.array(
            [initial.x, initial.y, initial.theta, math.log(initial_kappa)]
        )
        self.covariance = np.diag(np.array(config.initial_covariance_diag) ** 2)

    # --- the linearisation, now 4x4 ---------------------------------------

    def jacobian(self, mean: np.ndarray, control: Control, dt: float) -> np.ndarray:
        """4x4 Jacobian of the augmented motion model about `mean`.

        The 3x3 pose block is unchanged from Phase 3. What is new is the
        FOURTH COLUMN: how does each state respond to a change in log kappa?

        Because the state is log kappa rather than kappa, every entry in that
        column carries a chain-rule factor:

            d(kappa)/d(log kappa) = kappa

        so, writing k = exp(mean[3]):

            d(theta')/d(log k) = v * b * dt * k
            d(log k')/d(log k) = 1        -- log kappa does not evolve
            its ROW is [0, 0, 0, 1]       -- nothing changes kappa

        THE TWO ENTRIES THAT ARE EASY TO MISS, and they matter for the
        column's accuracy. A naive derivation puts zeros in the position rows
        of this column, on the grounds that x' and y' do not mention kappa.
        But WITHIN a step, a change in kappa perturbs theta, and the perturbed
        theta then affects where x and y land. Second order in dt:

            d(x')/d(log k) ~ -v*sin(theta) * (v*b*dt*k) * dt/2
            d(y')/d(log k) ~  v*cos(theta) * (v*b*dt*k) * dt/2

        MEASURED against numerical differentiation of the actual RK4 step
        (v=5, b=1, dt=0.05), looking at the FOURTH COLUMN specifically:
            without those terms:  ~4-6e-4
            with them:            ~2e-6   -- a >200x cut in the column error
        The residual scales linearly with kappa, as it must: the whole column
        carries a factor of k. Verified at both b = +1 and b = -1, which
        matters because the trajectories that make kappa observable flip b.

        A CAVEAT WORTH RECORDING (it caught a stale number). These terms do
        NOT move the WHOLE-MATRIX max error, which stays ~6e-4 with or without
        them: that max lives in the theta column (the Euler-vs-RK4 pose
        residual, same as Phase 3's 3x3 case), which these terms do not touch.
        So a whole-matrix tolerance cannot guard them -- the test asserts on
        the fourth column directly (see test_augmented_jacobian...). An earlier
        "3.1e-2 without the terms" figure was from a PLAIN-kappa
        parameterisation, where d(theta')/d(kappa) = v*b*dt = 0.25 makes the
        column large enough to dominate the matrix max; under log-kappa that
        factor becomes v*b*dt*k (~0.005 here) and the number no longer applies.
        It was carried over from before the parameterisation changed -- the
        lesson being to re-measure after the thing being measured changes.

        WHY THE RESIDUAL IS NOT ZERO: `step` uses RK4 while this is derived
        from the Euler form -- the same deliberate approximation documented in
        the Phase 3 filter.

        Note this reads kappa from the `mean` ARGUMENT, not from the filter's
        own state, so it linearises about whatever point it is given. The
        finite-difference test relies on that.
        """
        jac = np.array(
            [
                [
                    1,
                    0,
                    -1 * control.v * np.sin(mean[2]) * dt,
                    -1
                    * control.v
                    * np.sin(mean[2])
                    * (control.v * control.b * dt * math.exp(mean[3]))
                    * dt
                    / 2,
                ],
                [
                    0,
                    1,
                    control.v * np.cos(mean[2]) * dt,
                    control.v
                    * np.cos(mean[2])
                    * (control.v * control.b * dt * math.exp(mean[3]))
                    * dt
                    / 2,
                ],
                [0, 0, 1, control.v * control.b * dt * math.exp(mean[3])],
                [0, 0, 0, 1],
            ]
        )
        return jac

    # --- predict / update -------------------------------------------------

    def predict(self, control: Control, dt: float) -> None:
        """Advance the belief one step.

        Structurally identical to Phase 3's predict, with one crucial
        difference:

            THE MEAN IS PROPAGATED USING THE ESTIMATED KAPPA, not a fixed
            model parameter. In Phase 3 this was self.params.kappa, set once
            at construction. Here `self.params` exponentiates mean[3], so the
            filter's belief IS the model, and it moves. That self-consistency
            is the phase: as kappa_hat improves, its own predictions improve,
            which sharpens the innovation, which improves kappa_hat further.

        log kappa is carried through unchanged (it does not evolve); only the
        UNCERTAINTY about it grows, via Q. Covariance is re-symmetrised
        because floating-point asymmetry accumulates over F P F.T.
        """
        F = self.jacobian(self.mean, control, dt)
        Q = np.diag(np.array(self.config.process_noise_std) ** 2)
        new_mean = step(self.state, control, dt, self.params)
        self.mean = np.array([new_mean.x, new_mean.y, new_mean.theta, self.mean[3]])
        P = F @ self.covariance @ F.T + Q
        self.covariance = (P + P.T) / 2

    def update(self, measurement: tuple[float, float]) -> np.ndarray:
        """Correct from a noisy tip-position measurement.

        Same as Phase 3, with H now 2x4:

            H = [[1, 0, 0, 0],
                 [0, 1, 0, 0]]

        Note the fourth column of H is zero -- curvature is no more directly
        measured than heading was. It is corrected entirely through the
        covariance correlations built during predict. Worth checking that
        P[3, 0] and P[3, 1] actually become non-zero as you predict forward;
        if they stay zero, the Kalman gain's bottom row is zero, kappa can
        never be learned, and this phase does nothing.

        NO CLAMP IS NEEDED. Because the state is log(kappa), the update is
        unconstrained by construction and kappa = exp(state) stays positive
        however large the correction. See the log-parameterisation note in the
        module docstring for why that was preferred to clamping.

        Returns the innovation -- the filter's own error signal, and the thing
        Phase 3 showed goes systematically biased when the model is wrong.
        """
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.diag(np.array(self.config.measurement_noise_std) ** 2)
        y = np.array(measurement) - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.solve(S, np.eye(2))
        self.mean = self.mean + K @ y
        P = (np.eye(4) - K @ H) @ self.covariance
        self.covariance = (P + P.T) / 2
        return y

    # --- convenience ------------------------------------------------------

    @property
    def state(self) -> State:
        """Pose estimate as a State. Slices off the log-kappa component --
        `State` has three fields."""
        return State(*self.mean[:3])

    @property
    def kappa(self) -> float:
        """Current curvature estimate, exponentiated out of log space. This is
        what gets fed back to the planner."""
        return math.exp(self.mean[3])

    @property
    def params(self) -> NeedleParams:
        """The estimated kappa as a NeedleParams, for handing to `step` or to
        a planner. Constructed fresh each call -- NeedleParams is frozen, and
        construction is cheap relative to what consumes it."""
        return NeedleParams(kappa=self.kappa)


def run_augmented_estimation(
    true_params: NeedleParams,
    initial_kappa_guess: float,
    controls: list[tuple[Control, float]],
    start: State,
    config: AugmentedEKFConfig,
) -> list[AugmentedEKFRecord]:
    """Run an insertion with the augmented filter tracking pose AND kappa.

    IMPLEMENT ME. Same shape as Phase 3's run_estimation:

        1. true_state = start; ekf = AugmentedNeedleEKF(config, start,
           initial_kappa_guess)
        2. for each (control, dt), indexed by k:
             a. true_state = step(true_state, control, dt, TRUE_PARAMS)
             b. ekf.predict(control, dt)          # uses its ESTIMATED kappa
             c. if a measurement is due: ekf.update(noisy true position)
             d. record an AugmentedEKFRecord
        3. return the history

    THE TRUE/MODEL SPLIT is now sharper, not softer: 2a uses true_params; 2b
    uses whatever the filter currently believes. There is no fixed model
    parameter any more -- the filter's belief IS the model, and it moves.

    Measurement noise belongs to the WORLD: its own generator, as before.

    RECORDING kappa_variance: covariance[3, 3] is in LOG-kappa units. Either
    store it as-is and remember that when plotting, or convert to a kappa-space
    band via the delta method (sd_kappa ~ kappa * sd_log_kappa). Say which you
    did -- a variance plotted in the wrong units will look wrong by orders of
    magnitude.

    WHAT TO PLOT FIRST: kappa_estimate against the true value over time, with
    a +/- one standard deviation band. Two things decide whether this phase
    worked: whether the estimate converges, and whether the filter's stated
    confidence is honest (does the true value sit inside the band?).
    """
    filter_history = []
    true_state = start
    ekf = AugmentedNeedleEKF(config, start, initial_kappa_guess)
    world_rng = np.random.default_rng(config.seed + 1000)
    for k, (control, dt) in enumerate(controls):
        true_state = step(true_state, control, dt, true_params)
        ekf.predict(control, dt)
        z = None
        innovation = None
        if k % config.measurement_interval == 0:
            noise = world_rng.normal(
                0.0, config.measurement_noise_std
            )  # std, not variance
            z = np.array([true_state.x, true_state.y]) + noise
            innovation = ekf.update(z)
        kappa_variance = ekf.covariance[3, 3]
        filter_history.append(
            AugmentedEKFRecord(
                step_index=k,
                true_state=true_state,
                estimate=ekf.state,
                kappa_estimate=ekf.kappa,
                kappa_variance=kappa_variance,
                covariance=ekf.covariance.copy(),
                measurement=z,
                innovation=innovation,
            )
        )
    return filter_history
