"""Extended Kalman Filter for needle tip state estimation.

THIS IS YOUR PHASE 3 FILE. Interface and structure laid out; the load-bearing
math (the Jacobian, predict, update) is yours to implement. Per the working
agreement, filter math is yours; plain containers and plotting are delegable.

THE PROBLEM
-----------
Every planner so far has assumed perfect knowledge of the needle's pose. Real
systems do not have that. Imaging gives a noisy TIP POSITION at a few Hz,
while the needle advances at every simulation step. Heading is not measured at
all — it must be inferred from how position evolves.

The filter maintains a Gaussian belief over (x, y, theta): a mean estimate and
a 3x3 covariance. Between measurements it PREDICTS forward using the motion
model and its uncertainty grows; when a measurement arrives it UPDATES,
pulling the estimate toward the observation and shrinking the covariance. That
grow-then-collapse sawtooth is the behaviour worth plotting.

WHY THIS MATTERS BEYOND ESTIMATION
----------------------------------
This is where the true_needle vs model_needle separation, maintained since
Task 1, finally does real work. The SIMULATOR advances the needle with TRUE
kappa; the FILTER predicts with MODEL kappa. When they differ, the prediction
drifts systematically and measurements have to pull it back. How well the
filter copes under that mismatch is the Phase 4 setup — do not collapse the
two parameter objects into one for convenience.

DESIGN CHOICES (settled; see docs/roadmap.md)
---------------------------------------------
- EKF rather than a particle filter. The model is smooth, low-dimensional and
  only mildly nonlinear, which is the regime where linearisation works well
  and a PF's advantages do not materialise. Honest limitation to record: the
  EKF assumes a unimodal Gaussian belief, so it would fail if the state ever
  became genuinely ambiguous (two plausible tip locations). That does not
  arise with continuous position measurements, but it is the reason one might
  switch to a PF later.
- POSITION-ONLY measurements. This is what imaging provides, and it makes
  heading unobservable directly — the filter must infer it from the position
  history, which is a real estimation problem rather than bookkeeping.
- INTERMITTENT measurements. Real imaging runs at a few Hz while the sim steps
  at dt=0.05, so a measurement every ~20 steps is both realistic and produces
  the covariance sawtooth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from needlesim.models.unicycle_needle import Control, NeedleParams, State, step

# ---------------------------------------------------------------------------
# Containers -- scaffolding, delegable.
# ---------------------------------------------------------------------------


@dataclass
class EKFConfig:
    """Filter tunables. Config-driven + seeded, per CLAUDE.md conventions."""

    # Process noise: how much the filter distrusts its own prediction, as a
    # diagonal [x, y, theta]. THIS IS WHERE MODEL MISMATCH ENTERS. If true and
    # model kappa differ, predictions drift systematically, and process noise
    # is how you tell the filter "expect my predictions to be wrong". Too small
    # and the filter ignores measurements and diverges confidently; too large
    # and it chases measurement noise. Tuning this is part of the work.
    process_noise_std: tuple[float, float, float] = (0.1, 0.1, 0.01)

    # Measurement noise: imaging position accuracy, std in mm per axis.
    measurement_noise_std: tuple[float, float] = (1.0, 1.0)

    # Initial covariance: how uncertain the filter is about the START pose.
    # Nonzero even at the start -- insertion site is known but not exactly.
    initial_covariance_diag: tuple[float, float, float] = (1.0, 1.0, 0.05)

    # Steps between measurements. dt=0.05 with a measurement every 20 steps is
    # 1 Hz imaging against a 20 Hz sim.
    measurement_interval: int = 20

    seed: int = 0


@dataclass
class EKFRecord:
    """One timestep of filter history, for plotting and analysis."""

    step_index: int
    true_state: State  # simulator ground truth (NOT visible to filter)
    estimate: State  # filter mean
    covariance: np.ndarray  # 3x3
    measurement: tuple[float, float] | None = None  # None when no imaging
    innovation: tuple[float, float] | None = None  # measurement - prediction


# ---------------------------------------------------------------------------
# The filter. Methods marked IMPLEMENT ME are YOURS.
# ---------------------------------------------------------------------------


class NeedleEKF:
    def __init__(self, params: NeedleParams, config: EKFConfig, initial: State) -> None:
        """
        Args:
            params: the MODEL needle params -- the filter's BELIEF about
                curvature. NEVER the simulator's ground truth. Passing true
                params here silently removes the mismatch this phase exists to
                study.
            config: EKFConfig.
            initial: the filter's initial mean estimate (the known insertion
                pose). Its uncertainty is config.initial_covariance_diag.
        """
        self.params = params
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Filter state: mean as a length-3 array, covariance as 3x3.
        # IMPLEMENT (small): initialise from `initial` and the config diagonal.
        self.mean = np.array([initial.x, initial.y, initial.theta])
        self.covariance = np.diag(np.array(config.initial_covariance_diag) ** 2)

    # --- the linearisation ------------------------------------------------

    def jacobian(self, mean: np.ndarray, control: Control, dt: float) -> np.ndarray:
        """3x3 Jacobian of the motion model about `mean`.

        IMPLEMENT ME. This is the heart of the "extended" in EKF: the model is
        nonlinear, so you linearise about the current estimate to propagate the
        covariance.

        Differentiate the discrete step. Working from the Euler form
            x' = x + v*cos(theta)*dt
            y' = y + v*sin(theta)*dt
            theta' = theta + v*kappa*b*dt
        gives

            [ 1  0  -v*sin(theta)*dt ]
            [ 0  1   v*cos(theta)*dt ]
            [ 0  0          1        ]

        Note what the structure says: x and y do not depend on each other or on
        themselves beyond identity, and theta's evolution does not depend on
        ANY state (theta_dot = v*kappa*b is state-independent at constant
        kappa). That last fact is why heading integrates exactly here -- the
        same decoupling noted in the Task 1 docstring.

        A DELIBERATE APPROXIMATION, VERIFIED: `step` uses RK4, not Euler, so
        this Euler-form Jacobian is an approximation of the true linearisation.
        Measured discrepancy against numerical differentiation of the RK4 step
        at v=5, kappa=1/50, dt=0.05: ~6e-4 across headings. That is the right
        order for a first-order approximation of a fourth-order integrator over
        a small dt, and is far below the process noise, so it is acceptable.
        Verify it yourself rather than trusting this note -- see the finite-
        difference test in tests/test_ekf.py.

        WHEN THIS BREAKS: in Phase 4, kappa becomes position-dependent. Then
        theta_dot DOES depend on x and y, the bottom row stops being [0 0 1],
        and this Jacobian must gain those terms. Leave yourself a note.
        """
        jac = np.array(
            [
                [1, 0, -1 * control.v * np.sin(mean[2]) * dt],
                [0, 1, control.v * np.cos(mean[2]) * dt],
                [0, 0, 1],
            ]
        )
        return jac

    # --- the two filter steps ---------------------------------------------

    def predict(self, control: Control, dt: float) -> None:
        """Advance the belief one step under `control`, with no measurement.

        IMPLEMENT ME. Two things happen:

            mean     <- step(mean, control, dt, self.params)     [nonlinear]
            covariance <- F @ covariance @ F.T + Q               [linearised]

        where F = self.jacobian(...) and Q is the process noise covariance
        (diagonal, from config.process_noise_std, SQUARED -- the config stores
        standard deviations, the filter needs variances; getting this wrong by
        a square is a classic and silent error).

        Note the asymmetry, and it is the point of an EKF: the MEAN goes
        through the true nonlinear model, only the COVARIANCE is linearised.

        Covariance must stay symmetric. Floating-point asymmetry creeps in over
        many steps; consider re-symmetrising with (P + P.T)/2. Decide whether
        to do it every step or only occasionally, and say why.
        """
        F = self.jacobian(self.mean, control, dt)
        Q = np.diag(np.array(self.config.process_noise_std) ** 2)
        new_mean = step(self.state, control, dt, self.params)
        self.mean = np.array([new_mean.x, new_mean.y, new_mean.theta])
        P = F @ self.covariance @ F.T + Q
        self.covariance = (P + P.T) / 2

    def update(self, measurement: tuple[float, float]) -> np.ndarray:
        """Correct the belief using a noisy tip-position measurement.

        IMPLEMENT ME. Standard EKF update. The measurement model is linear
        here (you observe x and y directly), so H is a constant:

            H = [[1, 0, 0],
                 [0, 1, 0]]      # 2x3: position observed, heading is not

        Then:
            innovation  y = z - H @ mean                 (2-vector)
            innovation covariance  S = H @ P @ H.T + R   (2x2)
            Kalman gain  K = P @ H.T @ inv(S)            (3x2)
            mean  <- mean + K @ y
            P     <- (I - K @ H) @ P

        Return the innovation -- it is the filter's own error signal and is
        worth recording (a consistently biased innovation means the model is
        wrong, which is exactly what model mismatch looks like from inside the
        filter).

        THE INTERESTING PART: H has a zero column for theta, so heading is
        never measured. Yet K will have a nonzero bottom row -- the filter
        corrects heading from position observations, because the covariance
        carries the correlation between them, built up during predict. That
        correlation is the whole reason this works, and it is worth checking
        that P[0,2] and P[1,2] actually become nonzero as you predict forward.

        Numerically, prefer solving over inverting: np.linalg.solve rather than
        inv(S) where practical. At 2x2 it barely matters, but the habit is
        right. The Joseph form of the covariance update is more numerically
        stable than (I - K@H)@P; use it if you find P losing positive-
        definiteness over long runs.
        """
        H = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.diag(np.array(self.config.measurement_noise_std) ** 2)
        y = np.array(measurement) - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.solve(S, np.eye(2))
        self.mean = self.mean + K @ y
        P = (np.eye(3) - K @ H) @ self.covariance
        self.covariance = (P + P.T) / 2
        return y

    # --- convenience ------------------------------------------------------

    @property
    def state(self) -> State:
        """Current mean estimate as a State. IMPLEMENT (trivial)."""
        return State(*self.mean)


# ---------------------------------------------------------------------------
# The simulation loop that exercises the filter. Scaffolding-ish, but the
# true/model split inside it is load-bearing -- get it right.
# ---------------------------------------------------------------------------


def run_estimation(
    true_params: NeedleParams,
    model_params: NeedleParams,
    controls: list[tuple[Control, float]],
    start: State,
    config: EKFConfig,
) -> list[EKFRecord]:
    """Run a needle insertion with an EKF tracking it.

    IMPLEMENT ME. The loop:

        1. true_state = start;  filter = NeedleEKF(model_params, config, start)
        2. for each (control, dt), indexed by k:
             a. true_state = step(true_state, control, dt, TRUE_PARAMS)
             b. filter.predict(control, dt)              # uses MODEL params
             c. if k % config.measurement_interval == 0:
                    z = (true_state.x, true_state.y) + Gaussian noise
                    innovation = filter.update(z)
             d. record an EKFRecord

    THE CRITICAL LINE is 2a versus 2b: the simulator steps with TRUE params,
    the filter predicts with MODEL params. If both use the same object there
    is no mismatch and the experiment measures nothing. This is the single
    easiest thing to get wrong here.

    Measurement noise uses self.rng in the FILTER? No -- the noise is the
    WORLD's, not the filter's. Decide where the measurement RNG lives and
    document it; mixing the simulator's randomness into the filter's seed
    makes runs hard to reason about.
    """
    filter_history = []
    true_state = start
    ekf = NeedleEKF(model_params, config, start)
    # Measurement noise belongs to the WORLD, not the filter -- separate generator
    # so the simulator's randomness and the filter's are independently reasoned about.
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
        filter_history.append(
            EKFRecord(
                step_index=k,
                true_state=true_state,
                estimate=ekf.state,
                covariance=ekf.covariance.copy(),
                measurement=z,
                innovation=innovation,
            )
        )
    return filter_history
