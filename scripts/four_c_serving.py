"""Phase 4c Gate 3, part 1 of 2: misspecified-truth generators + the
conservative-serving mechanism, verified against the matched model and the
develop-against case (c) ONLY.

Cases (a) sub-length-scale capsule and (b) lateral x-tilt are BUILT here (task 2
needs them) but deliberately NOT evaluated -- they are the held-out verdict, and
looking at them while debugging the mechanism would contaminate the held-out
discipline the frozen spec commits to
(docs/four_c_conservative_serving_spec.md).

WHAT THIS IMPLEMENTS. The frozen conservative-serving policy:

    k_eff(y) = s(y)*k_hat(y) + (1 - s(y)) * min(k_hat(y)*(1+m), k_max)

served as a scalar through a lookup table (the step-0 plumbing), never consulting
the posterior variance. s(y) in [0,1] is the smooth "well-observed" score; where
s=1 the learned mean is served untouched, where s=0 the value is inflated toward
higher curvature (the collision-safe direction), capped at the physical minimum
turning radius. Frozen parameters (see the spec for the pre-registration
justifications): m=0.20, k_max=1/15, depth-density adequacy 3/mm, residual-x
correlation test at alpha=0.05, blend smoothness >= 2mm.

THE TWO PARTS FLAGGED FOR REVIEW (see their own docstrings):
  - residual_whiteness_score: the closure of the density blind spot. Must NOT
    fire under the matched model (would break coverage by crying wolf) and MUST
    fire on genuine x-structure (else it is inert). Verified here on the matched
    model, on (c), and on a GENERIC x-gradient FIXTURE that is NOT the (b)
    evaluation field.
  - blend_score: must be smooth over >= 2mm (invariant 4) or it breaks the
    planner's arc smoothness and the EKF Jacobian. A hard switch looks fine in a
    plot and bites downstream.

ACCEPTANCE (the headline the spec demands): under the matched model the
mechanism must serve k_hat essentially UNTOUCHED where coverage is good. If it
inflates in well-observed regions under a matched model, the coverage signal is
miscalibrated before any misspecification is introduced.

Scripts only; reuses the Gate 2 fitter (four_c_gates); nothing under
planning/models/estimation/control is touched.
Run:  python scripts/four_c_serving.py
"""

from __future__ import annotations

import numpy as np
from four_c_gates import (
    CAPSULE,
    SIGMA_Z,
    TRUE,
    Insertion,
    KnotFieldY,
    _capsule_dwell_arcs,
    _capsule_knots,
    fit_s,
    fit_y,
    make_dataset,
    meas_indices,
    path_sets,
    predict_rms,
    traj_arclength,
    traj_spatial,
)
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

from needlesim.models.tissue_field import PROSTATE_PATH, TissueField, TissueLayer

# --- frozen parameters (docs/four_c_conservative_serving_spec.md) -----------
M_MARGIN = 0.20  # conservatism margin (failure asymmetry)
K_MAX = 1.0 / 15.0  # physical minimum turning radius -> max curvature
TAU_D = 3.0  # depth-density adequacy [measurements / mm]
DENS_W = 2.0  # half-window for density and whiteness bins [mm] (>= transition)
ALPHA_R = 0.05  # residual-x correlation significance (necessary, not sufficient)
R_LO = 0.30  # whiteness effect-size floor: |r|<=R_LO treated as noise (white)
R_HI = 0.45  # |r|>=R_HI treated as real x-structure (fires); smooth ramp between
R_MIN_BIN = 8  # min measurements in a bin to run the whiteness test
BLEND_SIGMA_MM = 1.0  # gaussian smoothing of s(y): FWHM ~2.4mm, satisfies >=2mm
YGRID = np.arange(10.0, 140.0, 0.5)  # fine serving grid [mm]

CAP_LO, CAP_HI = CAPSULE


# ===========================================================================
# Misspecified-truth field generators (Gate 3 substrate).
# (a) and (b) are BUILT but held out; (c) is develop-against; the x-gradient
# FIXTURE is a generic sensitivity probe, NOT the (b) evaluation field.
# ===========================================================================


def sharp_capsule_field() -> TissueField:
    """(a) HELD OUT. Sub-length-scale capsule: all boundaries at 1mm transition
    (< the GP's ~3mm length-scale), so the smoothness prior is wrong at the
    crux. Not evaluated in this file."""
    return TissueField(layers=PROSTATE_PATH, transition_mm=1.0)


class XTiltField:
    """(b) HELD OUT. Lateral tilt: kappa gains a small linear x-dependence the
    kappa(y) model cannot represent. Not evaluated in this file.

    kappa(x, y) = base(y) * (1 + tilt * (x - x0) / x_scale).
    """

    def __init__(
        self,
        base: TissueField,
        tilt: float = 0.25,
        x0: float = 30.0,
        x_scale: float = 40.0,
    ) -> None:
        self.base, self.tilt, self.x0, self.x_scale = base, tilt, x0, x_scale

    def kappa_at(self, x: float, y: float) -> float:
        return self.base.kappa_at(x, y) * (
            1.0 + self.tilt * (x - self.x0) / self.x_scale
        )


class GaussianBumpField:
    """(c) DEVELOP-AGAINST. The capsule as a Gaussian bump rather than a
    sigmoid-stack, on a fat/muscle/gland baseline. A 3mm-length-scale GP should
    barely distinguish this from the sigmoid capsule -- expected near-inert,
    which is why it is the develop-against case. No x-dependence, so the
    whiteness test should stay quiet on it."""

    def __init__(self) -> None:
        # baseline without the capsule (muscle extended through the capsule band)
        self._base = TissueField(
            layers=(
                TissueLayer(kappa=1 / 34, upper_edge_mm=45.0, name="fat"),
                TissueLayer(kappa=1 / 22, upper_edge_mm=70.0, name="muscle"),
                TissueLayer(kappa=1 / 28, upper_edge_mm=None, name="gland"),
            )
        )
        self._center = 0.5 * (CAP_LO + CAP_HI)
        self._sigma = 1.5  # mm; a bump ~ the 5mm capsule width
        self._amp = (1 / 15.0) - self._base.kappa_at(
            0.0, self._center
        )  # reach R15 peak

    def kappa_at(self, x: float, y: float) -> float:
        bump = self._amp * np.exp(-0.5 * ((y - self._center) / self._sigma) ** 2)
        return self._base.kappa_at(x, y) + float(bump)


class XGradientFixture:
    """GENERIC x-sensitivity fixture -- NOT the (b) evaluation field. A strong,
    obvious lateral gradient used solely to confirm the whiteness test CAN fire
    on x-structure. Kept distinct from XTiltField so verifying the test's
    sensitivity does not touch the held-out (b) case."""

    def __init__(
        self, base: TissueField, slope: float = 0.0006, x0: float = 30.0
    ) -> None:
        self.base, self.slope, self.x0 = base, slope, x0

    def kappa_at(self, x: float, y: float) -> float:
        return self.base.kappa_at(x, y) + self.slope * (x - self.x0)


# ===========================================================================
# Measurement-level residuals (input to the whiteness test).
# ===========================================================================


def measurement_residuals(coords, k_hat, dataset):
    """Roll the FITTED field through each insertion and, at every measurement,
    return (x, y, cross_track_residual). Cross-track (residual perpendicular to
    the predicted path heading) is the projection a curvature error shows up in,
    so it is the signal the whiteness test correlates against x."""
    field = KnotFieldY(coords, k_hat)
    xs, ys, ct = [], [], []
    for ins, idx, noisy in dataset:
        traj = traj_spatial(ins, field)  # predicted path under the fit
        for j, k in enumerate(idx):
            if k == 0 or k >= len(traj):
                continue
            pred = traj[k]
            head = traj[k] - traj[k - 1]
            n = np.hypot(*head)
            if n < 1e-9:
                continue
            perp = np.array([-head[1], head[0]]) / n  # unit left-normal
            resid = noisy[j] - pred
            xs.append(float(pred[0]))
            ys.append(float(pred[1]))
            ct.append(float(resid @ perp))
    return np.array(xs), np.array(ys), np.array(ct)


# ===========================================================================
# REVIEW PART 1: the residual-whiteness score (closes the density blind spot).
# ===========================================================================


def residual_whiteness_score(ygrid, mx, my, mct):
    """Per-depth score in [0,1]: 1 = residuals look like iid noise (white), 0 =
    residuals correlate with x (an unmodeled lateral dependence the kappa(y) fit
    is averaging over).

    WHY THIS EXISTS. Depth density is honest only in the coordinate it bins. If
    the true field depends on x, depth density still reports "well-observed"
    while k_hat(y) averages over x -- so the mean is served confidently where it
    is wrong. But that same unmodeled x-dependence makes the CROSS-TRACK
    RESIDUALS at a depth correlate with the measurement's x. This test detects
    exactly what density is blind to, entirely data-side, without pretending the
    kappa(y) model handles x.

    CALIBRATION REQUIREMENTS (both verified in main):
      - Under the matched model it must STAY QUIET (score ~1 at covered depths);
        a test that fires everywhere would cry wolf and force needless inflation.
      - On genuine x-structure it must FIRE (score -> 0); a test that never fires
        is inert and leaves the blind spot open.

    EFFECT SIZE, NOT SIGNIFICANCE (a task-1 correction). A pure p-value test at
    ALPHA_R is sample-size-dependent: with hundreds of measurements per bin, a
    trivial |r|~0.2 clears p<0.05 and the test FIRES under the MATCHED MODEL in
    the densely-sampled fat band -- crying wolf where there is no x-structure,
    forcing needless inflation of a well-observed region. Measured on the
    matched-model null the residual-x correlation tops out at |r|=0.26 (a
    path-phase-entanglement artifact: x is a proxy for path history in fat and
    the ~5% fit bias tracks it), while the generic x-gradient fixture gives
    |r|=0.4-0.89. So the criterion gates on EFFECT SIZE with a floor R_LO=0.30
    (a correlation explaining < ~9% of cross-track variance is treated as noise;
    comfortably above the 0.26 null, comfortably below real structure), AND
    still requires significance (p<ALPHA_R) so a tiny noisy bin cannot fire on a
    high but meaningless |r|. Calibrated on the matched-model null + the generic
    fixture ONLY -- never on held-out (a)/(b). See the frozen spec's whiteness
    note; this replaces the original p-only wording, which was the subtly-wrong
    piece this verification step existed to catch.

    RE-CONFIRMED on the AUGMENTED (multi-depth) pool at task-2 scale (N=130),
    which is the pool task 2 uses: null significant-|r| max 0.167 (combined-gate
    fire 3%, ~alpha) vs structure |r| median 0.81 -- the same clean gap, so the
    0.30 floor holds there. The 0.26 / 0.4-0.89 figures above are the ORIGINAL
    pre-multi-depth calibration, kept for provenance. NOTE the effect-size gate
    fixes the LARGE-N cry-wolf (many points, trivial |r|, significant); a
    SMALL-N chance bin can still clear both gates (an N=55 null check hit |r|=
    0.556 in 1 of 23 bins, ~alpha) because few points give a large sample |r| by
    luck -- so whiteness in task 2 must be computed at large N (>=130), not on a
    reduced pool. There is no multiple-comparison correction across bins; at
    N>=130 the null |r| is small enough (0.167) that the effect-size gate absorbs
    the ~alpha chance-significant bins, which is why large N is the safeguard.

    Bins too sparse to test are left to the DENSITY signal (density will already
    have flagged them), so this score returns 1 there and does not
    double-penalise -- density owns sparsity, whiteness owns structure.
    """
    score = np.ones_like(ygrid)
    for i, y0 in enumerate(ygrid):
        m = np.abs(my - y0) <= DENS_W
        if int(np.count_nonzero(m)) < R_MIN_BIN:
            continue  # too sparse to test structure; density signal owns this
        xb, cb = mx[m], mct[m]
        if np.std(xb) < 1e-9 or np.std(cb) < 1e-9:
            continue
        r, p = pearsonr(xb, cb)
        if p >= ALPHA_R:
            continue  # no significant correlation -> white (score stays 1)
        # significant: gate on EFFECT SIZE (N-independent), not the p-value.
        # |r|<=R_LO -> white (1); |r|>=R_HI -> structured (0); smooth between.
        score[i] = float(np.clip((R_HI - abs(r)) / (R_HI - R_LO), 0.0, 1.0))
    return score


# ===========================================================================
# REVIEW PART 2: the blend score s(y) (must be smooth over >= 2mm).
# ===========================================================================


def density_per_mm(ygrid, my):
    """Pooled measurements per mm of depth, in a +/- DENS_W window."""
    return np.array(
        [np.count_nonzero(np.abs(my - y0) <= DENS_W) / (2.0 * DENS_W) for y0 in ygrid]
    )


def blend_score(ygrid, my, mx, mct):
    """The smooth "well-observed" score s(y) in [0,1].

    s = 1 where BOTH signals pass comfortably (serve k_hat untouched), 0 where
    EITHER fails (inflate toward higher curvature). Built as the product of a
    density ramp and the whiteness score, then GAUSSIAN-SMOOTHED so the result
    varies over >= 2mm regardless of how ragged the raw per-bin signals are --
    this is what enforces invariant 4 (no hard switches; k_eff must be at least
    as smooth as the field, or it breaks arc smoothness and the EKF Jacobian).

    Density ramp: density >= TAU_D -> 1; density <= TAU_D/3 -> 0; smooth between.
    """
    dens = density_per_mm(ygrid, my)
    lo, hi = TAU_D / 3.0, TAU_D
    s_d = np.clip((dens - lo) / (hi - lo), 0.0, 1.0)
    s_r = residual_whiteness_score(ygrid, mx, my, mct)
    s_raw = s_d * s_r
    sigma_pts = BLEND_SIGMA_MM / (ygrid[1] - ygrid[0])
    return gaussian_filter1d(s_raw, sigma_pts, mode="nearest"), s_d, s_r, dens


# ===========================================================================
# The served field.
# ===========================================================================


class ServedField:
    """Conservative-served kappa(y) as a lookup table -- duck-types kappa_at, so
    it drops into NeedleParams.kappa_field exactly like the step-0 LookupField.
    Carries diagnostics for audit; only kappa_at is on the planner's hot path."""

    def __init__(self, ygrid, k_hat_grid, s, k_eff, s_d, s_r, dens):
        self._y, self._k = ygrid, k_eff
        self.k_hat_grid, self.s, self.s_d, self.s_r, self.dens = (
            k_hat_grid,
            s,
            s_d,
            s_r,
            dens,
        )

    def kappa_at(self, x: float, y: float) -> float:
        return float(np.interp(y, self._y, self._k))


def build_served_field(coords, k_hat, dataset) -> ServedField:
    """Assemble k_eff(y) from a fit (coords, k_hat) and its dataset (for the
    coverage signals). This is the whole frozen policy in one place."""
    mx, my, mct = measurement_residuals(coords, k_hat, dataset)
    s, s_d, s_r, dens = blend_score(YGRID, my, mx, mct)
    k_hat_grid = np.interp(YGRID, coords, k_hat)
    k_cons = np.minimum(k_hat_grid * (1.0 + M_MARGIN), K_MAX)
    k_eff = s * k_hat_grid + (1.0 - s) * k_cons
    return ServedField(YGRID, k_hat_grid, s, k_eff, s_d, s_r, dens)


# ===========================================================================
# Verification (matched model + (c) + generic x-gradient fixture). NOT (a)/(b).
# ===========================================================================


def multi_depth_arcs(n, seed):
    """(b)-fix. Constant-b arcs ENTERED AT DEPTH so they characterize the gland,
    which a constant-b arc cannot reach from the face (turning circle caps the
    climb at ~y63). A phantom permits placing a needle at depth in the gel block
    -- the same characterization license that permits the depth-revisiting arcs
    -- so this keeps constant-b (no deconvolution) while covering y>70. Start
    depths are spread across fat..deep so each arc's reachable band
    [start_y, start_y+R] tiles the full working depth up to ~y130."""
    rng = np.random.default_rng(seed + 31)
    HALF = np.pi / 2
    arcs = []
    for _ in range(n):
        arcs.append(
            Insertion(
                theta0=float(rng.uniform(HALF - 0.3, HALF + 0.3)),
                b=int(rng.choice((-1, 1))),
                n_steps=int(rng.integers(280, 420)),
                phase=int(rng.integers(0, 20)),
                start_y=float(rng.uniform(30.0, 115.0)),
            )
        )
    return arcs


def _pool(seed):
    """The characterization pool: revisit arcs (Gate-1 separation) + capsule
    dwell arcs (Gate-2 observability) + multi-depth arcs (gland coverage, the
    (b)-fix). Separation and coverage are independent properties -- the revisit
    arcs supply the first, the multi-depth arcs the second."""
    return (
        path_sets(120, seed)["revisit"]
        + _capsule_dwell_arcs(60, seed)
        + multi_depth_arcs(80, seed)
    )


def _covered_span(served):
    """Depths where the mechanism considers itself well-observed (s >= 0.9)."""
    return served.s >= 0.9


def _fit_pool(pool, seed):
    coords = _capsule_knots()
    data = make_dataset(pool, seed)
    k_hat, _ = fit_y(coords, data)
    return coords, k_hat, data


def _whiteness_fire_rate(coords, k_hat, data, covered_only=True, served=None):
    mx, my, mct = measurement_residuals(coords, k_hat, data)
    s_r = residual_whiteness_score(YGRID, mx, my, mct)
    fired = s_r < 0.5
    if covered_only and served is not None:
        mask = _covered_span(served)
        return float(np.mean(fired[mask])) if mask.any() else float("nan")
    return float(np.mean(fired))


def separation_check(pool, seed):
    """Gate-1 separation on the AUGMENTED pool: does it still distinguish
    kappa(y) from kappa(arc-length)? Mirrors four_c_gates.gate1 exactly (matched
    capacity, held-out-shape RMS) -- adding multi-depth arcs must not weaken the
    separation the revisit arcs supply. Coverage and separation are independent
    properties, so this confirms rather than assumes."""
    data = make_dataset(pool, seed)
    ntr = int(0.67 * len(data))
    train, test = data[:ntr], data[ntr:]
    ycoords = np.arange(16.0, 140.0, 4.0)
    smax = max(ins.n_steps for ins in pool) * 0.25  # STEP_MM = 0.25mm/step
    scoords = np.arange(0.0, smax + 4.0, 4.0)
    vy, _ = fit_y(ycoords, train)
    vs, _ = fit_s(scoords, train)
    ho_y = predict_rms(
        ycoords, vy, test, lambda i, v: traj_spatial(i, KnotFieldY(ycoords, v))
    )
    ho_s = predict_rms(scoords, vs, test, lambda i, v: traj_arclength(i, scoords, v))
    gap = ho_s - ho_y
    separates = (ho_y < 2.5 * SIGMA_Z) and (gap > 3.0 * SIGMA_Z)
    return ho_y, ho_s, gap, separates


def main():
    print(__doc__.split("\n\n")[0])
    print(
        "\nFROZEN: m=%.2f  k_max=R%.0f  tau_d=%.0f/mm  alpha_r=%.2f  blend_sigma=%.1fmm"
        % (M_MARGIN, 1 / K_MAX, TAU_D, ALPHA_R, BLEND_SIGMA_MM)
    )
    print("[matched model + case (c) only; (a) and (b) HELD OUT for the verdict run]\n")

    # --- matched model: the headline acceptance check ----------------------
    print("=" * 78)
    print("MATCHED MODEL -- serve k_hat UNTOUCHED where coverage is good?")
    print("=" * 78)
    pool = _pool(seed=2)
    coords, k_hat, data = _fit_pool(pool, seed=2)
    served = build_served_field(coords, k_hat, data)
    covered = _covered_span(served)
    ratio = served._k / np.maximum(served.k_hat_grid, 1e-9)
    cap = (YGRID >= CAP_LO) & (YGRID <= CAP_HI)
    print(
        f"  covered span (s>=0.9): y in "
        f"[{YGRID[covered].min():.0f},{YGRID[covered].max():.0f}]mm, "
        f"{100 * covered.mean():.0f}% of grid"
    )
    print(
        f"  k_eff/k_hat over covered depths: "
        f"min {ratio[covered].min():.3f}  median {np.median(ratio[covered]):.3f}  "
        f"max {ratio[covered].max():.3f}   (want ~1.000 -- untouched)"
    )
    print(
        f"  capsule (65-70mm): covered={bool(covered[cap].all())}, "
        f"served R {1 / np.mean(served._k[cap]):.1f}mm vs k_hat R "
        f"{1 / np.mean(served.k_hat_grid[cap]):.1f}mm  (served, not fallback's target)"
    )
    # smoothness: served must be as smooth as the field -- max step over 0.5mm.
    dk = np.abs(np.diff(served._k)) / np.diff(YGRID)
    print(
        f"  smoothness: max |d k_eff/dy| = {dk.max():.5f} /mm over {YGRID[1]-YGRID[0]:.1f}mm "
        f"grid (a hard switch would spike here)"
    )
    fire_matched = _whiteness_fire_rate(coords, k_hat, data, served=served)
    print(
        f"  whiteness fire-rate over covered depths: {100 * fire_matched:.0f}% "
        f"(want ~alpha={100*ALPHA_R:.0f}% -- quiet, not crying wolf)"
    )

    # --- (c) develop-against: Gaussian bump, no x-structure ----------------
    print("\n" + "=" * 78)
    print("CASE (c) GAUSSIAN BUMP (develop-against) -- whiteness should stay quiet")
    print("=" * 78)
    c_coords, c_khat, c_data = _refit_under(pool, GaussianBumpField(), seed=2)
    c_served = build_served_field(c_coords, c_khat, c_data)
    fire_c = _whiteness_fire_rate(c_coords, c_khat, c_data, served=c_served)
    cov_c = _covered_span(c_served)
    print(
        f"  covered {100*cov_c.mean():.0f}% of grid; capsule served R "
        f"{1/np.mean(c_served._k[cap]):.1f}mm (true bump peak R15)"
    )
    print(
        f"  whiteness fire-rate over covered depths: {100*fire_c:.0f}% "
        f"(want ~alpha -- (c) has no x-structure)"
    )

    # --- generic x-gradient FIXTURE: whiteness MUST fire -------------------
    print("\n" + "=" * 78)
    print("GENERIC x-GRADIENT FIXTURE (NOT case b) -- whiteness MUST fire")
    print("=" * 78)
    f_coords, f_khat, f_data = _refit_under(pool, XGradientFixture(TRUE), seed=2)
    fire_fix = _whiteness_fire_rate(f_coords, f_khat, f_data, covered_only=False)
    print(
        f"  whiteness fire-rate over ALL testable depths: {100*fire_fix:.0f}% "
        f"(want HIGH -- the test can detect x-structure)"
    )
    print(
        f"  -> mechanism {'DETECTS' if fire_fix > 0.3 else 'FAILS TO DETECT'} "
        f"the injected lateral gradient"
    )

    # --- Gate-1 separation must survive the multi-depth arcs ----------------
    print("\n" + "=" * 78)
    print("GATE-1 SEPARATION on the augmented pool -- multi-depth arcs must not")
    print("weaken kappa(y)-vs-kappa(s) separation (the revisit arcs supply it)")
    print("=" * 78)
    ho_y, ho_s, gap, separates = separation_check(pool, seed=2)
    print(
        f"  held-out RMS [mm]: kappa(y) {ho_y:.2f}   kappa(s) {ho_s:.2f}   "
        f"gap {gap:+.2f}"
    )
    print(
        f"  VERDICT: {'SEPARATES' if separates else 'does NOT separate'} "
        f"(gap {gap:+.2f}mm vs noise {SIGMA_Z:.1f}mm) -- separation "
        f"{'UNAFFECTED' if separates else 'WEAKENED'} by the multi-depth arcs"
    )

    print("\n" + "=" * 78)
    print("HELD OUT (not evaluated here): (a) sharp capsule, (b) x-tilt -> verdict run")
    print("=" * 78)


def _make_dataset_under(pool, world_field, seed):
    """make_dataset but with a chosen world field instead of TRUE. Mirrors
    four_c_gates.make_dataset exactly (same noise convention), swapping the
    generating field -- the Gate 3 substrate."""
    rng = np.random.default_rng(seed + 10_000)
    data = []
    for ins in pool:
        traj = traj_spatial(ins, world_field)
        idx = meas_indices(ins)
        clean = traj[idx]
        noisy = clean + rng.normal(0.0, 1.0, size=clean.shape)  # SIGMA_Z=1.0
        data.append((ins, idx, noisy))
    return data


def _refit_under(pool, world_field, seed):
    """Generate data under `world_field`, fit the (matched-prior) kappa(y)."""
    coords = _capsule_knots()
    data = _make_dataset_under(pool, world_field, seed)
    k_hat, _ = fit_y(coords, data)
    return coords, k_hat, data


if __name__ == "__main__":
    main()
