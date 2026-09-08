"""Phase 4b: spatially varying tissue curvature.

THIS IS YOUR PHASE 4b FILE. The layer construction is scaffolding; the
integration with the model (`step`, and the Jacobians that depend on it) is
load-bearing and yours.

WHAT THIS IS FOR
----------------
Everything through 4a assumed kappa is a single number -- unknown, but
constant. Real tissue is not. 4b makes the SIMULATOR's kappa vary with
position, so that 4c has spatial structure to learn. On its own it is not a
result; it is the ground truth 4c is measured against.

Note the asymmetry, and keep it: this is the WORLD's kappa. The planner and
the filter still believe whatever they believe. Phase 4c's job is to close
that gap by learning the field; 4a's scalar estimate is the baseline it must
beat.

WHY LAYERED, AND WHY DEPTH
---------------------------
The transperineal prostate path passes through skin, subcutaneous fat, pelvic
floor muscle, the prostate capsule, and the gland -- a sequence of distinct
media, which is what the needle-steering literature actually models. Studies
of deflection under varied layer positions are a standard experimental setup,
and stiffness-sensing work demonstrates detecting layer boundaries during
insertion in phantoms and in bovine liver embedded in gelatin.

There is also a mechanism specific to DEPTH, independent of tissue type: at
shallow insertion the needle is barely supported by surrounding tissue and
deflects easily; deeper, more of its length is supported and resists
deformation. So curvature genuinely varies along the insertion axis for two
reasons at once, which is why kappa(y) is a defensible first model rather than
merely a convenient one.

WHY SMOOTHED BOUNDARIES, NOT SHARP
-----------------------------------
The physical argument, not the numerical one. A needle does not experience a
tissue interface at a point: the bevel has finite length, and the tip deforms
tissue ahead of itself before cutting it. The transition is felt over
millimetres. Even the capsule -- the sharpest real boundary on this path -- is
a membrane with thickness, not a discontinuity.

The numerical benefit is real but secondary: a discontinuous kappa makes
d(kappa)/dy a delta function at each interface, which the EKF's linearisation
cannot represent and the planner's Dubins geometry would see as an
instantaneous change in turning radius mid-arc.

HOW THE LAYERS BLEND -- a detail that matters for thin layers
--------------------------------------------------------------
The obvious construction, chaining sigmoids so each boundary blends the
running value into the next layer's, FAILS for a layer thinner than a couple
of transition widths: the two adjacent transitions overlap and the layer never
attains its own value. Measured with a 5mm capsule at R=15mm and 2mm
transitions, the chained version peaked at R=19.5mm -- the high-contrast layer
was washed out by its own smoothing, which would silently remove the most
interesting feature of the profile.

The construction used instead computes a MEMBERSHIP for each layer (a product
of the sigmoids at its lower and upper edges) and takes the
membership-weighted average of the layer kappas. Each layer keeps its value in
its interior; only the boundaries blend. The same 5mm capsule then reaches
R=15.1mm at its midpoint, as intended.

HONESTY ABOUT THE NUMBERS
-------------------------
The layer thicknesses and per-layer kappas below are PLAUSIBLE, not measured.
The anatomy (the sequence of media, the capsule as a distinct thin
high-contrast layer) is grounded; the specific millimetres and curvatures are
not. The kappa range is anchored to one real same-needle-different-medium
figure: a minimum turning radius of 1.5cm in artificial tissue versus 3.4cm in
liver.

Document them as what they are -- a synthetic ground truth standing in for an
imaging-derived tissue-type prior -- exactly as the obstacles are documented
as geometric abstractions rather than segmented anatomy. The clinical analogue
of 4c's learned field is a POPULATION model conditioned on segmentation, not a
per-patient measurement, and that uncertainty is what motivates 4d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TissueLayer:
    """One medium in the insertion path.

    upper_edge_mm: the y coordinate where this layer ends and the next begins.
        None for the deepest layer, which extends to the end of the workspace.
    kappa: the curvature the needle achieves in this medium [1/mm].
    name: for plots and debugging -- fat, muscle, capsule, gland.
    """

    kappa: float
    upper_edge_mm: float | None
    name: str


# A transperineal prostate path, as a first plausible profile. Insertion at
# y=20 heading into the tissue.
#
#   fat      y < 45     R = 34mm   soft, barely supported, deflects easily
#   muscle   45 - 65    R = 22mm   denser, more support
#   capsule  65 - 70    R = 15mm   thin and stiff; the literature singles it
#                                  out as mechanically distinct, and a needle
#                                  "anchors in the capsule" during insertion
#   gland    y > 70     R = 28mm   softer again than the capsule
#
# NOT MEASURED -- see the honesty note in the module docstring.
PROSTATE_PATH = (
    TissueLayer(kappa=1 / 34, upper_edge_mm=45.0, name="fat"),
    TissueLayer(kappa=1 / 22, upper_edge_mm=65.0, name="muscle"),
    TissueLayer(kappa=1 / 15, upper_edge_mm=70.0, name="capsule"),
    TissueLayer(kappa=1 / 28, upper_edge_mm=None, name="gland"),
)


@dataclass(frozen=True)
class TissueField:
    """A depth-dependent curvature field: kappa as a function of position.

    Only y is used for now -- see the depth argument in the module docstring.
    The signature takes both x and y so that a laterally varying field can be
    added later without changing every call site.

    FROZEN so it stays hashable. `NeedleParams` is `frozen=True` precisely so
    it is safe to hash; once it can carry a `TissueField` (Phase 4b), a
    non-frozen (unhashable) field would silently strip that guarantee -- a
    frozen dataclass holding an unhashable field raises only when hash() is
    actually called, so nothing would catch it until it bit. This holds only a
    tuple of frozen TissueLayer and a float, so freezing costs nothing.
    """

    layers: tuple[TissueLayer, ...] = PROSTATE_PATH

    # Width of the blend at each interface [mm]. Set from the needle's own
    # interaction length rather than for numerical convenience: the bevel is
    # finite and the tip deforms tissue ahead of itself, so a boundary is felt
    # over millimetres.
    transition_mm: float = 2.0

    def kappa_at(self, x: float, y: float) -> float:
        """Curvature at a position [1/mm].

        IMPLEMENT ME. Use the MEMBERSHIP-WEIGHTED construction, not chained
        sigmoids -- see the module docstring for why, and note the capsule is
        thin enough that the difference is not academic.

        For each layer, membership is the product of:
            sigmoid rising at its LOWER edge   (1 for the first layer)
            sigmoid falling at its UPPER edge  (1 for the last layer)
        with the sigmoid
            s(y, edge) = 1 / (1 + exp(-(y - edge) / (transition_mm / 4)))
        The /4 makes `transition_mm` roughly the full width of the blend rather
        than its scale parameter; state whatever convention you settle on so
        the number means something.

        Then return sum(membership * kappa) / sum(membership).

        Guard the degenerate case where all memberships underflow to zero far
        outside the layer stack -- return the nearest layer's kappa rather than
        dividing by zero.
        """

        def sig(edge):
            """Smooth step from 0 below `edge` to 1 above it.

            The exponent is clamped because math.exp raises OverflowError above
            ~709.78 (where the result exceeds float64's ~1.8e308). At |z| = 700
            the sigmoid has already saturated to full precision -- exp(700) is
            ~1e304, so 1/(1+1e304) is ~1e-304, indistinguishable from 0.0 for
            any purpose here -- so returning the limit directly loses nothing.

            The margin is smaller than it looks. The exponent is
            (y - edge) / (transition_mm / 4), so with transition_mm = 2 it
            overflows once y is ~355mm from an edge. The 150mm workspace and
            the -20..200 test sweep peak around 130, but halving transition_mm
            doubles every exponent, and a rescaled workspace would close the
            gap fast. The failure mode is a crash, not a wrong number, so the
            clamp is cheap insurance rather than premature defence.

            The z < -700 branch is strictly redundant -- exp underflows
            silently to 0.0, giving exactly 1.0 -- but it keeps the function
            visibly symmetric and skips a pointless exp call.
            """
            z = -(y - edge) / (self.transition_mm / 4)
            if z > 700:
                return 0.0
            if z < -700:
                return 1.0
            return 1.0 / (1.0 + math.exp(z))

        numer = 0.0
        denom = 0.0
        lower_edge_mm = None
        for tissue_layer in self.layers:
            m = 1.0
            if lower_edge_mm is not None:
                m *= sig(lower_edge_mm)
            if tissue_layer.upper_edge_mm is not None:
                m *= 1.0 - sig(tissue_layer.upper_edge_mm)
            numer += m * tissue_layer.kappa
            denom += m
            lower_edge_mm = tissue_layer.upper_edge_mm

        if denom == 0.0:
            if y < (self.layers[0].upper_edge_mm or 0):
                return self.layers[0].kappa
            else:
                return self.layers[-1].kappa

        return numer / denom

    def kappa_gradient_at(self, x: float, y: float) -> tuple[float, float]:
        """(d kappa/dx, d kappa/dy) at a position.

        IMPLEMENT ME, and it is needed for a reason worth understanding.

        Once kappa depends on position, theta_dot = v*kappa(x,y)*b depends on
        x and y, so the motion model's Jacobian gains terms it did not have:

          - `NeedleEKF.jacobian`'s theta row stops being [0, 0, 1]
          - `AugmentedNeedleEKF.jacobian`'s fourth row stops being [0,0,0,1]

        BOTH of those are currently asserted by tests
        (`test_jacobian_structure`, `test_log_kappa_row_is_identity`), and both
        SHOULD FAIL once a field is in use. That is the designed signal that
        the Jacobians need their new terms -- the Task 1 docstring anticipated
        it. Do not delete those tests; make them conditional on whether a
        field is in play, so they keep guarding the constant-kappa case.

        A finite-difference implementation is acceptable and is what the tests
        should check an analytic version against, if you write one.
        """
        # Central differences. An analytic derivative of the
        # membership-weighted blend is a quotient rule over a sum of sigmoid
        # products -- considerable algebra for something that would then need
        # checking against finite differences anyway. h = 1e-5 sits well below
        # the 2mm transition width (so the step does not smear the boundary)
        # and well above the scale where cancellation eats the result.
        h = 1e-5
        dk_dx = (self.kappa_at(x + h, y) - self.kappa_at(x - h, y)) / (2 * h)
        dk_dy = (self.kappa_at(x, y + h) - self.kappa_at(x, y - h)) / (2 * h)
        return (dk_dx, dk_dy)

    def layer_at(self, x: float, y: float) -> str:
        """Name of the dominant layer at a position -- for plots and for
        debugging a trajectory that behaves oddly at a particular depth.
        IMPLEMENT (small)."""
        for tissue_layer in self.layers:
            if tissue_layer.upper_edge_mm is None or y <= tissue_layer.upper_edge_mm:
                return tissue_layer.name
