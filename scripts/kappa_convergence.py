"""Phase 4a convergence + band-honesty check for the augmented EKF.

This is the runnable artifact behind the observability figures quoted in
`src/needlesim/estimation/ekf_augmented.py` (the OBSERVABILITY section) and in
`docs/roadmap.md`. It regenerates the two numbers that were previously produced
by a throwaway scratch script:

    from a 2x wrong prior (guess 1/25, true 1/50), over 1200 steps
        alternating b : ~5.4% of the initial error remaining (final ~1/47.4)
        pure arc      : ~2.1% remaining (final ~1/49.0)

The pure arc converges BETTER, which contradicts the classical observability
intuition that kappa needs sign-flipping to be identifiable -- see the
docstring correction in `ekf_augmented.py`. It does so here because the filter
starts with a confident heading prior (sigma 0.05 rad) and a bad kappa prior,
so heading is pinned and position evidence flows into kappa; flipping b
partially cancels the accumulating position error that carries the signal.

THE HONESTY CHECK (the reason this is a script and not just a plot). A
converging estimate means nothing if the reported uncertainty is dishonest, so
this prints the fraction of steps whose +/- 1 sigma band contains the true
kappa. For a calibrated filter that fraction should be roughly 0.68 (the 1
sigma coverage of a Gaussian) -- much lower means overconfident, ~1.0 means the
band is too wide. The band is MULTIPLICATIVE in kappa-space
(kappa * exp(+/- sd_log)) because the variance is in log units; plotting
kappa +/- sqrt(variance) would be wrong and could dip negative.

Run:  python scripts/kappa_convergence.py
"""

import math

import matplotlib.pyplot as plt

from needlesim.estimation.ekf_augmented import (
    AugmentedEKFConfig,
    run_augmented_estimation,
)
from needlesim.models.unicycle_needle import Control, NeedleParams, State

TRUE = 1.0 / 50.0
GUESS = 1.0 / 25.0  # deliberately 2x off, matching the Phase 3.5 mismatch
V, DT, N = 5.0, 0.05, 1200


def alternating(n, flip_every=100):
    """b flips sign every `flip_every` steps -- the trajectory shape the
    classical intuition says kappa needs to be observable."""
    out, b = [], 1
    for k in range(n):
        if k and k % flip_every == 0:
            b = -b
        out.append((Control(v=V, b=b), DT))
    return out


def pure_arc(n):
    """Constant-b arc -- the case the classical intuition says is poorly
    observable, and the one that in fact converges better here."""
    return [(Control(v=V, b=1), DT)] * n


def band_fraction_inside(history):
    """Fraction of steps whose multiplicative +/- 1 sigma band contains TRUE.
    This is the honesty check: ~0.68 is calibrated, much less is
    overconfident."""
    inside = 0
    for r in history:
        sd = math.sqrt(r.kappa_variance)  # log-kappa units
        lo = r.kappa_estimate * math.exp(-sd)
        hi = r.kappa_estimate * math.exp(+sd)
        if lo <= TRUE <= hi:
            inside += 1
    return inside / len(history)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, ctrls) in zip(
        axes, [("alternating b", alternating(N)), ("pure arc", pure_arc(N))]
    ):
        hist = run_augmented_estimation(
            true_params=NeedleParams(kappa=TRUE),
            initial_kappa_guess=GUESS,
            controls=ctrls,
            start=State(0.0, 0.0, 0.0),
            config=AugmentedEKFConfig(seed=1),
        )
        ks = [r.kappa_estimate for r in hist]
        # band is MULTIPLICATIVE in kappa-space: variance is in log units
        hi = [r.kappa_estimate * math.exp(+math.sqrt(r.kappa_variance)) for r in hist]
        lo = [r.kappa_estimate * math.exp(-math.sqrt(r.kappa_variance)) for r in hist]

        ax.plot(ks, color="tab:blue", label="kappa estimate")
        ax.fill_between(
            range(len(ks)), lo, hi, alpha=0.2, color="tab:blue", label="+/-1 sigma"
        )
        ax.axhline(TRUE, color="black", ls="--", label=f"true 1/{1/TRUE:.0f}")
        ax.axhline(GUESS, color="tab:red", ls=":", label=f"prior 1/{1/GUESS:.0f}")
        ax.set_title(f"{name}: 1/{1/ks[-1]:.1f} after {N} steps")
        ax.set_xlabel("step")
        ax.set_ylabel("kappa")
        ax.legend(fontsize=8)

        err0, err1 = abs(GUESS - TRUE), abs(ks[-1] - TRUE)
        frac = band_fraction_inside(hist)
        print(
            f"{name:14s}: final 1/{1/ks[-1]:6.1f}  "
            f"error {err0:.5f} -> {err1:.5f}  ({err1/err0:.1%} remaining)  "
            f"| true kappa inside +/-1 sigma band on {frac:.1%} of steps"
        )

    out = "docs/figures/kappa_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
