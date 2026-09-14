"""Phase 4c step 2, DIAGNOSTIC: which mechanism breaks the capsule variance?

Gate 2 found the capsule MEAN recovers (R 15.6-17.5 vs true 15.1) while the +/-1
sigma BAND collapses to zero in-capsule coverage as N grows. Three distinct
mechanisms produce that same symptom, and the three candidate fixes each address
only ONE -- so choosing a fix before pinning the cause repeats the
profile-before-optimise error. This script pins the cause before any quantifier
is built.

  (1) NONLINEARITY -- the true posterior is non-Gaussian and wider than the
      Laplace (J^T J)^{-1}. SIGNATURE: frequentist scatter of the estimate
      across independent data realisations >> the reported sigma, AND the mean
      is ~unbiased (converges to truth as N grows). FIX: sampling / RTO -- the
      one you'd naturally build first.
  (2) BIAS not in the probabilistic model -- the GP prior (length-scale 3mm,
      mean 1/29) oversmooths the 2mm capsule and the knot interpolation clips
      its peak. SIGNATURE: the mean is systematically off and FLOORS as N grows;
      the band measures spread around a biased point, so coverage -> 0 with no
      way for a variance fix to help. FIX: none honest -- bias reduction or the
      conservative fallback.
  (3) CONFOUNDING -- the capsule knot trades off against its muscle/gland
      neighbours along a poorly-determined ridge. SIGNATURE: per-knot scatter >>
      block-averaged scatter (aggregation stabilises it), and/or the capsule
      estimate anti-correlates with a neighbour across realisations. FIX: report
      at layer/block resolution, not per-knot.

These are not exclusive; the script prints all signatures and lets the pattern
speak. The N=40 row of Gate 2 (R jumped to 17.5 then back to 15.9 -- ~+/-15%
scatter against a ~+/-0.6% reported sigma) already hinted this is (2)/(3), not
(1); this measures it.

MATCHED MODEL. Data is generated with TissueField and inverted with the GP prior
tuned to it, so this diagnoses the INVERSION'S self-consistency, not transfer to
real tissue. Every verdict carries the label. Robustness to misspecification is
Gate 3 (the substrate that will also validate whatever step 2 produces).

Reuses the Gate 2 fitter wholesale (four_c_gates); nothing under
planning/models/estimation/control is touched.
Run:  python scripts/four_c_step2_diagnostic.py
"""

from __future__ import annotations

import numpy as np
from four_c_gates import (
    CAPSULE,
    TRUE,
    _capsule_dwell_arcs,
    _capsule_knots,
    fit_y,
    make_dataset,
    path_sets,
)

CAP_TRUE = TRUE.kappa_at(0.0, 67.5)  # ~0.0663 (R 15.1mm)
LABEL = "[MATCHED MODEL -- diagnoses the inversion, not transfer to real tissue]"


def build_pool(n_revisit: int, n_dwell: int, seed: int):
    """The Gate 2 path-set: separating revisit arcs + capsule-dwell arcs. Built
    large once, then subsampled per realisation."""
    return path_sets(n_revisit, seed)["revisit"] + _capsule_dwell_arcs(n_dwell, seed)


def _cap_mask(coords):
    return (coords >= CAPSULE[0]) & (coords <= CAPSULE[1])


def _central_knot(coords):
    return int(np.argmin(np.abs(coords - 67.5)))


def _fit_capsule(coords, insertions, seed):
    """One fit; return (block_mean_kappa, central_knot_kappa, mean_reported_sigma,
    full vals, neighbour kappas below/above the capsule)."""
    data = make_dataset(insertions, seed)
    vals, cov = fit_y(coords, data)
    cap = _cap_mask(coords)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    below = coords < CAPSULE[0]
    above = coords > CAPSULE[1]
    # nearest muscle knot below, nearest gland knot above
    musc = float(vals[below][-1]) if below.any() else float("nan")
    gland = float(vals[above][0]) if above.any() else float("nan")
    return (
        float(np.mean(vals[cap])),
        float(vals[_central_knot(coords)]),
        float(np.mean(std[cap])),
        musc,
        gland,
    )


# --- Diagnostic A: frequentist scatter vs reported sigma --------------------


def scatter_vs_sigma(coords, pool, N, M, seed0):
    """Re-collect the whole study M times (fresh arc subsample + fresh noise),
    measure the actual estimator scatter, compare to what the posterior claims,
    and test whether block-averaging stabilises it (confounding probe)."""
    rng = np.random.default_rng(seed0)
    blocks, centrals, sigmas, muscs, glands = [], [], [], [], []
    for m in range(M):
        pick = rng.choice(len(pool), size=N, replace=False)
        bm, ck, sig, mu, gl = _fit_capsule(
            coords, [pool[i] for i in pick], seed0 + 101 * m
        )
        blocks.append(bm)
        centrals.append(ck)
        sigmas.append(sig)
        muscs.append(mu)
        glands.append(gl)
    blocks = np.array(blocks)
    centrals = np.array(centrals)
    block_scatter = float(np.std(blocks))
    central_scatter = float(np.std(centrals))
    rep_sigma = float(np.mean(sigmas))
    bias = float(np.mean(blocks) - CAP_TRUE)
    # confounding sign: correlation of the capsule block with each neighbour.
    corr_musc = (
        float(np.corrcoef(blocks, muscs)[0, 1]) if np.std(muscs) > 0 else float("nan")
    )
    corr_gland = (
        float(np.corrcoef(blocks, glands)[0, 1]) if np.std(glands) > 0 else float("nan")
    )
    return dict(
        N=N,
        M=M,
        mean=float(np.mean(blocks)),
        bias=bias,
        block_scatter=block_scatter,
        central_scatter=central_scatter,
        rep_sigma=rep_sigma,
        scatter_over_sigma=block_scatter / rep_sigma if rep_sigma > 0 else float("inf"),
        bias_over_scatter=(
            abs(bias) / block_scatter if block_scatter > 0 else float("inf")
        ),
        central_over_block=(
            central_scatter / block_scatter if block_scatter > 0 else float("inf")
        ),
        corr_musc=corr_musc,
        corr_gland=corr_gland,
    )


# --- Diagnostic B: does the bias floor as N grows? --------------------------


def bias_floor(coords, pool, Ns, seed0):
    """Single fit at increasing N. If |mean - truth| shrinks toward 0 the bias is
    finite-sample (fixable with more data / a better posterior); if it FLOORS the
    bias is irreducible representation/prior smoothing and no variance fix is
    honest."""
    rng = np.random.default_rng(seed0)
    rows = []
    for N in Ns:
        pick = rng.choice(len(pool), size=N, replace=False)
        bm, _, sig, _, _ = _fit_capsule(coords, [pool[i] for i in pick], seed0 + N)
        rows.append((N, bm, 1.0 / bm, bm - CAP_TRUE, sig))
    return rows


# --- report -----------------------------------------------------------------


def main():
    print(__doc__.split("\n\n")[0])
    print("\n" + LABEL)
    coords = _capsule_knots()
    pool = build_pool(n_revisit=260, n_dwell=120, seed=2)  # ~380 arcs
    print(
        f"true capsule kappa {CAP_TRUE:.4f} (R {1 / CAP_TRUE:.1f}mm); "
        f"pool {len(pool)} arcs; {len(coords)} knots ({int(np.sum(_cap_mask(coords)))} in capsule)"
    )

    print("\n" + "=" * 78)
    print("DIAGNOSTIC A -- frequentist scatter vs the posterior's reported sigma")
    print("=" * 78)
    print(
        "  If scatter >> sigma: the band understates real variability (mechanism 1 or 3).\n"
        "  If |bias| >~ scatter: a systematic offset dominates (mechanism 2).\n"
        "  If central-knot scatter >> block scatter: per-knot confounding (mechanism 3).\n"
    )
    print(
        f"  {'N':>4} {'M':>3} {'mean R':>8} {'bias(k)':>9} {'scatter':>9} "
        f"{'rep_sig':>9} {'sc/sig':>7} {'bias/sc':>8} {'ctrl/blk':>9} "
        f"{'corr musc/gland':>16}"
    )
    for N in (20, 80):
        r = scatter_vs_sigma(coords, pool, N=N, M=12, seed0=400)
        print(
            f"  {r['N']:>4} {r['M']:>3} {1 / r['mean']:>7.1f}m {r['bias']:>+9.4f} "
            f"{r['block_scatter']:>9.4f} {r['rep_sigma']:>9.4f} "
            f"{r['scatter_over_sigma']:>7.1f} {r['bias_over_scatter']:>8.2f} "
            f"{r['central_over_block']:>9.1f} "
            f"{r['corr_musc']:>+7.2f}/{r['corr_gland']:>+6.2f}"
        )

    print("\n" + "=" * 78)
    print("DIAGNOSTIC B -- does the capsule bias FLOOR as N grows?")
    print("=" * 78)
    print(f"  {'N':>4} {'mean kappa':>11} {'mean R':>8} {'bias(k)':>9} {'rep_sig':>9}")
    rows = bias_floor(coords, pool, Ns=(40, 80, 160, 320), seed0=700)
    for N, bm, R, bias, sig in rows:
        print(f"  {N:>4} {bm:>11.4f} {R:>7.1f}m {bias:>+9.4f} {sig:>9.4f}")
    trend = rows[-1][3] / rows[0][3] if rows[0][3] != 0 else float("inf")
    floored = abs(trend) > 0.5  # |bias| at N=320 is still >50% of |bias| at N=40
    print(
        f"\n  |bias| at N={rows[-1][0]} is {abs(trend) * 100:.0f}% of |bias| at N={rows[0][0]} "
        f"-> bias {'FLOORS (irreducible)' if floored else 'shrinks (finite-sample)'}"
    )

    print("\n" + "=" * 78)
    print("READING (mechanism -> admissible fix)")
    print("=" * 78)
    print(
        "  Match the A/B signatures to the three mechanisms in the header. The\n"
        "  consequential distinction: if bias FLOORS (B) or scatter is driven by\n"
        "  bias not variance (A), then sampling/RTO (fix 1) buys nothing and no\n"
        "  variance quantifier is honest -- the density-keyed, risk-directional\n"
        "  conservative serving is the route, and it needs no calibrated posterior."
    )
    print("\n" + LABEL)


if __name__ == "__main__":
    main()
