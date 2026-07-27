"""Minimal plotting helpers. Kept separate from the model so the model
stays pure and importable without a display."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_trace(trace, ax=None, label=None, **kwargs):
    """Plot a needle trace (list of State) in the x-y plane."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    xs = [s.x for s in trace]
    ys = [s.y for s in trace]
    ax.plot(xs, ys, label=label, **kwargs)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if label:
        ax.legend()
    return ax


def plot_sdf_background(
    env, ax=None, vmin=-20.0, vmax=20.0, colorbar_label="signed distance [mm]"
):
    """Render a GridEnvironment's baked SDF as a diverging heatmap with the
    zero level set (obstacle boundary) drawn as a contour on top.

    Shared by scripts/eyeball_grid_environment.py and scripts/demo_rrt.py so
    every figure that shows an SDF uses identical colours/limits and any two
    such figures are legible side by side.

    Returns (ax, image) so callers can layer more on top (tree, path, markers)
    and access the image if they want a custom colorbar placement.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    extent = (0, env.width, 0, env.height)
    im = ax.imshow(
        env.sdf, origin="lower", extent=extent, cmap="RdBu", vmin=vmin, vmax=vmax
    )
    ax.contour(
        env.sdf, levels=[0.0], colors="k", linewidths=1.5, extent=extent, origin="lower"
    )
    ax.figure.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    return ax, im
