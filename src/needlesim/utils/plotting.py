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
