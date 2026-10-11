#!/usr/bin/env python3
"""
plot_style.py — One look for every figure the pipeline writes.

Rules followed here:
  - series colours are taken in a FIXED order (never cycled, never chosen by
    rank), and each series also gets its own marker shape, so a series can
    be told apart without colour
  - thin 2 px lines, markers of at least 8 px, light grid, no top/right box
  - labels and numbers are drawn in neutral ink, never in a series colour
  - every figure has a CSV file with the same numbers next to it
"""

from pathlib import Path

# Fixed categorical order (checked for colour-blind separation)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES_MARKERS = ["o", "s", "^", "D", "v"]

SURFACE = "#fcfcfb"
INK = "#1f1f1e"          # primary text
INK_MUTED = "#6b6a63"    # secondary text, axes
GRID = "#e4e3dd"
NEUTRAL_MARK = "#8a8980" # points that carry no category

FIGURE_DPI = 150


def use_style():
    """Apply the shared matplotlib settings.  Call once before plotting."""
    import matplotlib
    matplotlib.use("Agg")   # write files only; no window, works without a display
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
    })
    return plt


def save_figure(fig, path: Path):
    """Save a figure as PNG and close it (frees memory in long runs)."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
