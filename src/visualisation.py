"""
visualisation.py
----------------
Generates publication-quality figures suitable for IEEE conference papers.

All figures are saved at 300 dpi as PNG and as vector PDF.
Typography and layout follow IEEE double-column formatting conventions.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

if TYPE_CHECKING:
    from evaluation import GenerationStats

# ---------------------------------------------------------------------------
# Global style settings
# ---------------------------------------------------------------------------

# IEEE-appropriate font sizes and line widths
FONT_SIZE_LABEL  = 10
FONT_SIZE_TITLE  = 11
FONT_SIZE_TICK   = 9
FONT_SIZE_LEGEND = 9
LINE_WIDTH       = 1.5
MARKER_SIZE      = 5

# Figure dimensions in inches (fits a single column of an IEEE paper: ~3.5 in)
SINGLE_COL_W = 3.5
SINGLE_COL_H = 2.6

DOUBLE_COL_W = 7.16
DOUBLE_COL_H = 2.6

DPI = 300

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         FONT_SIZE_LABEL,
    "axes.titlesize":    FONT_SIZE_TITLE,
    "axes.labelsize":    FONT_SIZE_LABEL,
    "xtick.labelsize":   FONT_SIZE_TICK,
    "ytick.labelsize":   FONT_SIZE_TICK,
    "legend.fontsize":   FONT_SIZE_LEGEND,
    "lines.linewidth":   LINE_WIDTH,
    "lines.markersize":  MARKER_SIZE,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "figure.dpi":        DPI,
})

COLORS = {
    "best":    "#1f77b4",
    "avg":     "#ff7f0e",
    "worst":   "#d62728",
    "diversity": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------

def plot_best_fitness(stats: list, ax: plt.Axes):
    """Plot best-fitness evolution over generations."""
    gens  = [s.generation for s in stats]
    best  = [s.best_fitness for s in stats]
    ax.plot(gens, best, color=COLORS["best"], marker="o",
            label="Best BAS", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Blind-Spot Awareness Score")
    ax.set_title("Best Fitness vs. Generation")
    ax.set_ylim(bottom=max(0, min(best) - 0.3))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")


def plot_avg_fitness(stats: list, ax: plt.Axes):
    """Plot average fitness evolution over generations."""
    gens = [s.generation for s in stats]
    avg  = [s.avg_fitness for s in stats]
    ax.plot(gens, avg, color=COLORS["avg"], marker="s",
            label="Avg BAS", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Blind-Spot Awareness Score")
    ax.set_title("Average Fitness vs. Generation")
    ax.set_ylim(bottom=max(0, min(avg) - 0.3))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")


def plot_diversity(stats: list, ax: plt.Axes):
    """Plot population diversity over generations."""
    gens      = [s.generation for s in stats]
    diversity = [s.diversity for s in stats]
    ax.plot(gens, diversity, color=COLORS["diversity"], marker="^",
            label="Diversity", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity (unique / total)")
    ax.set_title("Population Diversity vs. Generation")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right")


def plot_combined(stats: list, ax: plt.Axes):
    """Overlay best and average fitness on a single axes."""
    gens  = [s.generation for s in stats]
    best  = [s.best_fitness for s in stats]
    avg   = [s.avg_fitness for s in stats]
    worst = [s.worst_fitness for s in stats]

    ax.plot(gens, best,  color=COLORS["best"],  marker="o",
            label="Best BAS",  linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.plot(gens, avg,   color=COLORS["avg"],   marker="s",
            label="Avg BAS",   linewidth=LINE_WIDTH, markersize=MARKER_SIZE, linestyle="--")
    ax.fill_between(gens, worst, best, alpha=0.10, color=COLORS["best"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Blind-Spot Awareness Score")
    ax.set_title("Fitness Evolution")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, results_dir: str, stem: str):
    """Save figure as PNG (300 dpi) and vector PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(results_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=ext)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def plot_all(stats: list, results_dir: str):
    """
    Generate and save all figures.

    Produces:
      - fitness_best.png / .pdf
      - fitness_avg.png  / .pdf
      - diversity.png    / .pdf
      - fitness_combined.png / .pdf  (suitable for a single IEEE column)

    Parameters
    ----------
    stats : list of GenerationStats
    results_dir : str
        Directory where figures are saved.
    """
    if not stats:
        return

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # --- 1. Best fitness ---
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_H))
    plot_best_fitness(stats, ax)
    fig.tight_layout()
    _save(fig, figures_dir, "fitness_best")

    # --- 2. Average fitness ---
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_H))
    plot_avg_fitness(stats, ax)
    fig.tight_layout()
    _save(fig, figures_dir, "fitness_avg")

    # --- 3. Population diversity ---
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_H))
    plot_diversity(stats, ax)
    fig.tight_layout()
    _save(fig, figures_dir, "diversity")

    # --- 4. Combined (for paper) ---
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, SINGLE_COL_H))
    plot_combined(stats, axes[0])
    plot_diversity(stats, axes[1])
    fig.tight_layout()
    _save(fig, figures_dir, "fitness_combined")

    print(f"Figures saved to {figures_dir}")


# ---------------------------------------------------------------------------
# Standalone demo (generates synthetic data for layout testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _MockStat:
        generation: int
        best_fitness: float
        avg_fitness: float
        worst_fitness: float
        diversity: float
        best_chromosome_id: str = "demo"

    rng = np.random.default_rng(0)
    mock_stats = []
    best = 2.1
    avg  = 1.8
    for g in range(11):
        best = min(best + rng.uniform(0.05, 0.25), 5.0)
        avg  = min(avg  + rng.uniform(0.02, 0.15), best)
        mock_stats.append(_MockStat(
            generation=g,
            best_fitness=round(best, 4),
            avg_fitness=round(avg,  4),
            worst_fitness=round(avg - rng.uniform(0.1, 0.4), 4),
            diversity=round(max(0.2, 1.0 - g * 0.06), 3),
        ))

    out = "/tmp/demo_figures"
    os.makedirs(out, exist_ok=True)
    plot_all(mock_stats, out)
    print("Demo figures written to", out)
