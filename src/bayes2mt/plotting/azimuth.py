"""ASTFs of one earthquake sorted by station azimuth (the style of Fig. 3 of
Seo et al., 2026).

One panel per phase (direct P, direct S, SmS by default).  Each ASTF is scaled
to unit area and drawn at its station azimuth; one station can be highlighted
(thick line and red label).  Short black bars mark the onset times predicted
from the relative location of the target and its EGF
(``relloc.predict_onsets``): with origin-time alignment, an ASTF starts at the
differential travel time.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from ..pld import normalize_area
from .style import use_style

PANELS = (("Direct P ASTFs", "g"), ("Direct S ASTFs", "b"), ("SmS ASTFs", "purple"))


def _panel(ax, astfs, azimuths, onsets, color, title, highlight, scale, xlim):
    for astf, az, onset in zip(astfs, azimuths, onsets):
        f = scale * normalize_area(astf.t, astf.f) + az
        ax.fill_between(astf.t, az, f, color=color, alpha=0.3, zorder=9)
        ax.vlines(x=onset, ymin=az - 2, ymax=az + 2, color="k", lw=2, zorder=20)
        if astf.code == highlight:
            ax.text(s=astf.code, x=xlim[0] + 0.005, y=az, color="r", fontsize=14,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5), zorder=30)
            ax.plot(astf.t, f, lw=3.0, color=color, alpha=1.0, zorder=10)
        else:
            ax.plot(astf.t, f, lw=1.5, color=color, alpha=1.0, zorder=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(-10, 380)
    ax.xaxis.set_major_locator(MultipleLocator(_time_step(xlim)))
    ax.set_yticks(np.arange(0, 390, 30))
    ax.tick_params(axis="y", left=True, labelleft=True)
    ax.set_xlabel("Time (s)", fontsize=20, labelpad=10)
    ax.set_title(title, fontsize=20, pad=10)
    ax.tick_params(axis="both", labelsize=16)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", length=2, width=0.5)
    ax.grid(True, alpha=0.5, linestyle="--", lw=0.5)


def _time_step(xlim):
    span = xlim[1] - xlim[0]
    return 0.05 if span <= 0.5 else 0.1 if span <= 1.0 else 0.2


def time_limits(panels, margin=0.02, step=0.05):
    """Time range covering every ASTF support and onset bar of ``panels``,
    widened by ``margin`` and rounded outward to ``step`` [s]."""
    lo, hi = [], []
    for astfs, _, onsets in panels:
        for a in astfs:
            nz = np.flatnonzero(a.f)
            if len(nz):
                lo.append(a.t[nz[0]]); hi.append(a.t[nz[-1]])
        lo.extend(np.atleast_1d(onsets).tolist())
    lo, hi = min(lo) - margin, max(hi) + margin
    return (np.floor(lo / step) * step, np.ceil(hi / step) * step)


def plot_astfs_by_azimuth(panels, highlight=None, scale=0.7, xlim=None, titles=PANELS):
    """ASTFs sorted by azimuth, one panel per phase.

    panels   : (astfs, azimuths [deg], onsets [s]) per panel, by default for
               direct P, direct S and SmS; ``astfs`` are ``io.ASTF`` objects
               (already quality-controlled), ``onsets`` the predicted onset
               times (``relloc.predict_onsets``)
    highlight: station drawn with a thick line and a red label (optional)
    scale    : height [deg of azimuth] of a unit-area ASTF per second
    xlim     : time range [s]; default: every support and onset (``time_limits``)
    titles   : (title, colour) per panel

    Returns (fig, axes).
    """
    use_style()
    xlim = time_limits(panels) if xlim is None else xlim
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 12), sharey=True,
                             gridspec_kw={"wspace": 0.20}, squeeze=False)
    axes = axes[0]
    for ax, (title, color), (astfs, azimuths, onsets) in zip(axes, titles, panels):
        _panel(ax, astfs, azimuths, onsets, color, title, highlight, scale, xlim)
    axes[0].set_ylabel("Azimuth (°)", fontsize=20, labelpad=10)
    # panel labels: left edge at the left edge of the y-axis label, baseline on the panel-title
    # baseline; the same offset from every panel
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dx = (axes[0].yaxis.label.get_window_extent(renderer).x0
          - axes[0].get_window_extent(renderer).x0)
    for i, ax in enumerate(axes):
        ax.text(dx / ax.get_window_extent(renderer).width, ax.title.get_position()[1],
                f"({chr(97 + i)})", transform=ax.title.get_transform(), ha="left", va="baseline",
                fontsize=20)
    return fig, axes
