"""Summary figure of a second-degree-moment (SDM) inversion of one earthquake
(the style of Fig. 4 of Seo et al., 2026).

Panels:
  (a) map-view location of the target relative to its EGF, with the target's
      focal mechanism; the identified fault plane is drawn as a thick red arc
  (b) a reference SDM ellipse (semi-axes Lc and Wc) and centroid-directivity
      arrow of length |v0| x tau_c, over posterior samples
  (c) observed and predicted apparent durations of direct S and SmS
  (d) the same for direct P
plus a symbol key.  Axis ranges follow the data unless given.
"""
from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import MaxNLocator, MultipleLocator

from ..config import SEED
from .style import use_style

TICK_SIZE = 16
LABEL_SIZE = 22
ENSEMBLE_COLOR = "#77C3EC"


# ------------------------------------------------------------------ helpers
def relative_offset(azimuth, dip, length):
    """Map-view offset (east, north) [m] of the target from the EGF.

    azimuth [deg] and dip [deg from vertical down] are the direction of the
    offset vector of length ``length`` [m] (``relloc.RelLoc`` fields).
    """
    az, dp = np.deg2rad(azimuth), np.deg2rad(dip)
    horizontal = length * np.sin(dp)
    return horizontal * np.sin(az), horizontal * np.cos(az)


def nodal_plane_curve(strike, dip, n=200):
    """Trace of a nodal plane on a lower-hemisphere equal-area (Schmidt)
    projection of unit radius; returns (x east, y north).

    Slip vectors in the plane are generated for rakes -180..180 deg
    (Aki & Richards convention, z down) and only those pointing down are kept.
    """
    rake = np.deg2rad(np.linspace(-180, 180, n))
    s, d = np.deg2rad(strike), np.deg2rad(dip)
    vn = np.cos(rake) * np.cos(s) - np.sin(rake) * np.cos(d) * np.sin(s)
    ve = np.cos(rake) * np.sin(s) + np.sin(rake) * np.cos(d) * np.cos(s)
    vd = np.sin(rake) * np.sin(d)
    keep = vd >= 0
    vn, ve, vd = vn[keep], ve[keep], vd[keep]
    radius = np.sqrt(2) * np.sin(np.arccos(vd) / 2)
    az = np.arctan2(ve, vn)
    return radius * np.sin(az), radius * np.cos(az)


class _BeachballHandler(HandlerBase):
    """Legend handle drawn as a small beach ball."""

    def __init__(self, mechanism, facecolor="k", **kwargs):
        self.mechanism = mechanism
        self.facecolor = facecolor
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        from obspy.imaging.beachball import beach
        return [beach(self.mechanism, xy=(width / 2, height / 2), width=height * 2.0,
                      linewidth=1, facecolor=self.facecolor)]


def _finish_axes(ax, label, label_x):
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.minorticks_on()
    ax.text(label_x, 1.05, label, transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="top", ha="right")


# ------------------------------------------------------------------ panels
def _nice(value, step):
    return float(np.ceil(value / step) * step)


def plot_relative_location(ax, offset, mechanism, extent=None, guide_length=None, ball_width=None):
    """Panel (a): EGF at the origin (red circle), target beach ball at
    ``offset`` = (east, north) [m], dashed line along strike, and the chosen
    nodal plane ``mechanism`` = (strike, dip, rake) as a thick red arc.
    ``extent`` [m] is the half-width of the panel (default: at least 200 m,
    enough for the offset); the strike line is 1.5 x and the beach ball
    0.3 x ``extent`` by default."""
    from obspy.imaging.beachball import beach
    x, y = offset
    if extent is None:
        extent = max(200.0, _nice(2.0 * np.hypot(x, y), 100.0))
    guide_length = 1.5 * extent if guide_length is None else guide_length
    ball_width = 0.3 * extent if ball_width is None else ball_width
    strike, dip, _ = mechanism
    s = np.deg2rad(strike)
    ax.plot([x + guide_length * np.sin(s), x - guide_length * np.sin(s)],
            [y + guide_length * np.cos(s), y - guide_length * np.cos(s)],
            color="gray", linestyle="--", lw=2, alpha=0.7, zorder=1)
    ax.scatter(0, 0, s=100, ec="r", facecolors="none", lw=2, zorder=10)
    ax.add_collection(beach(list(mechanism), xy=(x, y), width=ball_width, facecolor="k",
                            linewidth=1, alpha=1.0, zorder=20))
    cx, cy = nodal_plane_curve(strike, dip)
    r = ball_width / 2
    ax.plot(cx * r + x, cy * r + y, color="red", linewidth=3.5, zorder=25)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xlabel="Distance (m)", ylabel="Distance (m)")
    ax.set_xticks(np.linspace(-extent, extent, 5))
    ax.set_yticks(np.linspace(-extent, extent, 5))
    ax.grid(True, alpha=0.5, linestyle="--")


def plot_ellipses(ax, best, samples, indices, extent=None):
    """Panel (b): posterior samples ``indices`` (gray ellipses, light-blue
    arrows) and the best-fitting model (black ellipse, blue arrow).

    ``best`` and ``samples`` are outputs of ``moments.physical``.  Ellipse
    semi-axes are Lc and Wc rotated by theta^Lc; arrows are centred on the
    centroid, point along theta^v0 and are |v0| tau_c long.  The y axis
    points up-dip, so tick labels are negated to read 'along dip' downward.
    ``extent`` [m] is the half-width of the panel (default: 1.15 x the larger
    of the reference Lc and the 97.5th percentile of the drawn samples' Lc,
    rounded up to 100 m).
    """
    if extent is None:
        lc = max(best["Lc"], np.percentile(np.asarray(samples["Lc"])[indices], 97.5))
        extent = _nice(1.15 * lc, 100.0)
    for i in indices:
        lc, wc, rk = samples["Lc"][i], samples["Wc"][i], np.deg2rad(samples["rake"][i])
        half = lc / 2 * samples["dir_ratio"][i]
        ax.add_patch(patches.Ellipse((0, 0), 2 * lc, 2 * wc, angle=samples["angle"][i], ec="k",
                                     linewidth=0.1, fill=False, zorder=2, alpha=0.2))
        if samples["v0"][i] > 0:
            ax.arrow(-half * np.cos(rk), -half * np.sin(rk), 2 * half * np.cos(rk),
                     2 * half * np.sin(rk), width=10, color=ENSEMBLE_COLOR, zorder=10, alpha=0.05)
    rk = np.deg2rad(best["rake"])
    half = best["Lc"] / 2 * best["dir_ratio"]
    ax.add_patch(patches.Ellipse((0, 0), 2 * best["Lc"], 2 * best["Wc"], angle=best["angle"],
                                 linewidth=4, edgecolor="k", fill=False, zorder=20))
    if best["v0"] > 0:
        ax.arrow(-half * np.cos(rk), -half * np.sin(rk), 2 * half * np.cos(rk),
                 2 * half * np.sin(rk), width=10, color="b", zorder=20)
    ticks = np.arange(-extent, extent + 1e-9, 100.0 if extent <= 400 else _nice(extent / 3, 100.0))
    ticks = ticks.astype(int) if np.allclose(ticks, np.round(ticks)) else ticks
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xlabel="Along strike (m)", ylabel="Along dip (m)")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(-ticks)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_durations(ax, azimuth, phase, observed, predicted, phases, ymax=None):
    """Panels (c, d): observed (coloured) and predicted (black) apparent
    durations versus station azimuth for the listed ``phases``.  ``ymax`` [s]
    is the top of the axis (default: 1.15 x the largest duration, rounded up;
    the ticks are then chosen automatically)."""
    style = {"S": dict(marker="o", color="b"), "P": dict(marker="o", color="g"),
             "SmS": dict(marker="x", color="purple")}
    azimuth, phase = np.asarray(azimuth, float), np.asarray(phase)
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    for ph in phases:
        m = phase == ph
        st = style[ph]
        if st["marker"] == "o":
            ax.scatter(azimuth[m], observed[m], color=st["color"], s=70, lw=2.0, facecolors="none")
            ax.scatter(azimuth[m], predicted[m], color="k", s=70, lw=1.5, facecolors="none")
        else:
            ax.scatter(azimuth[m], observed[m], color=st["color"], s=70, lw=2.0, marker="x")
            ax.scatter(azimuth[m], predicted[m], color="k", s=70, lw=1.5, marker="x")
    if ymax is None:
        top = 1.15 * max(np.max(observed), np.max(predicted))
        ymax = _nice(top, 0.02 if top <= 0.2 else 0.1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    else:
        ax.set_yticks(np.arange(0.0, ymax + 0.02, 0.02))
    ax.set(xlim=(-5, 365), ylim=(0.0, ymax), xlabel="Azimuth (°)",
           ylabel=r"Apparent $\tau_c$ (s)")
    ax.set_xticks(np.arange(0, 405, 45))
    ax.grid(True, alpha=0.5, linestyle="--")


def plot_symbol_key(ax, mechanism, target_label, egf_label, rng, best_label="Best-fit solution"):
    """Right-hand symbol key (two-row panel) with the drawn ellipse symbols."""
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    ax.text(0.5, 0.975, "Symbols", ha="center", va="top", fontsize=15, fontweight="bold")
    target = mlines.Line2D([], [], color="none", label=target_label)
    handles = [
        target,
        mlines.Line2D([], [], color="white", marker="o", markerfacecolor="none", markeredgecolor="r",
                      markeredgewidth=2, markersize=10, label=egf_label),
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2, alpha=0.7,
                      label=f"Strike = {float(mechanism[0])}°"),
        mlines.Line2D([], [], color="none", marker="o", markeredgecolor="b", markerfacecolor="none",
                      markersize=8, markeredgewidth=2, label="Direct S (observed)"),
        mlines.Line2D([], [], color="none", marker="o", markeredgecolor="g", markerfacecolor="none",
                      markersize=8, markeredgewidth=2, label="Direct P (observed)"),
        mlines.Line2D([], [], color="none", marker="x", markeredgecolor="purple", markersize=8,
                      markeredgewidth=2, label="SmS (observed)"),
        mlines.Line2D([], [], color="none", marker="o", markeredgecolor="k", markerfacecolor="none",
                      markersize=8, markeredgewidth=1.5, label="Direct S/P (predicted)"),
        mlines.Line2D([], [], color="none", marker="x", markeredgecolor="k", markersize=8,
                      markeredgewidth=1.5, label="SmS (predicted)"),
    ]
    ax.legend(handles=handles, handler_map={target: _BeachballHandler(list(mechanism))},
              prop={"size": 13}, loc="upper left", bbox_to_anchor=(0.05, 0.93), frameon=False,
              labelspacing=2.0)
    y_best, y_ens = 0.25, 0.10          # as published (0.30, 0.15 in the export)
    ax.add_patch(patches.Ellipse((0.15, y_best), width=0.15, height=0.08, angle=30, ec="k",
                                 linewidth=3, fill=False, zorder=5))
    ax.arrow(0.15, y_best, 0.05, -0.04, head_width=0.02, color="b", zorder=6)
    ax.text(0.28, y_best, best_label, ha="left", va="center", fontsize=14)
    for _ in range(50):
        w, h = 0.15 + rng.uniform(-0.03, 0.03), 0.08 + rng.uniform(-0.02, 0.02)
        a = 30 + rng.uniform(-10, 10)
        ax.add_patch(patches.Ellipse((0.15, y_ens), width=w, height=h, angle=a, ec="k",
                                     linewidth=0.1, fill=False, alpha=0.3, zorder=1))
        dx, dy = 0.05 + rng.uniform(-0.01, 0.01), -0.04 + rng.uniform(-0.01, 0.01)
        ax.arrow(0.15, y_ens, dx, dy, head_width=0.02, color=ENSEMBLE_COLOR, alpha=0.1, zorder=2)
    ax.text(0.28, y_ens, "Posterior ensemble", ha="left", va="center", fontsize=14)


# ------------------------------------------------------------------ figure
def plot_sdm_summary(offset, mechanism, best, samples, azimuth, phase, observed, predicted,
                     target_label="Target", egf_label="EGF", best_label="Best-fit solution",
                     n_samples=500, seed=SEED, dpi=150, location_extent=None, ellipse_extent=None,
                     tauc_max=None):
    """Assemble the four-panel summary figure.

    offset      (east, north) of the target relative to the EGF [m]
                (``relative_offset`` of a ``relloc.RelLoc``)
    mechanism   (strike, dip, rake) of the identified fault plane [deg]
    best        ``moments.physical`` of the model drawn in black (e.g. the
                deterministic SDP solution or the posterior medoid); its
                predictions are the black symbols in (c) and (d)
    samples     ``moments.physical`` of the posterior ensemble
    azimuth, phase, observed, predicted
                one entry per measured duration [deg, -, s, s]
    target_label, egf_label, best_label
                texts of the symbol key
    n_samples   posterior samples drawn in (b), chosen without replacement
                with ``numpy.random.RandomState(seed)``
    location_extent, ellipse_extent [m], tauc_max [s]
                axis ranges of (a), (b) and (c, d); default from the data

    Returns (fig, axes) with axes keyed "a", "b", "c", "d", "key".
    """
    use_style()
    rng = np.random.RandomState(seed)
    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1.5, 0.8])
    axes = {"a": fig.add_subplot(gs[0, 0]), "c": fig.add_subplot(gs[0, 1]),
            "b": fig.add_subplot(gs[1, 0]), "d": fig.add_subplot(gs[1, 1]),
            "key": fig.add_subplot(gs[:, 2])}

    plot_relative_location(axes["a"], offset, mechanism, extent=location_extent)
    _finish_axes(axes["a"], "(a)", -0.20)

    if tauc_max is None:
        tauc_max = _nice(1.15 * max(np.max(observed), np.max(predicted)),
                         0.02 if np.max(observed) <= 0.17 else 0.1)
    plot_durations(axes["c"], azimuth, phase, observed, predicted, ["S", "SmS"], ymax=tauc_max)
    _finish_axes(axes["c"], "(c)", -0.11)

    indices = rng.choice(len(samples["Lc"]), n_samples, replace=False)
    plot_ellipses(axes["b"], best, samples, indices, extent=ellipse_extent)
    _finish_axes(axes["b"], "(b)", -0.20)

    plot_durations(axes["d"], azimuth, phase, observed, predicted, ["P"], ymax=tauc_max)
    _finish_axes(axes["d"], "(d)", -0.11)

    plot_symbol_key(axes["key"], mechanism, target_label, egf_label, rng, best_label=best_label)
    return fig, axes
