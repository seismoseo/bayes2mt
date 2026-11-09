"""Corner plot of the posterior of one earthquake (the style of Fig. 5 of
Seo et al., 2026).

Diagonal: marginal posterior PDFs as a Silverman kernel density estimate
(blue), the 95 % highest-density interval (HDI) shaded light blue with dashed
bounds, the MAP model (red) and the medoid (orange).  Below the diagonal:
joint PDFs as 2-D histograms annotated with the Pearson correlation r.
Angles are HDI'd on the circle, and correlated after centring each angle
sample on its circular mean so the wrap-around does not distort r.
Axis ranges and ticks are given per parameter by a ``Panel``;
``auto_panels`` derives them from the samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from ..moments import circular_hdi, hdi
from .style import use_style


@dataclass
class Panel:
    """One parameter of the corner plot.

    key     key in the samples dict (``moments.physical`` names, plus
            "stress_drop" in MPa)
    label   axis label
    limits  axis range (also the 2-D histogram range)
    major, minor, ticklabels
            tick positions and optional labels
    period  360 or 180 for angles (circular HDI and centring), else None
    log     plot log10 of the samples (stress drop)
    center  for angles: unwrap the samples to center +- period/2 (and use an
            ordinary HDI); None keeps them in [0, period)
    """
    key: str
    label: str
    limits: tuple
    major: list
    minor: list
    ticklabels: list | None = None
    period: float | None = None
    log: bool = False
    center: float | None = None


LABELS = {"tauc": r"$\tau_{c}$ (s)", "Lc": r"$L_{c}$ (m)", "Wc": r"$W_{c}$ (m)",
          "v0": r"$|\mathbf{v_0}|$ (m/s)", "rake": r"$\theta^{\mathbf{v_{0}}}$ (°)",
          "angle": r"$\theta^{L_{c}}$ (°)", "dir_ratio": r"$D$",
          "stress_drop": r"$\Delta\sigma$ (MPa)"}
PERIODS = {"rake": 360.0, "angle": 180.0}


_CHAR_EM = {".": 0.28, "-": 0.33}      # Helvetica advance widths [em]; digits 0.56


def _label_width(text, size):
    return size * sum(_CHAR_EM.get(c, 0.56) for c in text)


def _format(value, step):
    decimals = max(0, -int(np.floor(np.log10(step) + 1e-9)))
    if abs(step * 10 ** decimals - round(step * 10 ** decimals)) > 1e-6:    # 2.5 x 10^k
        decimals += 1
    text = f"{value:.{decimals}f}"
    return f"{abs(value):.{decimals}f}" if float(text) == 0 else text


def _edge_room(label, width_pt, gap_pt, size, width=None):
    """Fraction of the panel width between an edge tick and the panel edge
    such that its label reaches at most a quarter of the gap into the gap."""
    width = _label_width(label, size) if width is None else width
    return max(0.0, width / 2 - gap_pt / 4 + 2) / width_pt


def _ticks(lo, hi, width_pt, size=22, gap_pt=None):
    """Major ticks (steps 1, 2, 2.5, 5 x 10^k) whose labels fit side by side
    in a panel ``width_pt`` wide and do not reach into the neighbouring
    panels (``gap_pt`` apart): the range is padded so that an edge label
    stays clear, and widened outward to the nearest ticks if fewer than two
    fall inside.  Among the steps that fit with the range grown by at most
    60 % (else 200 %), the most ticks (2-4) win, then the least padding.
    Returns (lo, hi, major, minor, labels); minor ticks at half the step."""
    gap_pt = 0.16 * width_pt if gap_pt is None else gap_pt
    span = hi - lo
    k0 = int(np.floor(np.log10(span)))
    candidates = []
    for k in range(k0 - 2, k0 + 2):
        for m in (1, 2, 2.5, 5):
            step = m * 10.0 ** k
            a, b = np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9)
            if b - a < 1:
                a, b = np.floor(lo / step + 1e-9), np.ceil(hi / step - 1e-9)
            major = np.arange(a, b + 1) * step
            if not 2 <= len(major) <= 4:
                continue
            labels = [_format(v, step) for v in major]
            f_lo, f_hi = _edge_room(labels[0], width_pt, gap_pt, size), _edge_room(labels[-1], width_pt, gap_pt, size)
            lo2, hi2 = lo, hi
            for _ in range(5):
                s2 = hi2 - lo2
                lo2, hi2 = min(lo, major[0] - f_lo * s2), max(hi, major[-1] + f_hi * s2)
            room = step / (hi2 - lo2) * width_pt / (max(_label_width(t, size) for t in labels) + 0.5 * size)
            candidates.append((room >= 1, (hi2 - lo2) / span, room, len(major), m != 2.5,
                               lo2, hi2, step, major, labels))
    best = None
    for max_pad in (1.6, 3.0):              # prefer little padding; then more ticks
        fits = [c for c in candidates if c[0] and c[1] <= max_pad]
        if fits:
            best = max(fits, key=lambda c: (c[3], c[4], -c[1]))
            break
    if best is None:                        # labels too long for the panel: most room
        near = [c for c in candidates if c[1] <= 3.0] or candidates
        best = max(near, key=lambda c: c[2])
    lo, hi, step, major, labels = best[5:]
    minor = np.arange(major[0] - step, hi + step, step / 2)
    return float(lo), float(hi), [float(v) for v in major], [float(v) for v in minor], labels


def auto_panels(samples, keys=("tauc", "Lc", "Wc", "v0", "rake", "angle", "dir_ratio", "stress_drop"),
                coverage=(0.2, 99.8), pad=0.08):
    """Panels with ranges from the samples: the ``coverage`` percentiles
    widened by ``pad`` of their span.  Angles are unwrapped around their
    circular mean in [0, period) (so a posterior across 0/360 deg is not
    split); stress drop is shown as log10 over whole decades.  Ticks are
    chosen so that their labels fit the panel width of ``plot_corner``."""
    n = len(keys)
    width_pt = 72 * 24 * (0.985 - 0.085) / (n + 0.16 * (n - 1))    # panel width in plot_corner
    panels = []
    for k in keys:
        period, log = PERIODS.get(k), k == "stress_drop"
        center = None
        x = np.asarray(samples[k], float)
        if k == "angle":
            x = np.mod(x, 180.0)
        if period:
            ang = np.deg2rad(x * 360.0 / period)
            center = float(np.mod(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
                                  * period / 360.0, period))
            x = center + (np.mod(x - center + period / 2, period) - period / 2)
        if log:
            x = np.log10(x)
        lo, hi = np.percentile(x, coverage)
        if log:
            lo, hi = float(np.floor(lo)), float(np.ceil(hi))
            pad_log = _edge_room("", width_pt, 0.16 * width_pt, 22, width=1.6 * 22) * (hi - lo)
            decade_pt = width_pt / (hi - lo + 2 * pad_log)
            step = next(d for d in (1, 2, 3, 5, 10) if d * decade_pt >= 2.1 * 22)   # label 1.6 em + gap
            major = [float(v) for v in np.arange(lo, hi + 0.5, step)]
            minor = [j + np.log10(i) for j in range(int(lo) - 1, int(hi) + 1) for i in range(2, 10)]
            panels.append(Panel(k, LABELS.get(k, k), (lo - pad_log, hi + pad_log), major, minor,
                                [r"$\mathdefault{10^{%d}}$" % v for v in major], log=True))
            continue
        lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)
        if k in ("Wc", "Lc", "tauc", "v0", "dir_ratio"):
            lo = max(lo, 0.0)
        lo, hi, major, minor, labels = _ticks(lo, hi, width_pt)
        if k in ("Wc", "Lc", "tauc", "v0", "dir_ratio"):
            lo = max(lo, 0.0)
        panels.append(Panel(k, LABELS.get(k, k), (float(lo), float(hi)), major, minor, labels,
                            period=period, center=center))
    return panels


def _values(panel, samples):
    x = np.asarray(samples[panel.key], float)
    if panel.key == "angle":
        x = np.mod(x, 180.0)
    if panel.period and panel.center is not None:
        c, p = panel.center, panel.period
        x = c + (np.mod(x - c + p / 2, p) - p / 2)
    return np.log10(x) if panel.log else x


def _center_circular(x, period):
    ang = np.deg2rad(np.asarray(x) * (360.0 / period))
    mean = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) * (period / 360.0)
    return (np.asarray(x) - mean + period / 2) % period


def pearson_matrix(samples, panels):
    """Pearson r between all panel parameters (angles circular-centred)."""
    data = []
    for p in panels:
        x = _values(p, samples)
        data.append(_center_circular(x, p.period) if p.period else x)
    return np.corrcoef(np.vstack(data))


def _apply_ticks(ax, panel, axis):
    axis_obj = ax.xaxis if axis == "x" else ax.yaxis
    axis_obj.set_major_locator(ticker.FixedLocator(np.asarray(panel.major, float)))
    axis_obj.set_minor_locator(ticker.FixedLocator(np.asarray(panel.minor, float)))
    labels = list(panel.ticklabels) if panel.ticklabels is not None else \
        [("%g" % v) for v in panel.major]
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels)


def plot_marginal(ax, x, panel, map_value, medoid_value, prob=0.95):
    """Diagonal panel: KDE, shaded HDI with dashed bounds, MAP and medoid."""
    from arviz.stats import kde
    grid, pdf = kde(x, bw="Silverman")
    ax.plot(grid, pdf, color="blue", lw=2, zorder=20)
    circular = panel.period and panel.center is None
    bounds = circular_hdi(x, prob, panel.period) if circular else [hdi(x, prob)]
    for lo, hi in bounds:
        ax.fill_between(grid, pdf, where=(grid >= lo) & (grid <= hi), color="lightblue",
                        alpha=1.0, zorder=10)
        ax.vlines([lo, hi], ymin=0, ymax=[np.interp(lo, grid, pdf), np.interp(hi, grid, pdf)],
                  color="black", lw=1, linestyles="--", zorder=30)
    ax.axvline(map_value, color="r", lw=2, zorder=40)
    ax.axvline(medoid_value, color="orange", lw=2, zorder=50)
    ax.set_xlim(panel.limits)
    ax.minorticks_on()
    ax.grid(True, alpha=0.5, linestyle="--")


def plot_corner(samples, map_model, medoid_model, panels=None, bins=90, dpi=150):
    """Corner plot of the posterior (the style of Fig. 5 of Seo et al., 2026).

    samples       dict of per-sample arrays (``moments.physical`` keys and
                  "stress_drop" [MPa])
    map_model, medoid_model
                  dicts with the same keys for the two reference models
    panels        list of ``Panel`` (default: ``auto_panels(samples)``)
    Returns (fig, axes) with axes an (n, n) array.
    """
    use_style()
    panels = auto_panels(samples) if panels is None else panels
    n = len(panels)
    data = [_values(p, samples) for p in panels]
    ref = [(_values(p, {p.key: np.atleast_1d(map_model[p.key])})[0],
            _values(p, {p.key: np.atleast_1d(medoid_model[p.key])})[0]) for p in panels]
    r = pearson_matrix(samples, panels)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("white")                       # empty bins stay white

    fig, axes = plt.subplots(n, n, figsize=(24, 24), dpi=dpi)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                plot_marginal(ax, data[i], panels[i], *ref[i])
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.tick_params(axis="both", which="major", labelsize=22)
                _apply_ticks(ax, panels[i], "x")
            elif i > j:
                px, py = panels[j], panels[i]
                ax.hist2d(data[j], data[i], bins=bins, range=[list(px.limits), list(py.limits)],
                          cmap=cmap, vmin=1e-9, rasterized=True)
                ax.set_facecolor("white")
                ax.text(0.96, 0.94, rf"$r$={r[i, j]:.2f}", transform=ax.transAxes, ha="right",
                        va="top", fontsize=24, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
                ax.set_xlim(px.limits)
                ax.set_ylim(py.limits)
                ax.minorticks_on()
                ax.grid(True, alpha=0.5, linestyle="--")
                ax.tick_params(axis="both", which="major", labelsize=22)
                _apply_ticks(ax, px, "x")
                _apply_ticks(ax, py, "y")
            else:
                ax.axis("off")
            # tick labels and axis labels on the outer panels only
            if i == n - 1 and j <= i:
                ax.set_xlabel(panels[j].label, fontsize=26, labelpad=12)
            elif j <= i:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(panels[i].label, fontsize=26, labelpad=12)
            elif i > j:
                ax.set_yticklabels([])
    axes[0, 0].set_ylabel("PDF density", fontsize=26, labelpad=12)

    handles = [mlines.Line2D([], [], color="blue", lw=2, label="KDE"),
               mpatches.Patch(facecolor="lightblue", edgecolor="none", label="95% HDI"),
               mlines.Line2D([], [], color="r", lw=2, label="MAP"),
               mlines.Line2D([], [], color="orange", lw=2, label="Medoid")]
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.135, top=0.985, wspace=0.16,
                        hspace=0.16)
    box = axes[n - 1, 0].get_position()
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(box.x0, box.y0 - 0.052),
                     bbox_transform=fig.transFigure, ncol=4, prop={"size": 26}, frameon=True,
                     borderaxespad=0.0, handletextpad=0.6, columnspacing=2.0)
    leg.get_frame().set_edgecolor("k")
    return fig, axes
