"""The ASTF retrieval procedure at one station (the style of Fig. 2 of
Seo et al., 2026).

Panels:
  (a) target and EGF transverse waveforms around S, with the direct-S and SmS
      deconvolution windows highlighted
  (b) direct S: target window and its reconstruction (EGF convolved with the
      ASTF), with the variance reduction (VR)
  (c) the same for SmS
  (d) the direct-S and SmS ASTFs with their characteristic durations tau_c
      (bar length) and centroid times (dots)
  (e) duration-misfit trade-off curves; dashed lines mark the chosen support
      durations of the projected Landweber deconvolution (PLD)

The waveform panels use their own processing (1-Hz high-pass of the raw
trace, then the window is cut, demeaned and tapered), so the VRs printed in
(b) and (c) agree with the PLD values to ~1e-7 % but are not bit-identical.
The reconstruction in (b) is scaled by its own peak, the one in (c) by the
peak of the data.  Axis ranges of (d) and (e) follow the data unless given.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ..pld import centroid_time, characteristic_duration
from .style import use_style

HIGHPASS = 1.0          # Hz, high-pass applied to the raw traces in (a)-(c)
BEFORE, AFTER = 0.5, 1.0  # s, deconvolution window around each pick
LONG_AFTER = 3.5        # s, length of the panel (a) window after the S pick
COLOR_S, COLOR_SMS = "b", "purple"


@dataclass
class RetrievalWaveforms:
    """Windows of panels (a)-(c), cut as in the original figure notebook.

    Times are seconds from the start of each window.  ``sms_offset`` [s] is
    the SmS pick minus the S pick of the target, used to place the SmS
    window on the panel (a) time axis.
    """
    t_long_main: np.ndarray
    main_long: np.ndarray
    t_long_egf: np.ndarray
    egf_long: np.ndarray
    t_s_main: np.ndarray
    main_s: np.ndarray
    t_s_egf: np.ndarray
    egf_s: np.ndarray
    t_sms_main: np.ndarray
    main_sms: np.ndarray
    t_sms_egf: np.ndarray
    egf_sms: np.ndarray
    sms_offset: float
    t_s: np.ndarray            # direct S window used for (b)
    data_s: np.ndarray
    recon_s: np.ndarray
    vr_s: float
    t_sms: np.ndarray          # SmS window used for (c)
    data_sms: np.ndarray
    recon_sms: np.ndarray
    vr_sms: float


def _variance_reduction(recon, data):
    return (1 - np.sum((data - recon) ** 2) / np.sum(data ** 2)) * 100


def retrieval_waveforms(tr_main, tr_egf, astf_s, astf_sms, pick_s=None, pick_sms=None):
    """Cut and filter the windows of panels (a)-(c).

    tr_main, tr_egf : obspy Traces of the transverse component (accelerometer
                      records already integrated to velocity)
    astf_s, astf_sms: ``io.ASTF`` of direct S and SmS at this station
    pick_s, pick_sms: the target's S and SmS picks [s after the SAC reference
                      time]; default: SAC headers ``t0`` (S) and ``a`` (SmS)

    Both events are cut at the TARGET's pick times (origin-time alignment).
    """
    sac = tr_main.stats.sac
    t_s_pick = float(sac.t0) if pick_s is None else float(pick_s)
    t_sms_pick = float(sac.a) if pick_sms is None else float(pick_sms)

    def ref(tr, pick):
        return tr.stats.starttime - tr.stats.sac.b + pick

    s_main, s_egf = ref(tr_main, t_s_pick), ref(tr_egf, t_s_pick)
    sms_main, sms_egf = ref(tr_main, t_sms_pick), ref(tr_egf, t_sms_pick)
    long_main = (tr_main.copy().filter("highpass", freq=HIGHPASS, corners=4, zerophase=True)
                 .slice(s_main - BEFORE, s_main + LONG_AFTER))
    long_egf = (tr_egf.copy().filter("highpass", freq=HIGHPASS, corners=4, zerophase=True)
                .slice(s_egf - BEFORE, s_egf + LONG_AFTER))

    def cut(tr, t):
        return tr.copy().slice(t - BEFORE, t + AFTER)

    w_s_main, w_s_egf = cut(long_main, s_main), cut(long_egf, s_egf)
    w_sms_main, w_sms_egf = cut(long_main, sms_main), cut(long_egf, sms_egf)

    def reconstruct(w_main, w_egf, astf):
        data = w_main.copy().detrend("demean").taper(0.05)
        egf = w_egf.copy().detrend("demean").taper(0.05).data
        recon = np.convolve(egf, astf.f, "same")[:len(astf.f)]
        return data.times(), data.data, recon, _variance_reduction(recon, data.data)

    t_s, data_s, recon_s, vr_s = reconstruct(w_s_main, w_s_egf, astf_s)
    t_sms, data_sms, recon_sms, vr_sms = reconstruct(w_sms_main, w_sms_egf, astf_sms)
    return RetrievalWaveforms(
        long_main.times(), long_main.data, long_egf.times(), long_egf.data,
        w_s_main.times(), w_s_main.data, w_s_egf.times(), w_s_egf.data,
        w_sms_main.times(), w_sms_main.data, w_sms_egf.times(), w_sms_egf.data,
        t_sms_pick - t_s_pick, t_s, data_s, recon_s, float(vr_s),
        t_sms, data_sms, recon_sms, float(vr_sms))


def _label(ax, text, x, y):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="right")


def _panel_waveforms(ax, wf, station, distance_km, azimuth, ml_target, ml_egf):
    peak_main, peak_egf = np.max(np.abs(wf.main_long)), np.max(np.abs(wf.egf_long))
    ax.plot(wf.t_long_main, wf.main_long / peak_main + 2.0, color="k", lw=1.5)
    ax.plot(wf.t_long_egf, wf.egf_long / peak_egf, color="k", lw=1.5)
    ax.plot(wf.t_s_main, wf.main_s / peak_main + 2.0, color=COLOR_S, lw=2, label="Direct S")
    ax.plot(wf.t_s_egf, wf.egf_s / peak_egf, color=COLOR_S, lw=2)
    ax.plot(wf.t_sms_main + wf.sms_offset, wf.main_sms / peak_main + 2.0, color=COLOR_SMS, lw=2, label="SmS")
    ax.plot(wf.t_sms_egf + wf.sms_offset, wf.egf_sms / peak_egf, color=COLOR_SMS, lw=2)
    leg = ax.legend(prop={"size": 13}, loc="upper right", ncol=2)
    leg.get_frame().set_edgecolor("none")
    ticks = np.arange(0, 4.5, 0.5)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-1.2, 3.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.1f}" if x % 1 != 0 else f"{int(x)}" for x in ticks], fontsize=16)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=20, labelpad=15)
    ax.text(x=0.05, y=3.15, s=f"Station {station} ($\\Delta$={int(distance_km)} km / Az.={int(azimuth)}°)",
            fontsize=20, weight="bold")
    ax.text(x=-0.45, y=2.2, s="Target", fontsize=20)
    if ml_target is not None:
        ax.text(x=-0.45, y=1.7, s=f"$M_L$ {ml_target:.1f}", fontsize=20)
    ax.text(x=-0.45, y=0.2, s="EGF", fontsize=20)
    if ml_egf is not None:
        ax.text(x=-0.45, y=-0.3, s=f"$M_L$ {ml_egf:.1f}", fontsize=20)
    _label(ax, "(a)", -0.075, 1.02)
    ax.minorticks_on()


def _panel_reconstruction(ax, t, data, recon, vr, color, recon_scale, label, label_x):
    ax.plot(t, data / np.max(np.abs(data)), color="k", lw=2, label="Data")
    ax.plot(t, recon / recon_scale, color=color, lw=2, label=f"Reconstructed (VR={vr:.1f}%)")
    leg = ax.legend(prop={"size": 12.5}, loc="upper left", ncol=2)
    leg.get_frame().set_edgecolor("none")
    ticks = np.arange(0, 1.8, 0.3)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-1.1, 1.4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in ticks], fontsize=16)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=20, labelpad=10)
    _label(ax, label, label_x, 1.05)
    ax.minorticks_on()


def _astf_limits(astf_s, astf_sms):
    lo, hi = [], []
    for a in (astf_s, astf_sms):
        nz = np.flatnonzero(a.f)
        lo.append(a.t[nz[0]]); hi.append(a.t[nz[-1]])
    half = max(0.2, np.ceil(max(abs(min(lo)), abs(max(hi)) + 0.02) / 0.1) * 0.1)
    top = 1.25 * max(np.max(astf_s.f), np.max(astf_sms.f))
    return (-half, half), (-0.03 * top, top)


def _panel_astfs(ax, astf_s, astf_sms, xlim, ylim):
    ax.plot(astf_s.t, astf_s.f, color=COLOR_S, lw=2)
    ax.plot(astf_sms.t, astf_sms.f, color=COLOR_SMS, lw=2)
    for astf, color, height in ((astf_s, COLOR_S, 0.14), (astf_sms, COLOR_SMS, 0.08)):
        tc, t0 = characteristic_duration(astf.t, astf.f), centroid_time(astf.t, astf.f)
        y = np.max(astf.f) * height
        bar = np.linspace(t0 - tc / 2, t0 + tc / 2, 1000)
        ax.plot(bar, np.zeros(1000) + y, color=color, lw=4, label=f"$\\tau_{{c}}$ = {tc:.3f}s")
        ax.scatter(t0, y, s=120, color=color, zorder=10)
    ticks = np.arange(xlim[0], xlim[1] + 1e-9, 0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in ticks], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    leg = ax.legend(prop={"size": 13}, loc="upper left")
    leg.get_frame().set_edgecolor("k")
    ax.set_xlabel("Time (s)", fontsize=20, labelpad=10)
    ax.set_ylabel("Relative moment rate", fontsize=20, labelpad=10)
    _label(ax, "(d)", -0.18, 1.20)
    ax.minorticks_on()


def _tradeoff_limits(curve_s, curve_sms, duration_s, duration_sms):
    hi = max(0.24, np.ceil((max(duration_s, duration_sms) * 1.6) / 0.04) * 0.04)
    sel = [m for d, m in (curve_s, curve_sms) for dd, mm in [(np.asarray(d), np.asarray(m))]
           for m in [mm[(dd >= 0.04) & (dd <= hi)]]]
    top = max(0.1, np.ceil(1.1 * max(np.max(m) for m in sel if len(m)) / 0.1) * 0.1)
    return (0.04, hi), (0.0, top)


def _panel_tradeoff(ax, curve_s, curve_sms, duration_s, duration_sms, xlim, ylim):
    ax.plot(*curve_s, color=COLOR_S, lw=4)
    ax.plot(*curve_sms, color=COLOR_SMS, lw=4)
    ax.axvline(x=duration_s, color=COLOR_S, lw=2, linestyle="--")
    ax.axvline(x=duration_sms, color=COLOR_SMS, lw=2, linestyle="--")
    ticks = np.arange(xlim[0], xlim[1] + 0.08, 0.04)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in ticks], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("ASTF Duration (s)", fontsize=20, labelpad=10)
    ax.set_ylabel("Normalized misfit", fontsize=20, labelpad=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.minorticks_on()
    _label(ax, "(e)", -0.10, 1.20)


def plot_astf_retrieval(wf, astf_s, astf_sms, curve_s, curve_sms, duration_s, duration_sms,
                        station, distance_km, azimuth, ml_target=None, ml_egf=None,
                        astf_xlim=None, astf_ylim=None, tradeoff_xlim=None, tradeoff_ylim=None):
    """Draw the five-panel ASTF retrieval figure.

    wf                  : ``RetrievalWaveforms`` from :func:`retrieval_waveforms`
    astf_s, astf_sms    : ``io.ASTF`` of direct S and SmS
    curve_s, curve_sms  : (durations [s], misfit) from ``pld.tradeoff_curve``
    duration_s/_sms     : support durations chosen for the PLD [s]
    station, distance_km, azimuth [deg], ml_target, ml_egf : panel (a) text
                          (distance and azimuth are truncated to integers;
                          magnitudes are optional)
    astf_xlim, astf_ylim, tradeoff_xlim, tradeoff_ylim : axis ranges of (d)
                          and (e); default from the data

    Returns (fig, axes) with axes in panel order (a)-(e).
    """
    use_style()
    # font.size is set and restored by hand: rc_context would also restore the
    # interactive flag that IPython sets when this is the session's first
    # figure, and later figures of the notebook would then not be shown
    font_size = mpl.rcParams["font.size"]
    mpl.rcParams["font.size"] = 16
    try:
        fig = plt.figure(figsize=(12, 15), dpi=100)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.3, 0.7, 0.7], hspace=0.35, wspace=0.25)
        axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
        _panel_waveforms(axes[0], wf, station, distance_km, azimuth, ml_target, ml_egf)
        _panel_reconstruction(axes[1], wf.t_s, wf.data_s, wf.recon_s, wf.vr_s, COLOR_S,
                              np.max(np.abs(wf.recon_s)), "(b)", -0.18)
        _panel_reconstruction(axes[2], wf.t_sms, wf.data_sms, wf.recon_sms, wf.vr_sms, COLOR_SMS,
                              np.max(np.abs(wf.data_sms)), "(c)", -0.10)
        ax_x, ax_y = _astf_limits(astf_s, astf_sms)
        tr_x, tr_y = _tradeoff_limits(curve_s, curve_sms, duration_s, duration_sms)
        _panel_astfs(axes[3], astf_s, astf_sms, astf_xlim or ax_x, astf_ylim or ax_y)
        _panel_tradeoff(axes[4], curve_s, curve_sms, duration_s, duration_sms,
                        tradeoff_xlim or tr_x, tradeoff_ylim or tr_y)
    finally:
        mpl.rcParams["font.size"] = font_size
    return fig, axes
