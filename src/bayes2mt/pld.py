"""Apparent source time functions (ASTFs) by projected Landweber deconvolution
(PLD; Bertero et al., 1997; Lanza et al., 1999) of a target waveform by an
empirical Green's function (EGF), and their characteristic duration.

Waveforms are aligned on origin time, not on phase picks (Seo et al., 2026):
both SAC files have their origin as reference time, and the target event's
pick is used to cut BOTH windows, so the ASTF onset carries the differential
travel time between target and EGF.

Pick headers read from the SAC files:
    P   : header ``a`` on the vertical (Z) component, window -0.5 .. +0.5 s
    S   : header ``t0`` on the transverse (T) component, window -0.5 .. +1.0 s
    SmS : header ``a`` on the transverse (T) component, window -0.5 .. +1.0 s

The deconvolution iterates
    h   = F + tau conj(U0) (U1 - F U0)            (Landweber step, in frequency)
    F   = P_C[ P_+[h] ]                           (positivity, then compact support)
with tau = 1 / max|U0|^2, U0 = EGF spectrum, U1 = phase-shifted target spectrum.
The support [start, start + duration] of each station is chosen from the
duration-misfit trade-off curve (``tradeoff_curve``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft, ifft

from .io import ASTF

PHASE_SETUP = {          # component, SAC pick header, seconds after the pick
    "P": ("Z", "a", 0.5),
    "S": ("T", "t0", 1.0),
    "SmS": ("T", "a", 1.0),
}
WINDOW_BEFORE = 0.5      # seconds before the pick, all phases


# ------------------------------------------------------------- deconvolution
def _phase_shift_filter(nft, dt, tshift):
    freq = np.fft.fftfreq(nft, d=dt)
    return np.exp(-2j * np.pi * freq * tshift)


def _pl_deconv_many(y, x, dt, start, ends, pad_n, iter_max):
    """PLD for several supports [start, end) at once (one row per end).

    Returns (F, t_F, vr) with F of shape (len(ends), len(x) + pad_n), before
    the pad is removed.  The rows are independent; transforming them together
    gives the same numbers as one support at a time, only faster.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    n0 = len(x)
    x = np.append(x, np.zeros(pad_n))
    y = np.append(y, np.zeros(pad_n))
    n = len(x)
    sr = np.round(1 / dt).astype(int)
    U1 = _phase_shift_filter(n, dt, ((n - 2) / sr) / 2) * fft(y)
    U0 = fft(x)
    t_F = (np.arange(n) - n / 2) * dt
    i1 = np.argmin(np.abs(t_F - start))
    i2 = np.array([np.argmin(np.abs(t_F - end)) for end in ends])
    idx = np.arange(n)
    outside = (idx[None, :] < i1) | (idx[None, :] >= i2[:, None])
    tau_conj_U0 = (1 / (np.max(np.abs(U0)) ** 2)) * np.conjugate(U0)
    F = np.zeros((len(i2), n))
    for _ in range(iter_max):
        FF = fft(F, axis=-1)
        h = ifft(FF + tau_conj_U0 * (U1 - FF * U0), axis=-1).real
        h[h < 0] = 0
        h[outside] = 0
        F = h
    ss = np.sum(y[:n0] ** 2)
    vr = np.array([(1 - np.sum((y[:n0] - np.convolve(x, f, "same")[:n0]) ** 2) / ss) * 100 for f in F])
    return F, t_F, vr


def pl_deconv(y, x, dt, start, end, pad_n, iter_max=100):
    """Projected Landweber deconvolution of target ``y`` by EGF ``x``.

    Parameters
    ----------
    y, x     : target and EGF windows (same length, same sampling)
    dt       : sampling interval [s]
    start,end: support of the ASTF [s] on the deconvolution time axis
    pad_n    : number of zeros appended to both windows (the original
               analysis used ``pad_n = len(x)``)
    iter_max : number of Landweber iterations

    Returns (astf, t, vr): the ASTF and its time axis after removing the pad,
    and the variance reduction [%] of the waveform reconstruction
    ``x * astf`` over the unpadded window.

    Arithmetic is identical to the original notebook implementation; the
    reconstruction VR is evaluated once after the last iteration instead of
    after every iteration (only the final value was ever used).
    """
    F, t_F, vr = _pl_deconv_many(y, x, dt, start, [end], pad_n, iter_max)
    F, vr = F[0], float(vr[0])
    if pad_n == 0:
        return F, t_F, vr
    # (-pad_n)//2 as in the original slice F[pad_n//2:-pad_n//2]; for odd
    # pad_n this trims one more sample at the end than at the start
    return F[pad_n // 2:-pad_n // 2], t_F[pad_n // 2:-pad_n // 2], vr


def tradeoff_curve(y, x, dt, start, durations, iter_max=100):
    """Misfit (1 - VR/100) of PLD as a function of the support length.

    A typical scan is ``durations = arange(0.02, 0.41, dt)``; the duration is
    picked at the knee of this curve.  All durations are deconvolved
    together; each value equals a separate ``pl_deconv`` run.
    """
    durations = np.asarray(durations, float)
    vr = _pl_deconv_many(y, x, dt, start, start + durations, len(x), iter_max)[2]
    return durations, 1 - vr / 100


def default_durations(start, dt):
    """Default scan: ends from start+0.02 to start+0.40 s."""
    ends = np.arange(start + 0.02, start + 0.41, dt)
    return ends - start


# -------------------------------------------------------------- tau_c
def characteristic_duration(t, f):
    """tau_c = 2 sqrt(second central moment) of an ASTF [s]."""
    f = np.asarray(f, float)
    dt = float(t[1] - t[0])
    nz = np.flatnonzero(f)
    if len(nz) == 0:
        return np.nan
    i0, i1 = nz[0] - 1, nz[-1] + 1
    seg = f[i0:i1 + 1]
    idx = np.arange(i0, i1 + 1)
    m = np.sum(seg)
    c = np.sum(seg * idx) / m
    return 2 * np.sqrt((np.sum(seg * (idx - c) ** 2) / m) * dt * dt)


def onset_time(t, f):
    """Onset of an ASTF as recorded in the duration tables (column ``t1``): the
    time of the sample just before the first non-zero sample [s].  With
    origin-time alignment it is the differential travel time used by
    ``relloc``."""
    nz = np.flatnonzero(np.asarray(f))
    return float(t[max(nz[0] - 1, 0)]) if len(nz) else np.nan


def centroid_time(t, f):
    """Moment-weighted centroid time of an ASTF [s]."""
    f = np.asarray(f, float)
    return float(np.sum(f * t) / np.sum(f))


def normalize_area(t, f):
    """Scale an ASTF to unit area (trapezoid rule), for plotting."""
    area = np.trapezoid(f, t) if hasattr(np, "trapezoid") else np.trapz(f, t)
    return np.zeros_like(f) if area == 0 else f / area


# ---------------------------------------------------- waveform preparation
@dataclass
class Pick:
    """Processing choices for one station and phase (one row of a picks table).

    start, duration : ASTF support [s] on the deconvolution time axis
    shift           : extra time shift applied to the EGF window [s]
    channel         : instrument code; "HG" (accelerometer) is integrated to
                      velocity before processing
    interpolate     : resample both traces to 500 Hz (Lanczos, a=20)
    """
    station: str
    phase: str
    channel: str
    network: str = ""
    start: float = 0.0
    duration: float = 0.1
    shift: float = 0.0
    freqmin: float = 1.0
    freqmax: float = 100.0
    iter_max: int = 100
    interpolate: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Windows:
    """Target and EGF windows ready for deconvolution."""
    y: np.ndarray
    x: np.ndarray
    dt: float
    pick: Pick
    traces: tuple = ()        # unfiltered (target, egf) traces, for plotting


def _read_trace(path, channel, interpolate):
    from obspy import read
    tr = read(str(path))[0]
    if channel == "HG":
        tr = tr.detrend("demean").taper(0.05).integrate().detrend("demean").taper(0.05)
    if interpolate:
        tr = tr.interpolate(sampling_rate=500, method="lanczos", a=20)
    return tr


def prepare_windows(main_sac, egf_sac, pick):
    """Cut origin-time-aligned target and EGF windows for one pick."""
    _, header, after = PHASE_SETUP[pick.phase]
    tr_main = _read_trace(main_sac, pick.channel, pick.interpolate)
    tr_egf = _read_trace(egf_sac, pick.channel, pick.interpolate)
    t_pick = float(tr_main.stats.sac[header])
    out = []
    for tr, shift in ((tr_main, 0.0), (tr_egf, pick.shift)):
        t_ref = tr.stats.starttime - tr.stats.sac.b + t_pick + shift
        w = tr.copy().detrend("demean").taper(0.05)
        if pick.freqmax < 0.5 * tr.stats.sampling_rate:
            w = w.filter("bandpass", freqmin=pick.freqmin, freqmax=pick.freqmax, corners=4, zerophase=True)
        else:          # what ObsPy's bandpass does at or above Nyquist, without its warning
            w = w.filter("highpass", freq=pick.freqmin, corners=4, zerophase=True)
        w = w.slice(t_ref - WINDOW_BEFORE, t_ref + after).detrend("demean").taper(0.05)
        out.append(w)
    return Windows(out[0].data, out[1].data, 1 / out[0].stats.sampling_rate, pick, (tr_main, tr_egf))


def retrieve_astf(windows, pick=None):
    """Run PLD with the pick's support and return an ASTF (time axis shifted
    back by the EGF shift).  ``pick`` overrides
    the support (start, duration, iter_max) of the pick the windows were cut
    with, so cached windows can be reused while the support changes."""
    p = pick or windows.pick
    f, t, vr = pl_deconv(windows.y, windows.x, windows.dt, p.start, p.start + p.duration,
                         len(windows.x), p.iter_max)
    return ASTF(t - p.shift, f, float(vr), p.station, p.phase)


# ------------------------------------------------------------ picks tables
PICK_COLUMNS = ["phase", "network", "station", "channel", "start", "duration", "shift",
                "freqmin", "freqmax", "iter_max", "interpolate"]


def read_picks(path):
    """Read a picks table (CSV, one row per station and phase) into Picks."""
    import pandas as pd
    tab = pd.read_csv(path, keep_default_na=False)     # empty network stays ""
    return [Pick(station=str(r["station"]), phase=str(r["phase"]), channel=str(r["channel"]),
                 network=str(r.get("network", "")), start=float(r["start"]),
                 duration=float(r["duration"]), shift=float(r["shift"]),
                 freqmin=float(r["freqmin"]), freqmax=float(r["freqmax"]),
                 iter_max=int(r["iter_max"]), interpolate=bool(r["interpolate"]))
            for _, r in tab.iterrows()]


def write_picks(picks, path):
    """Write Picks to a CSV table (the reproducibility record of an analysis)."""
    import pandas as pd
    pd.DataFrame([{k: getattr(p, k) for k in PICK_COLUMNS} for p in picks]).to_csv(path, index=False)


def retrieve_from_picks(picks, waveform_dir):
    """Re-run PLD for every pick; waveform_dir holds Main/ and EGF/ SAC files.

    Returns a list of ASTFs in the order of ``picks``.
    """
    from .io import sac_pair_paths
    out = []
    for p in picks:
        main, egf = sac_pair_paths(waveform_dir, p.station, p.channel, PHASE_SETUP[p.phase][0],
                                   p.network or None)
        out.append(retrieve_astf(prepare_windows(main, egf, p)))
    return out


def duration_table(astfs, picks, waveform_dir, evla, evlo, evdp, qc=True, model=None):
    """Duration table of ASTFs retrieved with ``picks`` (columns of
    ``io.TABLE_COLUMNS``), ready for the inversion: station coordinates from
    the target SAC header, azimuth from the hypocenter, take-off angles from
    ``geometry.takeoff_angles`` with TauP ``model`` (default
    ``geometry.taup_model()``).  ``qc`` keeps ASTFs with VR >= VR_THRESHOLD."""
    import pandas as pd
    from obspy import read
    from obspy.geodetics import gps2dist_azimuth

    from .config import VR_THRESHOLD
    from .geometry import takeoff_angles
    from .io import TABLE_COLUMNS, sac_pair_paths
    rows = []
    for a, p in zip(astfs, picks):
        if qc and a.vr < VR_THRESHOLD:
            continue
        main, _ = sac_pair_paths(waveform_dir, p.station, p.channel, PHASE_SETUP[p.phase][0], p.network or None)
        h = read(str(main), headonly=True)[0].stats.sac
        lat, lon = (float(np.format_float_positional(np.float32(v))) for v in (h.stla, h.stlo))
        rows.append(dict(code=p.station, latitude=lat, longitude=lon,
                         azimuth=gps2dist_azimuth(evla, evlo, lat, lon)[1], vr=a.vr,
                         tauc=characteristic_duration(a.t, a.f), phase=p.phase,
                         Used=int(a.vr >= VR_THRESHOLD), t1=onset_time(a.t, a.f)))
    tab = pd.DataFrame(rows, columns=TABLE_COLUMNS[:-1])
    tab["toa"] = (takeoff_angles(evla, evlo, evdp, tab["latitude"], tab["longitude"], tab["phase"], model)
                 if len(tab) else [])
    order = tab["phase"].map({"P": 0, "S": 1, "SmS": 2})
    return tab.assign(_o=order).sort_values(["_o", "azimuth"]).drop(columns="_o").reset_index(drop=True)
