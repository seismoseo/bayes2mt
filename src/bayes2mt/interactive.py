"""Interactive picker for projected Landweber deconvolution (PLD) in Jupyter.

For each station, set the ASTF support (``start, duration``) and the EGF
``shift``, inspect the duration-misfit trade-off curve (full and zoomed), the
ASTF, the target and EGF windows and the waveform reconstruction, then accept
or reject.  :class:`PLDPicker` records every choice in a picks table so the
result can be re-run exactly (:func:`pld.retrieve_from_picks`).

Every widget callback is a plain method (``select``, ``set_params``,
``accept``, ``reject``, ``table``), so a session can be scripted and tested
without a browser.  Figures are drawn with matplotlib and shown as a PNG in
an ``ipywidgets.Image``: no ipympl and no particular matplotlib backend are
needed, so the widget works in the classic Notebook, JupyterLab, the VS Code
notebook UI and in headless execution (nbclient), where large outputs written
into an ``ipywidgets.Output`` stall until the cell timeout.  The live figure
is built once per station and then updated in place (fixed layout, fast PNG
encoding), so the sliders redraw it while they are dragged (about 0.1 s per
step; a new start or EGF shift adds one trade-off curve, about 0.1 s).
"""
from __future__ import annotations

import pickle
import warnings
from collections import OrderedDict
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from PIL import Image

from . import pld
from .config import VR_THRESHOLD
from .io import ASTF_LAYOUT, TABLE_COLUMNS, load_astf, sac_pair_paths

PARAMS = ("start", "duration", "shift", "freqmin", "freqmax", "iter_max", "interpolate")
# Starting values for a station without a saved pick.
DEFAULT_PICK = dict(start=-0.02, duration=0.10, shift=0.0, freqmin=1.0, freqmax=100.0,
                    iter_max=100, interpolate=False)
DISPLAY_AFTER = 1.0      # target/EGF display window ends 1 s after the pick (all phases)
ZOOM_MAX = 1.5           # upper end of the misfit-zoom slider (misfit > 1 means VR < 0)
# Slider ranges [s] (step 0.01 s = one sample at 100 Hz); widened automatically for a saved pick outside them.
SLIDERS = dict(start=(-0.20, 0.10), duration=(0.02, 0.40), shift=(-0.10, 0.10))
PHASE_COLOR = {"P": "g", "S": "b", "SmS": "purple"}
FIGURE_DPI = 80          # 16 x 9 in figure -> 1280 x 720 px in the widget
_CACHE_SIZE = 32


@dataclass
class StationEntry:
    """One station with a target ("Main") and EGF SAC file for the phase."""
    station: str
    network: str
    channel: str
    azimuth: float       # SAC header ``az`` [deg]
    latitude: float
    longitude: float
    main: Path
    egf: Path


def _sac_float(value):
    """SAC headers are float32; the shortest decimal that round-trips to the
    same float32 recovers the station-table value (e.g. 34.74789)."""
    return float(np.format_float_positional(np.float32(value)))


def find_stations(waveform_dir, phase, stations=None):
    """Stations with both target and EGF records of the phase's component,
    sorted by SAC-header azimuth.  Returns {station: [StationEntry, ...]}
    (more than one entry when a station has several channels)."""
    from obspy import read
    comp = pld.PHASE_SETUP[phase][0]
    found = {}
    for main in sorted((Path(waveform_dir) / "Main").glob(f"*{comp}.sac")):
        parts = main.name.split(".")      # <origin>.<net>.<sta>.<channel><comp>.sac
        if len(parts) != 5:
            continue
        net, sta, chan = parts[1], parts[2], parts[3][:-1]
        if stations is not None and sta not in stations:
            continue
        try:
            main_path, egf_path = sac_pair_paths(waveform_dir, sta, chan, comp, net)
        except FileNotFoundError:
            continue
        h = read(str(main_path), headonly=True)[0].stats.sac
        found.setdefault(sta, []).append(StationEntry(
            sta, net, chan, float(h.az), _sac_float(h.stla), _sac_float(h.stlo), main_path, egf_path))
    return dict(sorted(found.items(), key=lambda kv: kv[1][0].azimuth))


def _cached(cache, key, make):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = cache[key] = make()
    if len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return value


def _window_key(p):
    return (p.station, p.network, p.channel, p.shift, p.freqmin, p.freqmax, p.interpolate)


class PLDPicker:
    """Station-by-station PLD with live diagnostics.

    Parameters
    ----------
    waveform_dir : folder with ``Main/`` and ``EGF/`` SAC files of one event
    phase        : "P", "S" or "SmS"
    picks        : starting values, a picks CSV (any phases) or a list of
                   :class:`pld.Pick`; stations without a pick start from
                   ``defaults``.
    output_dir   : where accepted ASTFs are written (``P/``, ``S/direct/``,
                   ``S/moho/``, see ``io.ASTF_LAYOUT``) and ``picks.csv`` is
                   kept up to date; ``None`` keeps results in memory only.
                   Stations already accepted there (pick + pickle) are restored.
    stations     : optional subset of station codes
    evla, evlo, evdp : hypocenter (deg, deg, km), needed for the duration
                   table (azimuth from the hypocenter, take-off angles)
    defaults     : overrides of :data:`DEFAULT_PICK`
    """

    def __init__(self, waveform_dir, phase, picks=None, output_dir=None, stations=None,
                 evla=None, evlo=None, evdp=None, defaults=None):
        if phase not in pld.PHASE_SETUP:
            raise ValueError(f"phase must be one of {list(pld.PHASE_SETUP)}")
        self.waveform_dir, self.phase = Path(waveform_dir), phase
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.hypocenter = (evla, evlo, evdp)
        candidates = find_stations(waveform_dir, phase, stations)
        if not candidates:
            raise FileNotFoundError(f"no {phase} target/EGF SAC pairs in {waveform_dir}")
        given = {q.station: q for q in self._read_picks(picks)}
        start = {**DEFAULT_PICK, **(defaults or {})}
        self.entries, self.picks = {}, {}
        for sta, options in candidates.items():
            q = given.get(sta)
            e = next((o for o in options if q is not None and o.channel == q.channel), options[0])
            self.entries[sta] = e
            base = q if q is not None and q.channel == e.channel else pld.Pick(sta, phase, e.channel, **start)
            self.picks[sta] = replace(base, network=e.network, channel=e.channel)
        self.order = list(self.entries)
        self.status = dict.fromkeys(self.order, "pending")
        self.accepted = {}                      # station -> (Pick, ASTF)
        self.zoom = {}                          # station -> (lo, hi) misfit range of the zoomed curve
        self.current = self.order[0]
        self._windows, self._curves, self._astfs = OrderedDict(), OrderedDict(), OrderedDict()
        self._ui, self._syncing = None, False
        self._view, self._view_station = None, None     # live figure (artists updated in place)
        self._restore()

    # ------------------------------------------------------------ helpers
    def _read_picks(self, picks):
        if picks is None:
            return []
        items = pld.read_picks(picks) if isinstance(picks, (str, Path)) else list(picks)
        return [q for q in items if q.phase == self.phase]

    def _folder(self):
        subdirs, name, stations_file, azimuths_file = ASTF_LAYOUT[self.phase]
        return self.output_dir / subdirs[0], name, stations_file, azimuths_file

    def _restore(self):
        """Mark stations accepted in an earlier session of the same output_dir."""
        if self.output_dir is None or not (self.output_dir / "picks.csv").exists():
            return
        base, name, _, _ = self._folder()
        for q in self._read_picks(self.output_dir / "picks.csv"):
            path = base / name.format(code=q.station)
            if q.station in self.entries and q.channel == self.entries[q.station].channel and path.exists():
                q = replace(q, network=self.entries[q.station].network)
                self.picks[q.station] = q
                self.accepted[q.station] = (q, load_astf(path, q.station, self.phase))
                self.status[q.station] = "accepted"

    def station_azimuth(self, station):
        """Source-to-station azimuth [deg]: from the hypocenter when given (as
        in the duration tables), else the SAC header value."""
        evla, evlo, _ = self.hypocenter
        e = self.entries[station]
        if evla is None or evlo is None:
            return e.azimuth
        from obspy.geodetics import gps2dist_azimuth
        return gps2dist_azimuth(evla, evlo, e.latitude, e.longitude)[1]

    # -------------------------------------------------------- computation
    def windows(self, station=None):
        """Target and EGF windows of the current settings (cached)."""
        p = self.picks[station or self.current]
        e = self.entries[p.station]

        def make():
            with warnings.catch_warnings():
                # freqmax = 100 Hz at 100 samples/s is at Nyquist: ObsPy then
                # applies a high-pass at freqmin.
                warnings.filterwarnings("ignore", message="Selected high corner frequency")
                return pld.prepare_windows(e.main, e.egf, p)
        return _cached(self._windows, _window_key(p), make)

    def curve(self, station=None):
        """Duration-misfit trade-off curve over :func:`pld.default_durations`
        (recomputed only when the window, start or iteration count changes)."""
        p = self.picks[station or self.current]
        w = self.windows(p.station)
        return _cached(self._curves, _window_key(p) + (p.start, p.iter_max),
                       lambda: pld.tradeoff_curve(w.y, w.x, w.dt, p.start,
                                                  pld.default_durations(p.start, w.dt), p.iter_max))

    def compute(self, station=None):
        """ASTF of the current settings (cached), exactly as :func:`pld.retrieve_astf`."""
        p = self.picks[station or self.current]
        # cached windows carry the pick they were cut with; only the window
        # settings in _window_key must match, start/duration come from p
        w = self.windows(p.station)
        return _cached(self._astfs, _window_key(p) + (p.start, p.iter_max, p.duration),
                       lambda: pld.retrieve_astf(w, pick=p))

    # ------------------------------------------------------ public actions
    @property
    def pick(self):
        """Current settings of the selected station."""
        return self.picks[self.current]

    def select(self, station):
        """Show a station (its saved or starting values)."""
        if station not in self.entries:
            raise KeyError(f"{station} has no {self.phase} target/EGF pair")
        self.current = station
        self._refresh()

    def next(self):
        """Select the next station in azimuth order (wraps around)."""
        self.select(self.order[(self.order.index(self.current) + 1) % len(self.order)])

    def previous(self):
        """Select the previous station in azimuth order (wraps around)."""
        self.select(self.order[(self.order.index(self.current) - 1) % len(self.order)])

    def set_params(self, **kw):
        """Change the current station's settings (``start``, ``duration``,
        ``shift`` [s], ``freqmin``, ``freqmax`` [Hz], ``iter_max``,
        ``interpolate``) or the zoomed misfit range (``zoom=(lo, hi)``)."""
        if "zoom" in kw:
            zoom = kw.pop("zoom")
            self.zoom[self.current] = None if zoom is None else tuple(float(v) for v in zoom)
        unknown = set(kw) - set(PARAMS)
        if unknown:
            raise TypeError(f"unknown parameter(s) {sorted(unknown)}; allowed: {PARAMS}")
        cast = {k: (round(float(v), 6) if k in ("start", "duration", "shift") else
                    int(v) if k == "iter_max" else bool(v) if k == "interpolate" else float(v))
                for k, v in kw.items()}
        self.picks[self.current] = replace(self.pick, **cast)
        self._refresh()

    def accept(self, station=None):
        """Store the ASTF of the current settings; with ``output_dir`` write
        ``<STA>_pld_stf.pkl`` = [t, astf, vr], the station/azimuth lists and
        ``picks.csv``.  Returns the ASTF."""
        sta = station or self.current
        astf = self._store(sta)
        self._refresh(redraw=sta == self.current)
        return astf

    def reject(self, station=None):
        """Mark a station as not usable for this session; removes an earlier
        accepted ASTF (pickle, list entry and picks row)."""
        sta = station or self.current
        self._discard(sta)
        self._refresh(redraw=sta == self.current)

    def _store(self, sta):
        astf = self.compute(sta)
        self.accepted[sta] = (self.picks[sta], astf)
        self.status[sta] = "accepted"
        if self.output_dir is not None:
            base, name, _, _ = self._folder()
            base.mkdir(parents=True, exist_ok=True)
            with open(base / name.format(code=sta), "wb") as f:
                pickle.dump([astf.t, astf.f, astf.vr], f)
            self._write_records(sta, self.picks[sta])
        return astf

    def _discard(self, sta):
        self.status[sta] = "rejected"
        self.accepted.pop(sta, None)
        if self.output_dir is not None:
            base, name, _, _ = self._folder()
            (base / name.format(code=sta)).unlink(missing_ok=True)
            self._write_records(sta, None)

    def _write_records(self, station, pick):
        """Replace one station's row in picks.csv and the station/azimuth lists
        (rows of other stations and phases are kept)."""
        path = self.output_dir / "picks.csv"
        rows = [q for q in (pld.read_picks(path) if path.exists() else [])
                if not (q.phase == self.phase and q.station == station)]
        pld.write_picks(rows + ([pick] if pick is not None else []), path)
        base, _, stations_file, azimuths_file = self._folder()
        codes, azimuths = [], []
        if (base / stations_file).exists() and (base / azimuths_file).exists():
            with open(base / stations_file, "rb") as f:
                codes = [str(c) for c in pickle.load(f)]
            with open(base / azimuths_file, "rb") as f:
                azimuths = [float(a) for a in pickle.load(f)]
        keep = [(c, a) for c, a in zip(codes, azimuths) if c != station]
        if pick is not None:
            keep.append((station, self.station_azimuth(station)))
        base.mkdir(parents=True, exist_ok=True)
        with open(base / stations_file, "wb") as f:
            pickle.dump(np.array([c for c, _ in keep]), f)
        with open(base / azimuths_file, "wb") as f:
            pickle.dump(np.array([a for _, a in keep], float), f)

    def table(self):
        """Progress table: status, accepted settings, VR [%] and tau_c [s]."""
        rows = []
        for sta in self.order:
            p, a = self.accepted.get(sta, (self.picks[sta], None))
            rows.append(dict(station=sta, network=self.entries[sta].network, channel=p.channel,
                             azimuth=round(self.entries[sta].azimuth, 1), status=self.status[sta],
                             start=p.start, duration=p.duration, shift=p.shift,
                             vr=np.nan if a is None else a.vr,
                             tauc=np.nan if a is None else pld.characteristic_duration(a.t, a.f)))
        return pd.DataFrame(rows)

    def duration_table(self, path=None, qc=True, model=None):
        """Duration table of the accepted stations, ready for the inversion
        (columns of :data:`io.TABLE_COLUMNS`, sorted by azimuth).

        qc=True keeps ASTFs with VR >= VR_THRESHOLD only (all rows
        ``Used = 1``).  ``model``: TauP model for the take-off angles
        (default ``geometry.taup_model()``).  Written to ``path`` if given.
        """
        from .geometry import takeoff_angles
        evla, evlo, evdp = self.hypocenter
        if None in self.hypocenter:
            raise ValueError("pass evla, evlo, evdp to PLDPicker to export the duration table")
        rows = []
        for sta in self.order:
            if sta not in self.accepted:
                continue
            _, a = self.accepted[sta]
            if qc and a.vr < VR_THRESHOLD:
                continue
            e = self.entries[sta]
            rows.append(dict(code=sta, latitude=e.latitude, longitude=e.longitude,
                             azimuth=self.station_azimuth(sta), vr=a.vr,
                             tauc=pld.characteristic_duration(a.t, a.f), phase=self.phase,
                             Used=int(a.vr >= VR_THRESHOLD), t1=pld.onset_time(a.t, a.f)))
        tab = pd.DataFrame(rows, columns=TABLE_COLUMNS[:-1])
        tab["toa"] = (takeoff_angles(evla, evlo, evdp, tab["latitude"], tab["longitude"], self.phase,
                                     model) if len(tab) else [])
        if path is not None:
            tab.to_csv(path, index=False)
        return tab

    # -------------------------------------------------------------- figure
    def _display_traces(self, w):
        """Target and EGF filtered and cut from 0.5 s before to 1 s after the
        pick, without the final taper (the original 'Target - EGF pair' plot)."""
        p = w.pick
        tr_main, tr_egf = w.traces
        t_pick = float(tr_main.stats.sac[pld.PHASE_SETUP[p.phase][1]])
        out = []
        for tr, shift in ((tr_main, 0.0), (tr_egf, p.shift)):
            t_ref = tr.stats.starttime - tr.stats.sac.b + t_pick + shift
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Selected high corner frequency")
                out.append(tr.copy().detrend("demean").taper(0.05)
                           .filter("bandpass", freqmin=p.freqmin, freqmax=p.freqmax, corners=4,
                                   zerophase=True)
                           .slice(t_ref - pld.WINDOW_BEFORE, t_ref + DISPLAY_AFTER))
        return out

    def auto_zoom(self, station=None):
        """Default misfit range of the zoomed curve: 0.2 wide, around the
        misfit at the chosen duration."""
        a = self.compute(station)
        lo = round(min(max(0.0, round((1 - a.vr / 100) * 20) / 20 - 0.1), ZOOM_MAX - 0.2), 2)
        return lo, round(lo + 0.2, 2)

    def plot(self, station=None):
        """Diagnostic figure of one station; returns the matplotlib figure.

        (a, b) trade-off curve, full and zoomed, chosen duration circled;
        (c) ASTF on the deconvolution time axis with the support start (red
        band), tau_c (red bar) centered on the centroid (black dot);
        (d) normalized target and EGF windows; (e) target and reconstruction.
        """
        fig = plt.figure(figsize=(16, 9), layout="constrained")
        view = self._build_view(fig)
        self._update_view(view, station or self.current)
        return fig

    def _build_view(self, fig):
        """Axes and artists of the diagnostic figure, without data."""
        from .plotting import use_style
        use_style()
        ax = fig.subplot_mosaic([["full"] * 3 + ["zoom"] * 3,
                                 ["astf"] * 2 + ["pair"] * 2 + ["recon"] * 2])
        h = {"fig": fig, "ax": ax}
        for key, title in (("full", "Trade-off curve"), ("zoom", "Trade-off curve (zoomed)")):
            axk = ax[key]
            h[f"{key}_line"], = axk.plot([], [], color="k", lw=2)
            h[f"{key}_dots"] = axk.scatter([], [], color="r", s=25, zorder=3)
            h[f"{key}_chosen"] = axk.scatter([], [], s=180, facecolors="none", edgecolors="b", lw=2.5,
                                             zorder=4)
            h[f"{key}_vline"] = axk.axvline(0.0, color="b", lw=1.5, alpha=0.4)
            axk.set_xlim(0, 0.4)
            axk.xaxis.set_major_locator(MultipleLocator(0.04))
            axk.xaxis.set_minor_locator(MultipleLocator(0.01))
            axk.set_xlabel("ASTF duration (s)", fontsize=14)
            axk.set_ylabel("Misfit (1 - VR/100)", fontsize=14)
            axk.set_title(title, fontsize=16)

        axa = ax["astf"]
        h["astf_line"], = axa.plot([], [], lw=2)
        h["astf_dots"] = axa.scatter([], [], color="k", s=12)
        h["astf_fill"] = None
        h["start_line"] = axa.axvline(0.0, color="r", lw=5, alpha=0.25)
        h["support"] = None
        h["tau_bar"], = axa.plot([], [], color="r", lw=4)
        h["centroid"] = axa.scatter([], [], s=80, color="k", zorder=10)
        axa.set_xlim(-0.15, 0.3)
        axa.set_xlabel("Time (s)", fontsize=14)
        axa.set_title("ASTF", fontsize=16)

        axp = ax["pair"]
        h["target"], = axp.plot([], [], color="k", lw=1.5, label="Target")
        h["egf"], = axp.plot([], [], color="b", lw=1.5, label="EGF")
        axp.axvline(x=pld.WINDOW_BEFORE, color="r", lw=5, alpha=0.25)
        leg = axp.legend(prop={"size": 11}, loc="upper left", ncol=2)
        leg.get_frame().set_edgecolor("none")
        axp.set_xticks(np.arange(0, 1.6, 0.3))
        axp.set_yticks([])
        axp.set_xlim(0.0, pld.WINDOW_BEFORE + DISPLAY_AFTER)
        axp.set_xlabel("Time (s)", fontsize=14)
        axp.set_title("Target - EGF pair", fontsize=16)
        axp.grid()

        axr = ax["recon"]
        h["data"], = axr.plot([], [], color="k", lw=1.5, label="Data")
        h["recon"], = axr.plot([], [], lw=1.5)
        axr.set_yticks([])
        axr.set_xlabel("Time (s)", fontsize=14)

        for axk in ax.values():
            axk.tick_params(axis="both", labelsize=12)
        # one title, updated by set_text: calling suptitle again would move it back to its
        # default height after the layout has been fixed
        h["title"] = fig.suptitle(" ", fontsize=17)
        return h

    def _update_view(self, h, station):
        """Put the current settings of ``station`` into the artists of a view."""
        ax = h["ax"]
        sta = station
        p, e = self.picks[sta], self.entries[sta]
        w, (dur, misfit), a = self.windows(sta), self.curve(sta), self.compute(sta)
        zoom = self.zoom.get(sta) or self.auto_zoom(sta)
        color = PHASE_COLOR.get(p.phase, "b")

        for key in ("full", "zoom"):
            axk = ax[key]
            h[f"{key}_line"].set_data(dur, misfit)
            h[f"{key}_dots"].set_offsets(np.column_stack([dur, misfit]))
            h[f"{key}_chosen"].set_offsets([[p.duration, 1 - a.vr / 100]])
            h[f"{key}_chosen"].set_label(f"Chosen duration {p.duration:.2f} s")
            h[f"{key}_vline"].set_xdata([p.duration, p.duration])
            axk.legend(handles=[h[f"{key}_chosen"]], prop={"size": 12}, loc="upper right")
        ax["full"].relim()
        ax["full"].autoscale_view(scalex=False)
        ax["zoom"].set_ylim(*zoom)

        axa = ax["astf"]
        t_dec = a.t + p.shift
        h["astf_line"].set_data(t_dec, a.f)
        h["astf_line"].set_color(color)
        h["astf_line"].set_label(f"VR={a.vr:.1f}%")
        h["astf_dots"].set_offsets(np.column_stack([t_dec, a.f]))
        for name, make in (("astf_fill", lambda: axa.fill_between(t_dec, 0, a.f, color=color, alpha=0.2)),
                           ("support", lambda: axa.axvspan(p.start, p.start + p.duration, color="r",
                                                           alpha=0.06))):
            if h[name] is not None:
                h[name].remove()
            h[name] = make()
        h["start_line"].set_xdata([p.start, p.start])
        handles = [h["astf_line"]]
        if np.any(a.f > 0):
            tau = pld.characteristic_duration(a.t, a.f)
            tc, level = pld.centroid_time(a.t, a.f) + p.shift, 0.05 * a.f.max()
            h["tau_bar"].set_data([tc - tau / 2, tc + tau / 2], [level, level])
            h["tau_bar"].set_label(rf"$\tau_c$={tau:.4f} s")
            h["centroid"].set_offsets([[tc, level]])
            h["centroid"].set_label("Centroid")
            handles += [h["tau_bar"], h["centroid"]]
        else:
            h["tau_bar"].set_data([], [])
            h["centroid"].set_offsets(np.empty((0, 2)))
        axa.legend(handles=handles, prop={"size": 11}, loc="upper right")
        axa.relim()
        axa.autoscale_view(scalex=False)

        main, egf = self._display_traces(w)
        target, reference = main.data / np.abs(np.max(main.data)) + 1.5, egf.data / np.abs(np.max(egf.data))
        h["target"].set_data(main.times(), target)
        h["egf"].set_data(egf.times(), reference)
        ax["pair"].set_ylim(min(reference.min(), target.min()) - 0.2, target.max() + 0.8)   # room for the legend

        axr = ax["recon"]
        tw = np.arange(len(w.y)) * w.dt
        recon = np.convolve(w.x, a.f, "same")
        h["data"].set_data(tw, w.y)
        h["recon"].set_data(tw, recon)
        h["recon"].set_color(color)
        h["recon"].set_label(f"Reconstructed (VR={a.vr:.1f}%)")
        leg = axr.legend(handles=[h["data"], h["recon"]], prop={"size": 11}, loc="upper left", ncol=2)
        leg.get_frame().set_edgecolor("none")
        lo, hi = min(w.y.min(), recon.min()), max(w.y.max(), recon.max())
        axr.set_ylim(lo - 0.05 * (hi - lo), hi + 0.35 * (hi - lo))       # room for the legend
        axr.set_xlim(0, len(w.y) * w.dt)
        axr.set_xticks(np.arange(0, len(w.y) * w.dt + 0.01, 0.5))
        name = "SmS" if p.phase == "SmS" else f"Direct {p.phase}"
        axr.set_title(f"{name} reconstruction", fontsize=16)

        verdict = "passes" if a.vr >= VR_THRESHOLD else "fails"
        h["title"].set_text(f"{e.network}.{sta} ({e.channel}{pld.PHASE_SETUP[p.phase][0]}), {p.phase} phase, "
                            f"azimuth {e.azimuth:.1f}°: VR {a.vr:.1f} %, {verdict} QC "
                            f"(threshold {VR_THRESHOLD:.0f} %); status {self.status[sta]}")

    def _render(self):
        """PNG of the live figure for the current station.  The figure is
        built (and laid out) once per station; afterwards each change only
        updates the artists and redraws with the layout fixed."""
        if self._view is None or self._view_station != self.current:
            fig = Figure(figsize=(16, 9), dpi=FIGURE_DPI, layout="constrained")
            FigureCanvasAgg(fig)
            self._view, self._view_station = self._build_view(fig), self.current
            self._update_view(self._view, self.current)
            fig.canvas.draw()                     # constrained layout, once per station
            fig.set_layout_engine("none")
        else:
            self._update_view(self._view, self.current)
            self._view["fig"].canvas.draw()
        rgba = np.asarray(self._view["fig"].canvas.buffer_rgba())
        buf = BytesIO()
        Image.fromarray(rgba[..., :3]).save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    # ------------------------------------------------------------- widgets
    def _station_options(self):
        return [(f"{s}  ({self.entries[s].azimuth:.1f}°)  {self.status[s]}", s) for s in self.order]

    @property
    def ui(self):
        """The widget (built on first use)."""
        if self._ui is None:
            self._build_ui()
            self._refresh()
        return self._ui

    def _ipython_display_(self):
        from IPython.display import display
        display(self.ui)

    def _build_ui(self):
        import ipywidgets as W
        num = dict(continuous_update=False, layout=W.Layout(width="190px"),
                   style={"description_width": "85px"})
        # sliders redraw while dragged (the front end sends the latest value whenever the
        # kernel is free, so a slow step never queues up); the readout can also be typed
        # into, and the arrow keys move by one step
        slider = dict(continuous_update=True, readout_format=".2f", layout=W.Layout(width="420px"),
                      style={"description_width": "95px"})
        w = self._w = {
            "station": W.Dropdown(options=self._station_options(), value=self.current,
                                  description="Station", layout=W.Layout(width="330px")),
            "prev": W.Button(description="Previous", icon="arrow-left", layout=W.Layout(width="110px")),
            "next": W.Button(description="Next", icon="arrow-right", layout=W.Layout(width="110px")),
            "channel": W.Text(description="Channel", disabled=True, layout=W.Layout(width="150px"),
                              style={"description_width": "60px"}),
            "start": W.FloatSlider(min=SLIDERS["start"][0], max=SLIDERS["start"][1], step=0.01,
                                   description="Start (s)", **slider),
            "duration": W.FloatSlider(min=SLIDERS["duration"][0], max=SLIDERS["duration"][1], step=0.01,
                                      description="Duration (s)", **slider),
            "shift": W.FloatSlider(min=SLIDERS["shift"][0], max=SLIDERS["shift"][1], step=0.01,
                                   description="EGF shift (s)", **slider),
            "freqmin": W.FloatText(description="Low (Hz)", step=0.1, **num),
            "freqmax": W.FloatText(description="High (Hz)", step=1.0, **num),
            "iter_max": W.IntText(description="Iterations", step=10, **num),
            "interpolate": W.Checkbox(description="Resample to 500 Hz", indent=False),
            "zoom": W.FloatRangeSlider(min=0.0, max=ZOOM_MAX, step=0.01, description="Misfit zoom",
                                       continuous_update=True, readout_format=".2f",
                                       layout=W.Layout(width="420px"),
                                       style={"description_width": "85px"}),
            "accept": W.Button(description="Accept", button_style="success", icon="check"),
            "reject": W.Button(description="Reject", button_style="danger", icon="times"),
            "save": W.Button(description="Save table", icon="save",
                             tooltip="write the duration table of the accepted stations"),
            "badge": W.HTML(), "message": W.HTML(), "progress": W.HTML(),
            "figure": W.Image(format="png", layout=W.Layout(width="100%", max_width="1280px")),
        }
        for name in PARAMS:
            w[name].observe(lambda ch, n=name: self._on_change(lambda: self.set_params(**{n: ch["new"]})),
                            names="value")
        w["zoom"].observe(lambda ch: self._on_change(lambda: self.set_params(zoom=ch["new"])),
                          names="value")
        w["station"].observe(lambda ch: self._on_change(lambda: self.select(ch["new"])), names="value")
        w["prev"].on_click(lambda _: self._on_change(self.previous))
        w["next"].on_click(lambda _: self._on_change(self.next))
        w["accept"].on_click(lambda _: self._on_change(self._accept_and_advance))
        w["reject"].on_click(lambda _: self._on_change(self._reject_and_advance))
        w["save"].on_click(lambda _: self._on_change(self._save_table))
        table_box = W.Box([w["progress"]], layout=W.Layout(max_height="260px", overflow="auto",
                                                           border="1px solid #ccc"))
        self._ui = W.VBox([
            W.HBox([w["prev"], w["station"], w["next"], w["channel"]]),
            W.HBox([w["start"], w["duration"]]),
            W.HBox([w["shift"], w["zoom"]]),
            W.HBox([w["freqmin"], w["freqmax"], w["iter_max"], w["interpolate"]]),
            W.HBox([w["accept"], w["reject"], w["save"], w["badge"]]),
            w["message"], w["figure"], table_box,
        ])

    def _on_change(self, action):
        """Run a widget action unless the widgets are being synchronized;
        show errors in the widget instead of the browser console."""
        if self._syncing:
            return
        try:
            action()
        except Exception as err:
            self._w["message"].value = f"<span style='color:#b00'>{type(err).__name__}: {err}</span>"

    def _advance(self):
        """Select the next pending station after the current one (or redraw)."""
        pending = [s for s in self.order[self.order.index(self.current) + 1:] + self.order
                   if self.status[s] == "pending"]
        if pending:
            self.select(pending[0])
        else:
            self._refresh()

    def _accept_and_advance(self):
        sta = self.current
        astf = self._store(sta)
        where = ""
        if self.output_dir is not None:
            base, name, _, _ = self._folder()
            where = f" &rarr; {(base / name.format(code=sta)).relative_to(self.output_dir)}"
        self._advance()
        self._w["message"].value = f"Accepted {sta} (VR {astf.vr:.1f} %){where}"

    def _save_table(self):
        if self.output_dir is None:
            raise RuntimeError("no output_dir: give PLDPicker(..., output_dir=...) to save files")
        path = self.output_dir / f"tauc_table_{self.phase}.csv"
        tab = self.duration_table(path=path)
        self._w["message"].value = (f"Saved the duration table of {len(tab)} accepted station(s) "
                                    f"&rarr; {path.relative_to(self.output_dir.parent)}")

    def _reject_and_advance(self):
        sta = self.current
        self._discard(sta)
        self._advance()
        self._w["message"].value = f"Rejected {sta}"

    def _refresh(self, redraw=True):
        """Synchronize the widgets with the current station; redraw the
        figure unless only another station's status changed."""
        if self._ui is None:
            return
        w, p = self._w, self.pick
        self._syncing = True
        try:
            w["station"].options = self._station_options()
            w["station"].value = self.current
            w["channel"].value = p.channel
            for name in PARAMS:
                value = getattr(p, name)
                if name in SLIDERS:            # widen a slider for a saved value outside its range
                    w[name].min, w[name].max = min(w[name].min, value), max(w[name].max, value)
                w[name].value = value
        finally:
            self._syncing = False
        w["progress"].value = self._progress_html()
        if redraw:
            self._redraw()

    def _redraw(self):
        w = self._w
        w["message"].value = ""
        try:
            png = self._render()
        except Exception as err:
            self._view = None                     # rebuild on the next change
            w["badge"].value = ""
            w["message"].value = f"<span style='color:#b00'>{type(err).__name__}: {err}</span>"
            return
        w["figure"].value = png
        self._syncing = True
        try:
            w["zoom"].value = self.zoom.get(self.current) or self.auto_zoom()
        finally:
            self._syncing = False
        w["badge"].value = self._badge()

    def _badge(self):
        a = self.compute()
        ok = a.vr >= VR_THRESHOLD
        color = "#1a7f37" if ok else "#b00020"
        text = (f"VR {a.vr:.1f} % &nbsp;{'passes' if ok else 'fails'} QC (VR &ge; {VR_THRESHOLD:.0f} %)"
                f" &nbsp;|&nbsp; &tau;<sub>c</sub> {pld.characteristic_duration(a.t, a.f):.4f} s")
        saved = self.accepted.get(self.current)
        note = ""
        if saved is not None and saved[0] != self.pick:
            note = " &nbsp;<i>(accepted with other settings; press Accept to replace)</i>"
        return (f"<span style='padding:3px 8px;border-radius:4px;color:white;background:{color}'>"
                f"{text}</span> &nbsp;status: <b>{self.status[self.current]}</b>{note}")

    def _progress_html(self):
        tab = self.table()
        counts = tab["status"].value_counts()
        head = " / ".join(f"{counts.get(k, 0)} {k}" for k in ("accepted", "rejected", "pending"))
        return (f"<b>Progress:</b> {head}"
                + tab.to_html(index=False, float_format=lambda v: f"{v:.4g}", na_rep=""))

    def __repr__(self):
        counts = pd.Series(self.status).value_counts()
        return (f"PLDPicker(phase={self.phase!r}, stations={len(self.order)}, "
                f"accepted={counts.get('accepted', 0)}, rejected={counts.get('rejected', 0)}, "
                f"current={self.current!r})")
