"""Readers and writers for duration tables, ASTF pickles, posterior ensembles
and target/EGF SAC pairs."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_KEYS = [f"my_param_space.m{i}" for i in range(6)]
TABLE_COLUMNS = ["code", "latitude", "longitude", "azimuth", "vr", "tauc",
                 "phase", "Used", "t1", "toa"]

# Folder layout of the ASTFs of one event (written by ``interactive.PLDPicker``):
# sub-folders per phase, the file-name pattern of each station's pickle, and
# the pickled station and azimuth lists.  The first folder is written; when
# reading, "S/dirct" and "S" are also accepted.
ASTF_LAYOUT = {
    "P": (["P"], "{code}_pld_stf.pkl", "station_list.pkl", "azimuth_list.pkl"),
    "S": (["S/direct", "S/dirct", "S"], "{code}_pld_stf.pkl",
          "station_list.pkl", "azimuth_list.pkl"),
    "SmS": (["S/moho"], "{code}_moho_pld_stf.pkl",
            "moho_station_list.pkl", "moho_azimuth_list.pkl"),
}


# ----------------------------------------------------------------- tables
def read_tauc_table(path):
    """Read a duration (tau_c) table and normalise its columns.

    Columns: code, latitude, longitude, azimuth [deg], vr [%], tauc [s],
    phase (P, S or SmS), Used, t1 [s] (ASTF onset: the sample before the first
    non-zero sample, see ``pld.onset_time``), toa (take-off angle, deg from
    vertical down).
    A stray index column, a ``pcode`` header or a ``Phase`` header are
    accepted.
    """
    tab = pd.read_csv(path)
    tab = tab.loc[:, [c for c in tab.columns if not str(c).startswith("Unnamed")]]
    rename = {}
    for c in tab.columns:
        key = str(c).strip()
        if key.lower() in ("pcode", "code"):
            rename[c] = "code"
        elif key.lower() == "phase":
            rename[c] = "phase"
        else:
            rename[c] = key
    tab = tab.rename(columns=rename)
    tab["code"] = tab["code"].astype(str).str.strip()
    tab["phase"] = tab["phase"].astype(str).str.strip()
    return tab.reset_index(drop=True)


# ------------------------------------------------------------------- ASTFs
@dataclass
class ASTF:
    """One apparent source time function.

    t    : time axis [s], relative to the origin-time-aligned reference
    f    : ASTF samples (non-negative, not normalised)
    vr   : variance reduction of the waveform reconstruction [%]
    code : station code
    phase: "P", "S" or "SmS"
    """
    t: np.ndarray
    f: np.ndarray
    vr: float
    code: str = ""
    phase: str = ""

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])


def load_astf(path, code="", phase=""):
    """Read one ``*_pld_stf.pkl`` file, a pickled ``[t, astf, vr]`` list."""
    with open(path, "rb") as f:
        t, astf, vr = pickle.load(f)
    return ASTF(np.asarray(t), np.asarray(astf), float(vr), code, phase)


def load_event_astfs(astf_dir, phase):
    """All ASTFs of one phase of one event, sorted by station azimuth.

    Returns (list of ASTF, azimuths [deg]).
    """
    subdirs, name, stations_file, azimuths_file = ASTF_LAYOUT[phase]
    base = None
    for sub in subdirs:
        if (Path(astf_dir) / sub / stations_file).is_file():
            base = Path(astf_dir) / sub
            break
    if base is None:
        return [], np.array([])
    with open(base / stations_file, "rb") as f:
        codes = np.asarray(pickle.load(f))
    with open(base / azimuths_file, "rb") as f:
        azimuths = np.asarray(pickle.load(f), float)
    order = np.argsort(azimuths, kind="stable")
    astfs = [load_astf(base / name.format(code=c), c, phase) for c in codes[order]]
    return astfs, azimuths[order]


# --------------------------------------------------------------- posteriors
@dataclass
class Ensemble:
    """Posterior ensemble written by BayesBay (``get_results`` concatenated).

    x     : (N, 6) sampled Cholesky parameters m0..m5
    std   : (N,) sampled data-noise standard deviation [s] (hierarchical)
    dpred : (N, Ndata) predicted durations, if saved
    """
    x: np.ndarray
    std: np.ndarray | None = None
    dpred: np.ndarray | None = None

    def __len__(self):
        return len(self.x)


def load_ensemble(path, with_dpred=False):
    """Read a posterior ``.npz`` saved from ``BayesianInversion.get_results``."""
    with np.load(path) as d:
        x = np.column_stack([np.ravel(d[k]) for k in PARAM_KEYS])
        std = np.ravel(d["my_data.std"]) if "my_data.std" in d.files else None
        dpred = d["my_data.dpred"] if (with_dpred and "my_data.dpred" in d.files) else None
    return Ensemble(x, std, dpred)


def save_ensemble(path, ensemble):
    """Write an ensemble with the same keys BayesBay uses."""
    out = {k: ensemble.x[:, [i]] for i, k in enumerate(PARAM_KEYS)}
    if ensemble.std is not None:
        out["my_data.std"] = ensemble.std
    if ensemble.dpred is not None:
        out["my_data.dpred"] = ensemble.dpred
    np.savez_compressed(path, **out)


# ------------------------------------------------------------------ SAC pairs
def sac_pair_paths(waveform_dir, station, channel, component, network=None):
    """Target ("Main") and EGF SAC files of one station and component.

    File names are ``<origin>.<net>.<sta>.<channel><component>.sac``.
    """
    out = []
    for sub in ("Main", "EGF"):
        net = network or "*"
        hits = sorted((Path(waveform_dir) / sub).glob(
            f"*.{net}.{station}.{channel}{component}.sac"))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{sub}: expected one file for {station}.{channel}{component}, found {len(hits)}")
        out.append(hits[0])
    return tuple(out)
