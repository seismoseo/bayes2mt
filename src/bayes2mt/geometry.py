"""Ray geometry: take-off angles, the slowness (system) matrix G, and
azimuthal-coverage measures.

Coordinates follow McGuire (2004): x = north, y = east, z = down.  The
slowness vector of each ray is rotated into the fault frame
(x' = along strike, y' = fault normal, z' = down dip) by the strike and then
by (dip - 90).  Only the on-fault components s1 = s_x', s2 = s_z' enter the
second-degree-moment relation (McGuire, 2004)

    mu^(0,2)(s) = mu^(0,2) - 2 s.mu^(1,1) + s^T mu^(2,0) s ,

so each row of G is [1, -2 s1, -2 s2, s1^2, 2 s1 s2, s2^2].
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PACKAGE_DATA, TAUP_PHASES


# ------------------------------------------------------------ take-off angles
def _cache_dir():
    base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    path = Path(base) / "bayes2mt"
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=4)
def taup_model(nd_file=None):
    """TauP model built from a ``.nd`` velocity file (default: the 1-D gradient
    model of the southern Korean Peninsula interpolated from Kim & Kim, 1983).

    A gradient model is used instead of the layered one because take-off
    angles of first arrivals jump at sharp discontinuities.  The compiled
    ``.npz`` is cached in ``~/.cache/bayes2mt``.
    """
    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model

    nd = Path(nd_file) if nd_file else PACKAGE_DATA / "gradient.nd"
    out = _cache_dir() / (nd.stem + ".npz")
    if not out.exists() or out.stat().st_mtime < nd.stat().st_mtime:
        build_taup_model(str(nd), output_folder=str(_cache_dir()), verbose=False)
    return TauPyModel(model=str(out))


def takeoff_angles(evla, evlo, evdp, stla, stlo, phase, model=None):
    """Take-off angle [deg from vertical down] of the first matching ray.

    ``phase`` is "P", "S" or "SmS" (one value or one per station).
    """
    model = model or taup_model()
    stla, stlo = np.atleast_1d(stla), np.atleast_1d(stlo)
    phase = np.broadcast_to(np.asarray(phase), stla.shape)
    out = np.empty(len(stla))
    for i, (la, lo, ph) in enumerate(zip(stla, stlo, phase)):
        arr = model.get_travel_times_geo(evdp, evla, evlo, la, lo, phase_list=TAUP_PHASES[ph])
        out[i] = arr[0].takeoff_angle if arr else np.nan
    return out


# ------------------------------------------------------------ slowness matrix
def fault_slowness(azimuth, toa, phase, strike, dip, vp, vs):
    """On-fault slowness components (s1 along strike, s2 down dip) [s/km].

    azimuth, toa in degrees; vp, vs in km/s (so moments come out in km).
    """
    az, ta = np.deg2rad(np.asarray(azimuth, float)), np.deg2rad(np.asarray(toa, float))
    v = np.where(np.asarray(phase) == "P", vp, vs)
    s = np.column_stack([np.sin(ta) * np.cos(az), np.sin(ta) * np.sin(az), np.cos(ta)]) / v[:, None]
    z = np.deg2rad(strike)
    r_strike = np.array([[np.cos(z), np.sin(z), 0.0],
                         [-np.sin(z), np.cos(z), 0.0],
                         [0.0, 0.0, 1.0]])
    z = np.deg2rad(dip - 90.0)
    r_dip = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(z), np.sin(z)],
                      [0.0, -np.sin(z), np.cos(z)]])
    s = s @ (r_dip @ r_strike).T
    return s[:, 0], s[:, 2]


def slowness_matrix(table, strike, dip, vp, vs):
    """System matrix G (N x 6) from a duration table with columns
    ``azimuth``, ``toa`` and ``phase``.  Velocities in km/s."""
    s1, s2 = fault_slowness(table["azimuth"], table["toa"], table["phase"], strike, dip, vp, vs)
    return np.column_stack([np.ones_like(s1), -2 * s1, -2 * s2, s1 * s1, 2 * s1 * s2, s2 * s2])


# ------------------------------------------------------- azimuthal coverage
def _gaps(sorted_az):
    return np.append(np.diff(sorted_az), 360.0 - (sorted_az[-1] - sorted_az[0]))


def max_azimuthal_gap(azimuths):
    """Largest azimuthal gap [deg] between adjacent stations."""
    az = np.sort(np.mod(np.unique(azimuths), 360.0))
    if len(az) < 2:
        return 360.0
    return float(_gaps(az).max())


def mean_azimuthal_spacing(azimuths, threshold=10.0):
    """Average azimuthal spacing [deg] of 'effective' stations.

    Stations are merged greedily, in order of azimuth from 0 deg, into
    effective stations at the mean azimuth of each group (members within
    ``threshold`` deg of the group's first station); the first and last
    groups are merged if they are closer than ``threshold`` across north.
    The spacing is the covered sector (360 - largest gap) divided by the
    number of intervals (Seo et al. kept events with P or S spacing below
    30 deg).
    """
    az = np.sort(np.mod(np.unique(azimuths), 360.0))
    if len(az) == 0:
        return 360.0
    eff, remaining = [], list(az)
    while remaining:
        pivot = remaining[0]
        cluster = [a for a in remaining if a - pivot < threshold]
        remaining = [a for a in remaining if a - pivot >= threshold]
        eff.append(np.mean(cluster))
    if len(eff) > 1 and (eff[0] - eff[-1] + 360.0) % 360.0 < threshold:
        eff[0] = ((eff[-1] + eff[0] + 360.0) / 2.0) % 360.0
        eff.pop()
    if len(eff) < 2:
        return 360.0
    eff = np.sort(eff)
    return float((360.0 - _gaps(eff).max()) / (len(eff) - 1))
