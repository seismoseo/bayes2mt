"""Relative location of the target event with respect to its EGF from ASTF
onset times.

With origin-time alignment, the ASTF onset t1 at a station is the differential
travel time.  For a target offset by L metres in the direction
(azimuth phi, angle delta from vertical down) from the EGF, a ray with
take-off direction (az, toa) gives

    t1 = t0 - (L / V) cos(theta),
    cos(theta) = cos(toa) cos(delta) + sin(toa) sin(delta) cos(phi - az),

with V the P or S speed at the source and t0 a common shift.  ``locate``
searches a coarse grid (phi, delta every 5 deg, t0 every 10 ms, L every 10 m)
and then a fine grid around the best coarse point.  The misfit for all
(t0, L) pairs of a given direction is evaluated at once from sufficient
statistics, so both grids take seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cos_theta(phi, delta, az, toa):
    phi, delta, az, toa = map(np.deg2rad, (phi, delta, az, toa))
    return np.cos(toa) * np.cos(delta) + np.sin(toa) * np.sin(delta) * np.cos(phi - az)


def predict_onsets(phi, delta, t0, L, az, toa, v):
    """Predicted onset times t1 [s]; ``v`` in m/s per station, L in m."""
    return t0 - (L / np.asarray(v, float)) * _cos_theta(phi, delta, az, toa)


@dataclass
class RelLoc:
    azimuth: float
    dip: float          # angle from vertical down, deg (> 90 = above the EGF)
    t0: float
    L: float
    misfit: float       # RMS onset misfit [s]


def _grid_misfit(phi, delta, t0, L, az, toa, invv, t1):
    """RMS misfit on the grid phi x delta x t0 x L (C order)."""
    c = invv[None, None, :] * _cos_theta(phi[:, None, None], delta[None, :, None], az, toa)
    n = len(t1)
    sc, scc = c.sum(-1), (c * c).sum(-1)
    sd, scd, sdd = t1.sum(), (c * t1).sum(-1), (t1 * t1).sum()
    T, Lg = t0[:, None], L[None, :]
    ss = (n * T ** 2 + (Lg ** 2) * scc[..., None, None] + sdd - 2 * T * Lg * sc[..., None, None]
          - 2 * T * sd + 2 * Lg * scd[..., None, None])
    return np.sqrt(np.clip(ss, 0, None) / n)


def locate(table, vp=3600 * 1.73, vs=3600.0, refinement="minimum"):
    """Grid-search the target-EGF offset from a duration table with columns
    ``t1``, ``azimuth``, ``toa`` and ``phase`` (SmS uses the S speed).

    refinement : which point of the fine grid is returned; "minimum" (default)
                 the point with the smallest misfit, "last" the last point in
                 grid order whose misfit is below that of the coarse optimum
                 (the rule used by Seo et al., 2026).  If no fine-grid point
                 improves on the coarse optimum, the coarse optimum is returned.
    """
    if refinement not in ("minimum", "last"):
        raise ValueError("refinement must be 'minimum' or 'last'")
    az = np.asarray(table["azimuth"], float)
    toa = np.asarray(table["toa"], float)
    t1 = np.asarray(table["t1"], float)
    invv = np.where(np.asarray(table["phase"]) == "P", 1.0 / vp, 1.0 / vs)
    grids = [np.arange(0, 360, 5), np.arange(0, 185, 5), np.arange(-0.15, 0.16, 0.01), np.arange(10, 1010, 10)]
    mis = _grid_misfit(*grids, az, toa, invv, t1)
    i = np.unravel_index(np.argmin(mis), mis.shape)
    c = [float(g[k]) for g, k in zip(grids, i)]           # raw grid values (unrounded)
    coarse = RelLoc(*(round(v, 6) for v in c), float(mis[i]))
    fine = [np.arange(c[0] - 5, c[0] + 5, 0.1), np.arange(c[1] - 5, c[1] + 5, 0.1),
            np.arange(c[2] - 0.01, c[2] + 0.01, 0.001), np.arange(c[3] - 10, c[3] + 10, 1)]
    mis = _grid_misfit(*fine, az, toa, invv, t1)
    if refinement == "last":
        flat = np.flatnonzero(mis.ravel() < coarse.misfit)
        if len(flat) == 0:
            return coarse
        i = np.unravel_index(flat[-1], mis.shape)
    else:
        i = np.unravel_index(np.argmin(mis), mis.shape)
        if mis[i] >= coarse.misfit:
            return coarse
    # grid values are reported rounded (np.arange accumulates ~1e-13 noise)
    return RelLoc(*(round(float(g[k]), 6) for g, k in zip(fine, i)), float(mis[i]))
