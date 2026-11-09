"""Synthetic apparent durations from a kinematic elliptical rupture.

Slip D = D_max sqrt(1 - (x_s/a)^2 - (x_d/b)^2) on an ellipse with semi-axes
a (along strike) and b (along dip), on a square grid of cells.  Rupture
spreads from the hypocenter at constant speed; every cell slips with a
triangular slip-rate function.  The ASTF at a station is the sum of the cell
contributions, each delayed by its rupture time minus the ray delay r . s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SEED
from .pld import characteristic_duration, normalize_area


@dataclass
class Rupture:
    """Kinematic rupture on a fault plane (strike, dip in degrees).

    semi_strike, semi_dip : semi-axes of the slipping ellipse [km]
    hypocenter            : (along strike, up dip) from the ellipse center [km]
    rupture_speed         : [km/s]
    rise_time             : base of the triangular slip-rate function [s]
    """
    strike: float
    dip: float
    semi_strike: float
    semi_dip: float
    hypocenter: tuple = (0.0, 0.0)
    rupture_speed: float = 2.88
    rise_time: float = 0.04
    peak_slip: float = 0.5
    n_cells: int = 51
    sampling_rate: float = 100.0
    duration: float = 0.5

    def grid(self):
        """Cell coordinates [km], identical along strike and along dip."""
        half = 2.0 * max(self.semi_strike, self.semi_dip)
        return np.linspace(-half, half, self.n_cells)

    def slip(self):
        """Final slip [m], array [dip, strike]."""
        x = self.grid()
        r2 = (x[:, None] / self.semi_dip) ** 2 + (x[None, :] / self.semi_strike) ** 2
        return np.where(r2 <= 1.0, self.peak_slip * np.sqrt(np.clip(1.0 - r2, 0.0, None)), 0.0)


def astfs(rupture, azimuth, toa, velocity):
    """Unit-area ASTFs for rays with ``azimuth`` and take-off angle ``toa``
    [deg] and wave speed ``velocity`` [km/s].  Returns (t, f), f with one
    row per ray."""
    r = rupture
    x = r.grid()
    xs, xd = np.meshgrid(x - r.hypocenter[0], x - r.hypocenter[1])
    t_rup = np.sqrt(xs ** 2 + xd ** 2) / r.rupture_speed
    st, dp = np.deg2rad(r.strike), np.deg2rad(r.dip)
    east = xs * np.sin(st) - xd * np.cos(dp) * np.cos(st)
    north = xs * np.cos(st) + xd * np.cos(dp) * np.sin(st)
    up = xd * np.sin(dp)
    az, ta = np.deg2rad(np.atleast_1d(azimuth)), np.deg2rad(np.atleast_1d(toa))
    v = np.broadcast_to(velocity, az.shape)
    s = np.stack([np.sin(ta) * np.sin(az), np.sin(ta) * np.cos(az), -np.cos(ta)], axis=1) / v[:, None]
    onsets = t_rup.ravel()[None, :] - (east.ravel()[None, :] * s[:, :1]
                                       + north.ravel()[None, :] * s[:, 1:2] + up.ravel()[None, :] * s[:, 2:])
    t = np.arange(round(r.duration * r.sampling_rate) + 1) / r.sampling_rate
    w = r.slip().ravel()
    out = np.empty((len(onsets), len(t)))
    for k, onset in enumerate(onsets):
        start = onset[:, None]
        peak = start + r.rise_time / 2
        end = start + r.rise_time
        tt = t[None, :]
        tri = np.where((tt >= start) & (tt <= peak), (tt - start) / (peak - start),
                       np.where((tt > peak) & (tt <= end), (end - tt) / (end - peak), 0.0))
        out[k] = normalize_area(t, w @ tri)
    return t, out


def tauc_table(rupture, table, vp, vs):
    """Copy of a station table (columns ``azimuth``, ``toa``, ``phase``) with
    the noise-free duration ``tauc`` of every row [s]."""
    out = table.copy()
    velocity = np.where(out["phase"].to_numpy() == "P", vp, vs)
    t, f = astfs(rupture, out["azimuth"].to_numpy(), out["toa"].to_numpy(), velocity)
    out["tauc"] = [characteristic_duration(t, row) for row in f]
    return out


def add_noise(table, fraction=0.1, seed=SEED):
    """Copy with Gaussian noise of standard deviation ``fraction`` x tau_c."""
    out = table.copy()
    rng = np.random.default_rng(seed)
    out["tauc"] = out["tauc"] + rng.normal(0.0, fraction * out["tauc"].to_numpy())
    return out
