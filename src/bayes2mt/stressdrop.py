"""Static stress drop of the elliptical crack described by the second moments.

The second-degree moments give the characteristic length Lc and width Wc
(twice the standard deviations of the slip along its principal axes) and the
angle theta^Lc of the major axis.  The stress drop is that of a uniform
stress-drop elliptical shear crack (Eshelby, 1957) with semi-axes a = Lc and
b = Wc, major axis at theta^Lc, slip along the focal-mechanism rake and
seismic moment M0, for a Poisson's ratio of 0.25 (Madariaga, 1977):

    Delta sigma = M0 / (pi a b^2) [cos^2(psi) g_par + sin^2(psi) g_perp]
    g_par  = [3E + (b/a)^2 (K - E) / k^2] / 4     slip along the major axis
    g_perp = [3E + (E - (b/a)^2 K) / k^2] / 4     slip along the minor axis

with k^2 = 1 - (b/a)^2, K and E the complete elliptic integrals and
psi = rake - theta^Lc (both angles in the fault plane, from the strike
direction, positive up-dip).  For a = b this is Eshelby's 7 M0 / (16 a^3).
Over the crack the stress drop is uniform, so its area average, moment-based
and energy-based averages (Noda et al., 2013) coincide.  The expression
agrees with a numerical Ripperger & Mai (2004) solution to within 0.2 %.

Semi-axes a = Lc, b = Wc follow McGuire (2017) and McGuire & Kaneko (2018).
For the crack itself Lc = (2 / sqrt 5) a, so this convention gives
(sqrt 5 / 2)^3 = 1.40 times the crack's own stress drop; pass
``semi_axis_factor=np.sqrt(5) / 2`` for the latter.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk


def stress_drop(M0, Lc, Wc, rake, angle, semi_axis_factor=1.0):
    """Stress drop [MPa] for seismic moment ``M0`` [N m], ``Lc`` and ``Wc``
    [m], slip ``rake`` and major-axis ``angle`` [deg from strike].  Arrays
    broadcast (one value per posterior sample, for example)."""
    M0, Lc, Wc = (np.asarray(v, float) for v in (M0, Lc, Wc))
    a = semi_axis_factor * np.maximum(Lc, Wc)
    b = semi_axis_factor * np.minimum(Lc, Wc)
    r2 = (b / a) ** 2
    k2 = 1.0 - r2
    K, E = ellipk(k2), ellipe(k2)
    elongated = k2 > 1e-8
    k2_safe = np.where(elongated, k2, 1.0)
    g_par = np.where(elongated, (3 * E + r2 * (K - E) / k2_safe) / 4, 7 * np.pi / 16)
    g_perp = np.where(elongated, (3 * E + (E - r2 * K) / k2_safe) / 4, 7 * np.pi / 16)
    psi = np.radians(np.asarray(rake, float) - np.asarray(angle, float))
    return M0 / (np.pi * a * b ** 2) * (np.cos(psi) ** 2 * g_par + np.sin(psi) ** 2 * g_perp) / 1e6


def circular_stress_drop(M0, radius):
    """Eshelby (1957) circular crack, 7 M0 / (16 r^3) [MPa]; ``radius`` in m."""
    return 7.0 * np.asarray(M0, float) / (16.0 * np.asarray(radius, float) ** 3) / 1e6
