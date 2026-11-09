"""Second-degree moments: parameterisation, physical conversion, ensemble
statistics (medoid, highest-density intervals).

The six independent moments are ordered
    m2 = [mu02, mu11_x, mu11_y, mu20_xx, mu20_xy, mu20_yy]
with x along strike and y down dip (time in s, space in km).

Sampling parameterisation (Atterholt & Ross, 2022).  The 3x3 moment matrix
X = [[mu02, mu11^T], [mu11, mu20]] is written as X = L L^T with
    L = [[10**x0,       0,       0],
         [x1,      10**x3,       0],
         [x2,          x4,  10**x5]] ,
so every sample is positive semi-definite.  Expanding L L^T gives
    m2 = [10**(2 x0), 10**x0 x1, 10**x0 x2,
          x1^2 + 10**(2 x3), x1 x2 + 10**x3 x4, x2^2 + x4^2 + 10**(2 x5)].
"""
from __future__ import annotations

import math

import numpy as np

PHYSICAL_KEYS = ["tauc", "Lc", "Wc", "vx", "vy", "v0", "rake", "angle", "dir_ratio"]


# ------------------------------------------------------------ parameterisation
def cholesky_to_m2(x):
    """Moments from Cholesky parameters; ``x`` is (6,) or (N, 6)."""
    x = np.asarray(x, float)
    x0, x1, x2, x3, x4, x5 = np.moveaxis(x, -1, 0)
    a0, a3 = 10.0 ** x0, 10.0 ** x3
    return np.stack([10.0 ** (2 * x0), a0 * x1, a0 * x2,
                     x1 * x1 + 10.0 ** (2 * x3), x1 * x2 + a3 * x4,
                     x2 * x2 + x4 * x4 + 10.0 ** (2 * x5)], axis=-1)


def predict_tauc(G, m2):
    """Apparent durations tau_c(s) = 2 sqrt(G m2); ``m2`` (6,) or (N, 6)."""
    return 2.0 * np.sqrt(np.asarray(m2) @ np.asarray(G).T)


def variance_reduction(obs, pred):
    """Demeaned variance reduction [%] (= 100 R^2), along the last axis."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return (1.0 - np.sum((obs - pred) ** 2, axis=-1) / np.sum((obs - obs.mean()) ** 2)) * 100.0


# ---------------------------------------------------------- physical conversion
def physical(m2):
    """Rupture parameters from moments; ``m2`` is (6,) or (N, 6).

    Returns a dict of arrays (scalars for a single model):
      tauc      characteristic duration 2 sqrt(mu02) [s]
      Lc, Wc    characteristic length / width 2 sqrt(eigenvalues of mu20) [m]
      vx, vy    centroid velocity mu11 / mu02, along strike / down dip [km/s]
      v0        |v0| [m/s]
      rake      direction of v0, deg counter-clockwise from strike with
                up-dip at 90 deg (theta^v0), 0-360
      angle     direction of the major axis, same convention, 0-180
                (theta^Lc; an axis, so defined modulo 180)
      dir_ratio directivity ratio |v0| / (Lc / tauc)

    The eigen-decomposition of the 2x2 spatial block is done in closed form
    rather than by an SVD per sample; values agree with the SVD to rounding.
    """
    m2 = np.asarray(m2, float)
    single = m2.ndim == 1
    m2 = np.atleast_2d(m2)
    a, b, c = m2[:, 3], m2[:, 4], m2[:, 5]
    half_tr, root = 0.5 * (a + c), np.sqrt(0.25 * (a - c) ** 2 + b * b)
    lam1, lam2 = half_tr + root, np.clip(half_tr - root, 0.0, None)
    # eigenvector of lam1; pick the better-conditioned of the two forms
    use_c = a >= c
    px = np.where(use_c, lam1 - c, b)
    py = np.where(use_c, b, lam1 - a)
    flat = (px == 0) & (py == 0)                      # isotropic block
    px = np.where(flat, 1.0, px)
    angle = np.mod(np.rad2deg(np.arctan2(-py, px)), 180.0)
    Lc, Wc = 2 * np.sqrt(lam1) * 1000.0, 2 * np.sqrt(lam2) * 1000.0
    tauc = 2 * np.sqrt(m2[:, 0])
    vx, vy = m2[:, 1] / m2[:, 0], m2[:, 2] / m2[:, 0]
    rake = np.mod(np.rad2deg(np.arctan2(-vy, vx)), 360.0)
    v0 = np.sqrt(vx * vx + vy * vy) * 1000.0
    out = dict(tauc=tauc, Lc=Lc, Wc=Wc, vx=vx, vy=vy, v0=v0, rake=rake, angle=angle,
               dir_ratio=v0 / (Lc / tauc))
    if single:
        out = {k: float(v[0]) for k, v in out.items()}
    return out


# --------------------------------------------------------------------- medoid
_KERNEL = {}


def _weighted_distance_sums():
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def kernel(X, w):
        n, d = X.shape
        out = np.zeros(n)
        for i in prange(n):
            s = 0.0
            for j in range(n):
                q = 0.0
                for k in range(d):
                    diff = X[i, k] - X[j, k]
                    q += diff * diff
                s += w[j] * math.sqrt(q)
            out[i] = s
        return out

    return kernel


def medoid_index(m2, standardize=True):
    """Exact medoid: the sample minimising the summed Euclidean distance to
    all other samples, in z-scored moment space.

    MCMC ensembles repeat a state every time a proposal is rejected, so the
    search runs over the unique states with their multiplicities as weights
    (typically half of the samples are unique), with the same answer as the
    all-pairs search.  Distances are summed in float64.
    Returns the index of the first occurrence of the medoid state.
    """
    X = np.asarray(m2, float)
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    U, first, counts = np.unique(X, axis=0, return_index=True, return_counts=True)
    if "k" not in _KERNEL:
        _KERNEL["k"] = _weighted_distance_sums()
    sums = _KERNEL["k"](np.ascontiguousarray(U), counts.astype(float))
    return int(first[int(np.argmin(sums))])


# --------------------------------------------------------- credible intervals
def hdi(x, prob=0.95):
    """Highest-density interval of a 1-D sample (ArviZ convention)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    k = int(np.floor(prob * n))
    widths = x[k:] - x[:n - k]
    i = int(np.argmin(widths))
    return np.array([x[i], x[i + k]])


def circular_hdi(x, prob=0.95, modulus=360.0):
    """Highest-density interval of an angle sample on [0, modulus).

    Returns an (m, 2) array: one row, or two rows when the interval wraps
    through 0 (e.g. [[350, 360], [0, 10]]).
    """
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    k = math.ceil(prob * n)
    ext = np.concatenate([xs, xs + modulus])
    widths = ext[np.arange(n) + k - 1] - ext[:n]
    i = int(np.argmin(widths))
    lo, hi = ext[i], ext[i + k - 1]
    if hi >= modulus:
        return np.array([[lo, modulus], [0.0, hi - modulus]])
    return np.array([[lo, hi]])
