"""Deterministic second-degree-moment inversion as a semidefinite program
(McGuire, 2004), solved with CVXPY.

    minimise   || G m - b ||_2^2 ,   b = (tau_c / 2)^2
    subject to X = [[mu02, mu11^T], [mu11, mu20]]  positive semidefinite,
               mu02 <= 2 max(b)

The default solver is SCS with ``max_iters=10000``; the interior-point
solver ``solver="CLARABEL"`` converges more tightly (4th-5th significant
digit).  The problem is
built once with CVXPY parameters (disciplined parametrised programming), so
re-solving it for bootstrap resamples or the second nodal plane does not
rebuild it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import slowness_matrix
from .moments import physical, predict_tauc, variance_reduction


class SDP:
    """Reusable SDP for ``n`` observations."""

    def __init__(self, n, norm="l2"):
        import cvxpy as cp
        self.G = cp.Parameter((n, 6))
        self.b = cp.Parameter(n)
        self.bmax = cp.Parameter(nonneg=True)
        X = cp.Variable((3, 3), PSD=True)
        self.m = cp.hstack([X[0, 0], X[0, 1], X[0, 2], X[1, 1], X[1, 2], X[2, 2]])
        r = self.G @ self.m - self.b
        misfit = cp.sum_squares(r) if norm == "l2" else cp.sum(cp.abs(r))
        self.problem = cp.Problem(cp.Minimize(misfit), [X[0, 0] <= 2 * self.bmax])

    def solve(self, G, tauc, solver="SCS", **kwargs):
        if solver == "SCS":
            kwargs.setdefault("max_iters", 10000)
        b = (np.asarray(tauc, float) / 2) ** 2
        self.G.value, self.b.value, self.bmax.value = np.asarray(G, float), b, float(b.max())
        self.problem.solve(solver=solver, **kwargs)
        return np.asarray(self.m.value, float)


@dataclass
class SDPResult:
    m2: np.ndarray
    params: dict
    vr: float
    pred: np.ndarray
    G: np.ndarray


def invert(table, strike, dip, vp, vs, solver="SCS", norm="l2", **kwargs):
    """Invert one duration table on one fault plane; velocities in km/s."""
    G = slowness_matrix(table, strike, dip, vp, vs)
    tauc = np.asarray(table["tauc"], float)
    m2 = SDP(len(tauc), norm).solve(G, tauc, solver=solver, **kwargs)
    pred = predict_tauc(G, m2)
    return SDPResult(m2, physical(m2), float(variance_reduction(tauc, pred)), pred, G)


def compare_nodal_planes(table, plane1, plane2, vp, vs, threshold=10.0, **kwargs):
    """Invert on both nodal planes ((strike, dip) pairs) and compare misfit.

    The ambiguity is resolved when the variance reductions differ by more
    than ``threshold`` percentage points.  Returns
    (result1, result2, preferred) with preferred = 1, 2 or None.
    """
    r1 = invert(table, *plane1, vp, vs, **kwargs)
    r2 = invert(table, *plane2, vp, vs, **kwargs)
    d = r1.vr - r2.vr
    preferred = 1 if d > threshold else (2 if d < -threshold else None)
    return r1, r2, preferred


def bootstrap(table, strike, dip, vp, vs, n=1000, seed=620, solver="SCS", **kwargs):
    """Bootstrap the SDP by resampling stations with replacement.

    Resampling uses ``DataFrame.sample(frac=1, replace=True)`` with a
    ``RandomState(seed)``, the same draws as the original notebooks (which
    called ``np.random.seed(620)`` and sampled from the global state).
    Returns an (n, 6) array of moment vectors.
    """
    rng = np.random.RandomState(seed)
    sdp = SDP(len(table))
    out = np.empty((n, 6))
    for i in range(n):
        t = table.sample(frac=1.0, replace=True, random_state=rng).reset_index(drop=True)
        out[i] = sdp.solve(slowness_matrix(t, strike, dip, vp, vs), t["tauc"], solver=solver, **kwargs)
    return out
