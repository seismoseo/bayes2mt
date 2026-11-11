"""Deterministic second-degree-moment inversion as a semidefinite program
(McGuire, 2004), solved with CVXPY.

    minimise   || G m - b ||_2^2 ,   b = (tau_c / 2)^2
    subject to X = [[mu02, mu11^T], [mu11, mu20]]  positive semidefinite,
               mu02 <= 2 max(b)

The durations are divided by their median before solving and the moments
are scaled back afterwards (``m2 * scale**2``).  Without this the objective
is of order 1e-9 s^4, far below the default stopping tolerances of the conic
solvers, which then report an optimal solution that can be several
variance-reduction points from the optimum; with it, SCS (default,
``max_iters=10000``) and the interior-point solver CLARABEL agree to 0.001
points.  The variance reduction does not depend on the scale.  The problem is
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
        """Moments [s^2, s km, km^2] that fit the durations ``tauc`` [s]."""
        if solver == "SCS":
            kwargs.setdefault("max_iters", 10000)
        tauc = np.asarray(tauc, float)
        scale = float(np.median(tauc))
        b = (tauc / scale / 2) ** 2
        self.G.value, self.b.value, self.bmax.value = np.asarray(G, float), b, float(b.max())
        self.problem.solve(solver=solver, **kwargs)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SDP not solved: {self.problem.status}")
        return np.asarray(self.m.value, float) * scale ** 2


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
    ``RandomState(seed)``.
    Returns an (n, 6) array of moment vectors.
    """
    rng = np.random.RandomState(seed)
    sdp = SDP(len(table))
    out = np.empty((n, 6))
    for i in range(n):
        t = table.sample(frac=1.0, replace=True, random_state=rng).reset_index(drop=True)
        out[i] = sdp.solve(slowness_matrix(t, strike, dip, vp, vs), t["tauc"], solver=solver, **kwargs)
    return out
