"""bayes2mt: second-degree moment (SDM) inversion of small earthquakes from
apparent source time functions (ASTFs) retrieved with empirical Green's
functions, deterministic (CVXPY) and hierarchical Bayesian (BayesBay).

Modules
-------
pld            projected Landweber deconvolution, tau_c
interactive    station-by-station deconvolution in Jupyter
geometry       take-off angles, slowness matrix, azimuthal coverage
relloc         target-EGF relative location from ASTF onsets
deterministic  semidefinite-program inversion (CVXPY)
bayesian       hierarchical Bayesian inversion (BayesBay)
moments        Cholesky parameterisation, physical parameters, medoid, HDI
stressdrop     elliptical-crack stress drop
synthetic      kinematic synthetic ASTFs and durations
plotting       figures
io, config     file formats and constants
"""
from . import config, geometry, io, moments, pld, relloc, stressdrop, synthetic

__version__ = "0.1.0"

__all__ = ["config", "geometry", "io", "moments", "pld", "relloc", "stressdrop", "synthetic"]
