"""Package-wide constants, defined once with the reason for their value."""
from __future__ import annotations

from pathlib import Path

# Seismic moment from moment magnitude, M0 = 10**(1.5*Mw + C) [N m], with
# C = 9.05 as in Seo et al. (2026); Hanks & Kanamori (1979) give C = 9.1.
MOMENT_CONSTANT = 9.05

# Quality control of apparent source time functions (ASTFs): keep an ASTF only
# if the projected Landweber reconstruction explains at least this much of the
# target waveform (variance reduction, %).
VR_THRESHOLD = 70.0

# Default seed of every random draw (chain starting states, bootstrap
# resampling, posterior subsampling for plots).
SEED = 620

# TauP phase names used to compute take-off angles for each measured phase.
# "SmS" is the Moho-reflected S wave (SvmS in TauP notation).
TAUP_PHASES = {"P": ["p", "P"], "S": ["s", "S"], "SmS": ["SvmS"]}

PACKAGE_DATA = Path(__file__).resolve().parent / "data"


def moment_from_mw(mw, constant=MOMENT_CONSTANT):
    """Seismic moment [N m] from moment magnitude."""
    return 10.0 ** (1.5 * mw + constant)
