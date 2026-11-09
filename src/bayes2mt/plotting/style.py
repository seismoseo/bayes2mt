"""Figure style shared by all plotting functions.

Helvetica is used when available, as in the paper's figures.  Point the
``BAYES2MT_FONT_DIR`` environment variable at a folder of Helvetica font files
if they are not installed system-wide; otherwise Arial, then DejaVu Sans, is
used.  Labels are in sentence case and every panel keeps all four spines.
"""
from __future__ import annotations

import os

import matplotlib as mpl
from matplotlib import font_manager

FONT_ENV = "BAYES2MT_FONT_DIR"


def use_style():
    """Register fonts and set rcParams; returns the font family in use."""
    font_dir = os.environ.get(FONT_ENV)
    if font_dir and os.path.isdir(font_dir):
        for f in font_manager.findSystemFonts([font_dir]):
            font_manager.fontManager.addfont(f)
    names = {f.name.lower() for f in font_manager.fontManager.ttflist}
    family = next((n for n in ("helvetica", "arial") if n in names), "dejavu sans")
    mpl.rcParams["font.family"] = family
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["svg.fonttype"] = "none"
    return family


def panel_label(ax, text, x=-0.18, y=1.05, size=20):
    """Bold panel label such as "(a)" at the upper left of an axes."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="bottom", ha="right", zorder=100)
