"""Figures in the style of Seo et al. (2026): ASTF retrieval at one station
(Fig. 2), ASTFs by azimuth (Fig. 3), the SDM inversion summary (Fig. 4) and
the posterior corner plot (Fig. 5).  Axis ranges follow the data unless given.
"""
from .style import panel_label, use_style
from .retrieval import RetrievalWaveforms, plot_astf_retrieval, retrieval_waveforms
from .azimuth import plot_astfs_by_azimuth, time_limits
from .inversion import (nodal_plane_curve, plot_durations, plot_ellipses, plot_relative_location,
                        plot_sdm_summary, plot_symbol_key, relative_offset)
from .corner import LABELS, Panel, auto_panels, pearson_matrix, plot_corner, plot_marginal

__all__ = ["use_style", "panel_label", "RetrievalWaveforms", "retrieval_waveforms",
           "plot_astf_retrieval", "plot_astfs_by_azimuth", "time_limits", "relative_offset",
           "nodal_plane_curve", "plot_relative_location", "plot_ellipses", "plot_durations",
           "plot_symbol_key", "plot_sdm_summary", "Panel", "LABELS", "auto_panels",
           "pearson_matrix", "plot_marginal", "plot_corner"]
