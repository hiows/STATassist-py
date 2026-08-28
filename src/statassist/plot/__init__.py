"""The ``draw_*`` functions: a result in, a figure out.

Port of the ``draw_*.R`` files. The API and what comes back are R's - the same
arguments under the same names, the same tables and trees returned - and the
drawing itself is matplotlib's, since re-deriving base R's panel layout in a
library that lays panels out for you would be copying the workaround rather than
the plot.

Two conventions hold throughout:

* A ``draw_*`` call draws on the **current figure**, clearing it first, the way an
  R plot draws on the current device. Open a figure of your own with
  :func:`matplotlib.pyplot.figure` first to size it or to keep an earlier plot.
* ``cex_*`` arguments are multipliers of ``rcParams["font.size"]``, as R's
  ``cex.*`` are multipliers of the device's font size, and ``margin`` is still
  four numbers in lines of text.
"""

from __future__ import annotations

from .barplot import BAR_ERRORBARS, BAR_HEIGHTS, draw_grouped_barplot
from .boxplot import BOX_PANEL_AXES, draw_grouped_boxplot
from .butterfly import BUTTERFLY_SCALES, BUTTERFLY_TYPES, draw_butterfly_hist
from .corrplot import CORR_LIMITS, draw_corrplot
from .dim_reduction import SCATTER_VIEWS, draw_dim_reduction_plot
from .forest import FOREST_VIEWS, draw_forest_plot
from .heatmap import HCLUST_METHODS, HEATMAP_SCALES, Clustering, draw_heatmap
from .interaction import INTERACTION_VIEWS, draw_interaction_plot
from .mosaic import ANNO_MODES, MOSAIC_COLORS, RESIDUAL_BREAKS, RESIDUALS, draw_mosaic_plot
from .prediction import PREDICTION_VIEWS, draw_prediction_plot
from .roc import draw_roc_curve
from .volcano import draw_volcano_plot

__all__ = [
    "ANNO_MODES",
    "BAR_ERRORBARS",
    "BAR_HEIGHTS",
    "BOX_PANEL_AXES",
    "BUTTERFLY_SCALES",
    "BUTTERFLY_TYPES",
    "CORR_LIMITS",
    "FOREST_VIEWS",
    "HCLUST_METHODS",
    "HEATMAP_SCALES",
    "INTERACTION_VIEWS",
    "MOSAIC_COLORS",
    "PREDICTION_VIEWS",
    "RESIDUALS",
    "RESIDUAL_BREAKS",
    "SCATTER_VIEWS",
    "Clustering",
    "draw_butterfly_hist",
    "draw_corrplot",
    "draw_dim_reduction_plot",
    "draw_forest_plot",
    "draw_grouped_barplot",
    "draw_grouped_boxplot",
    "draw_heatmap",
    "draw_interaction_plot",
    "draw_mosaic_plot",
    "draw_prediction_plot",
    "draw_roc_curve",
    "draw_volcano_plot",
]
