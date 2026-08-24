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

from .butterfly import BUTTERFLY_SCALES, BUTTERFLY_TYPES, draw_butterfly_hist
from .forest import FOREST_VIEWS, draw_forest_plot
from .heatmap import HCLUST_METHODS, HEATMAP_SCALES, Clustering, draw_heatmap
from .interaction import INTERACTION_VIEWS, draw_interaction_plot
from .volcano import draw_volcano_plot

__all__ = [
    "BUTTERFLY_SCALES",
    "BUTTERFLY_TYPES",
    "FOREST_VIEWS",
    "HCLUST_METHODS",
    "HEATMAP_SCALES",
    "INTERACTION_VIEWS",
    "Clustering",
    "draw_butterfly_hist",
    "draw_forest_plot",
    "draw_heatmap",
    "draw_interaction_plot",
    "draw_volcano_plot",
]
