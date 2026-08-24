"""Python port of the R package STATassist.

The shared helper layer is :mod:`statassist.core` and the numeric engines are in
:mod:`statassist.kernel`; the public ``verb_object`` API is re-exported here as
each phase lands.

Two families are in place. The simulation family is what makes every later phase
testable: a comparison run on real data can only be judged against another
comparison, while one run on :func:`simulate_two_groups` can be judged against
the answer that was planted. The description family is what comes before a test
is chosen - what the data looks like, whether the assumptions hold, and what a
feature reads as once the control group is taken out of it.
"""

__version__ = "0.1.0.dev0"

from .compare import (
    compare_factorial_groups,
    compare_multiple_groups,
    compare_one_sample,
    compare_two_groups,
)
from .diagnose import diagnose_distribution, screen_outliers
from .estimate import estimate_significance
from .plot import (
    draw_butterfly_hist,
    draw_forest_plot,
    draw_heatmap,
    draw_interaction_plot,
    draw_volcano_plot,
)
from .simulate import (
    make_block_cor,
    simulate_categorical_groups,
    simulate_classification,
    simulate_factorial_groups,
    simulate_multiple_groups,
    simulate_regression,
    simulate_two_groups,
    split_data,
)
from .summarize import summarize_association_stats, summarize_descriptive_stats
from .transform import center_by_control

__all__ = [
    "__version__",
    "center_by_control",
    "compare_factorial_groups",
    "compare_multiple_groups",
    "compare_one_sample",
    "compare_two_groups",
    "diagnose_distribution",
    "draw_butterfly_hist",
    "draw_forest_plot",
    "draw_heatmap",
    "draw_interaction_plot",
    "draw_volcano_plot",
    "estimate_significance",
    "make_block_cor",
    "screen_outliers",
    "simulate_categorical_groups",
    "simulate_classification",
    "simulate_factorial_groups",
    "simulate_multiple_groups",
    "simulate_regression",
    "simulate_two_groups",
    "split_data",
    "summarize_association_stats",
    "summarize_descriptive_stats",
]
