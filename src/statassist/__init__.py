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

from .diagnose import diagnose_distribution, screen_outliers
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
    "diagnose_distribution",
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
