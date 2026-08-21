"""What the data looks like before a test is chosen.

Two public functions over the same split of the rows: one screens for outliers,
the other adds the normality and variance checks and reports them together.
Neither of them changes anything. A flagged observation is not removed and a
failed assumption does not swap one test for another; both are information the
user acts on, or decides not to.
"""

from __future__ import annotations

from .distribution import (
    diagnose_distribution,
    diagnose_samples,
    new_diagnosis,
    normality_table,
    variance_table,
)
from .outliers import screen_outliers, split_for_screening

__all__ = [
    "diagnose_distribution",
    "diagnose_samples",
    "new_diagnosis",
    "normality_table",
    "screen_outliers",
    "split_for_screening",
    "variance_table",
]
