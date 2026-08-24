"""The comparison family: one design per module, one result contract.

Ports of ``R/compare_*.R``. Each function runs a parametric, a rank-based and a
robust test side by side and reports all of them, so that disagreement between
them is visible rather than resolved by the package on the user's behalf.

They fill the slots :func:`~statassist.estimate_significance` reads back, which
is what lets a new design inherit the whole downstream path - the verdict table
and the plots - without either side knowing which tests were actually run.
"""

from __future__ import annotations

from .factorial_groups import compare_factorial_groups
from .multiple_groups import compare_multiple_groups
from .one_sample import compare_one_sample
from .two_groups import compare_two_groups

__all__ = [
    "compare_factorial_groups",
    "compare_multiple_groups",
    "compare_one_sample",
    "compare_two_groups",
]
