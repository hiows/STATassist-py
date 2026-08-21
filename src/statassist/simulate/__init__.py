"""Simulated data sets whose answer is known, and the split that precedes a fit.

A real data set can never say which features were really different, so a
comparison run on one can only be judged against another comparison. These
simulators plant a known answer and hand it back beside the data, which is what
makes recall, the false positive rate and the direction of an effect computable
rather than arguable.

Every simulator returns ``args`` named after the arguments of the function that
consumes it, so the analysis is one call away::

    sim = simulate_two_groups(seed=1)
    res = compare_two_groups(**sim.args)

The numbers are not R's numbers. Python and R have different generators, so a
seed reproduces a data set within this package and not across the two languages;
see :mod:`statassist.core.random`. What is the same is every contract around
them: slot names, column names and order, row order, and the planted quantities
the truth tables report.
"""

from __future__ import annotations

from .block_cor import make_block_cor
from .categorical_groups import simulate_categorical_groups
from .classification import simulate_classification
from .factorial_groups import simulate_factorial_groups
from .multiple_groups import simulate_multiple_groups
from .regression import simulate_regression
from .split import split_data
from .two_groups import simulate_two_groups

__all__ = [
    "make_block_cor",
    "simulate_categorical_groups",
    "simulate_classification",
    "simulate_factorial_groups",
    "simulate_multiple_groups",
    "simulate_regression",
    "simulate_two_groups",
    "split_data",
]
