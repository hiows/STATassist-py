"""Summaries that describe data rather than test a hypothesis.

Two public functions and the helpers they are built from. Neither reports a
decision: :func:`~statassist.summarize_descriptive_stats` reduces each feature to
a row of moments and quantiles, and
:func:`~statassist.summarize_association_stats` fills a correlation matrix. What
to conclude from either is left to the reader, which is why nothing here takes an
``alpha``.
"""

from __future__ import annotations

from ._correlation import cor_test_pvalue, kendall_tau, p_kendall, p_rho, spearman_rho
from .association import (
    association_matrices,
    pairwise_n,
    summarize_association_stats,
)
from .descriptive import (
    describe_columns,
    describe_vector,
    kurtosis,
    skewness,
    summarize_descriptive_stats,
)

__all__ = [
    "association_matrices",
    "cor_test_pvalue",
    "describe_columns",
    "describe_vector",
    "kendall_tau",
    "kurtosis",
    "p_kendall",
    "p_rho",
    "pairwise_n",
    "skewness",
    "spearman_rho",
    "summarize_association_stats",
    "summarize_descriptive_stats",
]
