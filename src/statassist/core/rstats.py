"""R's defaults where SciPy spells the same statistic differently.

These live here rather than beside any one caller because the callers have to
agree: a value the outlier screen calls a robust z of 3.6 and the summary table
reports a MAD for should be the same MAD, and a cell centre that feeds
:func:`~statassist.core.fact_term_effect` has to round the same way the
decomposition expects.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["MAD_CONSTANT", "mad", "r_mean"]

#: The scale factor ``stats::mad()`` applies by default.
#:
#: It makes the median absolute deviation estimate a standard deviation for
#: normal data, and the exact value would be ``1 / qnorm(0.75)`` =
#: 1.4826022185056. R's default is the rounded literal below, and the difference
#: shows: it moves every robust z by about 1.5e-6 relative, which is a hundred
#: times the tolerance the golden fixtures are graded at.
#:
#: :func:`scipy.stats.median_abs_deviation` offers ``scale="normal"``, which is
#: the unrounded constant, so it cannot be used here. R's literal is what this
#: package reports.
MAD_CONSTANT = 1.4826


def mad(v: Any, constant: float = MAD_CONSTANT) -> float:
    """Median absolute deviation about the median, scaled as R scales it.

    Port of ``stats::mad()`` at its defaults. Missing values are not handled: the
    caller has already decided which observations are usable.
    """
    array = np.asarray(v, dtype=float).reshape(-1)
    centre = float(np.median(array))
    return constant * float(np.median(np.abs(array - centre)))


def r_mean(v: Any) -> float:
    """Arithmetic mean rounded the way R's ``mean()`` aims to round.

    R accumulates in ``long double`` and then makes a correction pass over the
    residuals, so its answer is the correctly rounded one to within a hair.
    :func:`numpy.mean` sums pairwise in double and lands within an ULP of it,
    which is invisible at the tolerance the rest of this port is graded at and
    not invisible where a later step breaks a tie on absolute value: a two-level
    factor's ANOVA components are ``-d/2`` and ``+d/2`` by construction, and an
    ULP in a cell centre decides which of those two
    :func:`~statassist.core.fact_term_effect` reports as ``log2_effect``.

    The structure follows R's C code: a compensated sum, then a residual
    correction. :func:`math.fsum` is exactly rounded, so the result agrees with
    what R is aiming at rather than with how R gets there - which cannot be
    copied anyway, since ``long double`` is 80 bits where R is built and 64
    where CPython is.

    Missing values are not handled: the caller has already decided which
    observations are usable.
    """
    values = np.asarray(v, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    centre = math.fsum(values.tolist()) / values.size
    centre += math.fsum((values - centre).tolist()) / values.size
    return float(centre)
