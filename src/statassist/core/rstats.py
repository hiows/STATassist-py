"""R's defaults where SciPy spells the same statistic differently.

One function so far. It is here rather than beside either of its two callers -
:func:`~statassist.kernel.diagnostic.flag_outliers` and the descriptive summary -
because the two have to agree: a value the outlier screen calls a robust z of 3.6
and the summary table reports a MAD for should be the same MAD.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["MAD_CONSTANT", "mad"]

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
