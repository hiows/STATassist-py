"""Breaks, bins and a kernel density, on R's terms.

:func:`draw_butterfly_hist` reports the numbers behind its bars, so those
numbers have to be R's: the break points ``pretty()`` chooses, the counts
``hist()`` puts between them and the curve ``density()`` draws. numpy and scipy
have all three, and none of them the same way round - ``numpy.histogram`` closes
its bins on the left where R closes them on the right, and
:class:`scipy.stats.gaussian_kde` has neither R's default bandwidth nor its
default grid. The arithmetic is a few lines either way, so it is written here
against R's definitions rather than wrapped and then corrected.

The one deliberate difference is that the density is evaluated directly instead
of by the binned Fourier transform R uses. That is the same estimator computed
the slow way, which for the sample sizes one feature of one group holds is not
slow at all.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import SaValueError

__all__ = [
    "BREAK_RULES",
    "DENSITY_CUT",
    "DENSITY_N",
    "Density",
    "Histogram",
    "bw_nrd0",
    "density",
    "histogram",
    "nclass",
    "pretty",
]

#: The break rules :func:`graphics::hist` names, and this port with it.
BREAK_RULES = ("Sturges", "Scott", "FD")

#: How far past the data a density is evaluated, in bandwidths. R's ``cut``.
DENSITY_CUT = 3.0

#: How many points the density is evaluated at. R's ``n``.
DENSITY_N = 512

# R's pretty() weighs the wider unit against the closer fit with these, and the
# tick marks come out different if they are changed. `high.u.bias` and
# `u5.bias`, at their defaults.
_HIGH_U_BIAS = 1.5
_U5_BIAS = 0.5 + 1.5 * _HIGH_U_BIAS
_ROUNDING_EPS = 1e-10


def pretty(low: float, high: float, n: int = 5, min_n: int | None = None) -> np.ndarray:
    """Round numbers covering ``low`` to ``high``, as :func:`base::pretty` picks them.

    The unit is 1, 2, 5 or 10 times a power of ten, chosen by weighing how well
    it divides the range against how round it is, and the sequence is then
    extended outwards until it covers both ends.

    Args:
        low: Low end of the range to cover.
        high: High end of the range to cover.
        n: Roughly how many intervals are wanted.
        min_n: The fewest intervals accepted, ``n // 3`` as in R.

    Returns:
        The break points, from at or below ``low`` to at or above ``high``.
    """
    if min_n is None:
        min_n = n // 3
    if not math.isfinite(low) or not math.isfinite(high):
        raise SaValueError("`pretty()` needs a finite range.")
    if high < low:
        low, high = high, low

    span = high - low
    if span == 0:
        # No range to divide, so the unit comes from how large the value is
        # rather than from how wide the interval is.
        cell = max(abs(low), 1.0) * 0.75
        if min_n > 1:
            cell /= min_n
    else:
        cell = span / n if n > 1 else span

    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    if 2 * base - cell < _HIGH_U_BIAS * (cell - unit):
        unit = 2 * base
        if 5 * base - cell < _U5_BIAS * (cell - unit):
            unit = 5 * base
            if 10 * base - cell < _HIGH_U_BIAS * (cell - unit):
                unit = 10 * base

    start = math.floor(low / unit + _ROUNDING_EPS)
    stop = math.ceil(high / unit - _ROUNDING_EPS)
    while start * unit > low + _ROUNDING_EPS * unit:
        start -= 1
    while stop * unit < high - _ROUNDING_EPS * unit:
        stop += 1

    # Too few intervals is padded out on both sides, the low end first for a
    # range that starts below zero, so that zero stays a break point.
    short = min_n - (stop - start)
    if short > 0:
        if start >= 0:
            stop += short // 2
            start -= short // 2 + short % 2
        else:
            start -= short // 2
            stop += short // 2 + short % 2

    return np.arange(start, stop + 1, dtype=float) * unit


def nclass(x: np.ndarray, rule: str) -> int:
    """How many bins a break rule asks for.

    Ports ``nclass.Sturges``, ``nclass.scott`` and ``nclass.FD``, which is what
    :func:`graphics::hist` reads a character ``breaks`` as.
    """
    n = x.size
    if rule == "Sturges":
        return int(math.ceil(math.log2(n) + 1)) if n > 0 else 1
    spread = float(x.std(ddof=1)) if n > 1 else 0.0
    if rule == "Scott":
        width = 3.5 * spread * n ** (-1 / 3)
    else:
        # Freedman-Diaconis falls back on the standard deviation when the
        # interquartile range is zero, exactly as R does.
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        width = 2 * iqr * n ** (-1 / 3)
        if width == 0:
            width = 2 * spread * n ** (-1 / 3)
    if width <= 0:
        return 1
    return max(1, int(math.ceil((x.max() - x.min()) / width)))


class Histogram(dict):
    """One group binned on shared breaks, as :func:`graphics::hist` reports it.

    A ``dict`` with ``breaks``, ``counts``, ``density``, ``mids`` and ``xname``,
    which are the elements of R's ``"histogram"`` object that carry information.
    R returns an object a ``plot()`` method knows; here the caller has the
    numbers and matplotlib to draw them with, so the object is the numbers.
    """


class Density(dict):
    """A kernel density estimate, as :func:`stats::density` reports it.

    A ``dict`` with ``x``, ``y``, ``bw``, ``n`` and ``data_name``.
    """


def histogram(x: np.ndarray, breaks: np.ndarray, xname: str) -> Histogram:
    """Bin ``x`` on ``breaks``, closing every bin on the right.

    R's ``hist(right = TRUE, include.lowest = TRUE)``: a value falls in
    ``(b[i], b[i + 1]]``, and a value sitting exactly on the lowest break falls
    in the first bin rather than outside every one of them.

    Raises:
        SaValueError: If a value lies outside ``breaks``, which is what R
            reports as "some 'x' not counted".
    """
    at = np.searchsorted(breaks, x, side="left") - 1
    at[x == breaks[0]] = 0
    n_bins = breaks.size - 1
    if at.size > 0 and (at.min() < 0 or at.max() >= n_bins):
        raise SaValueError("`breaks` do not span the range of the values to bin.")

    counts = np.bincount(at, minlength=n_bins).astype(float)
    widths = np.diff(breaks)
    total = counts.sum()
    return Histogram(
        breaks=breaks,
        counts=counts,
        density=counts / (total * widths) if total > 0 else np.zeros(n_bins),
        mids=(breaks[:-1] + breaks[1:]) / 2,
        xname=xname,
    )


def bw_nrd0(x: np.ndarray) -> float:
    """Silverman's rule of thumb, as ``bw.nrd0`` computes it.

    The default bandwidth of :func:`stats::density`, including its fallbacks:
    the interquartile range is used when it is narrower than the standard
    deviation, and a sample with no spread at all falls back on its own
    magnitude so that the bandwidth is never zero.
    """
    n = x.size
    spread = float(x.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    lo = min(spread, iqr / 1.349)
    if lo == 0:
        lo = spread or abs(float(x[0])) or 1.0
    return float(0.9 * lo * n ** (-0.2))


def density(x: np.ndarray, adjust: float = 1.0, data_name: str = "x") -> Density:
    """A Gaussian kernel density estimate on R's default grid.

    The bandwidth is ``bw.nrd0`` times ``adjust``, and the estimate is evaluated
    at :data:`DENSITY_N` equally spaced points running :data:`DENSITY_CUT`
    bandwidths past each end of the data, which is where R evaluates it.
    """
    bw = bw_nrd0(x) * float(adjust)
    grid = np.linspace(x.min() - DENSITY_CUT * bw, x.max() + DENSITY_CUT * bw, DENSITY_N)
    z = (grid[:, None] - x[None, :]) / bw
    y = np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * math.sqrt(2 * math.pi))
    return Density(x=grid, y=y, bw=bw, n=x.size, data_name=data_name)
