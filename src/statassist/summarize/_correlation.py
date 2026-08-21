"""The three correlation tests exactly as ``stats::cor.test()`` runs them.

Port of the parts of R's ``cor.test.default()`` that
:func:`~statassist.summarize_association_stats` reaches, plus the two exact
distributions behind them: ``C_pRho`` from R's ``prho.c`` (Algorithm AS 89) and
``C_pKendall`` from its ``kendall.c``.

They are written out because SciPy answers a different question by default.
:func:`scipy.stats.spearmanr` always uses the asymptotic t approximation, where
R's default is the exact permutation distribution for any ``n <= 1290``, and
continuous simulated data is exactly the case that takes it. The two disagree in
the third decimal on a sample of eight, which is not a tolerance question. The
same holds for Kendall's test, where SciPy's ``method="exact"`` is the same
dynamic programme but its ``"asymptotic"`` differs from R's in the tie
correction.

Every function here takes two vectors that have already been reduced to their
complete cases, and returns the two-sided p-value alone. The one-sided
alternatives R offers have no caller in this package: a correlation screen asks
whether a pair is associated, not in which direction.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "KENDALL_EXACT_MAX_N",
    "METHODS",
    "RHO_EXACT_MAX_N",
    "RHO_SMALL_N",
    "cor_test_pvalue",
    "kendall_tau",
    "p_kendall",
    "p_rho",
    "spearman_rho",
]

#: The three coefficients, spelled as R spells them.
METHODS: tuple[str, ...] = ("pearson", "spearman", "kendall")

#: Up to this many observations ``prho`` enumerates every permutation; above it
#: the Edgeworth series takes over. R's ``prho.c`` calls this ``n_small`` and
#: notes that 10 already takes longer than the approximation.
RHO_SMALL_N = 9

#: Above this many observations R abandons the exact Spearman distribution
#: because ``n * (n^2 - 1)`` would overflow a 32-bit integer.
RHO_EXACT_MAX_N = 1290

#: Below this many observations ``cor.test`` defaults to the exact Kendall
#: distribution.
KENDALL_EXACT_MAX_N = 50

#: Edgeworth coefficients of Algorithm AS 89, in R's order.
_AS89 = (
    0.2274,
    0.2531,
    0.1745,
    0.0758,
    0.1033,
    0.3932,
    0.0879,
    0.0151,
    0.0072,
    0.0831,
    0.0131,
    4.6e-4,
)


@cache
def _rho_counts(n: int) -> tuple[np.ndarray, int]:
    """How many permutations of ``n`` reach each value of ``S``.

    R enumerates the permutations afresh on every call. Cached here instead,
    since a screen over many pairs of the same length would otherwise redo the
    same 362,880 permutations for each of them.

    Returns:
        The counts indexed by ``S``, and ``n!``.
    """
    from itertools import permutations

    positions = np.arange(1, n + 1)
    top = n * (n * n - 1) // 3
    counts = np.zeros(top + 1, dtype=np.int64)
    for permutation in permutations(positions):
        s = int(np.sum((positions - np.asarray(permutation)) ** 2))
        counts[s] += 1
    return counts, math.factorial(n)


def p_rho(s: float, n: int, lower_tail: bool) -> float:
    """The distribution of Spearman's ``S``, as R's ``prho`` computes it.

    Port of ``prho()`` in R's ``src/library/stats/src/prho.c``, itself Algorithm
    AS 89. ``S = (n^3 - n) * (1 - rho) / 6``, so it is small when the two
    rankings agree.

    Args:
        s: The observed ``S``. With ties it need not be an integer, which is why
            the comparison below is on the value rather than on a rounded one.
        n: Observations, at least 2.
        lower_tail: ``True`` for ``P[S < s]``, ``False`` for ``P[S >= s]``. Note
            the strict inequality on one side and not the other: that asymmetry
            is R's, and it is what makes ``cor.test`` pass ``round(q) + 2`` for
            the lower tail and ``round(q)`` for the upper one.

    Returns:
        The tail probability, clamped to ``[0, 1]``.

    References:
        Best, D. J. and Roberts, D. E. (1975). Algorithm AS 89: the upper tail
        probabilities of Spearman's rho. *Applied Statistics*, 24(3), 377-379.
    """
    if n <= 1:
        raise ValueError("`n` must be at least 2.")
    if s <= 0:
        return 0.0 if lower_tail else 1.0

    top = float(n) * (float(n) * n - 1.0) / 3.0
    if s > top:
        return 1.0 if lower_tail else 0.0

    if n <= RHO_SMALL_N:
        counts, factorial = _rho_counts(n)
        # The reversed permutation is the only one that reaches the maximum, and
        # R shortcuts to that count rather than scanning for it.
        at_least = 1 if s == top else int(counts[math.ceil(s) :].sum())
        return (factorial - at_least) / factorial if lower_tail else at_least / factorial

    b = 1.0 / n
    x = (6.0 * (s - 1) * b / (n * n - 1) - 1) * math.sqrt(n - 1.0)
    y = x * x
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = _AS89
    u = (
        x
        * b
        * (
            c1
            + b * (c2 + c3 * b)
            + y
            * (
                -c4
                + b * (c5 + c6 * b)
                - y * b * (c7 + c8 * b - y * (c9 - c10 * b + y * b * (c11 - c12 * y)))
            )
        )
    )
    correction = u / math.exp(y / 2.0)
    tail = float(stats.norm.cdf(x)) if lower_tail else float(stats.norm.sf(x))
    return min(1.0, max(0.0, (-correction if lower_tail else correction) + tail))


@cache
def _kendall_counts(n: int) -> np.ndarray:
    """How many permutations of ``n`` have each number of concordant pairs.

    R's ``ckendall`` is a memoised recursion over the same quantity. The
    coefficients of ``prod(1 + x + ... + x^(i-1))`` are the same numbers reached
    by convolution, which is what NumPy does in one pass per factor.
    """
    counts = np.array([1.0])
    for i in range(1, n + 1):
        counts = np.convolve(counts, np.ones(i))
    return counts


def p_kendall(q: float, n: int) -> float:
    """``P[T <= q]`` for Kendall's ``T``, as R's ``pkendall`` computes it.

    Port of ``pkendall()`` in R's ``src/library/stats/src/kendall.c``. ``T`` is
    the number of concordant ordered pairs, so it runs from 0 to
    ``n * (n - 1) / 2``.

    The ``1e-7`` added before the floor is R's, and it is not cosmetic: ``q``
    arrives as a double that a caller computed, so a value meant to be 14 can
    reach here as 13.999999999.
    """
    top = n * (n - 1) // 2
    floored = math.floor(q + 1e-7)
    if floored < 0:
        return 0.0
    if floored > top:
        return 1.0
    counts = _kendall_counts(n)
    return float(counts[: int(floored) + 1].sum()) / math.factorial(n)


def _tie_multiplicities(values: np.ndarray) -> np.ndarray:
    """The sizes of the tied groups, the way R's ``table(x[duplicated(x)]) + 1``
    reads them: one entry per repeated value, holding how often it occurs."""
    sizes = np.unique(values, return_counts=True)[1]
    return sizes[sizes > 1].astype(float)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho, which is Pearson's on the average ranks.

    ``cor(rank(x), rank(y))`` in R. Ranking first and correlating second is what
    gives the tie correction for free.
    """
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    if np.ptp(rank_x) == 0 or np.ptp(rank_y) == 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b, which is what ``cor(method = "kendall")`` returns."""
    n = x.size
    concordance = 0.0
    for i in range(n - 1):
        concordance += float(np.sum(np.sign(x[i + 1 :] - x[i]) * np.sign(y[i + 1 :] - y[i])))
    t0 = n * (n - 1) / 2
    t1 = float(np.sum(_tie_multiplicities(x) * (_tie_multiplicities(x) - 1))) / 2
    t2 = float(np.sum(_tie_multiplicities(y) * (_tie_multiplicities(y) - 1))) / 2
    denominator = math.sqrt((t0 - t1) * (t0 - t2))
    if denominator <= 0:
        return float("nan")
    return concordance / denominator


def _pearson_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 3:
        raise _Refused("not enough finite observations")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    df = n - 2
    if abs(r) >= 1:
        return 0.0
    statistic = math.sqrt(df) * r / math.sqrt(1 - r**2)
    return 2 * min(float(stats.t.cdf(statistic, df)), float(stats.t.sf(statistic, df)))


def _spearman_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 2:
        raise _Refused("not enough finite observations")
    r = spearman_rho(x, y)
    if not math.isfinite(r):
        return float("nan")

    q = (n**3 - n) * (1 - r) / 6
    ties = min(np.unique(x).size, np.unique(y).size) < n
    exact = not ties and n <= RHO_EXACT_MAX_N

    # Which tail to read is decided by where S falls relative to its own mean,
    # so the two-sided p-value doubles the smaller side rather than the nearer.
    lower_tail = q <= (n**3 - n) / 6

    if exact:
        p = p_rho(round(q) + 2 * lower_tail, n, lower_tail)
    else:
        # The asymptotic branch recovers rho from S rather than reusing it; on a
        # tied sample the two are the same number anyway.
        rho = 1 - q / ((n * (n**2 - 1)) / 6)
        df = n - 2
        if abs(rho) >= 1:
            p = 0.0
        else:
            statistic = rho / math.sqrt((1 - rho**2) / df)
            p = float(stats.t.sf(statistic, df) if lower_tail else stats.t.cdf(statistic, df))
    return min(2 * p, 1.0)


def _kendall_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 2:
        raise _Refused("not enough finite observations")
    r = kendall_tau(x, y)
    if not math.isfinite(r):
        return float("nan")

    ties = min(np.unique(x).size, np.unique(y).size) < n
    if n < KENDALL_EXACT_MAX_N and not ties:
        q = round((r + 1) * n * (n - 1) / 4)
        p = 1 - p_kendall(q - 1, n) if q > n * (n - 1) / 4 else p_kendall(q, n)
        return min(2 * p, 1.0)

    x_ties = _tie_multiplicities(x)
    y_ties = _tie_multiplicities(y)
    t0 = n * (n - 1) / 2
    t1 = float(np.sum(x_ties * (x_ties - 1))) / 2
    t2 = float(np.sum(y_ties * (y_ties - 1))) / 2
    s = r * math.sqrt((t0 - t1) * (t0 - t2))

    v0 = n * (n - 1) * (2 * n + 5)
    vt = float(np.sum(x_ties * (x_ties - 1) * (2 * x_ties + 5)))
    vu = float(np.sum(y_ties * (y_ties - 1) * (2 * y_ties + 5)))
    v1 = float(np.sum(x_ties * (x_ties - 1))) * float(np.sum(y_ties * (y_ties - 1)))
    v2 = float(np.sum(x_ties * (x_ties - 1) * (x_ties - 2))) * float(
        np.sum(y_ties * (y_ties - 1) * (y_ties - 2))
    )
    var_s = (v0 - vt - vu) / 18 + v1 / (2 * n * (n - 1)) + v2 / (9 * n * (n - 1) * (n - 2))
    if var_s <= 0:
        return float("nan")
    # R's `continuity` defaults to FALSE here, so no half-step is taken off `s`.
    statistic = s / math.sqrt(var_s)
    return 2 * min(float(stats.norm.cdf(statistic)), float(stats.norm.sf(statistic)))


class _Refused(Exception):
    """What R raises as an error from inside ``cor.test``."""


def cor_test_pvalue(u: Any, v: Any, method: str) -> float:
    """The p-value of one correlation test, or ``nan`` when there is no test.

    Port of ``sa_cor_test_pvalue()``. ``stats::cor.test()`` refuses a pair it
    cannot test - fewer than three complete observations, or a vector with no
    variance - and the refusal is an error rather than a p-value. A screen over
    every pair cannot stop at the first such pair, so the refusal becomes a
    missing value and the rest of the matrix is still computed.

    The tie warning the exact branches of Spearman's and Kendall's tests emit is
    expected on real data and is not passed on. What comes back when it fires is
    the normal approximation, which is the p-value the engine itself falls back
    to.

    Args:
        u: The first column, with non-finite values already missing.
        v: The second column, likewise.
        method: One of :data:`METHODS`.

    Returns:
        The two-sided p-value, or ``nan``.
    """
    if method not in METHODS:
        raise ValueError("`method` must be one of: " + ", ".join(METHODS) + ".")

    first = np.asarray(u, dtype=float).reshape(-1)
    second = np.asarray(v, dtype=float).reshape(-1)
    complete = np.isfinite(first) & np.isfinite(second)
    x = first[complete]
    y = second[complete]

    engine = {
        "pearson": _pearson_pvalue,
        "spearman": _spearman_pvalue,
        "kendall": _kendall_pvalue,
    }[method]
    try:
        pvalue = engine(x, y)
    except _Refused:
        return float("nan")
    return pvalue if math.isfinite(pvalue) else float("nan")
