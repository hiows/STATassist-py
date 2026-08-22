"""The rank-based family of a comparison, with the interval R reports.

Port of what ``stats::wilcox.test()`` does, which is more than SciPy offers.
``scipy.stats.mannwhitneyu`` and ``scipy.stats.wilcoxon`` give a statistic and a
p-value; neither reports the Hodges-Lehmann location estimate or its confidence
interval, and those two are columns of the comparison contract
(``hl_shift``, ``lower_conf``, ``upper_conf``). Wrapping SciPy for the p-value
and writing the interval here would also mean two different decisions about when
the exact distribution is used, which is exactly the kind of disagreement the
kernel layer exists to avoid. So the whole test is written out once.

Three parts of R 4.x are reproduced deliberately.

The exact distribution is tie-aware
    Older R fell back on the normal approximation as soon as two observations
    tied. R 4.x instead conditions on the observed rank vector: the exact null
    distribution is the one induced by *those* ranks, half-integers included.
    :func:`_subset_sum_counts` and :func:`_choose_sum_counts` are the two
    convolutions R runs in C as ``dpermdist1`` and ``dpermdist2``.

The interval is a step function inverted by search
    A rank statistic moves in jumps, so there is no formula for the endpoint.
    Without ties R reads it off an order statistic of the Walsh averages or the
    pairwise differences; with ties it walks the midpoints between them. Both are
    here, and which one runs is decided the way R decides it.

``uniroot`` is Brent's method with a stated tolerance
    The asymptotic interval solves ``W(d) = z`` numerically at ``tol = 1e-4``, so
    the value returned depends on the iterates and not only on the root. R's
    ``zeroin`` is transcribed in :func:`_zeroin` rather than replaced by
    ``scipy.optimize.brentq``, whose convergence test is not the same one, which
    is what lets this column be compared against R at the same tolerance as the
    closed-form ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from ..core.errors import SaValueError, warn
from ._shared import as_sample, check_alternative

__all__ = [
    "EXACT_MAX_N",
    "TOL_ROOT",
    "psignrank",
    "pwilcox",
    "qsignrank",
    "qwilcox",
    "rank_sum",
    "signed_rank",
]

#: Largest sample the exact distribution is used for, per group.
#:
#: R's ``exact <- (n < 50)``. Beyond it the normal approximation takes over, for
#: the cost of the exact convolution rather than for any change in its validity.
EXACT_MAX_N = 50

#: Tolerance the asymptotic interval's root is located to.
#:
#: ``wilcox.test(tol.root = 1e-4)``. Not the accuracy of the interval, which is
#: set by the discreteness of the statistic; this is only where the search stops.
TOL_ROOT = 1e-4

#: Slack in the "at or below" test on the exact support.
#:
#: R writes ``sum(d[s < e + 1e-08])``: the support holds halves once there are
#: ties, and comparing accumulated halves for equality is what this avoids.
_SUPPORT_TOL = 1e-8

#: Slack added to alpha when the interval endpoints are searched for.
#:
#: R's ``toler <- 10 * .Machine$double.eps``.
_ALPHA_TOL = 10 * np.finfo(float).eps


# --------------------------------------------------------------------------- #
# Exact null distributions
# --------------------------------------------------------------------------- #


def _rank_scale(z: np.ndarray) -> int:
    """Whether the rank vector has to be doubled to become whole.

    R's ``f <- 2 - all(z == floor(z))``. Average ranks over a tie of even width
    land on a half, and the convolutions below index an array by the sum.
    """
    return 1 if bool(np.all(z == np.floor(z))) else 2


def _subset_sum_counts(scaled: np.ndarray) -> np.ndarray:
    """How many subsets of ``scaled`` add up to each total.

    R's ``dpermdist1``, before the division by ``2 ** n``. Every observation is
    either signed positive or not, so the generating function is the product of
    ``1 + t ** z`` over the ranks.
    """
    counts = np.zeros(int(scaled.sum()) + 1)
    counts[0] = 1.0
    reach = 0
    for step in scaled:
        width = int(step)
        counts[width : reach + width + 1] += counts[0 : reach + 1]
        reach += width
    return counts


def _choose_sum_counts(scaled: np.ndarray, m: int) -> np.ndarray:
    """How many ``m``-element subsets of ``scaled`` add up to each total.

    R's ``dpermdist2``, before the division by ``choose(m + n, m)``. The second
    dimension is the reason this one is not a plain polynomial product: which
    ranks fell to the first sample is a choice of exactly ``m`` of them.
    """
    total = int(scaled.sum())
    # by_size[j, s]: ways to pick j ranks summing to s. Only the last row is
    # wanted, but every row feeds the next observation's update.
    by_size = np.zeros((m + 1, total + 1))
    by_size[0, 0] = 1.0
    for step in scaled:
        width = int(step)
        # Descending in j so each observation is used at most once.
        for size in range(m, 0, -1):
            by_size[size, width:] += by_size[size - 1, : total + 1 - width]
    counts: np.ndarray = by_size[m]
    return counts


def _signrank_support(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact null distribution of ``V`` for a given rank vector.

    Returns:
        The values ``V`` can take, ascending, and their probabilities.
    """
    scale = _rank_scale(z)
    counts = _subset_sum_counts(np.rint(z * scale))
    values = np.arange(counts.size) / scale
    return values, counts / 2.0**z.size


def _wilcox_support(z: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """The exact null distribution of ``W`` for a given rank vector.

    ``W`` is the first sample's rank sum less its own minimum, so the support is
    the rank-sum support shifted by ``m * (m + 1) / 2``.
    """
    scale = _rank_scale(z)
    counts = _choose_sum_counts(np.rint(z * scale), m)
    values = np.arange(counts.size) / scale - m * (m + 1) / 2
    return values, counts / math.comb(z.size, m)


def _cdf(values: np.ndarray, probs: np.ndarray, q: float, lower: bool) -> float:
    """``P(stat <= q)`` off a tabulated distribution, or its complement."""
    below = float(probs[values < q + _SUPPORT_TOL].sum())
    return below if lower else 1.0 - below


def psignrank(q: float, n: int, lower: bool = True) -> float:
    """Distribution function of the Wilcoxon signed rank statistic.

    Port of ``stats::psignrank()``, the untied case: the ranks are ``1..n``.
    """
    values, probs = _signrank_support(np.arange(1, n + 1, dtype=float))
    return _cdf(values, probs, q, lower)


def qsignrank(p: float, n: int) -> float:
    """Quantile function of the Wilcoxon signed rank statistic.

    Port of ``stats::qsignrank()``: the smallest ``q`` whose lower tail reaches
    ``p``.
    """
    values, probs = _signrank_support(np.arange(1, n + 1, dtype=float))
    reached = np.cumsum(probs) >= p - _SUPPORT_TOL
    hit = np.flatnonzero(reached)
    return float(values[hit[0]]) if hit.size else float(values[-1])


def pwilcox(q: float, m: int, n: int, lower: bool = True) -> float:
    """Distribution function of the Wilcoxon rank sum statistic.

    Port of ``stats::pwilcox()``, the untied case.
    """
    values, probs = _wilcox_support(np.arange(1, m + n + 1, dtype=float), m)
    return _cdf(values, probs, q, lower)


def qwilcox(p: float, m: int, n: int) -> float:
    """Quantile function of the Wilcoxon rank sum statistic.

    Port of ``stats::qwilcox()``.
    """
    values, probs = _wilcox_support(np.arange(1, m + n + 1, dtype=float), m)
    reached = np.cumsum(probs) >= p - _SUPPORT_TOL
    hit = np.flatnonzero(reached)
    return float(values[hit[0]]) if hit.size else float(values[-1])


# --------------------------------------------------------------------------- #
# Root finding
# --------------------------------------------------------------------------- #


def _zeroin(
    f: Callable[[float], float],
    ax: float,
    bx: float,
    fa: float,
    fb: float,
    tol: float = TOL_ROOT,
    max_iter: int = 1000,
) -> float:
    """Brent's method as R's ``uniroot()`` runs it.

    A transcription of ``C_zeroin2`` rather than a call to
    ``scipy.optimize.brentq``, because the two are not the same search. Both are
    Brent's method, but they keep Brent's two acceptance conditions in different
    places: ``brentq`` compares the trial step against the step of the last
    *sign change*, and Brent - which is what R follows - compares it against the
    last step taken. On a smooth function the difference washes out in the last
    digits. On a step function, which is what a rank statistic is, the two take
    different steps and stop at different iterates, and an interval located to
    ``tol = 1e-4`` is that iterate.

    So the interpolation is accepted only when it stays inside the bracket,
    ``2p < 3mq - |tol q|``, *and* at least halves the previous step,
    ``p < |e q / 2|``. The second is the one ``brentq`` spells differently.

    The endpoint values are taken as arguments rather than recomputed, since the
    caller has them already and each costs a full re-ranking of the sample.
    """
    eps = np.finfo(float).eps
    a, b, c = ax, bx, ax
    fc = fa

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    for _ in range(max_iter + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            # Keep b as the best approximation so far.
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * eps * abs(b) + tol / 2
        new_step = (c - b) / 2

        if abs(new_step) <= tol_act or fb == 0.0:
            return b

        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                # Linear interpolation: only two distinct points are known.
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            # p is carried positive and the sign moved into q, so the two
            # acceptance tests can be written without absolute values.
            if p > 0.0:
                q = -q
            else:
                p = -p
            inside = p < (0.75 * cb * q - abs(tol_act * q) / 2)
            halves = p < abs(prev_step * q / 2)
            if inside and halves and p > 0.0:
                new_step = p / q

        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act

        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa

    return b


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #


def _ranks(values: np.ndarray) -> np.ndarray:
    """R's ``rank()``: average ranks within a tie."""
    return np.asarray(stats.rankdata(values, method="average"), dtype=float)


def _tie_term(ranks: np.ndarray) -> float:
    """``sum(NTIES ** 3 - NTIES)`` over the tie widths of a rank vector."""
    _, widths = np.unique(ranks, return_counts=True)
    sizes = widths.astype(float)
    return float(np.sum(sizes**3 - sizes))


def _correction(z: float, alternative: str, correct: bool) -> float:
    """The continuity correction, signed the way the alternative asks."""
    if not correct:
        return 0.0
    if alternative == "two.sided":
        return float(np.sign(z)) * 0.5
    return 0.5 if alternative == "greater" else -0.5


def _normal_pval(z: float, alternative: str) -> float:
    """Tail probability of a standardised statistic."""
    if alternative == "less":
        return float(stats.norm.cdf(z))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    lower = float(stats.norm.cdf(z))
    return 2 * min(lower, 1 - lower)


def _walsh_averages(x: np.ndarray) -> np.ndarray:
    """The pairwise means ``(x[i] + x[j]) / 2`` with ``i <= j``, ascending.

    What the signed rank interval and the pseudomedian are read off. R builds
    the whole outer sum and keeps the upper triangle including the diagonal.
    """
    total = x[:, None] + x[None, :]
    keep = np.triu(np.ones(total.shape, dtype=bool))
    return np.sort(total[keep] / 2.0)


def _step_endpoints(
    diffs: np.ndarray,
    tail: Callable[[float, bool], float],
    alpha: float,
    alternative: str,
) -> tuple[float, float, float]:
    """Invert a tied exact distribution by walking the candidate endpoints.

    The tied counterpart of reading an order statistic off ``diffs``: with
    half-integer ranks there is no quantile to look the endpoint up at, so R
    tests the midpoint between each neighbouring pair of candidates and takes the
    first one whose tail clears alpha.

    Returns:
        The lower endpoint, the upper endpoint and the alpha actually achieved.
    """
    n_d = diffs.size

    def lower(level: float) -> tuple[float, float]:
        level = level + _ALPHA_TOL
        reached = 0.0
        if tail(float(diffs[0]) - 1, False) > level:
            return -math.inf, reached
        for k in range(n_d - 1):
            p = tail(float(diffs[k] + diffs[k + 1]) / 2, False)
            if p > level:
                return float(diffs[k]), reached
            reached = p
        return float(diffs[-1]), reached

    def upper(level: float) -> tuple[float, float]:
        level = level + _ALPHA_TOL
        reached = 0.0
        if tail(float(diffs[-1]) + 1, True) > level:
            return math.inf, reached
        for k in range(n_d - 2, -1, -1):
            p = tail(float(diffs[k] + diffs[k + 1]) / 2, True)
            if p > level:
                return float(diffs[k + 1]), reached
            reached = p
        return float(diffs[0]), reached

    if alternative == "two.sided":
        low, low_p = lower(alpha / 2)
        high, high_p = upper(alpha / 2)
        return low, high, low_p + high_p
    if alternative == "greater":
        low, low_p = lower(alpha)
        return low, math.inf, low_p
    high, high_p = upper(alpha)
    return -math.inf, high, high_p


def _order_endpoints(
    diffs: np.ndarray,
    cdf: Callable[[float], float],
    quantile: Callable[[float], float],
    n_support: float,
    alpha: float,
    alternative: str,
) -> tuple[float, float]:
    """Read the interval off an order statistic of ``diffs``, as R does untied.

    Args:
        diffs: The candidate endpoints, ascending.
        cdf: Lower tail of the exact null distribution.
        quantile: Its quantile function.
        n_support: The largest value the statistic can take, which the upper
            endpoint is counted back from.
        alpha: One minus the confidence level.
        alternative: Which side is being bounded.
    """
    level = alpha / 2 if alternative == "two.sided" else alpha
    qu = quantile(level)
    if cdf(qu) <= level + _ALPHA_TOL:
        qu = qu + 1
    if qu == 0:
        return -math.inf, math.inf

    at = int(math.trunc(qu))
    ql = int(n_support - at)
    if alternative == "greater":
        return float(diffs[at - 1]), math.inf
    if alternative == "less":
        return -math.inf, float(diffs[ql])
    return float(diffs[at - 1]), float(diffs[ql])


# --------------------------------------------------------------------------- #
# The two tests
# --------------------------------------------------------------------------- #


def signed_rank(
    x: Any,
    mu: float = 0.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
    exact: bool | None = None,
) -> dict[str, float]:
    """Wilcoxon signed rank test with a location estimate and interval.

    Port of the one-sample branch of ``stats::wilcox.test()``. Used both for a
    paired two-group comparison, where ``x`` holds the within-pair differences,
    and for a one-sample comparison against ``mu``.

    Args:
        x: The sample, missing values already removed.
        mu: The location the null puts it at.
        alternative: ``"two.sided"``, ``"less"`` or ``"greater"``.
        conf_level: Confidence level of the reported interval.
        correct: Whether the normal approximation gets a continuity correction.
            Ignored on the exact path, which needs none.
        exact: Force the exact or the approximate distribution. ``None`` decides
            it as R does, by sample size.

    Returns:
        ``v_stat``, ``hl_shift`` (the pseudomedian), ``pval``, ``lower_conf`` and
        ``upper_conf``.
    """
    check_alternative(alternative)
    sample = as_sample(x, "x")
    if sample.size < 1:
        raise SaValueError("not enough (non-missing) observations.")

    shifted = sample - mu
    n = shifted.size
    use_exact = (n < EXACT_MAX_N) if exact is None else bool(exact)

    if use_exact:
        zero = shifted == 0
        ranks = _ranks(np.abs(shifted))
        tied = np.unique(ranks).size != ranks.size
        v_stat = float(ranks[shifted > 0].sum())
        # R conditions on the observed ranks once they are not 1..n, which is
        # what keeps the exact test available in the presence of ties.
        held = ranks[~zero] if (tied or bool(zero.any())) else None

        if held is None:
            centre = n * (n + 1) / 4
            if v_stat > centre:
                tail_p = psignrank(v_stat - 0.25, n, lower=False)
            else:
                tail_p = psignrank(v_stat, n, lower=True)
            pval = (
                min(2 * tail_p, 1.0)
                if alternative == "two.sided"
                else (
                    psignrank(v_stat - 0.25, n, lower=False)
                    if alternative == "greater"
                    else psignrank(v_stat, n, lower=True)
                )
            )
        else:
            values, probs = _signrank_support(held)
            centre = float(held.sum()) / 2
            if alternative == "two.sided":
                if v_stat > centre:
                    tail_p = _cdf(values, probs, v_stat - 0.25, lower=False)
                else:
                    tail_p = _cdf(values, probs, v_stat, lower=True)
                pval = min(2 * tail_p, 1.0)
            elif alternative == "greater":
                pval = _cdf(values, probs, v_stat - 0.25, lower=False)
            else:
                pval = _cdf(values, probs, v_stat, lower=True)

        walsh = _walsh_averages(shifted)
        alpha = 1 - conf_level
        if held is None:
            low, high = _order_endpoints(
                walsh,
                lambda q: psignrank(q, n),
                lambda p: qsignrank(p, n),
                n * (n + 1) / 2,
                alpha,
                alternative,
            )
        else:

            def tail(at: float, lower: bool) -> float:
                moved = shifted - at
                ranks_at = _ranks(np.abs(moved))
                v_at = float(ranks_at[moved > 0].sum())
                values_at, probs_at = _signrank_support(ranks_at)
                return _cdf(values_at, probs_at, v_at, lower)

            low, high, _ = _step_endpoints(walsh, tail, alpha, alternative)

        return {
            "v_stat": v_stat,
            "hl_shift": float(np.median(walsh)) + mu,
            "pval": float(pval),
            "lower_conf": low + mu,
            "upper_conf": high + mu,
        }

    # Zeros carry no sign, so they leave the sample and the sample size with it.
    nonzero = shifted[shifted != 0]
    n_used = nonzero.size
    ranks = _ranks(np.abs(nonzero))
    v_stat = float(ranks[nonzero > 0].sum())
    expected = n_used * (n_used + 1) / 4
    sigma = math.sqrt(n_used * (n_used + 1) * (2 * n_used + 1) / 24 - _tie_term(ranks) / 48)
    centred = v_stat - expected
    z = (centred - _correction(centred, alternative, correct)) / sigma
    pval = _normal_pval(z, alternative)

    low, high, estimate = _signed_rank_interval(shifted, alternative, conf_level, correct)
    return {
        "v_stat": v_stat,
        "hl_shift": estimate + mu,
        "pval": float(pval),
        "lower_conf": low + mu,
        "upper_conf": high + mu,
    }


def _signed_rank_statistic(
    x: np.ndarray,
    at: float,
    alternative: str,
    correct: bool,
) -> float:
    """The standardised signed rank statistic of ``x`` shifted to ``at``.

    R's local ``W()`` inside the asymptotic interval. Recomputing the ranks at
    every candidate is what makes the function a step function of ``at``, which
    is the function being inverted.
    """
    moved = x - at
    moved = moved[moved != 0]
    n = int(moved.size)
    ranks = _ranks(np.abs(moved))
    centred = float(ranks[moved > 0].sum()) - n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - _tie_term(ranks) / 48)
    if sigma == 0:
        warn("cannot compute confidence interval when all observations are zero or tied")
    return float(centred - _correction(centred, alternative, correct)) / sigma


def _signed_rank_interval(
    shifted: np.ndarray,
    alternative: str,
    conf_level: float,
    correct: bool,
) -> tuple[float, float, float]:
    """The asymptotic interval and pseudomedian of a signed rank test."""
    alpha = 1 - conf_level
    lo, hi = float(shifted.min()), float(shifted.max())

    def at(value: float) -> float:
        return _signed_rank_statistic(shifted, value, alternative, correct)

    w_lo = at(lo)
    w_hi = at(hi) if math.isfinite(w_lo) else math.nan
    if not math.isfinite(w_hi):
        # Every observation tied, so the statistic never moves and there is no
        # interval to report. R answers with an empty one at conf.level 0.
        low = -math.inf if alternative == "less" else math.nan
        high = math.inf if alternative == "greater" else math.nan
        return low, high, (lo + hi) / 2

    def root(zq: float) -> float:
        return _zeroin(lambda value: at(value) - zq, lo, hi, w_lo - zq, w_hi - zq)

    alpha = _widen_alpha(alpha, w_lo, w_hi, alternative, conf_level)
    if alpha >= 1:
        centre = float(np.median(shifted))
        low = -math.inf if alternative == "less" else centre
        high = math.inf if alternative == "greater" else centre
    elif alternative == "two.sided":
        low = root(float(stats.norm.isf(alpha / 2)))
        high = root(float(stats.norm.ppf(alpha / 2)))
    elif alternative == "greater":
        low, high = root(float(stats.norm.isf(alpha))), math.inf
    else:
        low, high = -math.inf, root(float(stats.norm.ppf(alpha)))

    # R turns the correction off before reading the estimate, so the point
    # estimate is the root of the uncorrected statistic whichever tail was asked.
    def plain(value: float) -> float:
        return _signed_rank_statistic(shifted, value, alternative, False)

    estimate = _zeroin(plain, lo, hi, plain(lo), plain(hi))
    return low, high, estimate


def _widen_alpha(
    alpha: float,
    w_lo: float,
    w_hi: float,
    alternative: str,
    conf_level: float,
) -> float:
    """Widen alpha until the requested quantile lies inside the bracket.

    R's ``repeat`` loop: a sample too small for the level asked for cannot
    bracket the root, and doubling alpha is how R reports that rather than
    failing.
    """
    while True:
        if alternative == "two.sided":
            outside = w_lo - float(stats.norm.isf(alpha / 2)) < 0 or (
                w_hi - float(stats.norm.ppf(alpha / 2)) > 0
            )
        elif alternative == "greater":
            outside = w_lo - float(stats.norm.isf(alpha)) < 0
        else:
            outside = w_hi - float(stats.norm.ppf(alpha / 2)) > 0
        if not outside:
            break
        alpha = alpha * 2

    if alpha >= 1 or 1 - conf_level < alpha * 0.75:
        warn("requested conf.level not achievable")
    return alpha


def rank_sum(
    x: Any,
    y: Any,
    mu: float = 0.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
    exact: bool | None = None,
) -> dict[str, float]:
    """Wilcoxon rank sum test with a location shift estimate and interval.

    Port of the two-sample branch of ``stats::wilcox.test()``, the Mann-Whitney
    U test. ``x`` is the sample every reported quantity is read in the direction
    of, so ``hl_shift`` is above zero when ``x`` is the larger sample.

    Args:
        x: First sample, missing values already removed.
        y: Second sample, the one subtracted.
        mu: The location shift the null puts between them.
        alternative: ``"two.sided"``, ``"less"`` or ``"greater"``.
        conf_level: Confidence level of the reported interval.
        correct: Whether the normal approximation gets a continuity correction.
        exact: Force the exact or the approximate distribution. ``None`` decides
            it as R does, by sample size.

    Returns:
        ``w_stat``, ``hl_shift``, ``pval``, ``lower_conf`` and ``upper_conf``.
    """
    check_alternative(alternative)
    first = as_sample(x, "x")
    second = as_sample(y, "y")
    if first.size < 1:
        raise SaValueError("not enough (non-missing) observations in the first sample.")
    if second.size < 1:
        raise SaValueError("not enough observations in the second sample.")

    m, n = first.size, second.size
    use_exact = (m < EXACT_MAX_N and n < EXACT_MAX_N) if exact is None else bool(exact)

    pooled = np.concatenate([first - mu, second])
    ranks = _ranks(pooled)
    w_stat = float(ranks[:m].sum()) - m * (m + 1) / 2
    tied = np.unique(ranks).size != ranks.size

    if use_exact:
        held = ranks if tied else None
        if held is None:
            if alternative == "two.sided":
                pval = min(
                    2 * pwilcox(w_stat, m, n),
                    2 * pwilcox(w_stat - 0.25, m, n, lower=False),
                    1.0,
                )
            elif alternative == "greater":
                pval = pwilcox(w_stat - 0.25, m, n, lower=False)
            else:
                pval = pwilcox(w_stat, m, n)
        else:
            values, probs = _wilcox_support(held, m)
            if alternative == "two.sided":
                pval = min(
                    2 * _cdf(values, probs, w_stat, True),
                    2 * _cdf(values, probs, w_stat - 0.25, False),
                    1.0,
                )
            elif alternative == "greater":
                pval = _cdf(values, probs, w_stat - 0.25, False)
            else:
                pval = _cdf(values, probs, w_stat, True)

        diffs = np.sort((first[:, None] - second[None, :]).reshape(-1))
        alpha = 1 - conf_level
        if held is None:
            low, high = _order_endpoints(
                diffs,
                lambda q: pwilcox(q, m, n),
                lambda p: qwilcox(p, m, n),
                m * n,
                alpha,
                alternative,
            )
        else:

            def tail(at: float, lower: bool) -> float:
                ranks_at = _ranks(np.concatenate([first - at, second]))
                w_at = float(ranks_at[:m].sum()) - m * (m + 1) / 2
                values_at, probs_at = _wilcox_support(ranks_at, m)
                if lower:
                    return _cdf(values_at, probs_at, w_at, True)
                return _cdf(values_at, probs_at, w_at - 0.25, False)

            low, high, _ = _step_endpoints(diffs, tail, alpha, alternative)

        return {
            "w_stat": w_stat,
            "hl_shift": float(np.median(diffs)),
            "pval": float(pval),
            "lower_conf": low,
            "upper_conf": high,
        }

    expected = m * n / 2
    sigma = math.sqrt((m * n / 12) * ((m + n + 1) - _tie_term(ranks) / ((m + n) * (m + n - 1))))
    centred = w_stat - expected
    z = (centred - _correction(centred, alternative, correct)) / sigma
    pval = _normal_pval(z, alternative)

    low, high, estimate = _rank_sum_interval(first, second, alternative, conf_level, correct)
    return {
        "w_stat": w_stat,
        "hl_shift": estimate,
        "pval": float(pval),
        "lower_conf": low,
        "upper_conf": high,
    }


def _rank_sum_statistic(
    x: np.ndarray,
    y: np.ndarray,
    at: float,
    alternative: str,
    correct: bool,
) -> float:
    """The standardised rank sum statistic with ``x`` shifted by ``at``."""
    m, n = x.size, y.size
    ranks = _ranks(np.concatenate([x - at, y]))
    centred = float(ranks[:m].sum()) - m * (m + 1) / 2 - m * n / 2
    sigma = math.sqrt((m * n / 12) * ((m + n + 1) - _tie_term(ranks) / ((m + n) * (m + n - 1))))
    if sigma == 0:
        warn("cannot compute confidence interval when all observations are tied")
    return (centred - _correction(centred, alternative, correct)) / sigma


def _rank_sum_interval(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
    conf_level: float,
    correct: bool,
) -> tuple[float, float, float]:
    """The asymptotic interval and location shift of a rank sum test."""
    alpha = 1 - conf_level
    lo = float(x.min()) - float(y.max())
    hi = float(x.max()) - float(y.min())

    def at(value: float) -> float:
        return _rank_sum_statistic(x, y, value, alternative, correct)

    w_lo, w_hi = at(lo), at(hi)

    def root(zq: float) -> float:
        # An endpoint outside the bracket is the bracket, which is what R
        # returns rather than letting the root finder refuse the interval.
        f_lower = w_lo - zq
        if f_lower <= 0:
            return lo
        f_upper = w_hi - zq
        if f_upper >= 0:
            return hi
        return _zeroin(lambda value: at(value) - zq, lo, hi, f_lower, f_upper)

    if alternative == "two.sided":
        low = root(float(stats.norm.isf(alpha / 2)))
        high = root(float(stats.norm.ppf(alpha / 2)))
    elif alternative == "greater":
        low, high = root(float(stats.norm.isf(alpha))), math.inf
    else:
        low, high = -math.inf, root(float(stats.norm.ppf(alpha)))

    def plain(value: float) -> float:
        return _rank_sum_statistic(x, y, value, alternative, False)

    estimate = _zeroin(plain, lo, hi, plain(lo), plain(hi))
    return low, high, estimate
