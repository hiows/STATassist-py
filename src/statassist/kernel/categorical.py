"""Contingency table kernels: plain counts in, one named row out.

Port of ``R/kernel_categorical.R``, written to the same rule as
:mod:`statassist.kernel.anova`: no fitted object kept anywhere and nothing said to
the user. That last part is a rule and not an accident. A kernel is a function of
its arguments, so anything it would want to tell the caller - that a correction
was applied, that a branch was taken - is a number it can return instead, and the
scenario function is the one place that decides what is worth reporting. Every
fact this file could have printed is in the row it hands back.

Two things differ from the numeric kernels. There is one table rather than one
sample per feature, so a kernel here is called once and an unusable table is an
error the caller sees rather than a missing row in a long scan. And where R
suppresses its engines' warnings about the expected counts, nothing is suppressed
here because nothing warns: the statistic is written out and the same fact reaches
the caller as the smallest expected count.

None of these tests reports an interval for the association itself. A test of
independence says the two variables are not independent; it does not say by how
much, and the measures that do say it are the business of
:func:`assoc_measures`. The ``lower_conf`` and ``upper_conf`` columns are
therefore present and missing except where the test itself defines an interval,
which only Fisher's exact test on a 2 x 2 table does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from scipy.stats.contingency import odds_ratio as _conditional_odds_ratio

from ..core.contingency import DISCORDANT_PAIR_MIN, expected_independence, finite_or_na
from ..core.contracts import association_columns
from ..core.errors import SaValueError
from ..core.random import SaRandom
from ..core.tables import stat_row
from ._fisher import fisher_exact_rxc
from ._shared import as_matrix

__all__ = [
    "ASSOC_COLUMNS",
    "MCNEMAR_EXACT_MAX_DISCORDANT",
    "assoc_measures",
    "assoc_measures_paired",
    "assoc_measures_repeated",
    "assoc_row",
    "chisq",
    "cochran_q",
    "fisher",
    "has_zero_cell",
    "mcnemar",
    "odds_ratio",
    "phi",
]

#: Below this many discordant pairs, McNemar's test is taken exactly.
#:
#: The rule ``Configuration/registry/assumptions.yaml`` records as
#: ``discordant_pair_count``. R writes the number out in two places, here and in
#: the diagnostic that reports the same rule; this side states it once, in
#: :data:`~statassist.core.contingency.DISCORDANT_PAIR_MIN`, so that the branch
#: this test takes and the check reported beside it cannot come apart.
MCNEMAR_EXACT_MAX_DISCORDANT = DISCORDANT_PAIR_MIN

#: Columns of an association table, in order.
#:
#: The contract itself is :func:`~statassist.core.contracts.association_columns`,
#: which is what the result object of the scenario is held to. This is the same
#: list as a tuple, so a kernel and the table it feeds cannot come to disagree
#: about the order.
ASSOC_COLUMNS: tuple[str, ...] = tuple(association_columns())

#: How much slack R leaves when it counts a Monte Carlo table as "at least as
#: extreme". ``chisq.test()`` writes it ``1 - 64 * .Machine$double.eps``, so that a
#: replicate which equals the observed statistic by construction is not dropped
#: by a rounding.
_ALMOST_ONE = 1 - 64 * float(np.finfo(float).eps)


def _as_counts(counts: Any) -> np.ndarray:
    """Read a table as a two-dimensional float array."""
    if isinstance(counts, pd.DataFrame):
        table = counts.to_numpy(dtype=float)
    else:
        table = np.asarray(counts, dtype=float)
    if table.ndim != 2 or min(table.shape) < 2:
        raise SaValueError("a contingency table must be two-dimensional and at least 2 x 2.")
    return table


def _is_square_2x2(table: np.ndarray) -> bool:
    """Whether R's ``identical(dim(counts), c(2L, 2L))`` would hold."""
    return table.shape == (2, 2)


def _resample_fixed_margins(
    table: np.ndarray,
    n_resamples: int,
    rng: SaRandom,
) -> np.ndarray:
    """Draw tables with the observed margins, uniformly over the conditional law.

    R draws these with Patefield's algorithm. Permuting one variable's labels
    across the observations while holding the other's fixed reaches the same
    distribution by a shorter route: every assignment of labels is equally likely,
    and the number of assignments producing a given table is what the multiple
    hypergeometric density counts. So the draws are from the same law, which is
    all the Monte Carlo p-value rests on - the individual tables cannot match R's
    in any case, since the two languages' generators differ.

    Returns:
        One row per replicate, each the table flattened row-major.
    """
    integer = np.rint(table).astype(np.int64)
    n_rows, n_cols = integer.shape
    flat = integer.reshape(-1)
    row_of = np.repeat(np.arange(n_rows).repeat(n_cols), flat)
    col_of = np.repeat(np.tile(np.arange(n_cols), n_rows), flat)

    out = np.empty((n_resamples, n_rows * n_cols), dtype=float)
    for replicate in range(n_resamples):
        shuffled = rng.rng.permutation(row_of)
        out[replicate] = np.bincount(shuffled * n_cols + col_of, minlength=n_rows * n_cols).astype(
            float
        )
    return out


def chisq(
    counts: Any,
    correct: bool = True,
    simulate_p_value: bool = False,
    n_resamples: int = 9999,
    seed: Any = None,
) -> dict[str, float]:
    """Pearson's chi-square test of independence.

    Port of ``sa_chisq()``.

    Args:
        counts: Two-dimensional table of counts.
        correct: Whether to apply Yates' continuity correction. R's
            ``chisq.test()`` applies it to **2 x 2 tables only**, whatever it is
            told, so it changes nothing on a larger table.
            :func:`scipy.stats.chi2_contingency` keys the same correction on
            having one degree of freedom instead, which is not the same rule: a
            2 x N table has more than one and a 2 x 2 is not the only shape with
            one. The statistic is written out here so that the rule is R's.
        simulate_p_value: Monte Carlo p-value for a sparse table, drawn with the
            margins held fixed.
        n_resamples: How many tables to draw for it.
        seed: Seeds the draw, and is ignored unless ``simulate_p_value``. R has one
            global stream and no argument here; this port seeds per call, so the
            Monte Carlo branch is reproducible within Python without reproducing
            R's numbers.

    Returns:
        ``n_used``, ``statistic``, ``df``, ``pval``, ``lower_conf``,
        ``upper_conf``. A simulated p-value is not referred to a chi-square
        distribution, so there are no degrees of freedom to report rather than
        zero of them.

    Raises:
        SaValueError: If a row or column holds no observation.

    References:
        Pearson, K. (1900). On the criterion that a given system of deviations from
        the probable in the case of a correlated system of variables is such that
        it can be reasonably supposed to have arisen from random sampling.
        *Philosophical Magazine*, 50(302), 157-175.
    """
    table = _as_counts(counts)
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        raise SaValueError(
            "a row or column holding no observation leaves an expected count of zero, "
            "which the chi-square statistic divides by. Drop the level from "
            "`category_lv`."
        )

    expected = expected_independence(table)
    deviation = np.abs(table - expected)
    n_used = float(table.sum())

    if simulate_p_value:
        # R computes the statistic without the correction here and sums it from
        # the largest term down, so that the observed value and a replicate that
        # equals it are added the same way.
        statistic = float(np.sum(np.sort(((table - expected) ** 2 / expected).reshape(-1))[::-1]))
        draws = _resample_fixed_margins(table, n_resamples, SaRandom(seed))
        simulated = np.sum((draws - expected.reshape(-1)) ** 2 / expected.reshape(-1), axis=1)
        at_least = np.count_nonzero(simulated >= _ALMOST_ONE * statistic)
        pval = float((1 + at_least) / (n_resamples + 1))
        return stat_row(
            n_used=n_used,
            statistic=statistic,
            df=np.nan,
            pval=pval,
            lower_conf=np.nan,
            upper_conf=np.nan,
        )

    # R's `YATES <- min(0.5, abs(x - E))` takes the minimum over the whole matrix
    # as well as against a half, so a cell already within half an observation of
    # its expectation caps the correction below 0.5.
    yates = min(0.5, float(deviation.min())) if (correct and _is_square_2x2(table)) else 0.0
    statistic = float(np.sum((deviation - yates) ** 2 / expected))
    df = float((table.shape[0] - 1) * (table.shape[1] - 1))
    return stat_row(
        n_used=n_used,
        statistic=statistic,
        df=df,
        pval=float(stats.chi2.sf(statistic, df)),
        lower_conf=np.nan,
        upper_conf=np.nan,
    )


def fisher(
    counts: Any,
    conf_level: float = 0.95,
    simulate_p_value: bool = False,
    n_resamples: int = 9999,
    seed: Any = None,
) -> dict[str, float]:
    """Fisher's exact test on a contingency table.

    Port of ``sa_fisher()``. Conditions on both margins and reads the p-value off
    the hypergeometric distribution, so it needs no expected count to be large.
    There is no statistic referred to a null distribution, which is why
    ``statistic`` and ``df`` are missing rather than zero.

    On a 2 x 2 table the test defines an odds ratio of its own, the conditional
    maximum likelihood estimate, and an interval for it. That is reported as
    ``odds_ratio_cond`` and is **not** the sample odds ratio :func:`odds_ratio`
    carries: the conditional estimate is pulled towards 1 and the two differ most
    where the counts are smallest, which is the situation this test exists for.

    The enumeration has a size limit, and reaching it is not an error in the data.
    There are more tables with the observed margins than can be walked, so the
    p-value comes back missing with ``enumerated = 0`` rather than as a raised
    condition, because the chi-square test standing beside it in the same result
    was computed and losing it would be the more expensive failure. The caller is
    what says so and points at ``simulate_p_value=True``. See
    :data:`._fisher.FISHER_TABLE_LIMIT` for where this port's limit sits relative
    to R's.

    Args:
        counts: Two-dimensional table of counts.
        conf_level: Confidence level for the conditional odds ratio interval.
        simulate_p_value: Monte Carlo variant for a large r x c table, where the
            exact enumeration is infeasible. Ignored on a 2 x 2 table, where the
            exact test always is feasible - which is R's behaviour too.
        n_resamples: How many tables to draw for it.
        seed: Seeds the draw. See :func:`chisq`.

    Returns:
        ``n_used``, ``statistic``, ``df``, ``pval``, ``lower_conf``,
        ``upper_conf``, ``odds_ratio_cond``, ``enumerated``. ``enumerated`` is
        whether the test could be computed at all, which the caller reports and
        does not carry into the test table.

    References:
        Fisher, R. A. (1935). The logic of inductive inference. *Journal of the
        Royal Statistical Society*, 98(1), 39-82.
    """
    table = _as_counts(counts)
    n_used = float(table.sum())

    if _is_square_2x2(table):
        integer = np.rint(table).astype(np.int64)
        pval = float(stats.fisher_exact(integer).pvalue)
        conditional = _conditional_odds_ratio(integer, kind="conditional")
        interval = conditional.confidence_interval(confidence_level=conf_level)
        return stat_row(
            n_used=n_used,
            statistic=np.nan,
            df=np.nan,
            pval=pval,
            lower_conf=float(interval.low),
            upper_conf=float(interval.high),
            odds_ratio_cond=float(conditional.statistic),
            enumerated=1.0,
        )

    if simulate_p_value:
        # R compares the negated sum of log factorials, which is the log of a
        # table's probability up to the constant every table with these margins
        # shares. A replicate is at least as extreme when its probability is no
        # greater than the observed one's.
        log_weight = -float(np.sum(gammaln(table.reshape(-1) + 1)))
        draws = _resample_fixed_margins(table, n_resamples, SaRandom(seed))
        simulated = -np.sum(gammaln(draws + 1), axis=1)
        pval = float(
            (1 + np.count_nonzero(simulated <= log_weight / _ALMOST_ONE)) / (n_resamples + 1)
        )
        return stat_row(
            n_used=n_used,
            statistic=np.nan,
            df=np.nan,
            pval=pval,
            lower_conf=np.nan,
            upper_conf=np.nan,
            odds_ratio_cond=np.nan,
            enumerated=1.0,
        )

    pval = fisher_exact_rxc(np.rint(table).astype(np.int64))
    return stat_row(
        n_used=n_used,
        statistic=np.nan,
        df=np.nan,
        pval=pval,
        lower_conf=np.nan,
        upper_conf=np.nan,
        odds_ratio_cond=np.nan,
        enumerated=float(np.isfinite(pval)),
    )


def mcnemar(
    counts: Any,
    correct: bool = True,
    exact: bool | None = None,
) -> dict[str, float]:
    """McNemar's test of symmetry.

    Port of ``sa_mcnemar()``. Reads only the two discordant cells. A pair that
    answered the same way under both conditions carries no information about which
    condition raises the response, so the concordant cells do not enter the
    statistic and the number of pairs does not either, beyond fixing how many
    discordant ones there could have been.

    The uncorrected statistic is ``(b - c)^2 / (b + c)``, which is exactly the sum
    of squared Pearson residuals of the table against the symmetry expectation
    :func:`~statassist.core.expected_symmetry` builds. The cell table of the
    result and the p-value here are the same arithmetic read two ways.

    Args:
        counts: A 2 x 2 table crossing the two conditions.
        correct: Whether to apply the continuity correction to the chi-square
            approximation. Ignored by the exact branch, which needs none.
        exact: ``True`` for the exact binomial test on the discordant pairs,
            ``False`` for the chi-square approximation, or ``None`` to take the
            exact test when there are fewer than
            :data:`MCNEMAR_EXACT_MAX_DISCORDANT` discordant pairs. ``None`` is
            R's ``NULL`` here rather than a sentinel, because "decide from the
            data" is what it means in both languages.

    Returns:
        ``n_used``, ``n_discordant``, ``exact_used``, ``statistic``, ``df``,
        ``pval``, ``lower_conf``, ``upper_conf``. ``exact_used`` is how the branch
        taken under ``exact=None`` reaches the caller, which records it as a
        parameter; it is a setting rather than a finding, so it does not travel on
        into the test table.

    Raises:
        SaValueError: If no pair is discordant.

    References:
        McNemar, Q. (1947). Note on the sampling error of the difference between
        correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
    """
    table = _as_counts(counts)
    b = float(table[0, 1])
    c = float(table[1, 0])
    n_discordant = b + c

    if n_discordant == 0:
        raise SaValueError(
            "every pair answered the same way under both conditions, so there is no "
            "discordance for McNemar's test to be about."
        )

    use_exact = (n_discordant < MCNEMAR_EXACT_MAX_DISCORDANT) if exact is None else bool(exact)
    if use_exact:
        pval = float(stats.binomtest(int(round(b)), int(round(n_discordant)), 0.5).pvalue)
        statistic = np.nan
        df = np.nan
    else:
        adjust = 1.0 if correct else 0.0
        statistic = max(abs(b - c) - adjust, 0.0) ** 2 / n_discordant
        df = 1.0
        pval = float(stats.chi2.sf(statistic, 1))

    return stat_row(
        n_used=float(table.sum()),
        n_discordant=n_discordant,
        exact_used=float(use_exact),
        statistic=statistic,
        df=df,
        pval=pval,
        lower_conf=np.nan,
        upper_conf=np.nan,
    )


def cochran_q(responses: Any) -> dict[str, float]:
    """Cochran's Q test for three or more repeated binary conditions.

    Port of ``sa_cochran_q()``. The extension of McNemar's test past two
    conditions. Written out rather than taken from an engine because the statistic
    is also what Kendall's W is built from, and computing it twice from two code
    paths is how the two end up disagreeing.

    A subject who answered the same way under every condition contributes nothing
    to the numerator, which is the same fact that makes the concordant cells drop
    out of McNemar's test. Such subjects are kept in the denominator, where the
    formula puts them.

    Args:
        responses: Subjects-by-conditions matrix of 0 and 1, no missing values.

    Returns:
        ``n_used``, ``n_conditions``, ``statistic``, ``df``, ``pval``,
        ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If no subject varies across the conditions.

    References:
        Cochran, W. G. (1950). The comparison of percentages in matched samples.
        *Biometrika*, 37(3-4), 256-266.
    """
    matrix = as_matrix(responses, "responses")
    n_subjects, k = matrix.shape
    col_n = matrix.sum(axis=0)
    row_n = matrix.sum(axis=1)
    total = float(matrix.sum())

    denominator = k * total - float(np.sum(row_n**2))
    if denominator <= 0:
        raise SaValueError(
            "every subject answered the same way under every condition, leaving "
            "Cochran's Q undefined: there is no within-subject variation for it to "
            "compare across conditions."
        )

    statistic = k * (k - 1) * float(np.sum((col_n - total / k) ** 2)) / denominator
    return stat_row(
        n_used=float(n_subjects),
        n_conditions=float(k),
        statistic=statistic,
        df=float(k - 1),
        pval=float(stats.chi2.sf(statistic, k - 1)),
        lower_conf=np.nan,
        upper_conf=np.nan,
    )


def assoc_row(
    measure: str,
    estimate: Any,
    lower_conf: Any = np.nan,
    upper_conf: Any = np.nan,
) -> pd.DataFrame:
    """One row of an association table.

    Port of ``sa_assoc_row()``. Every number passes through
    :func:`~statassist.core.finite_or_na`, so an unbounded estimate is reported as
    absent rather than as an infinity a plot would have to cope with.
    """
    values = finite_or_na([estimate, lower_conf, upper_conf])
    return pd.DataFrame(
        {
            "measure": pd.Series([measure], dtype=object),
            "estimate": [float(values[0])],
            "lower_conf": [float(values[1])],
            "upper_conf": [float(values[2])],
        }
    )


def _assoc_table(rows: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack association rows and renumber them."""
    return pd.concat(rows, ignore_index=True)[list(ASSOC_COLUMNS)]


def assoc_measures(counts: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Association measures for an independent r x c table.

    Port of ``sa_assoc_measures()``. An effect-size builder rather than a test
    kernel, so it returns a table rather than a row.

    Every measure is built from the **uncorrected** chi-square statistic, whichever
    way ``correct`` was set for the test. Yates' correction is about the tail
    probability of a discrete statistic referred to a continuous distribution; it
    is not about how far the table sits from independence, and letting it into the
    effect size would make the reported strength of an association depend on a
    choice made about its p-value.

    Args:
        counts: Two-dimensional table of counts.
        conf_level: Confidence level for the odds ratio interval.

    Returns:
        ``measure``, ``estimate``, ``lower_conf``, ``upper_conf``. A 2 x 2 table
        adds the signed phi coefficient and the odds ratio, which only exist
        there.

    References:
        Cramer, H. (1946). *Mathematical Methods of Statistics*. Princeton
        University Press.

        Agresti, A. (2002). *Categorical Data Analysis*, 2nd ed. Wiley.
    """
    table = _as_counts(counts)
    n = float(table.sum())
    expected = expected_independence(table)
    chi_sq = float(np.sum((table - expected) ** 2 / expected))
    min_df = min(table.shape) - 1

    rows = [
        assoc_row("cramers_v", np.sqrt(chi_sq / (n * min_df))),
        assoc_row("contingency_coefficient", np.sqrt(chi_sq / (chi_sq + n))),
    ]
    if _is_square_2x2(table):
        rows.append(assoc_row("phi_coefficient", phi(table)))
        rows.append(odds_ratio(table, conf_level))
    return _assoc_table(rows)


def assoc_measures_paired(counts: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Association measures for a matched 2 x 2 table.

    Port of ``sa_assoc_measures_paired()``. All three read the discordant cells,
    which is where the whole of a matched comparison lives. ``b`` is the pairs that
    answered the second level under the second condition and the first level under
    the first, and ``c`` the reverse.

    When every discordant pair moved the same way the odds ratio is unbounded, so
    its estimate and one of its limits are absent. The other two measures are
    bounded by construction and still carry the finding: ``cohens_g`` reaches its
    extreme of 0.5 and ``risk_difference_paired`` reports how large a share of the
    pairs moved.

    Args:
        counts: A 2 x 2 table crossing the two conditions.
        conf_level: Confidence level for the intervals.
    """
    table = _as_counts(counts)
    n = float(table.sum())
    b = float(table[0, 1])
    c = float(table[1, 0])
    n_discordant = b + c

    if n_discordant == 0:
        share = np.array([np.nan, np.nan, np.nan])
    else:
        # Clopper-Pearson on the split of the discordant pairs, which is exactly
        # the quantity the exact branch of McNemar's test is about, so the interval
        # and the p-value cannot disagree about whether 0.5 is plausible.
        test = stats.binomtest(int(round(b)), int(round(n_discordant)), 0.5)
        interval = test.proportion_ci(confidence_level=conf_level, method="exact")
        share = np.array([b / n_discordant, float(interval.low), float(interval.high)])

    # The paired odds ratio is a monotone transform of that same share, so its
    # interval is the transformed one rather than a second calculation.
    with np.errstate(divide="ignore", invalid="ignore"):
        paired_or = share / (1 - share)

    difference = (b - c) / n
    standard_error = np.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    z = float(stats.norm.ppf(1 - (1 - conf_level) / 2))

    return _assoc_table(
        [
            assoc_row("odds_ratio_paired", paired_or[0], paired_or[1], paired_or[2]),
            assoc_row(
                "risk_difference_paired",
                difference,
                max(difference - z * standard_error, -1.0),
                min(difference + z * standard_error, 1.0),
            ),
            assoc_row("cohens_g", share[0] - 0.5, share[1] - 0.5, share[2] - 0.5),
        ]
    )


def assoc_measures_repeated(q_stat: float, n_subjects: int, k: int) -> pd.DataFrame:
    """Association measure for three or more repeated binary conditions.

    Port of ``sa_assoc_measures_repeated()``. Kendall's W rescales Cochran's Q by
    the largest value it could have taken for this many subjects and conditions,
    which is what turns a statistic that grows with the sample into a measure that
    does not.

    Args:
        q_stat: Cochran's Q, as :func:`cochran_q` returns it.
        n_subjects: Subjects the statistic was computed on.
        k: Conditions it was computed over.
    """
    return _assoc_table([assoc_row("kendalls_w", q_stat / (n_subjects * (k - 1)))])


def phi(counts: Any) -> float:
    """The signed phi coefficient of a 2 x 2 table.

    Port of ``sa_phi()``. Its magnitude is ``sqrt(chi_sq / n)`` on the uncorrected
    statistic, so phi says nothing Cramer's V does not on a 2 x 2 table, where the
    two are equal in size. What it adds is the sign, and the sign is the finding:
    phi above zero means the second level of each variable occurs with the second
    level of the other more often than independence would give, which is the same
    direction the odds ratio reads above 1.
    """
    table = _as_counts(counts)
    a, b = float(table[0, 0]), float(table[0, 1])
    c, d = float(table[1, 0]), float(table[1, 1])
    denominator = np.sqrt(float(np.prod([a + b, c + d, a + c, b + d])))
    if denominator == 0:
        return float("nan")
    return float((a * d - b * c) / denominator)


def has_zero_cell(counts: Any) -> bool:
    """Whether a 2 x 2 table holds a cell the odds ratio cannot be read off.

    Port of ``sa_has_zero_cell()``. The scenario asks this rather than reading it
    off the measure, so that the one message about the Haldane-Anscombe correction
    is raised where every other message to the user is raised and
    :func:`odds_ratio` stays a function of its arguments.
    """
    table = _as_counts(counts)
    return bool(_is_square_2x2(table) and np.any(table == 0))


def odds_ratio(counts: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """The sample odds ratio of a 2 x 2 table, with a Wald interval.

    Port of ``sa_odds_ratio()``. Read against the first level of each variable,
    which the reference level fixes. An odds ratio above 1 says the two second
    levels go together, and pointing the reference at the other level of either
    variable inverts it.

    The interval is built on the log scale and exponentiated, so it is asymmetric
    about the estimate on the ratio scale and cannot reach below zero. A zero cell
    leaves the log undefined, so the Haldane-Anscombe correction of half an
    observation per cell is applied: the estimate is then a shrunken one rather
    than an infinite one. :func:`has_zero_cell` is how the scenario knows to say
    so.
    """
    table = _as_counts(counts)
    cells = np.array(
        [table[0, 0], table[0, 1], table[1, 0], table[1, 1]],
        dtype=float,
    )
    if np.any(cells == 0):
        cells = cells + 0.5

    log_or = float(np.log(cells[0] * cells[3] / (cells[1] * cells[2])))
    standard_error = float(np.sqrt(np.sum(1 / cells)))
    z = float(stats.norm.ppf(1 - (1 - conf_level) / 2))
    return assoc_row(
        "odds_ratio",
        np.exp(log_or),
        np.exp(log_or - z * standard_error),
        np.exp(log_or + z * standard_error),
    )
