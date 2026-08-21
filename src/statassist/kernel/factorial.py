"""The kernels of a fully crossed between-subject analysis.

Port of ``R/kernel_factorial.R``, written to the same rule as
:mod:`statassist.kernel.anova`: numbers in, a named row or a table out, no fitted
model kept anywhere.

Two things are different here. A factorial analysis answers on two axes at once,
the whole model and the individual terms, and both come out of one call for the
reason :func:`~statassist.kernel.oneway_anova` and :func:`~statassist.kernel.tukey`
share a mean square error: the same sums of squares computed twice in two code
paths is how the two ends of a result object come to disagree.

And the arithmetic runs on the cell means rather than on the observations. A fully
crossed model gives every row of a cell the same predictor values, so the residual
sum of squares of any sub-model is the within-cell sum of squares plus the
weighted residual sum of squares of the cell means, with the cell counts as
weights. Every sum of squares here is a difference of two such residuals, so the
within-cell part cancels and what is left is a regression of ``n_cells`` numbers
rather than of ``n_used`` of them. It is exact rather than approximate: the two
formulations have the same normal equations.

Column and term indices are zero-based, and the intercept's ``assign`` entry is
``-1`` rather than R's ``0``, so that ``terms[assign[column]]`` reads the term a
column belongs to without an off-by-one correction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats

from ..core.errors import SaValueError
from ..core.factorial import fact_term_labels, fact_terms
from ..core.tables import stat_row
from .anova import oneway_anova
from .posthoc import posthoc_columns

__all__ = [
    "QR_RANK_TOL",
    "SS_TYPES",
    "TERM_COLUMNS",
    "CellMatrix",
    "FactorialFit",
    "FactorialPlan",
    "SsPair",
    "contr_sum",
    "fact_cell_matrix",
    "fact_ss_plan",
    "factorial_anova",
    "factorial_plan",
    "factorial_tukey",
]

#: The three sums of squares a crossed model can be read with, as R spells them.
SS_TYPES: tuple[str, ...] = ("III", "II", "I")

#: Relative tolerance R's ``qr()`` counts a rank with.
#:
#: R's ``qr()`` runs LINPACK's ``dqrdc2``, which treats a column whose residual
#: norm has fallen to ``1e-7`` of the largest as having no rank left to give.
#: :func:`numpy.linalg.matrix_rank` uses an SVD cutoff scaled by the matrix
#: dimensions instead, so the two disagree on a design that is not of full rank -
#: and the degrees of freedom of every term are read off this number.
QR_RANK_TOL = 1e-7


def contr_sum(n_levels: int) -> np.ndarray:
    """Sum-to-zero contrast codes for a factor of ``n_levels`` levels.

    Port of ``stats::contr.sum()``: ``n_levels`` rows and ``n_levels - 1``
    columns, the identity on every level but the last and ``-1`` throughout on
    the last. Written out rather than imported because it is four lines and
    because ``patsy`` and ``statsmodels`` spell the same coding with different
    column orders.

    >>> contr_sum(3)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [-1., -1.]])
    """
    if n_levels < 2:
        raise SaValueError(f"a factor needs at least two levels to be coded, got {n_levels}.")
    codes = np.zeros((n_levels, n_levels - 1), dtype=float)
    codes[: n_levels - 1, :] = np.eye(n_levels - 1)
    codes[n_levels - 1, :] = -1.0
    return codes


class CellMatrix(NamedTuple):
    """The model matrix of the cells of a crossed design.

    Attributes:
        x: One row per cell and one column per model coefficient, the intercept
            first and then the terms in :func:`~statassist.core.fact_terms` order.
        assign: Which term each column belongs to, as a zero-based position into
            ``terms``. The intercept belongs to no term and is ``-1``.
        terms: The terms themselves.
    """

    x: np.ndarray
    assign: np.ndarray
    terms: list[tuple[str, ...]]


def fact_cell_matrix(
    factor_lv: Mapping[str, Sequence[str]],
    cells: pd.DataFrame,
) -> CellMatrix:
    """Sum-to-zero model matrix of the cells of a crossed design.

    Port of ``sa_fact_cell_matrix()``. Sum-to-zero (``contr.sum``) coding is what
    makes a term's columns orthogonal to the terms it does not contain in a
    balanced design, which is why Type I, II and III agree there and why a Type
    III sum of squares means what the unweighted marginal means say it does.

    The columns of an interaction are the elementwise products of the columns of
    the factors it is over, which is what R's ``model.matrix()`` builds for a
    ``:`` term. Built here instead so that the column order is the term order and
    no formula has to be parsed at run time. Within an interaction block the
    columns of the earlier factor vary fastest, which is R's own order and what
    the golden fixture holds.

    Args:
        factor_lv: Factors and their levels.
        cells: Grid of level indices, as :func:`~statassist.core.fact_grid`
            returns.
    """
    terms = fact_terms(list(factor_lv))
    codes = {name: contr_sum(len(levels)) for name, levels in factor_lv.items()}
    n_cells = len(cells)

    blocks: list[np.ndarray] = [np.ones((n_cells, 1), dtype=float)]
    assign: list[int] = [-1]
    for position, term in enumerate(terms):
        block = np.ones((n_cells, 1), dtype=float)
        for name in term:
            coded = codes[name][np.asarray(cells[name], dtype=np.int64), :]
            width = block.shape[1]
            block = np.tile(block, (1, coded.shape[1])) * np.repeat(coded, width, axis=1)
        blocks.append(block)
        assign.extend([position] * block.shape[1])

    return CellMatrix(
        x=np.hstack(blocks),
        assign=np.asarray(assign, dtype=np.int64),
        terms=terms,
    )


class SsPair(NamedTuple):
    """The two models whose difference is one term's sum of squares.

    Attributes:
        base: Zero-based column positions of the model the term is out of.
        full: The same columns plus the term's own, so ``full`` always contains
            ``base``.
    """

    base: np.ndarray
    full: np.ndarray


def fact_ss_plan(
    terms: Sequence[Sequence[str]],
    assign: Any,
    ss_type: str,
) -> list[SsPair]:
    """The two models whose difference is a term's sum of squares.

    Port of ``sa_fact_ss_plan()``. Every sum of squares in an ANOVA table is a
    model comparison, and the three types differ only in which pair of models is
    compared. Naming the pairs once means the three types share one arithmetic
    path and cannot come to disagree about anything except what they are meant to
    disagree about.

    ``"III"``
        Everything else stays in and the term comes out, so the sum of squares is
        what this term explains that no other term can. Unweighted by cell size,
        which is what makes it the type to score a simulation against: the planted
        main effect is a statement about the levels, not about how many
        observations happened to land in each.
    ``"II"``
        The term is added to the model holding every term that does not contain
        it, so a main effect is adjusted for the other main effects but not for
        the interaction it is part of.
    ``"I"``
        Sequential. Each term is added to the ones before it, so the sums of
        squares add up to the between-cell sum of squares exactly and the answer
        depends on the order the factors were declared in. This is what R's
        ``aov()`` reports, which is what makes it the type an external check can
        be matched against on unbalanced data.

    Args:
        terms: One entry per model term.
        assign: Which term each column of the model matrix belongs to, ``-1`` for
            the intercept.
        ss_type: One of :data:`SS_TYPES`.

    Returns:
        One :class:`SsPair` per term, in ``terms`` order.

    Raises:
        SaValueError: If ``ss_type`` is not one of :data:`SS_TYPES`.
    """
    if ss_type not in SS_TYPES:
        raise SaValueError("`ss_type` must be one of: " + ", ".join(SS_TYPES) + ".")

    labels = np.asarray(assign, dtype=np.int64)
    n_terms = len(terms)
    cols_of = [np.flatnonzero(labels == position) for position in range(n_terms)]
    intercept = np.flatnonzero(labels == -1)

    plan: list[SsPair] = []
    for position in range(n_terms):
        if ss_type == "III":
            keep = [other for other in range(n_terms) if other != position]
        elif ss_type == "II":
            # A term contains this one when its factors include all of them,
            # which is also true of the term itself, so it drops out without a
            # second condition.
            keep = [
                other
                for other in range(n_terms)
                if not set(terms[position]).issubset(set(terms[other]))
            ]
        else:
            keep = list(range(position))

        base = np.sort(np.concatenate([intercept, *[cols_of[other] for other in keep]]))
        full = np.sort(np.concatenate([base, cols_of[position]]))
        plan.append(SsPair(base=base, full=full))
    return plan


class FactorialPlan(NamedTuple):
    """Everything about a crossed model that does not depend on the data.

    The design matrix and the model comparisons are the same for every feature, so
    they are settled once and handed to the kernel rather than rebuilt per
    feature. With hundreds of features that is the difference between building one
    matrix and building hundreds of identical ones.

    Attributes:
        x: The cell model matrix.
        assign: Which term each column belongs to, ``-1`` for the intercept.
        terms: One entry per term.
        labels: The ``a:b`` label of each term.
        orders: How many factors each term spans.
        ss: The model comparison each term's sum of squares is.
    """

    x: np.ndarray
    assign: np.ndarray
    terms: list[tuple[str, ...]]
    labels: list[str]
    orders: list[int]
    ss: list[SsPair]


def factorial_plan(
    factor_lv: Mapping[str, Sequence[str]],
    cells: pd.DataFrame,
    ss_type: str = "III",
) -> FactorialPlan:
    """Settle the design matrix and the model comparisons once.

    Port of ``sa_factorial_plan()``.

    Args:
        factor_lv: Factors and their levels.
        cells: Grid of level indices, as :func:`~statassist.core.fact_grid`
            returns.
        ss_type: One of :data:`SS_TYPES`.
    """
    matrix = fact_cell_matrix(factor_lv, cells)
    return FactorialPlan(
        x=matrix.x,
        assign=matrix.assign,
        terms=matrix.terms,
        labels=fact_term_labels(matrix.terms),
        orders=[len(term) for term in matrix.terms],
        ss=fact_ss_plan(matrix.terms, matrix.assign, ss_type),
    )


def _cell_samples(samples: Any) -> tuple[list[str], list[np.ndarray]]:
    """Read one sample per cell, in cell order, keeping the empty ones.

    A cell holding nothing is the error :func:`factorial_anova` reports in its own
    words, so the samples cannot go through
    :func:`~statassist.kernel._shared.as_samples`, which refuses an empty one with
    a message about a group.
    """
    if isinstance(samples, Mapping):
        items = list(samples.items())
    elif isinstance(samples, Sequence) and not isinstance(samples, str | bytes):
        items = [(str(index + 1), values) for index, values in enumerate(samples)]
    else:
        raise SaValueError("`samples` must be a mapping of cell label to sample, or a sequence.")

    names = [str(name) for name, _ in items]
    arrays = [np.asarray(values, dtype=float).reshape(-1) for _, values in items]
    for name, array in zip(names, arrays, strict=True):
        if array.size and not np.isfinite(array).all():
            raise SaValueError(
                f"`samples[{name!r}]` holds a missing or infinite value; the caller is "
                "expected to have dropped those according to its own design."
            )
    return names, arrays


def _weighted_fit(xw: np.ndarray, yw: np.ndarray, columns: np.ndarray) -> tuple[float, int]:
    """Residual sum of squares and rank of one weighted sub-model.

    R runs ``qr()`` and ``qr.resid()``; the pivoted QR here is the same
    decomposition, and the residual is taken against the span of the first
    ``rank`` pivoted columns exactly as ``qr.resid()`` does. The rank itself uses
    :data:`QR_RANK_TOL` so that a design which is not of full rank loses the same
    degrees of freedom it loses in R.
    """
    a = xw[:, columns]
    q, r = scipy.linalg.qr(a, mode="economic", pivoting=True)[:2]
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0 or diagonal[0] == 0:
        rank = 0
    else:
        rank = int(np.count_nonzero(diagonal > QR_RANK_TOL * diagonal[0]))

    projected = (q.T @ yw)[:rank]
    # Subtracting two nearly equal sums of squares can land a hair below zero on
    # a sub-model that explains everything there is to explain.
    rss = max(float(yw @ yw) - float(projected @ projected), 0.0)
    return rss, rank


#: Columns of the term table :func:`factorial_anova` returns, in order.
TERM_COLUMNS: tuple[str, ...] = (
    "n_used",
    "df",
    "ss",
    "ms",
    "f_stat",
    "df_error",
    "eta_sq",
    "partial_eta_sq",
    "pval",
)


class FactorialFit(NamedTuple):
    """What one feature's crossed model came to.

    Attributes:
        model: The whole-model test, the named row
            :func:`~statassist.kernel.oneway_anova` returns with ``n_groups``
            renamed ``n_cells``.
        terms: One row per term, indexed by term label, with the columns of
            :data:`TERM_COLUMNS`.
        means: The cell means, in cell order.
        n: The cell counts, in cell order.
        ms_error: The mean square error the F tests were scaled by, which the
            post-hoc stage reuses so that it is scaled by the same one.
        df_error: Its degrees of freedom.
    """

    model: dict[str, float]
    terms: pd.DataFrame
    means: np.ndarray
    n: np.ndarray
    ms_error: float
    df_error: float


def factorial_anova(samples: Any, plan: FactorialPlan) -> FactorialFit:
    """Factorial analysis of variance, whole model and term by term.

    Port of ``sa_factorial_anova()``. The whole-model test is the one-way ANOVA
    that treats the cells as groups, which is the same test as the F test of a
    fully crossed model: the crossed model is the cell means model written in
    another basis, so it fits the same values and leaves the same residuals. That
    is what lets one feature-wise row stand for "this feature responds to the
    design" while the term rows say which part of it responds.

    The term sums of squares are model comparisons on the weighted cell means, as
    described at the top of this module, so ``df`` comes from the ranks of the two
    matrices rather than from a formula and stays right if a design ever arrives
    that is not of full rank.

    Args:
        samples: One sample per cell, in cell order, no missing values. A mapping
            keyed by cell label is the usual form, since that is what an empty
            cell is reported by name from.
        plan: The plan :func:`factorial_plan` settled.

    Returns:
        A :class:`FactorialFit`.

    Raises:
        SaValueError: If any cell holds no usable observation.
    """
    names, arrays = _cell_samples(samples)
    n = np.array([array.size for array in arrays], dtype=float)
    if np.any(n == 0):
        empty = [name for name, size in zip(names, n, strict=True) if size == 0]
        raise SaValueError(
            "cell(s) with no usable observation, which leaves a crossed model with "
            "nothing to estimate there: " + ", ".join(empty) + "."
        )

    # The whole-model row is delegated rather than rewritten, so a factorial
    # result and a one-way result over the same cells cannot report different F
    # values.
    oneway = oneway_anova(dict(zip(names, arrays, strict=True)))
    model = {("n_cells" if key == "n_groups" else key): value for key, value in oneway.items()}

    means = np.array([float(np.mean(array)) for array in arrays])
    ss_within = float(
        sum(float(np.sum((array - mean) ** 2)) for array, mean in zip(arrays, means, strict=True))
    )
    df_error = float(model["df2"])
    ms_error = ss_within / df_error
    grand = float(np.sum(n * means) / np.sum(n))
    ss_total = ss_within + float(np.sum(n * (means - grand) ** 2))

    # One weighted least squares problem per distinct sub-model. Several terms ask
    # for the same one under Type I and II, so the answers are kept.
    sw = np.sqrt(n)
    xw = plan.x * sw[:, None]
    yw = means * sw
    seen: dict[tuple[int, ...], tuple[float, int]] = {}

    def fit(columns: np.ndarray) -> tuple[float, int]:
        key = tuple(int(column) for column in columns)
        if key not in seen:
            seen[key] = _weighted_fit(xw, yw, columns)
        return seen[key]

    n_used = float(np.sum(n))
    rows: list[dict[str, float]] = []
    for pair in plan.ss:
        base_rss, base_rank = fit(pair.base)
        full_rss, full_rank = fit(pair.full)
        df = full_rank - base_rank
        # Subtracting two residual sums of squares of nearly equal size can land a
        # hair below zero on a term that explains nothing at all.
        ss = max(base_rss - full_rss, 0.0)
        if df < 1:
            rows.append(
                stat_row(
                    n_used=n_used,
                    df=0.0,
                    ss=ss,
                    ms=np.nan,
                    f_stat=np.nan,
                    df_error=df_error,
                    eta_sq=np.nan,
                    partial_eta_sq=np.nan,
                    pval=np.nan,
                )
            )
            continue
        ms = ss / df
        f_stat = ms / ms_error
        rows.append(
            stat_row(
                n_used=n_used,
                df=float(df),
                ss=ss,
                ms=ms,
                f_stat=f_stat,
                df_error=df_error,
                eta_sq=ss / ss_total,
                partial_eta_sq=ss / (ss + ss_within),
                pval=float(stats.f.sf(f_stat, df, df_error)),
            )
        )

    return FactorialFit(
        model=model,
        terms=pd.DataFrame(rows, index=pd.Index(plan.labels), columns=list(TERM_COLUMNS)),
        means=means,
        n=n,
        ms_error=ms_error,
        df_error=df_error,
    )


def factorial_tukey(
    fit: FactorialFit,
    sel1: Sequence[np.ndarray],
    sel2: Sequence[np.ndarray],
    nmeans: Sequence[int],
    rows: Sequence[int],
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Tukey-Kramer comparisons of marginal means and of simple effects.

    Port of ``sa_factorial_tukey()``. The post-hoc stage of a factorial model.
    Every contrast is scaled by the mean square error of the whole model, which is
    what makes these comparisons consistent with the F tests they follow rather
    than a second analysis of the same data.

    A marginal mean is the **unweighted** mean of the cell means, not the mean of
    the observations, so a level's mean is not pulled towards whichever
    combination of the other factors happened to be sampled most. That is also
    the quantity a factorial simulator's contrast truth table records, so an
    estimate here and a planted delta there are the same quantity and can be
    compared without a correction. The weights being ``1/m``, the variance of a
    difference is ``MSE * (sum(w^2 / n_c) + sum(w^2 / n_c))``, the Kramer form,
    which reduces to Tukey's own when the cells are equal in size.

    The family is one block of contrasts rather than the whole table: the
    studentised range is over the number of levels of the factor being compared,
    so a marginal comparison of a four-level factor and a simple comparison of the
    same factor within one stratum are judged against the same distribution, and
    the p-values are family-wise within each block without further adjustment.

    R takes the whole skeleton object; the two cell selections are passed
    separately here, the way :func:`~statassist.core.fact_contrast_skeleton`
    hands them back.

    Args:
        fit: What :func:`factorial_anova` returned.
        sel1: Per skeleton row, the cells whose mean is the left side.
        sel2: Per skeleton row, the cells whose mean is the right side.
        nmeans: Number of means the studentised range spans, one per skeleton row.
        rows: Zero-based skeleton rows to compute, in the order they are wanted.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per entry of ``rows``, with the columns of
        :func:`~statassist.kernel.posthoc_columns`.

    Raises:
        SaValueError: If the mean square error of the model is zero.

    References:
        Tukey, J. W. (1949). Comparing individual means in the analysis of
        variance. *Biometrics*, 5(2), 99-114.

        Kramer, C. Y. (1956). Extension of multiple range tests to group means
        with unequal numbers of replications. *Biometrics*, 12(3), 307-310.
    """
    means = fit.means
    n = fit.n
    mse = fit.ms_error
    df = fit.df_error
    if mse <= 0:
        raise SaValueError(
            "the mean square error of the model is zero, so no contrast can be scaled."
        )

    wanted = posthoc_columns()
    out: list[dict[str, float]] = []
    for row in rows:
        first = np.asarray(sel1[row], dtype=np.int64)
        second = np.asarray(sel2[row], dtype=np.int64)
        estimate = float(np.mean(means[first]) - np.mean(means[second]))
        variance = mse * (
            float(np.sum((1 / first.size) ** 2 / n[first]))
            + float(np.sum((1 / second.size) ** 2 / n[second]))
        )
        # The studentised range is the range of the means over the standard error
        # of one of them, so the divisor carries a 1/2 that a t statistic does
        # not.
        stderr = float(np.sqrt(variance / 2))
        q_stat = estimate / stderr
        span = int(nmeans[row])
        critical = float(stats.studentized_range.ppf(conf_level, span, df))
        out.append(
            stat_row(
                n1=float(np.sum(n[first])),
                n2=float(np.sum(n[second])),
                estimate=estimate,
                stderr=stderr,
                statistic=q_stat,
                df=df,
                pval=float(stats.studentized_range.sf(abs(q_stat), span, df)),
                lower_conf=estimate - critical * stderr,
                upper_conf=estimate + critical * stderr,
            )
        )
    return pd.DataFrame(out, columns=wanted, dtype=float)
