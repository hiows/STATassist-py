"""Performance kernels for a two-class outcome, written out rather than borrowed.

Port of ``R/kernel_performance.R``, and the R file's reasoning is what decided the
Python side too. ``pROC`` covers the first three of these and none of the last
three: IDI and NRI have no implementation there at all, and DeLong's test has no
counterpart in ``scikit-learn`` or ``scipy`` for this side to call. Depending on a
library in each language would therefore buy the easy half and leave the hard half
to be written twice against two different sets of defaults. Written out once, the
formula is the specification and ``testdata/golden/`` is what holds this side to
it.

``response`` is always 0/1 with 1 for the event, which is the class a model's
predicted probability is the probability *of*. The callers convert once and these
never see a label.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import SaInternalError, SaValueError
from ..core.tables import stat_row

__all__ = [
    "Placement",
    "auc",
    "auc_delong",
    "brier",
    "check_response",
    "delong_test",
    "idi",
    "nri",
    "placement_values",
    "roc_points",
    "threshold_scores",
]


def _as_pair(response: Any, predictor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read the two vectors as float arrays of the same length."""
    outcome = np.asarray(response, dtype=float).reshape(-1)
    scores = np.asarray(predictor, dtype=float).reshape(-1)
    if outcome.size != scores.size:
        raise SaInternalError("internal error: `response` and `predictor` differ in length.")
    return outcome, scores


def check_response(response: Any, predictor: Any) -> None:
    """Refuse a pair of vectors no performance measure can be read off.

    Port of ``sa_check_response()``. R tests ``response %in% c(0, 1)``, which a
    double ``0.0``/``1.0`` passes as readily as an integer, and the same
    tolerance is kept here: the callers convert a two-level factor to a numeric
    indicator and there is no reason for a kernel to care which numeric type
    came out of that.

    Raises:
        SaInternalError: If the two vectors differ in length, or ``response``
            holds anything but 0 and 1. Both are the caller's contract rather
            than the user's input.
        SaValueError: If only one of the two classes is present, which the user
            can see and fix.
    """
    outcome, _ = _as_pair(response, predictor)
    if not np.all((outcome == 0) | (outcome == 1)):
        raise SaInternalError("internal error: `response` must be 0/1 with 1 for the event.")

    n_event = int(np.count_nonzero(outcome == 1))
    if n_event == 0 or n_event == outcome.size:
        raise SaValueError(
            "the scored rows hold a single class, so there is nothing to discriminate "
            "between. Both classes have to be present in `answer`."
        )


def roc_points(response: Any, predictor: Any) -> pd.DataFrame:
    """Operating points of a ROC curve.

    Port of ``sa_roc_points()``. One row per distinct predicted value, plus the
    point above all of them, so a curve of ``k`` distinct predictions has
    ``k + 1`` points running from ``(0, 1)`` to ``(1, 0)``. A row is positive
    when its prediction is greater than or equal to the threshold.

    Ties are one point rather than several. Rows sharing a predicted value cannot
    be separated by any threshold, so the curve steps diagonally through them,
    which is the same thing counting them as half a concordant pair does in
    :func:`auc`.

    Args:
        response: 0/1, 1 for the event.
        predictor: Predicted probability of the event.

    Returns:
        ``threshold``, ``sensitivity`` and ``specificity``, one row per point.
    """
    check_response(response, predictor)
    outcome, scores = _as_pair(response, predictor)
    n_event = float(np.count_nonzero(outcome == 1))
    n_other = float(outcome.size - n_event)

    # R's `order(decreasing = TRUE)` leaves ties in the order they arrived; a
    # stable sort of the negated scores is the same permutation.
    at = np.argsort(-scores, kind="stable")
    sorted_scores = scores[at]
    hit = np.cumsum(outcome[at])
    miss = np.cumsum(1.0 - outcome[at])
    # The last index of each run of equal predictions is the only one a threshold
    # can stop at, since every row of the run crosses together.
    last = np.ones(sorted_scores.size, dtype=bool)
    last[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    return pd.DataFrame(
        {
            "threshold": np.concatenate(([np.inf], sorted_scores[last])),
            "sensitivity": np.concatenate(([0.0], hit[last])) / n_event,
            "specificity": 1.0 - np.concatenate(([0.0], miss[last])) / n_other,
        }
    )


class Placement(NamedTuple):
    """DeLong's structural components, one array per class.

    Attributes:
        event: Per event, the share of non-events it outranks, a tie counting as
            half.
        other: Per non-event, the share of events that outrank it.
    """

    event: np.ndarray
    other: np.ndarray


def placement_values(response: Any, predictor: Any) -> Placement:
    """Placement values, the per-row terms an AUC is the mean of.

    Port of ``sa_placement_values()``. Both classes' values average to the AUC,
    and their variances are what its standard error is built from, which is what
    makes the statistic a mean of independent terms rather than a U statistic to
    be approximated.

    Computed from ranks rather than from the ``n_event * n_other`` comparisons.
    The rank of an event within the pooled sample, less its rank within the
    events alone, is the number of non-events below it plus half the number tied
    with it, which is that row's placement value times ``n_other``.

    R's ``rank()`` averages ties, which is
    :func:`scipy.stats.rankdata`'s default and the reason a tie ends up counted
    as half without being special-cased anywhere.

    Like R's, this does not validate: it is called from functions that already
    have.
    """
    outcome, scores = _as_pair(response, predictor)
    is_event = outcome == 1
    x = scores[is_event]
    y = scores[~is_event]
    n_event = x.size
    n_other = y.size

    pooled = stats.rankdata(np.concatenate([x, y]))
    return Placement(
        event=(pooled[:n_event] - stats.rankdata(x)) / n_other,
        other=1.0 - (pooled[n_event:] - stats.rankdata(y)) / n_event,
    )


def auc(response: Any, predictor: Any) -> float:
    """Area under the ROC curve.

    Port of ``sa_auc()``. The Mann-Whitney statistic scaled to a probability: the
    chance that a randomly drawn event is ranked above a randomly drawn
    non-event, with a tie counted as half. The same quantity
    :func:`~statassist.kernel.brunner_munzel` reports as its relative effect,
    read here as a statement about a classifier rather than about two samples.
    """
    check_response(response, predictor)
    outcome, scores = _as_pair(response, predictor)
    is_event = outcome == 1
    n_event = float(np.count_nonzero(is_event))
    n_other = float(outcome.size - n_event)
    ranks = stats.rankdata(scores)
    return float((np.sum(ranks[is_event]) - n_event * (n_event + 1) / 2) / (n_event * n_other))


def auc_delong(response: Any, predictor: Any) -> dict[str, float]:
    """Area under the ROC curve with DeLong's standard error.

    Port of ``sa_auc_delong()``. The variance of a mean of placement values, one
    variance per class, which is the non-parametric standard error of an AUC.

    Returns:
        ``auc`` and ``se``. A single row of either class leaves that class's
        variance undefined, and an AUC resting on one observation has no spread
        to report rather than a spread of zero, so ``se`` is missing there.

    References:
        DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing
        the areas under two or more correlated receiver operating characteristic
        curves. *Biometrics*, 44(3), 837-845.
    """
    check_response(response, predictor)
    placement = placement_values(response, predictor)
    n_event = placement.event.size
    n_other = placement.other.size

    area = float(np.mean(placement.event))
    if n_event > 1 and n_other > 1:
        variance = float(
            np.var(placement.event, ddof=1) / n_event + np.var(placement.other, ddof=1) / n_other
        )
        standard_error = float(np.sqrt(variance))
    else:
        standard_error = np.nan
    return stat_row(auc=area, se=standard_error)


def delong_test(response: Any, predictor_1: Any, predictor_2: Any) -> dict[str, float]:
    """DeLong's test for two AUCs measured on the same rows.

    Port of ``sa_delong_test()``. Paired, because both models ranked the same
    rows and their placement values therefore covary. Ignoring that covariance
    would treat two models that agree almost everywhere as two independent
    estimates and make every difference between them look more surprising than it
    is.

    Args:
        response: 0/1, 1 for the event.
        predictor_1: The first set of predicted probabilities.
        predictor_2: The second, in the direction ``predictor_1 - predictor_2``.

    Returns:
        ``delta``, ``se``, ``statistic`` and ``pval``. Two models that rank every
        row identically differ by exactly zero with a standard error of exactly
        zero, and the ratio of the two is not a number: reporting 1 would claim a
        test was run against a distribution that has no spread, so the statistic
        and the p-value say there is nothing here instead.

    References:
        DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing
        the areas under two or more correlated receiver operating characteristic
        curves. *Biometrics*, 44(3), 837-845.
    """
    check_response(response, predictor_1)
    check_response(response, predictor_2)

    first = placement_values(response, predictor_1)
    second = placement_values(response, predictor_2)
    n_event = first.event.size
    n_other = first.other.size

    delta = float(np.mean(first.event) - np.mean(second.event))
    if n_event < 2 or n_other < 2:
        return stat_row(delta=delta, se=np.nan, statistic=np.nan, pval=np.nan)

    s_event = np.cov(np.vstack([first.event, second.event]), ddof=1)
    s_other = np.cov(np.vstack([first.other, second.other]), ddof=1)
    s = s_event / n_event + s_other / n_other
    variance = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])

    if not np.isfinite(variance) or variance <= 0:
        return stat_row(delta=delta, se=0.0, statistic=np.nan, pval=np.nan)

    standard_error = float(np.sqrt(variance))
    statistic = delta / standard_error
    return stat_row(
        delta=delta,
        se=standard_error,
        statistic=statistic,
        pval=float(2 * stats.norm.cdf(-abs(statistic))),
    )


def idi(response: Any, predictor_old: Any, predictor_new: Any) -> dict[str, float]:
    """Integrated discrimination improvement.

    Port of ``sa_idi()``. How much further apart the two classes' predicted
    probabilities moved. The mean predicted probability rises among events and
    falls among non-events for a model that discriminates better, and the IDI is
    the sum of those two movements, so it is on the probability scale rather than
    on the scale of a rank.

    It answers something a difference of AUCs cannot. An AUC sees only the order
    of the predictions, so a new model that pushes every event's probability up by
    a tenth without reordering anything leaves it untouched and moves this.

    Returns:
        ``idi``, ``se``, ``statistic`` and ``pval``. The standard error is
        Pencina's paired form: the two class-wise mean changes are independent, so
        their variances add.

    References:
        Pencina, M. J., D'Agostino, R. B., D'Agostino, R. B. and Vasan, R. S.
        (2008). Evaluating the added predictive ability of a new marker: from area
        under the ROC curve to reclassification and beyond. *Statistics in
        Medicine*, 27(2), 157-172.
    """
    check_response(response, predictor_old)
    check_response(response, predictor_new)
    outcome, old = _as_pair(response, predictor_old)
    _, new = _as_pair(response, predictor_new)

    is_event = outcome == 1
    moved = new - old
    moved_event = moved[is_event]
    moved_other = moved[~is_event]
    n_event = moved_event.size
    n_other = moved_other.size

    estimate = float(np.mean(moved_event) - np.mean(moved_other))
    if n_event < 2 or n_other < 2:
        return stat_row(idi=estimate, se=np.nan, statistic=np.nan, pval=np.nan)

    variance = float(np.var(moved_event, ddof=1) / n_event + np.var(moved_other, ddof=1) / n_other)
    if not np.isfinite(variance) or variance <= 0:
        return stat_row(idi=estimate, se=0.0, statistic=np.nan, pval=np.nan)

    standard_error = float(np.sqrt(variance))
    statistic = estimate / standard_error
    return stat_row(
        idi=estimate,
        se=standard_error,
        statistic=statistic,
        pval=float(2 * stats.norm.cdf(-abs(statistic))),
    )


def nri(response: Any, predictor_old: Any, predictor_new: Any) -> dict[str, float]:
    """Continuous net reclassification improvement.

    Port of ``sa_nri()``. How often a probability moved the right way, which is
    the third question and the coarsest. Only the direction of each row's change
    is counted, so it is unmoved by how large the changes were and answers where
    the IDI does not: a new model that helps most rows a little and hurts a few a
    great deal has a positive NRI and can have a negative IDI.

    Category-free, so no risk strata are named. A stratified NRI depends on cut
    points that are a clinical convention rather than a property of the data, and
    there is no default for them this package could pick.

    The standard error is the non-null one, the variance of the difference of two
    proportions with the ``(p_up - p_down)^2`` term kept. Pencina's published test
    drops that term, which is correct under the null the test is against, where
    the two proportions are equal and it vanishes. Keeping it is what makes the
    interval and the p-value here come from one standard error rather than two, at
    the cost of a p-value very slightly different from the one
    ``Hmisc::improveProb()`` reports.

    Returns:
        ``nri``, ``nri_event``, ``nri_other``, ``se``, ``statistic`` and ``pval``.
        The two class-wise components are reported beside the total, since they
        are what it is made of and they routinely point opposite ways.

    References:
        Pencina, M. J., D'Agostino, R. B. and Steyerberg, E. W. (2011). Extensions
        of net reclassification improvement calculations to measure usefulness of
        new biomarkers. *Statistics in Medicine*, 30(1), 11-21.
    """
    check_response(response, predictor_old)
    check_response(response, predictor_new)
    outcome, old = _as_pair(response, predictor_old)
    _, new = _as_pair(response, predictor_new)

    is_event = outcome == 1
    moved = new - old
    moved_event = moved[is_event]
    moved_other = moved[~is_event]
    n_event = moved_event.size
    n_other = moved_other.size

    up_event = float(np.mean(moved_event > 0))
    down_event = float(np.mean(moved_event < 0))
    up_other = float(np.mean(moved_other > 0))
    down_other = float(np.mean(moved_other < 0))

    # An event whose probability rose was reclassified towards the truth; a
    # non-event whose probability rose was reclassified away from it, which is why
    # the second component reads the other way round.
    nri_event = up_event - down_event
    nri_other = down_other - up_other
    estimate = nri_event + nri_other

    variance = (up_event + down_event - nri_event**2) / n_event + (
        up_other + down_other - nri_other**2
    ) / n_other

    if not np.isfinite(variance) or variance <= 0:
        return stat_row(
            nri=estimate,
            nri_event=nri_event,
            nri_other=nri_other,
            se=0.0,
            statistic=np.nan,
            pval=np.nan,
        )

    standard_error = float(np.sqrt(variance))
    statistic = estimate / standard_error
    return stat_row(
        nri=estimate,
        nri_event=nri_event,
        nri_other=nri_other,
        se=standard_error,
        statistic=statistic,
        pval=float(2 * stats.norm.cdf(-abs(statistic))),
    )


def brier(response: Any, predictor: Any) -> float:
    """Brier score.

    Port of ``sa_brier()``. Mean squared distance between the predicted
    probability and the outcome, so it is to a classification what the root mean
    squared error squared is to a regression: a model that ranks perfectly but
    predicts every event at 0.6 is scored here and not by an AUC.

    References:
        Brier, G. W. (1950). Verification of forecasts expressed in terms of
        probability. *Monthly Weather Review*, 78(1), 1-3.
    """
    check_response(response, predictor)
    outcome, scores = _as_pair(response, predictor)
    return float(np.mean((scores - outcome) ** 2))


def threshold_scores(response: Any, predictor: Any, threshold: float) -> dict[str, float]:
    """Accuracy, sensitivity and specificity at one stated threshold.

    Port of ``sa_threshold_scores()``. A row is called an event when its predicted
    probability is greater than or equal to ``threshold``, the same direction
    :func:`roc_points` steps in.
    """
    check_response(response, predictor)
    outcome, scores = _as_pair(response, predictor)
    called = (scores >= threshold).astype(float)
    is_event = outcome == 1
    return stat_row(
        accuracy=float(np.mean(called == outcome)),
        sensitivity=float(np.mean(called[is_event] == 1)),
        specificity=float(np.mean(called[~is_event] == 0)),
    )
