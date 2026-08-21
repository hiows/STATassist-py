"""Internal helpers shared by the supervised learning simulators.

The port of ``R/utils_simulate.R``. The two simulators differ only in what they
do with the linear predictor: one adds noise to it and one runs it through a
logistic link. Everything before that is the same question twice over - which
coefficients were planted, what the predictors look like, which rows belong to
which subject - so it is answered once here.

The correlated draw is done with a Cholesky factor rather than
``numpy.random.Generator.multivariate_normal``, following R. It is the stricter
of the two: a correlation matrix that no data could have is rejected instead of
quietly being projected onto one that could.

**R's ``chol()`` is upper triangular and NumPy's is lower.** ``mvnorm`` right
multiplies by the factor, so it needs the upper one, which is
``np.linalg.cholesky(a).T``. Getting this backwards produces draws whose
covariance is the wrong matrix rather than an error, so the transpose is applied
once, here, and every caller takes the factor as given.

Every argument is checked before any of them is drawn on. A simulator that
consumed the random stream and then rejected its own arguments would give a
different data set to the seed that fixed it, depending on which call failed
first.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from ..core.errors import SaValueError
from ..core.validate import check_count, check_range, check_scalar_num, fmt_num
from ._patterns import pick_up_down

__all__ = [
    "PredSpec",
    "SupervisedDesign",
    "add_intercept",
    "balanced_levels",
    "chol_or_none",
    "cor_root",
    "factor_offsets",
    "mask_missing",
    "mvnorm",
    "plant_beta",
    "pred_spec",
    "recycle",
    "solve_intercept",
    "split_args",
    "subject_sizes",
    "supervised_design",
    "truth_pred",
    "truth_term",
]

#: Share of the numeric predictors given a coefficient in each direction when
#: the caller names no count. A fraction rather than a fixed number, so that
#: asking for fewer predictors plants fewer coefficients instead of failing.
_PLANT_SHARE = 0.25

#: How far the intercept search widens its bracket before giving up. An event
#: rate the predictors cannot reach is a statement about the arguments, so it is
#: reported as one rather than chased to infinity.
_BRACKET_LIMIT = 1e4

#: R's ``uniroot()`` default tolerance, kept so the intercept is solved to the
#: same precision on both sides of the port.
_ROOT_TOL = math.sqrt(np.finfo(float).eps)


def chol_or_none(cor_mat: np.ndarray) -> np.ndarray | None:
    """Factorise a correlation matrix, or say that it cannot be.

    Port of ``sa_sim_chol()``. A symmetric matrix with a unit diagonal and no
    entry outside -1 and 1 can still describe no data at all: ask for three
    predictors each correlated 0.9 with the other two and no set of vectors
    satisfies it. The factorisation is what finds that out, and the factor it
    produces is also what the draw needs, so the two questions are answered by
    one call.

    The failure comes back as ``None`` rather than as an error, because the
    sentence worth printing depends on whether the caller wrote the matrix down
    or asked for blocks that add up to it.

    Returns:
        The **upper** triangular factor ``r`` with ``r.T @ r == cor_mat``, or
        ``None`` when the matrix is not positive definite.
    """
    try:
        return np.linalg.cholesky(np.asarray(cor_mat, dtype=float)).T
    except np.linalg.LinAlgError:
        return None


def cor_root(cor_mat: object, n_pred: int, arg: str = "cor_mat") -> np.ndarray:
    """Check a correlation matrix and return the factor the draw needs.

    Port of ``sa_sim_cor_root()``. The three properties checked first are the
    ones that make a matrix a correlation matrix, and the fourth is the one that
    makes it a possible one.

    Args:
        cor_mat: The matrix as received, or ``None`` for independence.
        n_pred: Number of numeric predictors it has to describe.
        arg: Argument name to name in the error.

    Returns:
        The upper triangular Cholesky factor.
    """
    if cor_mat is None:
        return np.eye(n_pred)

    array = np.asarray(cor_mat)
    is_matrix = array.ndim == 2 and array.dtype.kind in "iuf" and array.dtype != bool
    if not is_matrix or array.shape != (n_pred, n_pred):
        raise SaValueError(
            f"`{arg}` must be a numeric {n_pred} x {n_pred} matrix, one row and "
            "column per numeric predictor."
        )
    values = np.asarray(array, dtype=float)
    if not np.isfinite(values).all():
        raise SaValueError(f"`{arg}` must not contain missing or non-finite values.")
    if not np.array_equal(values, values.T):
        raise SaValueError(f"`{arg}` must be symmetric.")
    if not bool((np.diag(values) == 1).all()):
        raise SaValueError(
            f"`{arg}` must have 1 on its diagonal, since a variable is "
            "perfectly correlated with itself."
        )
    if bool((np.abs(values) > 1).any()):
        raise SaValueError(f"`{arg}` holds correlation(s) outside [-1, 1].")

    root = chol_or_none(values)
    if root is None:
        raise SaValueError(
            f"`{arg}` is not positive definite, so no data has these "
            "correlations. Build it with make_block_cor(), which says which of "
            "the blocks cannot hold."
        )
    return root


def recycle(x: object, n: int, arg: str, lower: float = -math.inf) -> np.ndarray:
    """Check a numeric argument that may be given once or once per predictor.

    Port of ``sa_sim_recycle()``. The alternative would be to draw the means and
    standard deviations from a range, the way the expression simulators draw
    theirs. It is not done here because ``beta`` is a coefficient per unit of its
    predictor, so the size of the effect that is planted depends on the spread of
    the column it is planted on. A spread that was drawn at random would leave
    the signal-to-noise ratio of the simulation unreadable from its arguments.
    """
    array = np.asarray(x)
    numeric = array.dtype.kind in "iuf" and array.dtype != bool
    values = np.asarray(array, dtype=float).reshape(-1) if numeric else np.empty(0)
    if not numeric or values.size not in (1, n) or not np.isfinite(values).all():
        raise SaValueError(
            f"`{arg}` must be a finite numeric vector of length 1 or {n}, "
            "the number of numeric predictors."
        )
    if bool((values < lower).any()):
        raise SaValueError(f"`{arg}` must not go below {fmt_num(lower)}.")
    # rep_len(): a single value covers every predictor.
    return np.resize(values, n).astype(float)


def mvnorm(
    n: int,
    value_mean: np.ndarray,
    value_sd: np.ndarray,
    root: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw correlated normal predictors.

    Port of ``sa_sim_mvnorm()``.

    Args:
        n: Rows to draw.
        value_mean: One mean per predictor.
        value_sd: One standard deviation per predictor.
        root: Upper triangular Cholesky factor of the correlation matrix, as
            :func:`cor_root` returns it.
        rng: The generator to draw from.

    Returns:
        An ``n`` by ``len(value_mean)`` array.
    """
    means = np.asarray(value_mean, dtype=float)
    sds = np.asarray(value_sd, dtype=float)
    z = rng.standard_normal((n, means.size))
    # The columns are scaled after the rotation rather than by assembling a
    # covariance matrix, so that a single standard deviation stays a scale on the
    # columns instead of becoming a matrix of its own.
    return np.asarray(z @ root * sds + means, dtype=float)


class PredSpec(NamedTuple):
    """How many numeric predictors there are and where their coefficients come from.

    Attributes:
        n_pred: Number of numeric predictors.
        n_pos: How many carry a planted positive coefficient.
        n_neg: How many carry a planted negative one.
        beta: The coefficients the caller stated, or ``None`` when they are to be
            planted.
        value_mean: One mean per predictor.
        value_sd: One standard deviation per predictor.
    """

    n_pred: int
    n_pos: int
    n_neg: int
    beta: np.ndarray | None
    value_mean: np.ndarray
    value_sd: np.ndarray


class Planted(NamedTuple):
    """The coefficients that were planted, and which way each one points."""

    beta: np.ndarray
    direction: list[str]


class SupervisedDesign(NamedTuple):
    """Everything the two supervised simulators share.

    A design and not yet a data set: the outcome is the one thing each of them
    adds for itself, since a continuous outcome is this linear predictor plus
    noise and a class is a draw from the logistic function of it.

    Attributes:
        x: The predictor frame, numeric then categorical then constant.
        predictors: Every predictor name, in the column order of ``x``.
        numeric_pred: The numeric predictor names.
        factor_pred: The categorical predictor names.
        constant_pred: The single-valued predictor names.
        beta: One coefficient per numeric predictor.
        direction: ``"up"``, ``"down"`` or ``"none"`` per numeric predictor.
        offsets: Per categorical predictor, the offset of each of its levels.
        eta: The linear predictor **without** an intercept, one entry per row.
        subject: Subject label per row, or ``None`` when there are no subjects.
        subject_offset: The subject's offset per row, zeros without subjects.
        n_samples: Rows in the design.
        sizes: Rows per subject, or ``None`` when there are no subjects.
        truth: The per-predictor answer.
        truth_term: The per-model-term answer, without the intercept row.
    """

    x: pd.DataFrame
    predictors: list[str]
    numeric_pred: list[str]
    factor_pred: list[str]
    constant_pred: list[str]
    beta: np.ndarray
    direction: list[str]
    offsets: dict[str, dict[str, float]]
    eta: np.ndarray
    subject: list[str] | None
    subject_offset: np.ndarray
    n_samples: int
    sizes: list[int] | None
    truth: pd.DataFrame
    truth_term: pd.DataFrame


def pred_spec(
    n_pred: Any,
    beta: Any,
    n_pos: int | None,
    n_neg: int | None,
    value_mean: Any,
    value_sd: Any,
    explicit: Collection[str],
) -> PredSpec:
    """Settle how many numeric predictors there are, and how their coefficients come.

    Port of ``sa_sim_pred_spec()``. Two ways of saying what the coefficients are,
    and they are not alternatives that need reconciling: one plants a known
    number of them and leaves the rest at exactly zero, and the other states all
    of them. Naming both is refused rather than resolved, the same way
    :func:`~statassist.simulate.simulate_multiple_groups` refuses a ``group_lv``
    and an ``n_treat`` that count differently.

    Nothing is drawn here. The counts and the lengths are settled, and
    :func:`plant_beta` is what turns them into numbers once every other argument
    has been checked too.

    Args:
        n_pred: The argument as received, already resolved from its sentinel.
        beta: The coefficients the caller stated, or ``None``.
        n_pos: The argument as received, or ``None`` for its default share.
        n_neg: The same for the negative coefficients.
        value_mean: One mean, or one per predictor.
        value_sd: One standard deviation, or one per predictor.
        explicit: Names of the arguments the caller actually supplied, which is
            how a default is told apart from a value that was asked for.
    """
    if beta is None:
        count = check_count(n_pred, "n_pred", 1)
        # R evaluates the `round(0.25 * n_pred)` defaults here, where `n_pred` is
        # a checked count rather than whatever was passed in.
        n_pos = round(_PLANT_SHARE * count) if n_pos is None else n_pos
        n_neg = round(_PLANT_SHARE * count) if n_neg is None else n_neg
        pos = check_count(n_pos, "n_pos")
        neg = check_count(n_neg, "n_neg")
        if pos + neg > count:
            raise SaValueError(
                f"`n_pos` + `n_neg` is {pos + neg}, which is more coefficients "
                f"than the {count} numeric predictor(s) that `n_pred` asks for."
            )
        coefs = None
    else:
        clash = [name for name in ("n_pos", "n_neg") if name in explicit]
        if clash:
            raise SaValueError(
                "`beta` states every coefficient, so there is nothing left for "
                + " and ".join(f"`{name}`" for name in clash)
                + " to plant. Drop one of the two."
            )
        array = np.atleast_1d(np.asarray(beta))
        numeric = array.ndim == 1 and array.dtype.kind in "iuf" and array.dtype != bool
        if not numeric or array.size == 0 or not bool(np.isfinite(array).all()):
            raise SaValueError(
                "`beta` must be a finite numeric vector, one coefficient per "
                "numeric predictor and no intercept among them."
            )
        coefs = np.asarray(array, dtype=float)
        # `beta` holds one entry per predictor, which makes its length the number
        # of them, exactly as the length of `n_treat` is the number of treatment
        # groups in simulate_multiple_groups().
        if "n_pred" in explicit and coefs.size != n_pred:
            raise SaValueError(
                f"`n_pred` asks for {n_pred} numeric predictor(s) but `beta` gives "
                f"{coefs.size} coefficient(s). The intercept is not one of them: it "
                "is `intercept` for a regression and `event_rate` for a "
                "classification."
            )
        count = int(coefs.size)
        pos = 0
        neg = 0

    return PredSpec(
        n_pred=count,
        n_pos=pos,
        n_neg=neg,
        beta=coefs,
        value_mean=recycle(value_mean, count, "value_mean"),
        value_sd=recycle(value_sd, count, "value_sd", 0),
    )


def plant_beta(
    spec: PredSpec,
    beta_range: tuple[float, float],
    rng: np.random.Generator,
) -> Planted:
    """Turn the settled counts into coefficients.

    Port of ``sa_sim_plant_beta()``. A planted coefficient lands on a predictor
    drawn at random, but how many are positive and how many are negative is a
    function of the arguments alone. Drawing the signs instead would make those
    two counts move with the seed, which is the kind of thing about a simulation
    that should not have to be looked up.
    """
    if spec.beta is not None:
        coefs = spec.beta
        return Planted(
            beta=coefs,
            direction=["up" if c > 0 else "down" if c < 0 else "none" for c in coefs],
        )

    coefs = np.zeros(spec.n_pred)
    direction = ["none"] * spec.n_pred
    if spec.n_pos + spec.n_neg > 0:
        pos_idx, neg_idx = pick_up_down(spec.n_pred, spec.n_pos, spec.n_neg, rng)
        for i in pos_idx:
            direction[int(i)] = "up"
        for i in neg_idx:
            direction[int(i)] = "down"
        coefs[pos_idx] = rng.uniform(beta_range[0], beta_range[1], spec.n_pos)
        coefs[neg_idx] = -rng.uniform(beta_range[0], beta_range[1], spec.n_neg)
    return Planted(beta=coefs, direction=direction)


def subject_sizes(
    n_samples: Any,
    n_per_subject: Any,
    use_default_n: bool,
) -> tuple[list[int] | None, int]:
    """Work out how many subjects there are and how many rows each one carries.

    Port of ``sa_sim_subject_sizes()``. ``n_per_subject`` carries one row count
    per subject, which makes its length the number of subjects, the same rule
    ``n_treat`` follows in
    :func:`~statassist.simulate.simulate_multiple_groups`. ``n_samples`` says the
    same thing from the other side, so the two are settled together rather than
    in two passes that could disagree, and a single count has an obvious number
    of subjects to be spread over as soon as ``n_samples`` says how many rows
    there are in all.

    Args:
        n_samples: The argument as received, already resolved from its sentinel.
        n_per_subject: The argument as received, or ``None``.
        use_default_n: Whether ``n_samples`` was left at its default, which is
            what R reads off ``missing(n_samples)``.

    Returns:
        The row count of each subject, or ``None`` when there are no subjects,
        and the resulting row total.
    """
    if n_per_subject is None:
        return None, check_count(n_samples, "n_samples", 2)

    array = np.atleast_1d(np.asarray(n_per_subject))
    numeric = array.ndim == 1 and array.dtype.kind in "iuf" and array.dtype != bool
    if not numeric or array.size == 0:
        raise SaValueError(
            "`n_per_subject` must be one or more row counts, one per subject, or "
            "None for one row per subject."
        )

    if array.size == 1:
        total = check_count(n_samples, "n_samples", 2)
        per = check_count(array[0], "n_per_subject", 1)
        if total % per != 0:
            raise SaValueError(
                f"`n_per_subject` = {per} does not divide the {total} row(s) "
                "`n_samples` asks for. Pass a row count per subject, such as "
                f"`n_per_subject = [{per}] * {total // per}`."
            )
        sizes = [per] * (total // per)
    else:
        sizes = [check_count(value, f"n_per_subject[{k}]", 1) for k, value in enumerate(array)]
        total = sum(sizes)
        # The counts already say how many rows there are, so a default
        # `n_samples` has nothing to add. One that was asked for and disagrees is
        # not guessed at.
        if not use_default_n:
            asked = check_count(n_samples, "n_samples", 2)
            if asked != total:
                raise SaValueError(
                    f"`n_per_subject` gives {len(sizes)} subject(s) holding {total} "
                    f"row(s) in all, but `n_samples` asks for {asked}. Drop one of "
                    "the two."
                )

    if len(sizes) < 2:
        raise SaValueError(
            f"`n_per_subject` describes {len(sizes)} subject(s), and a split taken "
            "over subjects needs at least 2."
        )
    return sizes, total


def balanced_levels(
    n: int,
    levels: Sequence[str],
    rng: np.random.Generator,
) -> pd.Categorical:
    """Hand out factor levels in counts that do not depend on the seed.

    Port of ``sa_sim_balanced_levels()``. A permutation of a balanced vector
    rather than a draw with replacement, so that which unit gets which level is
    random while how many of each there are is not. Recycling the levels to fill
    ``n`` leaves the counts differing by at most one.
    """
    codes = rng.permutation(np.resize(np.arange(len(levels)), n))
    return pd.Categorical([levels[code] for code in codes], categories=list(levels))


def factor_offsets(
    factor_lv: Sequence[str],
    beta_range: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Plant an offset on every factor level beyond the reference.

    Port of ``sa_sim_factor_offsets()``. The reference level carries no offset,
    because it is what the intercept absorbs and what the other levels are
    contrasts against. The magnitudes are drawn from the same range the numeric
    coefficients use, and the signs alternate rather than being drawn, for the
    reason the counts of positive and negative coefficients are not drawn either.
    """
    k = len(factor_lv)
    signs = np.resize(np.array([1.0, -1.0]), k - 1)
    drawn = signs * rng.uniform(beta_range[0], beta_range[1], k - 1)
    return {level: 0.0 if i == 0 else float(drawn[i - 1]) for i, level in enumerate(factor_lv)}


def mask_missing(x: pd.DataFrame, p_missing: float, rng: np.random.Generator) -> pd.DataFrame:
    """Punch holes in the predictors after the outcome has been computed.

    Port of ``sa_sim_mask_missing()``. The outcome is generated from the complete
    predictors and the holes are made afterwards, which is what missing
    completely at random means: the value was there and doing its work, and it is
    the record of it that is gone. Computing the outcome from the holed frame
    instead would make the missingness part of the truth rather than something
    the analysis has to survive.

    Only the numeric predictors are holed. A hole in the factor predictor would
    make it unusable as the stratifier of a split, and a hole in a constant
    predictor would stop it being constant, since a missing value counts as one
    of the values a column takes.

    The number of cells is a function of ``p_missing`` and the size of the frame;
    only which cells they are is drawn.
    """
    n_col = len(x.columns)
    if p_missing == 0 or n_col == 0:
        return x
    n_row = len(x.index)
    n_na = round(p_missing * n_row * n_col)
    if n_na == 0:
        return x

    at = rng.permutation(n_row * n_col)[:n_na]
    values = x.to_numpy(dtype=float, copy=True)
    # Column-major, as R's linear index into a matrix is.
    values[at % n_row, at // n_row] = np.nan
    return pd.DataFrame(values, columns=x.columns, index=x.index)


def solve_intercept(eta: np.ndarray, event_rate: float) -> float:
    """Find the intercept that gives the requested event rate.

    Port of ``sa_sim_solve_intercept()``. Solved on the linear predictor that was
    actually drawn rather than on its expectation, so the rate the data comes out
    with is the rate that was asked for up to the Bernoulli draw itself. The
    logistic function is increasing in the intercept, so the root is unique and
    the only work is finding a bracket that contains it.

    Args:
        eta: Linear predictor without an intercept, one entry per row.
        event_rate: Target proportion of events, strictly between 0 and 1.
    """

    def gap(a: float) -> float:
        return float(np.mean(expit(a + eta))) - event_rate

    lower = -1.0
    upper = 1.0
    while gap(lower) > 0 and lower > -_BRACKET_LIMIT:
        lower *= 2
    while gap(upper) < 0 and upper < _BRACKET_LIMIT:
        upper *= 2
    if gap(lower) > 0 or gap(upper) < 0:
        raise SaValueError(
            f"no intercept gives an event rate of {fmt_num(event_rate)} on these "
            "predictors. A rate this far from a half needs a smaller `beta_range` "
            "or a smaller `subject_sd`."
        )
    return float(brentq(gap, lower, upper, xtol=_ROOT_TOL))


def supervised_design(
    *,
    n_samples: Any,
    n_pred: Any,
    beta: Any,
    n_pos: int | None,
    n_neg: int | None,
    beta_range: Any,
    value_mean: Any,
    value_sd: Any,
    cor_mat: Any,
    n_factor_pred: Any,
    factor_lv: Any,
    n_constant_pred: Any,
    p_missing: Any,
    n_per_subject: Any,
    subject_sd: Any,
    subject_share: Any,
    pred_prefix: Any,
    explicit: Collection[str],
    use_default_n: bool,
    rng: np.random.Generator,
) -> SupervisedDesign:
    """Build the predictors, the subjects and the linear predictor they imply.

    Port of ``sa_sim_supervised_design()``, everything the two supervised
    simulators share.

    The subject offset is inside ``eta`` rather than added to the outcome
    afterwards, so that in a classification it moves the probability of the class
    rather than the class itself. Either way it is the between-subject variation
    that makes a row-wise split leak: a subject seen in training is partly known
    before its test rows are read, and ``subject_sd`` is how much of it is.

    Factor predictors are drawn per subject rather than per row when there are
    subjects. A subject attribute is what a factor predictor usually is in a
    repeated-measures design, and it is also the only thing in the frame that can
    stratify a split taken over subjects, since a stratifier has to be constant
    within a unit.

    The numeric predictors are split between a subject level and a row level by
    ``subject_share``, which is what makes two rows of one subject resemble each
    other rather than merely share an outcome offset. Without that resemblance a
    row-wise split has nothing to give away: a model cannot recognise a subject
    it was trained on if its rows look like anyone else's. The two parts are
    drawn through the same correlation factor and their variances add to
    ``value_sd ** 2``, so ``subject_share`` moves the intraclass correlation of a
    column without moving its distribution.
    """
    spec = pred_spec(n_pred, beta, n_pos, n_neg, value_mean, value_sd, explicit)
    n_pred = spec.n_pred
    # Checked whether or not it planted the coefficients: the factor offsets are
    # drawn from it either way, so a rejected value must not depend on `beta`.
    beta_lo, beta_hi = check_range(beta_range, "beta_range", 0)
    root = cor_root(cor_mat, n_pred)

    n_factor_pred = check_count(n_factor_pred, "n_factor_pred")
    n_constant_pred = check_count(n_constant_pred, "n_constant_pred")
    p_missing = check_scalar_num(p_missing, "p_missing", 0, 1, upper_open=True)
    subject_sd = check_scalar_num(subject_sd, "subject_sd", 0)
    subject_share = check_scalar_num(subject_share, "subject_share", 0, 1)
    given = (
        list(factor_lv)
        if isinstance(factor_lv, Sequence) and not isinstance(factor_lv, str)
        else []
    )
    if n_factor_pred > 0 and (
        len(given) < 2
        or not all(isinstance(level, str) for level in given)
        or len(set(given)) != len(given)
    ):
        raise SaValueError(
            "`factor_lv` must be at least two distinct non-missing level names, "
            "the first being the reference."
        )
    levels = [str(level) for level in given]
    if not isinstance(pred_prefix, str) or not pred_prefix:
        raise SaValueError("`pred_prefix` must be a single non-empty string.")

    sizes, n_samples = subject_sizes(n_samples, n_per_subject, use_default_n)
    n_unit = n_samples if sizes is None else len(sizes)

    numeric_pred = [f"{pred_prefix}_{i + 1}" for i in range(n_pred)]
    factor_pred = [f"{pred_prefix}_cat_{i + 1}" for i in range(n_factor_pred)]
    constant_pred = [f"{pred_prefix}_const_{i + 1}" for i in range(n_constant_pred)]

    # Nothing above this line has drawn anything.
    planted = plant_beta(spec, (beta_lo, beta_hi), rng)
    if sizes is None:
        values = mvnorm(n_samples, spec.value_mean, spec.value_sd, root, rng)
    else:
        between = mvnorm(
            len(sizes), spec.value_mean, spec.value_sd * math.sqrt(subject_share), root, rng
        )
        within = mvnorm(
            n_samples, np.zeros(n_pred), spec.value_sd * math.sqrt(1 - subject_share), root, rng
        )
        values = np.repeat(between, sizes, axis=0) + within

    x = pd.DataFrame(values, columns=numeric_pred)
    eta = values @ planted.beta

    offsets: dict[str, dict[str, float]] = {}
    for name in factor_pred:
        level = balanced_levels(n_unit, levels, rng)
        if sizes is not None:
            level = pd.Categorical(
                np.repeat(np.asarray(level, dtype=object), sizes), categories=levels
            )
        offsets[name] = factor_offsets(levels, (beta_lo, beta_hi), rng)
        x[name] = level
        eta = eta + np.array([offsets[name][value] for value in level])

    for name in constant_pred:
        x[name] = np.ones(n_samples)

    subject = None
    subject_offset = np.zeros(n_samples)
    if sizes is not None:
        subject = [f"subject_{i + 1}" for i, size in enumerate(sizes) for _ in range(size)]
        subject_offset = np.repeat(rng.normal(0, subject_sd, len(sizes)), sizes)
    eta = eta + subject_offset

    x[numeric_pred] = mask_missing(x[numeric_pred], p_missing, rng).to_numpy()

    return SupervisedDesign(
        x=x,
        predictors=numeric_pred + factor_pred + constant_pred,
        numeric_pred=numeric_pred,
        factor_pred=factor_pred,
        constant_pred=constant_pred,
        beta=planted.beta,
        direction=planted.direction,
        offsets=offsets,
        eta=eta,
        subject=subject,
        subject_offset=subject_offset,
        n_samples=n_samples,
        sizes=sizes,
        truth=truth_pred(planted, spec, numeric_pred, factor_pred, constant_pred, cor_mat),
        truth_term=truth_term(planted.beta, numeric_pred, offsets),
    )


def truth_pred(
    planted: Planted,
    spec: PredSpec,
    numeric_pred: list[str],
    factor_pred: list[str],
    constant_pred: list[str],
    cor_mat: Any,
) -> pd.DataFrame:
    """Per predictor answer, and what it is up against.

    Port of ``sa_sim_truth_pred()``. ``max_cor_signal`` is the reason
    :func:`~statassist.simulate.make_block_cor` exists. A null predictor
    correlated with a planted one is the case where a coefficient of exactly zero
    is estimated well away from zero, and looking the correlation up accounts for
    that rather than leaving it as a false positive with no explanation.

    A factor predictor has one offset per level rather than one coefficient, so
    its ``beta`` is missing here and its answer is in :func:`truth_term`.
    """
    n_pred = spec.n_pred
    signal = set(np.flatnonzero(planted.beta != 0).tolist())
    cors = np.eye(n_pred) if cor_mat is None else np.abs(np.asarray(cor_mat, dtype=float))
    max_cor = np.array(
        [max((cors[i, j] for j in signal if j != i), default=0.0) for i in range(n_pred)]
    )

    n_other = len(factor_pred) + len(constant_pred)
    nan_other = [np.nan] * n_other
    return pd.DataFrame(
        {
            "predictors": numeric_pred + factor_pred + constant_pred,
            "role": ["null" if b == 0 else "signal" for b in planted.beta]
            + ["factor"] * len(factor_pred)
            + ["constant"] * len(constant_pred),
            "beta": list(planted.beta) + [np.nan] * len(factor_pred) + [0.0] * len(constant_pred),
            "direction": planted.direction
            + [None] * len(factor_pred)
            + ["none"] * len(constant_pred),
            "value_mean": list(spec.value_mean) + nan_other,
            "value_sd": list(spec.value_sd) + nan_other,
            "max_cor_signal": list(max_cor) + nan_other,
        }
    )


def truth_term(
    beta: np.ndarray,
    numeric_pred: list[str],
    offsets: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Per model term answer, in the row order the coefficient table follows.

    Port of ``sa_sim_truth_term()``. The predictors that were passed in and the
    terms that come back are not the same list: a factor with ``k`` levels
    becomes ``k - 1`` terms named after the level each stands for, and a
    predictor that takes one value becomes no term at all. So the table that
    scores the coefficients is built on the term axis rather than reindexed from
    the predictor axis afterwards, which is why ``truth_contrast`` exists beside
    ``truth`` in :func:`~statassist.simulate.simulate_multiple_groups`.

    The intercept row is added by :func:`add_intercept`, since a regression and a
    classification arrive at their intercept by different routes.
    """
    terms = list(numeric_pred)
    values = list(beta)
    predictors = list(numeric_pred)

    for name, offset in offsets.items():
        # A dummy column is named by pasting the level onto the column name, so
        # the term is predicted here rather than read back off a fit.
        beyond_reference = list(offset)[1:]
        terms += [f"{name}{level}" for level in beyond_reference]
        values += [offset[level] for level in beyond_reference]
        predictors += [name] * len(beyond_reference)

    return pd.DataFrame({"terms": terms, "predictors": predictors, "beta": values})


def add_intercept(terms: pd.DataFrame, intercept: float) -> pd.DataFrame:
    """Put the intercept at the top of the term answer.

    Port of ``sa_sim_add_intercept()``.
    """
    head = pd.DataFrame(
        {"terms": ["(Intercept)"], "predictors": [None], "beta": [float(intercept)]}
    )
    return pd.concat([head, terms], ignore_index=True)


def split_args(
    data: pd.DataFrame,
    design: SupervisedDesign,
    stratify_outcome: bool,
) -> dict[str, Any]:
    """What a split of this data set should be told to preserve.

    Port of ``sa_sim_split_args()``. A stratifier has to be constant within a
    sampling unit, since a unit goes to one side of the split as a whole. That
    rules the outcome out for a regression measured repeatedly, because it varies
    from row to row within a subject, and the subject-level factor predictor is
    what is left. A classification has no such problem: a subject is a case or a
    control as a whole.

    Args:
        data: The frame the split is to be taken of.
        design: The design it was built from.
        stratify_outcome: Whether the outcome is constant within a subject.
    """
    if stratify_outcome or design.subject is None:
        stratified: str | None = "y"
    elif design.factor_pred:
        stratified = design.factor_pred[0]
    else:
        stratified = None

    return {
        "data": data,
        "stratified": stratified,
        "id": None if design.subject is None else "subject",
    }
