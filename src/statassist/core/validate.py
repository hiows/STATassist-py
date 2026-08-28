"""Argument validation, shared by every public function in the package.

The port of ``R/utils_validate.R``. Every check runs at the boundary, before
anything is computed, so a bad call fails where it was made rather than halfway
through a feature loop.

Three R behaviours are reproduced deliberately, because the rest of the package
was written against them:

``is.numeric(TRUE)`` is ``FALSE`` in R
    A logical is not a number there. In Python :class:`bool` is a subclass of
    :class:`int`, so it has to be rejected by name or ``check_count(True)``
    would quietly return ``1``.

A length-one vector is a scalar in R
    ``sa_check_scalar_num(c(0.5))`` is a valid call, so a one-element array,
    list or Series is unwrapped here rather than refused.

``is.na(NaN)`` is ``TRUE`` in R
    So ``NaN`` is refused wherever R refuses a missing value, while ``Inf`` is
    only refused where R actually tests ``is.finite``. ``check_scalar_num`` and
    ``check_margin`` accept ``Inf``; ``check_count``, ``check_num_vector``,
    ``check_range`` and ``check_lim`` do not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .errors import SaValueError
from .padjust import P_ADJUST_METHODS

__all__ = [
    "UNSET",
    "Alignment",
    "Pairing",
    "RowVector",
    "WideInput",
    "align_by_subject",
    "check_count",
    "check_feat_names",
    "check_flag",
    "check_lim",
    "check_margin",
    "check_num_vector",
    "check_p_adjust",
    "check_pvalues",
    "check_range",
    "check_scalar_num",
    "control_first",
    "fmt_est",
    "fmt_num",
    "pair_by_id",
    "pair_by_order",
    "resolve_row_vector",
    "validate_wide_input",
]


class _Unset:
    """The type of :data:`UNSET`. One instance, so ``is`` is the test."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


#: Default for an argument that has to know whether it was supplied.
#:
#: R answers that with ``missing(arg)``, and several functions need the answer
#: for an argument whose default is a real value rather than ``NULL``:
#: ``simulate_multiple_groups()`` spreads a default ``n_treat`` over as many
#: labels as ``group_lv`` names but refuses a supplied one that counts
#: differently, and ``simulate_regression()`` refuses a supplied ``n_pred`` that
#: disagrees with ``beta`` while accepting the default. ``None`` cannot carry
#: that distinction where it is already a meaningful value, so this sentinel does.
UNSET: Any = _Unset()


# --------------------------------------------------------------------------- #
# Internal coercion and formatting
# --------------------------------------------------------------------------- #


def fmt_num(value: Any) -> str:
    """Render a number the way R's ``as.character()`` does.

    R prints a whole-number double without a decimal point, so ``[0, 1]`` rather
    than ``[0.0, 1.0]``, and spells the infinities ``Inf`` and ``-Inf``. The
    error messages are carried over verbatim from R, so the numbers in them are
    too, which is why this is shared rather than private to the checks.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


#: Significant digits an estimate is reported to. R's ``format()`` default.
_EST_DIGITS = 3


def fmt_est(value: Any, digits: int = _EST_DIGITS) -> str:
    """Render an estimate the way R's ``format(x, digits = 3, trim = TRUE)`` does.

    Port of ``sa_fmt_est()``. Different from :func:`fmt_num`, which is R's
    ``as.character()`` and keeps every digit: this one is what a number reads as
    in a sentence a person is meant to read, and the sentences it appears in are
    carried over from R verbatim.

    R's rule is not "round to three decimals". It is three *significant* digits,
    never fewer than the integer part needs, with the trailing zeros dropped and
    with scientific notation chosen only when it is strictly shorter. So a count
    of ``4.16666`` reads as ``4.17``, one of ``12345.7`` reads as ``12346``
    rather than as ``12300``, and one of ``0.0000123`` turns the corner into
    ``1.23e-05``.

    ``digits`` is what R's ``sa_fmt_num(x, digits)`` takes, for the callers whose
    numbers are not estimates: a criterion sits on the scale of the row count
    while the differences that decided a search are of order one, so it needs
    more digits than a coefficient does.

    >>> [fmt_est(x) for x in (4.16666, 12345.678, 2.5, 5.0, 0.0)]
    ['4.17', '12346', '2.5', '5', '0']
    >>> [fmt_est(x) for x in (0.000123456, 0.0000123456, float("nan"), None)]
    ['0.000123', '1.23e-05', 'NA', 'NA']
    >>> [fmt_est(158.6931, digits) for digits in (3, 6)]
    ['159', '158.693']
    """
    if value is None or _is_na(value):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    if number == 0:
        return "0"

    decimals = max(0, digits - 1 - math.floor(math.log10(abs(number))))
    fixed = f"{number:.{decimals}f}"
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")

    mantissa, _, exponent = f"{number:.{digits - 1}e}".partition("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    scientific = f"{mantissa}e{exponent[0]}{int(exponent[1:]):02d}"

    # R weighs the two against each other by width and keeps the fixed one on a
    # tie, which is what ``options(scipen = 0)`` means.
    return fixed if len(fixed) <= len(scientific) else scientific


def _is_bool(value: Any) -> bool:
    """Whether the value is a logical, which R does not count as a number."""
    return isinstance(value, bool | np.bool_)


def _scalar_number(value: Any) -> float | None:
    """Read a value as R would read a length-one numeric vector.

    Returns ``None`` when the value is not one, which the callers turn into
    their own message. A one-element container is unwrapped, since that is a
    scalar in R; a longer one is not.
    """
    if isinstance(value, str | bytes):
        return None

    if isinstance(value, np.ndarray):
        if value.dtype == bool or value.size != 1:
            return None
        value = value.reshape(-1)[0]
    elif isinstance(value, Sequence | set | frozenset):
        items = list(value)
        if len(items) != 1:
            return None
        value = items[0]
    elif hasattr(value, "to_numpy"):  # pandas Series / Index
        array = np.asarray(value)
        if array.dtype == bool or array.size != 1:
            return None
        value = array.reshape(-1)[0]

    if _is_bool(value):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return None


def _float_array(value: Any) -> np.ndarray | None:
    """Read a value as R would read a numeric vector, or return ``None``.

    A logical vector is refused, matching ``is.numeric()``. The result is always
    a fresh, writable ``float64`` array, so a caller that goes on to mutate it
    is not handed a read-only view of somebody's DataFrame column.
    """
    if isinstance(value, str | bytes):
        return None
    if _is_bool(value):
        return None

    if isinstance(value, int | float | np.integer | np.floating):
        return np.array([float(value)], dtype=float)

    array = np.asarray(value)
    if array.dtype == bool:
        return None
    if array.dtype.kind in "iuf":
        # copy=True: the callers hand this on, and a view of a DataFrame column
        # can be read-only.
        return np.array(array, dtype=float, copy=True).reshape(-1)
    if array.dtype.kind == "O":
        # A plain list of numbers holding a missing one, which is what R's `NA`
        # in a numeric vector looks like coming from Python. Refused if anything
        # in it is not a number.
        try:
            return np.array(
                [np.nan if _is_na(item) else float(item) for item in array.reshape(-1)],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None
    return None


def _is_na(value: Any) -> bool:
    """Whether a single value stands for R's ``NA``.

    ``None``, ``nan``, ``pd.NA`` and ``pd.NaT`` all count.
    """
    if value is None:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, bool | np.bool_) else False


# --------------------------------------------------------------------------- #
# Scalar and vector argument checks
# --------------------------------------------------------------------------- #


def check_flag(x: Any, arg: str) -> bool:
    """Check a length-one logical argument.

    Port of ``sa_check_flag()``. ``0`` and ``1`` are refused: R's ``is.logical``
    does not accept a number, and accepting one here would let a mistyped
    argument through.
    """
    if isinstance(x, np.ndarray) and x.dtype == bool and x.size == 1:
        x = bool(x.reshape(-1)[0])
    if not _is_bool(x):
        raise SaValueError(f"`{arg}` must be TRUE or FALSE.")
    return bool(x)


def check_scalar_num(
    x: Any,
    arg: str,
    lower: float = -math.inf,
    upper: float = math.inf,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    """Check a length-one numeric argument against a range.

    Port of ``sa_check_scalar_num()``. ``lower_open`` / ``upper_open`` make the
    corresponding bound exclusive.

    ``Inf`` is not refused here, only by the bounds. That is R's behaviour, and
    ``check_count()`` is what rules it out where a whole number is wanted.
    """
    value = _scalar_number(x)
    if value is None or math.isnan(value):
        raise SaValueError(f"`{arg}` must be a single non-missing number.")

    too_low = value <= lower if lower_open else value < lower
    too_high = value >= upper if upper_open else value > upper
    if too_low or too_high:
        raise SaValueError(
            f"`{arg}` must be in "
            f"{'(' if lower_open else '['}{fmt_num(lower)}, {fmt_num(upper)}"
            f"{')' if upper_open else ']'}, but is {fmt_num(value)}."
        )
    return value


def check_count(x: Any, arg: str, lower: float = 0) -> int:
    """Check a length-one whole number argument.

    Port of ``sa_check_count()``. Sample sizes and feature counts read as
    integers but arrive as doubles from a literal like ``100``, so the
    whole-number requirement is checked on the value rather than on the type.
    """
    value = check_scalar_num(x, arg, lower)
    if not math.isfinite(value) or value != math.trunc(value):
        raise SaValueError(f"`{arg}` must be a finite whole number, but is {fmt_num(value)}.")
    return int(value)


def check_num_vector(
    x: Any,
    arg: str,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> np.ndarray:
    """Check a numeric vector argument against inclusive bounds.

    Port of ``sa_check_num_vector()``. A hyperparameter grid is a vector rather
    than a scalar, and the whole of it is checked at once so that the offending
    values can be named: a grid of fifty values is not something to bisect by
    hand.
    """
    array = _float_array(x)
    if array is None or array.size == 0 or not np.isfinite(array).all():
        raise SaValueError(f"`{arg}` must be a non-empty numeric vector of finite values.")

    outside = array[(array < lower) | (array > upper)]
    if outside.size > 0:
        # R's unique() keeps first appearance order.
        _, first = np.unique(outside, return_index=True)
        bad = outside[np.sort(first)]
        raise SaValueError(
            f"`{arg}` must be in [{fmt_num(lower)}, {fmt_num(upper)}], but holds "
            + ", ".join(fmt_num(v) for v in bad)
            + "."
        )
    return array


def check_range(x: Any, arg: str, lower: float = -math.inf) -> tuple[float, float]:
    """Check a length-two increasing range argument.

    Port of ``sa_check_range()``. The two ends go on to a uniform draw, which
    takes a reversed pair without complaint and draws from it anyway, so this is
    the only place the caller finds out that the range they wrote is not the
    range they get.
    """
    array = _float_array(x)
    if array is None or array.size != 2 or not np.isfinite(array).all():
        raise SaValueError(f"`{arg}` must be a finite numeric vector of length 2.")

    low, high = float(array[0]), float(array[1])
    if low > high:
        raise SaValueError(
            f"`{arg}` must be increasing, but is c({fmt_num(low)}, {fmt_num(high)})."
        )
    if low < lower:
        raise SaValueError(
            f"`{arg}` must not go below {fmt_num(lower)}, but starts at {fmt_num(low)}."
        )
    return low, high


def check_margin(x: Any, arg: str = "margin") -> tuple[float, float, float, float]:
    """Check a plot margin argument.

    Port of ``sa_check_margin()``. Four non-negative values; R tests ``anyNA``
    rather than ``is.finite`` here, so an infinite margin is not refused.
    """
    array = _float_array(x)
    invalid = (
        array is None or array.size != 4 or bool(np.isnan(array).any()) or bool((array < 0).any())
    )
    if invalid:
        raise SaValueError(f"`{arg}` must be a numeric vector of 4 non-negative values.")
    assert array is not None  # narrowed by `invalid`
    return float(array[0]), float(array[1]), float(array[2]), float(array[3])


def check_lim(x: Any, arg: str) -> tuple[float, float] | None:
    """Check an optional axis range argument.

    Port of ``sa_check_lim()``. ``None`` means the range is derived from the
    data, so it is always accepted.
    """
    if x is None:
        return None
    array = _float_array(x)
    if array is None or array.size != 2 or not np.isfinite(array).all():
        raise SaValueError(f"`{arg}` must be NULL or a finite numeric vector of length 2.")
    return float(array[0]), float(array[1])


def check_p_adjust(x: Any, arg: str) -> str:
    """Check a multiplicity adjustment method name.

    Port of ``sa_check_p_adjust()``. Validated against the one list of methods
    the package holds, rather than a hand-written set of choices per call site:
    a misspelled entry in such a list once got through here and was refused much
    later, by the adjustment itself.
    """
    if not isinstance(x, str) or x not in P_ADJUST_METHODS:
        raise SaValueError(f"`{arg}` must be one of: " + ", ".join(P_ADJUST_METHODS) + ".")
    return x


def check_pvalues(pvalue: Any, arg: str = "pvalue") -> np.ndarray:
    """Check a vector of p-values.

    Port of ``sa_check_pvalues()``. A missing p-value is allowed, because a
    feature the test could not be run on has one; an infinite or out-of-range
    one is not.

    The positions in the message are zero-based, unlike R's, so that they index
    the array the caller passed in.
    """
    array = _float_array(pvalue)
    if array is None:
        raise SaValueError(f"`{arg}` must be a numeric vector.")

    present = ~np.isnan(array)
    bad = np.flatnonzero(present & (np.isinf(array) | (array < 0) | (array > 1)))
    if bad.size > 0:
        shown = ", ".join(str(int(i)) for i in bad[:5])
        ellipsis = ", ..." if bad.size > 5 else ""
        raise SaValueError(f"`{arg}` must lie in [0, 1]. Offending position(s): {shown}{ellipsis}.")
    return array


def check_feat_names(feats: Any) -> list[str]:
    """Check that ``feats`` is a usable vector of feature names.

    Port of ``sa_check_feat_names()``. Shared by the functions that take ``data``
    and by those that only take per-feature vectors, so that a feature name is
    refused for the same reasons everywhere.

    A bare string is read as one feature name, which is what a length-one
    character vector means in R. Iterating its characters instead would be an
    accident of Python's sequence protocol, not a behaviour to reproduce.
    """
    if isinstance(feats, str):
        feats = [feats]

    if feats is None or isinstance(feats, bytes):
        raise SaValueError("`feats` must be a non-empty character vector of feature names.")

    try:
        items = list(feats)
    except TypeError:
        raise SaValueError(
            "`feats` must be a non-empty character vector of feature names."
        ) from None

    if len(items) == 0:
        raise SaValueError("`feats` must be a non-empty character vector of feature names.")
    if any(_is_na(item) for item in items):
        raise SaValueError("`feats` must not contain NA.")
    if not all(isinstance(item, str | np.str_) for item in items):
        raise SaValueError("`feats` must be a non-empty character vector of feature names.")

    names = [str(item) for item in items]
    seen: set[str] = set()
    duplicated: list[str] = []
    for name in names:
        if name in seen and name not in duplicated:
            duplicated.append(name)
        seen.add(name)
    if duplicated:
        raise SaValueError("`feats` contains duplicated names: " + ", ".join(duplicated))
    return names


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #


class RowVector(NamedTuple):
    """A resolved argument that describes the rows of ``data``.

    Attributes:
        value: The vector itself, or ``None`` when the argument was not given.
        label: The column name it came from, ``"<vector>"`` when it arrived as a
            vector of its own, or ``None`` when the argument was not given. R
            spells the last case ``NA_character_``; a result object records this
            so it can report what the analysis was made on, and a resolved
            vector no longer remembers whether it arrived as a column or not.
    """

    value: pd.Series | None
    label: str | None


class WideInput(NamedTuple):
    """Validated wide-format input, with the unusable rows already gone.

    Attributes:
        data: The row-filtered frame, re-indexed from zero.
        feats: The feature names, in the order they were given.
        group: The grouping as a Categorical whose categories are ``group_lv``,
            in display order, or ``None`` when the input was ungrouped.
        id: The pairing key as strings, filtered alongside ``data``, or ``None``.
        n_dropped: How many rows were removed for belonging to a level outside
            ``group_lv``.
    """

    data: pd.DataFrame
    feats: list[str]
    group: pd.Categorical | None
    id: list[str] | None
    n_dropped: int


def _as_frame(data: Any) -> pd.DataFrame:
    """Read ``data`` as a wide frame, the way R accepts a data.frame or matrix."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return pd.DataFrame(data)
    raise SaValueError("`data` must be a data.frame or a matrix.")


def _as_str_or_none(values: Any) -> list[str | None]:
    """R's ``as.character()``, keeping a missing value missing.

    ``str(float('nan'))`` is ``'nan'``, which would then match a group level
    literally called ``"nan"``. Missing entries are carried through as ``None``
    instead, so they simply fail every membership test, which is what
    ``as.character(NA)`` does in R.
    """
    out: list[str | None] = []
    for value in values:
        out.append(None if _is_na(value) else str(value))
    return out


def _row_count(value: Any) -> int:
    """Length of a row-aligned argument."""
    if isinstance(value, pd.DataFrame):
        return len(value.index)
    return len(np.asarray(value).reshape(-1)) if not isinstance(value, Sequence) else len(value)


def resolve_row_vector(
    x: Any,
    arg: str,
    data: pd.DataFrame,
    allow_na: bool = False,
) -> RowVector:
    """Resolve an argument that names something about the rows.

    Port of ``sa_resolve_row_vector()``. ``stratified``, ``id`` and ``outcome``
    all describe the rows of ``data``. A vector is the form the rest of the
    package takes, and a column name is the form that does not repeat the object
    name. A length-one string matching a column is read as that column; anything
    else is read as a vector of its own.

    Args:
        x: The argument as received, or ``None``.
        arg: What to call it in an error message.
        data: The frame its length is measured against.
        allow_na: Whether a missing entry is acceptable. It is not when the
            argument decides where a row goes, and it is when the row will be
            dropped by the listwise deletion the model input goes through anyway.
    """
    if x is None:
        return RowVector(value=None, label=None)

    label = "<vector>"
    if isinstance(x, str) and x in data.columns:
        label = x
        x = data[x]

    series = x if isinstance(x, pd.Series) else pd.Series(list(x) if _is_iterable(x) else [x])
    if len(series) != len(data.index):
        raise SaValueError(
            f"`{arg}` must name a column of `data` or hold one entry per row of it: "
            f"got {len(series)} for {len(data.index)} row(s)."
        )
    if not allow_na and bool(series.isna().any()):
        raise SaValueError(
            f"`{arg}` must not contain NA: a row it does not describe cannot be "
            "assigned to a side of the split."
        )
    return RowVector(value=series.reset_index(drop=True), label=label)


def _is_iterable(value: Any) -> bool:
    """Whether the value is a vector rather than a single entry."""
    if isinstance(value, str | bytes):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def validate_wide_input(
    data: Any,
    feats: Any,
    group: Any,
    group_lv: Any,
    id: Any = None,  # noqa: A002 - matches the R argument name
    n_levels: int | None = None,
    min_levels: int = 2,
) -> WideInput:
    """Validate wide-format grouped input.

    Port of ``sa_validate_wide_input()``, the single entry point twelve public
    functions share.

    Rows whose group falls outside ``group_lv`` are **dropped** rather than
    coerced to missing: leaving them in as ``NaN`` would inject missing values
    into the test samples through what looks like a plain boolean mask.

    Args:
        data: Wide frame, one row per observation.
        feats: Names of the numeric columns to analyse.
        group: Grouping vector, one entry per row of ``data``.
        group_lv: Group levels to keep, in display order.
        id: Optional pairing key, filtered alongside ``data``.
        n_levels: If given, ``group_lv`` must hold exactly this many levels.
        min_levels: Fewest levels a grouped analysis will accept.

    Returns:
        A :class:`WideInput`. Its ``data`` is re-indexed from zero, unlike R
        which leaves the original row names in place. Nothing downstream reads
        those names, and a gapped index would make a positional group vector and
        a label-indexed column silently misalign under pandas' auto-alignment.
    """
    frame = _as_frame(data)
    n_rows = len(frame.index)
    if n_rows == 0:
        raise SaValueError("`data` has zero rows.")

    names = check_feat_names(feats)
    unknown = [name for name in names if name not in frame.columns]
    if unknown:
        raise SaValueError("`feats` not found in `data`: " + ", ".join(unknown))
    # is.numeric() is FALSE for a logical in R, and pandas counts bool as numeric.
    non_numeric = [
        name
        for name in names
        if not pd.api.types.is_numeric_dtype(frame[name]) or frame[name].dtype == bool
    ]
    if non_numeric:
        raise SaValueError(
            "`feats` must refer to numeric columns. Not numeric: " + ", ".join(non_numeric)
        )

    if group is None and group_lv is None:
        if id is not None and _row_count(id) != n_rows:
            raise SaValueError(
                f"`id` must have one entry per row of `data`: got {_row_count(id)} "
                f"for {n_rows} rows."
            )
        return WideInput(
            data=frame.reset_index(drop=True),
            feats=names,
            group=None,
            id=None if id is None else [str(v) for v in _as_str_or_none(id)],
            n_dropped=0,
        )
    if group is None or group_lv is None:
        raise SaValueError("`group` and `group_lv` must both be supplied or both be `None`.")

    if _row_count(group) != n_rows:
        raise SaValueError(
            f"`group` must have one entry per row of `data`: got {_row_count(group)} "
            f"for {n_rows} rows."
        )
    group_chr = _as_str_or_none(group)

    if id is not None and _row_count(id) != n_rows:
        raise SaValueError(
            f"`id` must have one entry per row of `data`: got {_row_count(id)} for {n_rows} rows."
        )

    levels_in = list(group_lv) if _is_iterable(group_lv) else [group_lv]
    if len(levels_in) == 0:
        raise SaValueError("`group_lv` must be a non-empty vector of group levels.")
    if any(_is_na(level) for level in levels_in):
        raise SaValueError("`group_lv` must not contain NA.")
    levels = [str(level) for level in levels_in]

    seen: set[str] = set()
    duplicated: list[str] = []
    for level in levels:
        if level in seen and level not in duplicated:
            duplicated.append(level)
        seen.add(level)
    if duplicated:
        raise SaValueError("`group_lv` contains duplicated levels: " + ", ".join(duplicated))

    if n_levels is not None and len(levels) != n_levels:
        raise SaValueError(
            f"`group_lv` must contain exactly {n_levels} levels, but {len(levels)} "
            "were given: " + ", ".join(levels)
        )
    if len(levels) < min_levels:
        raise SaValueError(f"`group_lv` must contain at least {min_levels} levels.")

    observed = set(group_chr)
    absent = [level for level in levels if level not in observed]
    if absent:
        raise SaValueError("`group_lv` level(s) absent from `group`: " + ", ".join(absent))

    wanted = set(levels)
    keep = np.array([value in wanted for value in group_chr], dtype=bool)
    n_dropped = int((~keep).sum())

    ids: list[str] | None = None
    if id is not None:
        id_chr = _as_str_or_none(id)
        ids = [str(value) for value, take in zip(id_chr, keep, strict=True) if take]

    return WideInput(
        data=frame.loc[keep].reset_index(drop=True),
        feats=names,
        group=pd.Categorical(
            [value for value, take in zip(group_chr, keep, strict=True) if take],
            categories=levels,
        ),
        id=ids,
        n_dropped=n_dropped,
    )


def control_first(
    group_lv: Any,
    control_label: Any,
    arg: str = "control_label",
    lv_arg: str = "group_lv",
) -> list[str]:
    """Move the named level to the front of the display order.

    Port of ``sa_control_first()``. ``group_lv`` is a required argument of every
    comparison, so the reference is already stated by its first element and
    ``control_label`` cannot be a way of stating it for the first time. What it
    is instead is a way of re-pointing it: the named level moves to the front and
    the rest keep the order they were given.

    Args:
        group_lv: Group levels in display order, already validated.
        control_label: The level to hold as the reference, or ``None`` to leave
            the order as it arrived.
        arg: What to call ``control_label`` in an error message.
        lv_arg: What to call ``group_lv`` in an error message. A crossed design
            names one reference per factor, so the two messages can name the
            element that was wrong (``control_label['sex']``) rather than the
            whole argument.
    """
    levels = [str(level) for level in (group_lv if _is_iterable(group_lv) else [group_lv])]
    if control_label is None:
        return levels

    label_items = [control_label] if not _is_iterable(control_label) else list(control_label)
    if len(label_items) != 1 or _is_na(label_items[0]):
        raise SaValueError(
            f"`{arg}` must be a single level name, the one to hold as the reference."
        )
    label = str(label_items[0])

    if label not in levels:
        raise SaValueError(
            f"`{arg}` names a level `{lv_arg}` does not hold: {label}. "
            "Present: " + ", ".join(levels) + "."
        )
    return [label] + [level for level in levels if level != label]


# --------------------------------------------------------------------------- #
# Pairing and alignment
#
# Every index these three return is a zero-based row position, where R's
# `which()` returns a one-based one. Nothing else about them changes: which rows
# pair with which, and the order they come out in, is the same.
# --------------------------------------------------------------------------- #


class Pairing(NamedTuple):
    """Two groups matched into pairs.

    Attributes:
        idx_x: Zero-based row positions of the first group, in pair order.
        idx_y: Zero-based row positions of the second group, in pair order.
        unmatched: Pairing keys that appeared in only one group.
    """

    idx_x: np.ndarray
    idx_y: np.ndarray
    unmatched: list[str]


class Alignment(NamedTuple):
    """Three or more repeated conditions lined up by subject.

    Attributes:
        idx: Subjects by conditions, holding zero-based row positions. The index
            is the subject and the columns follow ``group_lv``. R returns a
            matrix with dimnames; a frame is what carries the same two labels in
            pandas.
        subjects: The complete subjects, in the order they first appear.
        unmatched: Subjects dropped for missing at least one condition.
    """

    idx: pd.DataFrame
    subjects: list[str]
    unmatched: list[str]


def _group_positions(group: Any, level: str) -> np.ndarray:
    """R's ``which(group == level)``, as zero-based positions."""
    values = _as_str_or_none(group)
    return np.flatnonzero(np.array([value == level for value in values], dtype=bool))


def _repeated(values: Sequence[str]) -> list[str]:
    """Values that occur more than once, in the order they first repeat."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [value for value, count in counts.items() if count > 1]


def pair_by_order(group: Any, group_lv: Sequence[str]) -> Pairing:
    """Pair up two groups by row order.

    Port of ``sa_pair_by_order()``. Used when no ``id`` is supplied: order is all
    the information available, so unequal group sizes mean the pairs cannot be
    formed at all.

    ``group_lv`` arrives in ``x``, ``y`` order rather than in the display order
    the user supplied, since the caller has already resolved which level is the
    reference. Which side a level lands on does not change the pairs.
    """
    levels = [str(level) for level in group_lv]
    idx_x = _group_positions(group, levels[0])
    idx_y = _group_positions(group, levels[1])
    if idx_x.size != idx_y.size:
        raise SaValueError(
            "`paired = True` without `id` pairs observations by row order, which "
            "requires the same number of rows per group. Got "
            f"{levels[0]} = {idx_x.size}, {levels[1]} = {idx_y.size}. "
            "Supply `id` to match on a pairing key instead."
        )
    return Pairing(idx_x=idx_x, idx_y=idx_y, unmatched=[])


def pair_by_id(id: Any, group: Any, group_lv: Sequence[str]) -> Pairing:  # noqa: A002
    """Pair up two groups by an explicit pairing key.

    Port of ``sa_pair_by_id()``. Row order carries no meaning here, so a
    reordered or partially incomplete data set still yields the correct pairs.
    Pairs come out in the row order of ``group_lv[0]``, so the result is
    deterministic.
    """
    levels = [str(level) for level in group_lv]
    keys = _as_str_or_none(id)
    if any(key is None for key in keys):
        raise SaValueError("`id` must not contain NA when it is used to form pairs.")

    idx_x = _group_positions(group, levels[0])
    idx_y = _group_positions(group, levels[1])
    id_x = [str(keys[i]) for i in idx_x]
    id_y = [str(keys[i]) for i in idx_y]

    repeated = list(dict.fromkeys(_repeated(id_x) + _repeated(id_y)))
    if repeated:
        raise SaValueError(
            "`id` must be unique within each group, otherwise the pairing is "
            "ambiguous. Repeated id(s): " + ", ".join(repeated) + "."
        )

    in_y = set(id_y)
    common = [key for key in dict.fromkeys(id_x) if key in in_y]
    if len(common) < 2:
        raise SaValueError(
            f"only {len(common)} id(s) appear in both `{levels[0]}` and "
            f"`{levels[1]}`; at least 2 pairs are needed."
        )

    where_x = {key: position for position, key in enumerate(id_x)}
    where_y = {key: position for position, key in enumerate(id_y)}
    matched = set(common)
    return Pairing(
        idx_x=idx_x[[where_x[key] for key in common]],
        idx_y=idx_y[[where_y[key] for key in common]],
        unmatched=[key for key in dict.fromkeys(id_x + id_y) if key not in matched],
    )


def align_by_subject(id: Any, group: Any, group_lv: Sequence[str]) -> Alignment:  # noqa: A002
    """Line up three or more repeated conditions by subject.

    Port of ``sa_align_by_subject()``, the many-condition counterpart of
    :func:`pair_by_id`. A within-subject omnibus test needs a complete rectangle,
    so subjects missing any condition are dropped rather than partially used, and
    the caller is told how many went.

    Row-order pairing is deliberately not offered here. With two groups it is at
    least well defined; with ``k`` conditions it would also have to assume the
    groups are stored in the same subject order, and there is no way to notice
    when they are not.
    """
    levels = [str(level) for level in group_lv]
    keys = _as_str_or_none(id)
    if any(key is None for key in keys):
        raise SaValueError("`id` must not contain NA when it is used to align conditions.")

    per_level: dict[str, dict[str, int]] = {}
    for level in levels:
        positions = _group_positions(group, level)
        ids = [str(keys[i]) for i in positions]
        repeated = _repeated(ids)
        if repeated:
            raise SaValueError(
                "`id` must be unique within each condition, otherwise the design "
                f"is ambiguous. Repeated id(s) in `{level}`: " + ", ".join(repeated) + "."
            )
        per_level[level] = {
            name: int(position) for name, position in zip(ids, positions, strict=True)
        }

    # First appearance across the levels in `group_lv` order, so a numeric id is
    # not silently reordered as text.
    all_ids: list[str] = []
    for level in levels:
        for name in per_level[level]:
            if name not in all_ids:
                all_ids.append(name)

    complete = [name for name in all_ids if all(name in per_level[level] for level in levels)]
    if len(complete) < 2:
        raise SaValueError(
            f"only {len(complete)} subject(s) have all {len(levels)} condition(s); "
            "at least 2 complete subjects are needed."
        )

    idx = pd.DataFrame(
        {level: [per_level[level][name] for name in complete] for level in levels},
        index=complete,
        columns=levels,
        dtype=int,
    )
    return Alignment(
        idx=idx,
        subjects=complete,
        unmatched=[name for name in all_ids if name not in set(complete)],
    )
