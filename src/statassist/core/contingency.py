"""Resolving categorical input, laying it out as a table, and what it was expected to hold.

The port of ``R/utils_categorical.R``. :func:`validate_wide_input` cannot serve
here: it requires every named column to be numeric, which is exactly the
requirement this scenario inverts. What it can share is everything after the
columns are read - the level ordering rules, the ``control_label`` reordering, the
row dropping and the counts that report it - so those come from
:func:`~statassist.core.validate.control_first` and
:func:`~statassist.core.factorial.fact_control_first` rather than being written a
third time.

The expected counts are built by one function per null hypothesis rather than by
one function with a branch, because "expected" is not a property of a table. It is
a property of a table and a claim about it, and the three claims this scenario
tests make three different numbers out of the same counts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .contracts import categorical_cell_columns, categorical_nulls
from .errors import SaInternalError, SaValueError
from .factorial import fact_control_first
from .validate import check_count, control_first, fmt_est

__all__ = [
    "DISCORDANT_PAIR_MIN",
    "EXPECTED_COUNT_FLOOR",
    "EXPECTED_COUNT_MIN",
    "EXPECTED_LT5_MAX_PROP",
    "MAX_CATEGORY_LEVELS",
    "REPEATED_CELL_MIN",
    "TABLE_AXIS_NAMES",
    "CategoricalInput",
    "categorical_cells",
    "categorical_condition_counts",
    "categorical_counts",
    "categorical_shared_lv",
    "categorical_table",
    "check_level_count",
    "diagnose_discordance",
    "diagnose_expected",
    "diagnose_repeated",
    "expected_independence",
    "expected_symmetry",
    "finite_or_na",
    "validate_categorical_input",
]

#: How many levels a variable may take before it is refused as a category.
#:
#: Reading a column as labels turns a continuous measurement into one label per
#: observation, and a table with a cell per observation is not a thing a test of
#: association has anything to say about. This is a default rather than a
#: constant, because a genuinely many-levelled category exists and the caller is
#: the one who knows whether theirs is one.
MAX_CATEGORY_LEVELS = 20


class CategoricalInput(NamedTuple):
    """Validated categorical input, with the unusable rows already gone.

    The counterpart of :class:`~statassist.core.validate.WideInput` for a design
    whose columns are the answers rather than the measurements.

    Attributes:
        data: The row-filtered frame holding only the used columns, each a
            :class:`pandas.Categorical` at its resolved levels, re-indexed from
            zero.
        variables: The variable names, in the order the analysis reads them.
        category_lv: The resolved levels of each variable, reference first.
        n_used: How many rows hold a level of every variable.
        n_dropped: How many rows were removed for naming a level outside
            ``category_lv``.
        n_incomplete: How many rows were removed for missing a value.
    """

    data: pd.DataFrame
    variables: list[str]
    category_lv: dict[str, list[str]]
    n_used: int
    n_dropped: int
    n_incomplete: int


def _as_frame(data: Any) -> pd.DataFrame:
    """Read ``data`` as a wide frame, the way R accepts a data.frame or matrix."""
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        frame = pd.DataFrame(data)
    else:
        raise SaValueError("`data` must be a data.frame or a matrix.")
    return frame.rename(columns=str)


def _is_missing(value: Any) -> bool:
    """Whether a single entry stands for R's ``NA``."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _as_labels(values: Iterable[Any]) -> list[str | None]:
    """R's ``as.character()`` on a column, keeping a missing value missing.

    A factor arrives as its labels and a 0/1 coded variable as ``"0"`` and
    ``"1"``, so both are categorical here and neither keeps a storage mode the
    table would have to remember. ``str(float("nan"))`` is ``"nan"``, which would
    then match a level literally called ``"nan"``, so a missing entry is carried
    through as ``None`` and simply fails every membership test.
    """
    return [None if _is_missing(value) else str(value) for value in values]


def _named_levels(value: Any, arg: str) -> list[str]:
    """Read one variable's levels, refusing what R's checks refuse."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise SaValueError(f"`{arg}` must hold that variable's levels.")
    levels = _as_labels(value)
    if len(levels) < 2 or any(level is None for level in levels) or len(set(levels)) != len(levels):
        raise SaValueError(f"`{arg}` must hold at least two distinct non-missing levels.")
    return [str(level) for level in levels]


def check_level_count(
    category_lv: Mapping[str, Sequence[str]],
    max_levels: Any,
    named_lv: bool,
) -> None:
    """Refuse a variable that is a measurement rather than a category.

    Port of ``sa_check_level_count()``. Reading a column as labels is what lets a
    factor, a logical and a 0/1 code all count as categorical. It also turns a
    continuous measurement into one label per observation, and a table with a cell
    per observation is not a thing a test of association has anything to say
    about: every expected count is a fraction, Fisher's enumeration does not
    finish, and the answer that comes back is arithmetic on noise.

    So there is a ceiling, and it is an argument rather than a constant, because a
    genuinely many-levelled category exists and the caller is the one who knows
    whether theirs is one.

    Args:
        category_lv: The resolved levels of each variable.
        max_levels: The ceiling, checked here rather than by the caller.
        named_lv: Whether the levels were named rather than taken from the data,
            which decides whether naming them is offered as a way through.

    Raises:
        SaValueError: If any variable takes more levels than ``max_levels``.
    """
    ceiling = check_count(max_levels, "max_levels", 2)

    over = [name for name, levels in category_lv.items() if len(levels) > ceiling]
    if not over:
        return

    counted = ", ".join(f"{name} ({len(category_lv[name])})" for name in over)
    raise SaValueError(
        f"variable(s) taking more levels than the {ceiling} `max_levels` allows: "
        f"{counted}. A measurement read as a category makes a table with a cell per "
        "observation, which no test of association is about. Bin the variable "
        "first, or name the levels to keep in `category_lv`"
        + ("" if named_lv else " (which also drops the rows at the rest)")
        + ". Raise `max_levels` if the variable really has this many."
    )


def categorical_shared_lv(
    category_lv: Mapping[str, Sequence[str]],
    observed: Mapping[str, Sequence[str]],
    named_lv: bool,
    control_label: Any,
) -> dict[str, list[str]]:
    """Unify the level sets of a matched design.

    Port of ``sa_categorical_shared_lv()``. Repeated measurements of one thing
    take one set of levels, so a matched design has one to resolve rather than one
    per condition. Taking the union when the levels come from the data is what
    keeps a condition in which nobody answered ``"n"`` from silently producing a
    table that is not square.

    Args:
        category_lv: The levels resolved so far, one entry per variable.
        observed: The levels each variable was actually seen to take.
        named_lv: Whether the caller named the levels.
        control_label: The single level to hold as the reference, or ``None``.

    Raises:
        SaValueError: If named level sets disagree, or if ``control_label`` is not
            one name.
    """
    variables = list(category_lv)

    if named_lv:
        first = [str(level) for level in category_lv[variables[0]]]
        disagree = [name for name in variables if set(category_lv[name]) != set(first)]
        if disagree:
            raise SaValueError(
                "`paired = TRUE` reads the columns as repeated measurements of one "
                "thing, so every variable takes the same levels and the table is "
                "square. `category_lv` disagrees at: " + ", ".join(disagree) + "."
            )
        shared = first
    else:
        shared = sorted({str(level) for name in variables for level in observed[name]})

    if control_label is not None:
        if isinstance(control_label, Mapping):
            labels = list(control_label.values())
        elif isinstance(control_label, str) or not isinstance(control_label, Iterable):
            labels = [control_label]
        else:
            labels = list(control_label)
        if len(labels) != 1:
            raise SaValueError(
                "a matched design has one level set shared by every condition, so "
                "`control_label` is a single level name rather than one per "
                f"variable. Got {len(labels)}."
            )
        shared = control_first(shared, labels[0], lv_arg="category_lv" if named_lv else "data")

    return {name: list(shared) for name in variables}


def validate_categorical_input(
    data: Any,
    category_lv: Mapping[str, Sequence[str]] | None = None,
    control_label: Any = None,
    paired: bool = False,
    max_levels: Any = MAX_CATEGORY_LEVELS,
) -> CategoricalInput:
    """Resolve the categorical variables and the levels of each.

    Port of ``sa_validate_categorical_input()``. Everything the analysis needs to
    know about where an observation sits, settled in one pass, so that the levels,
    the table and the row selection cannot be derived from each other twice in
    different orders. This is the counterpart of the factorial layout for a design
    whose variables are the answers rather than the strata.

    Two kinds of unusable row are counted apart rather than together. A row naming
    a level ``category_lv`` leaves out was measured and excluded; a row missing a
    value was not measured. Both leave the table, and a membership test would make
    them indistinguishable, so they are found before it is used.

    A matched design is the one place the level sets are not resolved per
    variable. Repeated measurements of one thing share a level set by definition,
    and a square table is what McNemar's test needs, so the levels are unified
    across the conditions and ``control_label`` is a single name rather than one
    per variable.

    Args:
        data: Wide frame or matrix, one row per observation.
        category_lv: Levels per variable, or ``None`` to take every column of
            ``data`` with its levels in sorted order.
        control_label: The level to hold first: a mapping of one name per variable
            for an independent design, a single name for a matched one, or
            ``None``.
        paired: Whether the columns are repeated conditions on the same rows.
        max_levels: How many levels a variable may take before it is refused as a
            category. See :func:`check_level_count`.

    Returns:
        A :class:`CategoricalInput`.

    Raises:
        SaValueError: If the input is not a table this scenario can be run on.
    """
    frame = _as_frame(data)
    if len(frame.index) == 0:
        raise SaValueError("`data` has zero rows.")

    named_lv = category_lv is not None
    if named_lv:
        assert category_lv is not None
        if not isinstance(category_lv, Mapping) or len(category_lv) == 0:
            raise SaValueError(
                "`category_lv` must be a named mapping, one entry per categorical "
                "variable, holding that variable's levels: "
                '{"cat_1": ["y", "n"], "cat_2": ["high", "low"]}.'
            )
        variables = [str(name) for name in category_lv]
        unknown = [name for name in variables if name not in frame.columns]
        if unknown:
            raise SaValueError(
                "`category_lv` names column(s) absent from `data`: "
                + ", ".join(unknown)
                + ". Present: "
                + ", ".join(str(name) for name in frame.columns)
                + "."
            )
    else:
        variables = [str(name) for name in frame.columns]

    if len(variables) < 2:
        source = "`category_lv` names " if named_lv else "`data` holds "
        raise SaValueError(
            "a categorical comparison crosses at least two variables, and "
            f"{source}{len(variables)}."
        )

    not_atomic = [
        name
        for name in variables
        if any(isinstance(value, list | dict | set | np.ndarray) for value in frame[name])
    ]
    if not_atomic:
        raise SaValueError(
            "a categorical variable must be an atomic column. Not atomic: "
            + ", ".join(not_atomic)
            + "."
        )

    values = {name: _as_labels(frame[name]) for name in variables}
    observed = {
        name: sorted({label for label in column if label is not None})
        for name, column in values.items()
    }
    empty = [name for name in variables if not observed[name]]
    if empty:
        raise SaValueError("variable(s) holding no non-missing value: " + ", ".join(empty) + ".")

    if named_lv:
        assert category_lv is not None
        resolved: dict[str, list[str]] = {}
        for name in variables:
            levels = _named_levels(category_lv[name], f"category_lv${name}")
            absent = [level for level in levels if level not in observed[name]]
            if absent:
                raise SaValueError(
                    f"`category_lv${name}` level(s) absent from `data${name}`: "
                    + ", ".join(absent)
                    + "."
                )
            resolved[name] = levels
    else:
        resolved = {name: list(observed[name]) for name in variables}

    # Checked on the resolved levels rather than on the column, so naming three
    # levels of a variable that happens to take fifty is a way through rather than
    # a second thing to argue with.
    check_level_count(resolved, max_levels, named_lv)

    # The levels are settled here and nowhere later, so the table, the cell labels
    # and the row selection are all built from the order the reference ended up in
    # rather than corrected afterwards.
    if paired:
        resolved = categorical_shared_lv(resolved, observed, named_lv, control_label)
    else:
        resolved = fact_control_first(
            resolved, control_label, "category_lv" if named_lv else "data"
        )

    incomplete = np.zeros(len(frame.index), dtype=bool)
    for name in variables:
        incomplete |= np.array([label is None for label in values[name]], dtype=bool)
    outside = np.zeros(len(frame.index), dtype=bool)
    for name in variables:
        allowed = set(resolved[name])
        outside |= np.array([label not in allowed for label in values[name]], dtype=bool)
    # A missing entry is absent from every level set, so the two counts are kept
    # apart by masking the missing rows out before the membership test is read.
    outside &= ~incomplete
    keep = ~incomplete & ~outside

    n_used = int(keep.sum())
    if n_used < 2:
        raise SaValueError(
            f"only {n_used} row(s) hold a level of every variable, which is not a "
            f"table. Dropped: {int(incomplete.sum())} for a missing value and "
            f"{int(outside.sum())} for a level outside `category_lv`."
        )

    out = pd.DataFrame(
        {
            name: pd.Categorical(
                [label for label, kept in zip(values[name], keep, strict=True) if kept],
                categories=resolved[name],
            )
            for name in variables
        }
    )

    return CategoricalInput(
        data=out,
        variables=variables,
        category_lv=resolved,
        n_used=n_used,
        n_dropped=int(outside.sum()),
        n_incomplete=int(incomplete.sum()),
    )


def _levels_of(column: Any) -> list[str]:
    """The levels a column takes, the way R's ``table()`` reads them.

    A :class:`pandas.Categorical` keeps its own order, which is what
    :func:`validate_categorical_input` settled. Anything else is sorted, which is
    what ``table()`` does to a character vector.
    """
    if isinstance(getattr(column, "dtype", None), pd.CategoricalDtype):
        return [str(level) for level in column.cat.categories]
    return sorted({str(value) for value in column if not _is_missing(value)})


def _codes_of(column: Any, levels: Sequence[str]) -> np.ndarray:
    """Zero-based level positions, with a missing entry at ``-1``."""
    at = {level: position for position, level in enumerate(levels)}
    return np.array(
        [-1 if label is None else at.get(label, -1) for label in _as_labels(column)],
        dtype=np.int64,
    )


def categorical_counts(data: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Cross two variables into a contingency table.

    Port of ``sa_categorical_counts()``. Every level of both variables is a row
    or a column whether or not any observation landed there, which is what makes
    the table's shape a property of the design rather than of the draw.

    R returns a ``table`` with named dimnames. The counterpart here is a frame
    whose index and columns are the two level sets, because those labels are the
    key ``cells`` and a simulator's ``truth_cell`` are both read on, and a frame
    is what carries them through :mod:`statassist.kernel.categorical` unchanged.

    Args:
        data: Validated data whose columns hold the levels.
        variables: The two variable names, row first.
    """
    if len(variables) != 2:
        raise SaInternalError(
            f"internal error: a contingency table crosses two variables, got {len(variables)}."
        )
    row_name, col_name = str(variables[0]), str(variables[1])
    row_lv = _levels_of(data[row_name])
    col_lv = _levels_of(data[col_name])

    row_at = _codes_of(data[row_name], row_lv)
    col_at = _codes_of(data[col_name], col_lv)
    usable = (row_at >= 0) & (col_at >= 0)
    counts = np.bincount(
        row_at[usable] * len(col_lv) + col_at[usable],
        minlength=len(row_lv) * len(col_lv),
    ).reshape(len(row_lv), len(col_lv))

    out = pd.DataFrame(counts, index=pd.Index(row_lv, name=row_name), columns=col_lv)
    out.columns.name = col_name
    return out


def categorical_condition_counts(
    data: pd.DataFrame,
    variables: Sequence[str],
    levels: Sequence[str],
) -> pd.DataFrame:
    """Summarise repeated binary conditions as a condition-by-response table.

    Port of ``sa_categorical_condition_counts()``. Cochran's Q is asked of a
    subjects-by-conditions matrix, which has no two-variable cross-classification
    to tabulate. What it is asked *about* is whether the conditions share a
    marginal response rate, and that question has a table: one row per condition
    and one column per level of the shared response.

    It is the table to plot for the same reason - it holds the rates the test
    compares - and it is not the paired table McNemar's test is read from, which
    crosses two conditions against each other and only exists when there are two.
    """
    wanted = [str(level) for level in levels]
    rows = []
    for name in variables:
        at = _codes_of(data[str(name)], wanted)
        rows.append(np.bincount(at[at >= 0], minlength=len(wanted)))

    out = pd.DataFrame(
        np.vstack(rows) if rows else np.zeros((0, len(wanted)), dtype=np.int64),
        index=pd.Index([str(name) for name in variables], name="condition"),
        columns=wanted,
    )
    out.columns.name = "response"
    return out


def finite_or_na(x: Any) -> np.ndarray:
    """Replace every non-finite value with a missing one, keeping the shape.

    Port of ``sa_finite_or_na()``. Dividing by an empty margin is not a residual
    of zero, it is a residual that does not exist, so an infinity is turned into a
    missing value rather than passed on as one. R's ``!is.finite()`` catches
    ``NA``, ``NaN``, ``Inf`` and ``-Inf`` alike, and so does this.

    >>> finite_or_na([1.5, float("inf"), 0.0]).tolist()
    [1.5, nan, 0.0]
    """
    values = np.array(x, dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def expected_independence(counts: Any) -> np.ndarray:
    """The counts a table was expected to hold if the two variables were independent.

    Port of ``sa_expected_independence()``. The product of the margins over the
    total. This is what the chi-square test of independence and Fisher's exact
    test are both about, and it is also what the condition-by-response table of a
    repeated design is held against: there the row margins are fixed at the number
    of subjects, so the same arithmetic says that every condition shows the pooled
    response rate, which is marginal homogeneity rather than independence. The
    formula does not distinguish the two claims; which one was made is recorded
    beside it rather than here.
    """
    table = np.asarray(counts, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / float(table.sum())
    return np.asarray(expected, dtype=float)


def expected_symmetry(counts: Any) -> np.ndarray:
    """The counts a square table was expected to hold if it were symmetric.

    Port of ``sa_expected_symmetry()``. The average of each cell and its
    transpose. Two things follow, and both are the whole content of a matched
    comparison.

    The diagonal is expected at exactly what it holds, so it carries no residual.
    That is not an approximation: a pair that answered the same way under both
    conditions says nothing about which condition raises the response, which is
    the same fact that drops the concordant cells out of McNemar's statistic.

    And the residuals of a discordant pair square and sum to that statistic. For a
    2 x 2 table each of ``b`` and ``c`` is expected at ``(b + c) / 2``, so the two
    Pearson residuals square to ``(b - c)^2 / (2 * (b + c))`` each and to
    ``(b - c)^2 / (b + c)`` together, which is McNemar's uncorrected chi-square.
    The cell table and the p-value beside it are therefore about the same
    hypothesis.

    Raises:
        SaInternalError: If the table is not square. Which table reaches here is
            the caller's contract, not the user's input.
    """
    table = np.asarray(counts, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        shape = " x ".join(str(size) for size in table.shape)
        raise SaInternalError(
            f"internal error: symmetry is a claim about a square table, and this one is {shape}."
        )
    return np.asarray((table + table.T) / 2, dtype=float)


def _labels_of(counts: Any, axis: int) -> list[str]:
    """The level labels of one axis of a table, or its positions when it has none.

    R has no second branch here: ``expand.grid()`` on a table without dimnames
    builds a frame of no rows and the assembly fails. Every table this is called
    on inside the package is labelled, so the difference is only reachable by a
    caller passing a bare array, and numbering the axis is a more useful answer
    there than a message about a grid.
    """
    if isinstance(counts, pd.DataFrame):
        labels = counts.index if axis == 0 else counts.columns
        return [str(label) for label in labels]
    size = np.asarray(counts).shape[axis]
    return [str(position + 1) for position in range(size)]


def categorical_cells(counts: Any, null: str = "independence") -> pd.DataFrame:
    """Expand a contingency table into the canonical cell table.

    Port of ``sa_categorical_cells()``. A matrix vectorises down its columns and
    the label grid varies its first argument fastest, so the labels and the counts
    line up without either being matched by name. That is the row order of every
    cell table in the package, and it is what a simulator's ``truth_cell`` merges
    onto.

    Args:
        counts: A two-dimensional table with row and column labels.
        null: Which null hypothesis the expected counts state. One of
            :func:`~statassist.core.contracts.categorical_nulls`.

    Returns:
        A frame with the columns of
        :func:`~statassist.core.contracts.categorical_cell_columns`, one row per
        cell.

    Raises:
        SaInternalError: If ``null`` is not one of the three. Which null is being
            tested is the caller's contract, not the user's input.
    """
    if null not in categorical_nulls():
        raise SaInternalError(
            "internal error: `null` must name one of " + ", ".join(categorical_nulls()) + "."
        )

    table = np.asarray(
        counts.to_numpy() if isinstance(counts, pd.DataFrame) else counts, dtype=float
    )
    total = table.sum()
    row_n = table.sum(axis=1)
    col_n = table.sum(axis=0)

    expected = expected_symmetry(table) if null == "symmetry" else expected_independence(table)

    # Dividing by an empty margin is not a residual of zero, it is a residual that
    # does not exist, so the non-finite results are kept missing rather than
    # passed on. numpy says so out loud where R is silent, hence the suppression.
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = finite_or_na((table - expected) / np.sqrt(expected))

        # The variance correction the standardized residual divides by is derived
        # for a two-way table held against its own margins. Under symmetry the
        # comparison is a cell against its transpose and that correction has no
        # counterpart, so the column is missing there rather than a number that
        # looks referable to a standard normal and is not.
        if null == "symmetry":
            std_residual = np.full(table.shape, np.nan)
        else:
            spread = expected * np.outer(1 - row_n / total, 1 - col_n / total)
            std_residual = finite_or_na((table - expected) / np.sqrt(spread))

        prop_row = finite_or_na(table / row_n[:, None])
        prop_col = finite_or_na(table / col_n[None, :])

    row_lv = _labels_of(counts, 0)
    col_lv = _labels_of(counts, 1)

    def down(values: np.ndarray) -> np.ndarray:
        """Flatten the way R's ``as.numeric()`` reads a matrix: down the columns."""
        return np.asarray(values, dtype=float).reshape(-1, order="F")

    return pd.DataFrame(
        {
            "row_level": row_lv * len(col_lv),
            "col_level": [level for level in col_lv for _ in row_lv],
            "observed": down(table),
            "expected": down(expected),
            "residual": down(residual),
            "std_residual": down(std_residual),
            "prop_total": down(table) / total,
            "prop_row": down(prop_row),
            "prop_col": down(prop_col),
        },
        columns=categorical_cell_columns(),
    )


#: What the two axes of a folded table are called when nothing names them.
#:
#: R's defaults for ``sa_categorical_table()``. Every call inside the package
#: passes ``design$row_var`` and ``design$col_var``, so these are only reached by
#: a caller folding a bare cell table of their own.
TABLE_AXIS_NAMES = ("row", "column")


def categorical_table(
    cells: pd.DataFrame,
    row_var: str = TABLE_AXIS_NAMES[0],
    col_var: str = TABLE_AXIS_NAMES[1],
) -> pd.DataFrame:
    """Fold a cell table back into a contingency table.

    Port of ``sa_categorical_table()``. ``cells`` is the canonical form because
    that is the shape which survives being written out as JSON with its labels
    attached; a table is the shape to read it in, so it is built on request
    rather than stored twice and left to drift.

    Indexed by level name rather than by position, so the result does not depend
    on the order the cells happen to be in.

    Args:
        cells: One row per cell, as
            :func:`~statassist.core.contracts.categorical_cell_columns` names.
        row_var: What to call the row axis, normally ``design["row_var"]``.
        col_var: The same for the columns.

    Returns:
        A frame of counts whose index and columns are the two level sets, named
        after the axes. The counts come back as integers when every one of them
        is whole, which is what the engines are handed everywhere else.
    """
    row_lv = list(dict.fromkeys(str(level) for level in cells["row_level"]))
    col_lv = list(dict.fromkeys(str(level) for level in cells["col_level"]))

    out = pd.DataFrame(
        np.zeros((len(row_lv), len(col_lv))),
        index=pd.Index(row_lv, name=row_var),
        columns=pd.Index(col_lv, name=col_var),
    )
    at_row = {level: position for position, level in enumerate(row_lv)}
    at_col = {level: position for position, level in enumerate(col_lv)}
    values = np.asarray(cells["observed"], dtype=float)
    for position, (row, column) in enumerate(
        zip(cells["row_level"], cells["col_level"], strict=True)
    ):
        out.iloc[at_row[str(row)], at_col[str(column)]] = values[position]

    if bool(np.all(values == np.round(values))):
        return out.astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# The approximation rule each design rests on
#
# Reported rather than enforced, which is the same choice `diagnose_distribution`
# makes about normality: a failed check does not change which tests run, it
# changes which of them deserves the weight. Here it is a short walk from the
# check to the answer, because the test that does not need the approximation is
# already in the result beside it.
#
# Each design rests on a different rule, so the three builders below report
# different numbers, and every one of them ends in the same two fields:
# `approx_ok`, and a `note` that says in one sentence what to read instead when it
# is False. That is what lets a printed result report the check without knowing
# which design produced it. Each `rule` is an id
# `Configuration/registry/assumptions.yaml` records.
# --------------------------------------------------------------------------- #

#: An expected count at or above this needs no apology.
EXPECTED_COUNT_MIN = 5

#: No expected count may fall below this, whatever share of the cells is small.
EXPECTED_COUNT_FLOOR = 1

#: What share of the cells may sit below :data:`EXPECTED_COUNT_MIN`.
EXPECTED_LT5_MAX_PROP = 0.2

#: Below this many discordant pairs, McNemar's chi-square is not to be trusted.
#:
#: The rule ``Configuration/registry/assumptions.yaml`` records as
#: ``discordant_pair_count``. It is also what ``mcnemar(exact=None)`` decides its
#: branch on - the diagnostic and the choice of test are the same number, so they
#: are the same constant.
DISCORDANT_PAIR_MIN = 25

#: Subjects times conditions below this and Cochran's Q is an indication only.
REPEATED_CELL_MIN = 24


def diagnose_expected(cells: pd.DataFrame) -> dict[str, Any]:
    """The expected-count rule the chi-square approximation rests on.

    Port of ``sa_diagnose_expected()``. The rule ``assumptions.yaml`` records as
    ``expected_count_min``: every expected count at least 5, or at most a fifth of
    the cells below 5 with none below 1.
    """
    expected = np.asarray(cells["expected"], dtype=float)
    n_cells = len(expected)
    n_lt5 = int(np.count_nonzero(expected < EXPECTED_COUNT_MIN))
    n_lt1 = int(np.count_nonzero(expected < EXPECTED_COUNT_FLOOR))
    ok = n_lt5 == 0 or (n_lt1 == 0 and n_lt5 / n_cells <= EXPECTED_LT5_MAX_PROP)

    return {
        "rule": "expected_count_min",
        "min_expected": float(expected.min()),
        "n_cells": n_cells,
        "n_expected_lt5": n_lt5,
        "prop_expected_lt5": n_lt5 / n_cells,
        "approx_ok": ok,
        "note": (
            f"The smallest expected count is {fmt_est(expected.min())} and {n_lt5} of "
            f"{n_cells} cell(s) fall below 5, so the chi-square approximation is "
            "doubtful here. Read `$tests$fisher_test`, which needs no "
            "approximation, or set `simulate_p_value = TRUE`."
        ),
    }


def diagnose_discordance(n_discordant: int) -> dict[str, Any]:
    """The discordant pair rule McNemar's approximation rests on.

    Port of ``sa_diagnose_discordance()``. ``assumptions.yaml`` records it as
    ``discordant_pair_count``: at least 25 discordant pairs for the chi-square
    approximation. The exact branch is what runs below that under the default
    ``exact=None``, so this reports whether the approximation *would* have been
    sound rather than whether the answer is.
    """
    count = int(n_discordant)
    return {
        "rule": "discordant_pair_count",
        "n_discordant": count,
        "approx_ok": count >= DISCORDANT_PAIR_MIN,
        "note": (
            f"Only {count} discordant pair(s) carry the comparison, which is below "
            "the 25 the chi-square approximation asks for. The exact binomial "
            "branch is the one to read; `parameters$exact` says whether it ran."
        ),
    }


def diagnose_repeated(n_subjects: int, k: int) -> dict[str, Any]:
    """The sample size rule Cochran's Q rests on.

    Port of ``sa_diagnose_repeated()``. Q is referred to a chi-square distribution
    on ``k - 1`` degrees of freedom, and the usual rule of thumb for that
    approximation is that the number of subjects times the number of conditions
    reaches 24. ``assumptions.yaml`` records it as ``sample_size_repeated``. There
    is no exact test in the result to fall back on, so the note says what the
    number means rather than where to look instead.
    """
    subjects = int(n_subjects)
    conditions = int(k)
    cells = subjects * conditions
    return {
        "rule": "sample_size_repeated",
        "n_subjects": subjects,
        "n_conditions": conditions,
        "n_cells": cells,
        "approx_ok": cells >= REPEATED_CELL_MIN,
        "note": (
            f"{subjects} subject(s) over {conditions} condition(s) is {cells} "
            "observation(s), below the 24 the chi-square approximation for Q asks "
            "for, so read its p-value as an indication rather than as a rate."
        ),
    }
