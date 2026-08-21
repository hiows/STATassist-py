"""Assembling a per-feature result table.

The port of the table-building half of ``R/utils_validate.R``. Every comparison
in the package runs one test across hundreds of features and reports the result
as one rectangular table, and the two things that make that work are here:

* A single unusable feature must not abort the scan. It becomes an all-missing
  row and is reported afterwards, with every other feature intact.
* Engine noise is reported once. A tie warning raised for two hundred features
  is one note about two hundred features, not two hundred notes.

Both are contracts the callers rely on, which is why the aggregation lives here
rather than in each ``compare_*`` function.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .contracts import posthoc_table_columns
from .errors import SaInternalError, notify, warn
from .padjust import p_adjust

__all__ = [
    "add_padj",
    "feature_table",
    "level_pairs",
    "na_row",
    "posthoc_table",
    "stat_row",
]

#: The label columns of a post-hoc table, which carry text rather than a number.
_POSTHOC_LABELS = ("features", "contrast", "group1", "group2")


def na_row(names: Iterable[str]) -> dict[str, float]:
    """Build an all-missing result row with the expected names.

    Port of ``sa_na_row()``. What :func:`feature_table` substitutes for a feature
    the test could not be run on.
    """
    return dict.fromkeys(names, float("nan"))


def stat_row(**values: Any) -> dict[str, float]:
    """Assemble one result row from named scalars.

    Port of ``sa_row()``. Engines attach their own names to what they return -
    a t-test names its statistic ``t`` and its parameter ``df`` - and in R
    ``c(a = x)`` on a named ``x`` yields ``a.t`` rather than ``a``. R forces every
    value through ``as.numeric()`` to strip that.

    Here the equivalent problem is shape rather than naming: a SciPy result may
    hand back a zero-dimensional array or a one-element array where a scalar is
    wanted. Every value is reduced to its first element, so the keys are exactly
    as written at the call site.
    """
    row: dict[str, float] = {}
    for name, value in values.items():
        if value is None:
            row[name] = float("nan")
            continue
        flat = np.asarray(value, dtype=float).reshape(-1)
        row[name] = float("nan") if flat.size == 0 else float(flat[0])
    return row


def add_padj(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Append a multiplicity adjusted p-value next to ``pval``.

    Port of ``sa_add_padj()``. The position matters: ``pval_adj`` sits
    immediately after ``pval`` so a reader scanning the table finds the raw and
    the adjusted value side by side, and a consumer selecting a column range gets
    both or neither.
    """
    if "pval" not in df.columns:
        raise SaInternalError("internal error: a table given to `add_padj` has no `pval` column.")

    out = df.copy()
    out["pval_adj"] = p_adjust(out["pval"].to_numpy(dtype=float), method)

    others = [name for name in out.columns if name != "pval_adj"]
    at = others.index("pval") + 1
    return out[others[:at] + ["pval_adj"] + others[at:]]


def level_pairs(group_lv: Sequence[str]) -> pd.DataFrame:
    """Every unordered pair of group levels, in display order.

    Port of ``sa_level_pairs()``. Pairs are generated so the first member is
    always the *later* level of ``group_lv``. Every post-hoc estimate therefore
    reads as ``group_lv[j] - group_lv[i]`` with ``i < j``, which puts the
    reference - being the first level - on the right of every contrast it takes
    part in. A treatment level then reads against the control the same way
    ``effect['log2fc']`` already divides it, so a feature the treatment raised is
    positive in both.

    Returns:
        A frame with ``i``, ``j``, ``group1``, ``group2`` and ``contrast``.
        ``i`` and ``j`` are **zero-based** positions into ``group_lv``, where R
        stores one-based ones.
    """
    levels = [str(level) for level in group_lv]
    rows = [
        {
            "i": j,
            "j": i,
            "group1": levels[j],
            "group2": levels[i],
            "contrast": f"{levels[j]} - {levels[i]}",
        }
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]
    return pd.DataFrame(rows, columns=["i", "j", "group1", "group2", "contrast"])


def _report_failures(label: str, failures: dict[str, str], n_feats: int) -> None:
    """One warning for the whole scan, however many features failed."""
    if not failures:
        return
    detail = "\n".join(f"  {name}: {message}" for name, message in failures.items())
    # 4 frames out lands on whoever called feature_table / posthoc_table, which is
    # the analysis function the user actually named.
    warn(
        f"{label} could not be computed for {len(failures)} of {n_feats} "
        f"feature(s); those rows are NA:\n{detail}",
        stacklevel=4,
    )


def feature_table(
    feats: Sequence[str],
    columns: Sequence[str],
    label: str,
    fun: Callable[[int], dict[str, Any] | pd.Series],
    p_adjust_method: str | None = "none",
) -> pd.DataFrame:
    """Run one test across all features and assemble a result table.

    Port of ``sa_feature_table()``.

    Args:
        feats: Feature names, one output row per entry, in this order.
        columns: The numeric column names ``fun`` is expected to return.
        label: Human readable test name, used in the aggregated messages.
        fun: Called with the **zero-based** index of the feature (R passes the
            one-based one) and returning a mapping from every name in ``columns``
            to a number.
        p_adjust_method: Method for the multiplicity adjustment, or ``None`` for
            a table that holds no p-value at all, such as the effect estimates.

    Returns:
        A frame with ``features``, then ``columns`` in the order given, and -
        unless ``p_adjust_method`` is ``None`` - ``pval_adj`` after ``pval``.
    """
    names = [str(name) for name in feats]
    wanted = [str(name) for name in columns]

    failures: dict[str, str] = {}
    notes: dict[str, str] = {}
    rows: list[dict[str, float]] = []

    for index, name in enumerate(names):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                produced: dict[str, Any] | pd.Series = fun(index)
            except Exception as error:  # noqa: BLE001 - one bad feature must not stop the scan
                failures[name] = str(error)
                produced = na_row(wanted)
        if caught:
            # Muffled, as R muffles them, and reported once below.
            unique = list(dict.fromkeys(str(entry.message) for entry in caught))
            notes[name] = "; ".join(unique)

        row = dict(produced)
        absent = [column for column in wanted if column not in row]
        if absent:
            raise SaInternalError(
                f"internal error: {label} row for `{name}` is missing column(s): "
                + ", ".join(absent)
                + "."
            )
        rows.append({column: _as_float(row[column]) for column in wanted})

    out = pd.DataFrame(rows, columns=wanted, dtype=float)
    out.insert(0, "features", names)

    if p_adjust_method is not None:
        out = add_padj(out, p_adjust_method)

    if notes:
        grouped = Counter(notes.values())
        detail = "\n".join(
            f"  [{count} feature(s)] {text}" for text, count in sorted(grouped.items())
        )
        notify(f"{label}: engine note(s) for {len(notes)} of {len(names)} feature(s):\n{detail}")
    _report_failures(label, failures, len(names))

    return out


def _as_float(value: Any) -> float:
    """Coerce one cell of a result row, mapping a missing value to ``nan``."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _empty_posthoc_table() -> pd.DataFrame:
    """A zero-row post-hoc table carrying the full column contract.

    A scenario in which no feature cleared the omnibus stage still has to return
    a table of the agreed shape, or every consumer would need a special case for
    it.
    """
    contract = posthoc_table_columns()
    data = {
        name: pd.Series([], dtype=object if name in _POSTHOC_LABELS else float) for name in contract
    }
    return pd.DataFrame(data, columns=contract)


def posthoc_table(
    feats: Sequence[str],
    group_lv: Sequence[str],
    columns: Sequence[str],
    label: str,
    fun: Callable[[str], pd.DataFrame | np.ndarray],
    p_adjust_method: str = "holm",
) -> pd.DataFrame:
    """Run one post-hoc procedure across features and assemble a table.

    Port of ``sa_posthoc_table()``, the pairwise counterpart of
    :func:`feature_table`. Two things differ.

    A post-hoc row is a feature *and a pair*, so ``fun`` returns every pair of
    one feature at once, in the row order of :func:`level_pairs`. And a feature
    whose omnibus test was not significant is left out of the table entirely
    rather than filled with missing values, because an absent row means "not
    asked" while a missing row means "asked and failed".

    Args:
        feats: Features to run, already reduced to those that qualified.
        group_lv: Group levels, fixing the pair order and the direction.
        columns: Numeric column names ``fun`` is expected to return per pair.
        label: Human readable procedure name, used in the aggregated warning.
        fun: Called with the feature **name** - not its index, unlike
            :func:`feature_table` - and returning one row per pair.
        p_adjust_method: Applied across the pairs *within* each feature. The
            pairwise family is the set of contrasts of one feature, so adjusting
            across features here as well would mix two families.
    """
    pairs = level_pairs(group_lv)
    n_pairs = len(pairs.index)
    names = [str(name) for name in feats]
    wanted = [str(name) for name in columns]

    # Checked before anything runs, and before the no-features shortcut, so a
    # caller naming the wrong columns hears about it whether or not any feature
    # qualified. Without this the final contract selection would fail with a
    # bare KeyError several hundred features later.
    uncovered = [
        name
        for name in posthoc_table_columns()
        if name not in _POSTHOC_LABELS and name != "pval_adj" and name not in wanted
    ]
    if uncovered:
        raise SaInternalError(
            f"internal error: `columns` for {label} does not cover the post-hoc "
            "contract: " + ", ".join(uncovered) + "."
        )

    if not names:
        return _empty_posthoc_table()

    failures: dict[str, str] = {}
    blocks: list[pd.DataFrame] = []

    for name in names:
        try:
            produced = fun(name)
        except Exception as error:  # noqa: BLE001 - one bad feature must not stop the scan
            failures[name] = str(error)
            produced = pd.DataFrame(np.full((n_pairs, len(wanted)), np.nan), columns=wanted)

        frame = (
            produced
            if isinstance(produced, pd.DataFrame)
            else pd.DataFrame(np.asarray(produced, dtype=float), columns=wanted)
        )
        if len(frame.index) != n_pairs:
            raise SaInternalError(
                f"internal error: {label} returned {len(frame.index)} row(s) for "
                f"`{name}`, expected {n_pairs}."
            )
        absent = [column for column in wanted if column not in frame.columns]
        if absent:
            raise SaInternalError(
                f"internal error: {label} table for `{name}` is missing column(s): "
                + ", ".join(absent)
                + "."
            )

        block = pd.DataFrame({"features": [name] * n_pairs})
        for column in ("contrast", "group1", "group2"):
            block[column] = pairs[column].to_numpy()
        for column in wanted:
            block[column] = frame[column].to_numpy(dtype=float)
        block["pval_adj"] = p_adjust(block["pval"].to_numpy(dtype=float), p_adjust_method)
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    _report_failures(label, failures, len(names))
    return out[posthoc_table_columns()]
