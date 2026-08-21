"""The result contract every analysis fills.

The port of ``R/result.R``. ``compare_two_groups()`` fills these slots and
``estimate_significance()`` reads them back, so a second scenario reusing the
same slots inherits the whole downstream path without either side knowing which
tests were actually run.

Nothing in a result object is an engine object: every slot holds a scalar, a
string, a list, a mapping or a DataFrame. That is what lets the object be written
out as JSON and rebuilt in another language, and it is why the fitted model is
not kept here.

R uses a named list with an S3 class. :class:`SaResult` is the counterpart:
attribute access for reading (``res.effect``) and key access for the R spelling
(``res["effect"]``), over the same mapping.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from .contracts import (
    cell_table_columns,
    pairwise_table_columns,
    posthoc_table_columns,
    term_table_columns,
    test_table_columns,
)
from .errors import SaInternalError, SaValueError

__all__ = [
    "SaComparison",
    "SaDiagnosis",
    "SaResult",
    "SaSimulation",
    "SaSplit",
    "metadata",
    "new_comparison",
    "pick_test",
]


def metadata() -> dict[str, str]:
    """Reproducibility metadata attached to every result.

    Port of ``sa_metadata()``. The timestamp is ISO-8601 with an explicit offset,
    so the value survives a trip through JSON into a language with a different
    default timezone.
    """
    from .. import __version__

    return {
        "package_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


class SaResult(Mapping[str, Any]):
    """A result object: named slots, no engine objects, JSON-shaped.

    Reading works two ways on purpose. ``res.effect`` is what a Python caller
    will write; ``res["effect"]`` is R's ``res$effect``, which keeps the port
    line-comparable with the R sources it came from.

    Slots are stored exactly as handed over. Deciding that an axis does not
    apply to a scenario is the constructing function's job, not this class's:
    :func:`new_comparison` leaves out the four slots R leaves out and keeps
    ``diagnostics`` even when it is ``None``, because a reader asking
    ``res.diagnostics is None`` is asking a different question from one asking
    ``"posthoc" in res``.
    """

    __slots__ = ("_slots",)

    def __init__(self, slots: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_slots", dict(slots))

    # -- Mapping interface (R's `$` and `[[`) ------------------------------- #

    def __getitem__(self, name: str) -> Any:
        try:
            return self._slots[name]
        except KeyError:
            raise KeyError(name) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._slots[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no slot {name!r}. Present: " + ", ".join(self._slots)
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """A shallow copy of the slots, for serialisation."""
        return dict(self._slots)

    def __repr__(self) -> str:
        analysis = self._slots.get("analysis", "?")
        return f"<{type(self).__name__} {analysis}: " + ", ".join(self._slots) + ">"


class SaSimulation(SaResult):
    """A simulated data set together with the answer that was planted in it.

    R gives the simulators no class at all: they return a plain list of ``args``
    and one or more ``truth`` tables. The class is added here for the two-way
    slot access the rest of the port uses, and for nothing else - there is no
    contract to check, because a simulator wrote every slot itself.

    ``args`` is named after the arguments of the function that consumes it, so
    the analysis is one call away:

        sim = simulate_two_groups(seed=1)
        res = compare_two_groups(**sim.args)

    which is what ``do.call(compare_two_groups, sim$args)`` says in R.
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: " + ", ".join(self) + ">"


class SaSplit(SaResult):
    """A train/test partition, drawn once or several times over."""

    def __repr__(self) -> str:
        """The summary ``print.sa_split`` prints, not the data.

        The frames themselves are in ``split.datasets[name]["train_data"]``, and
        printing a hundred and fifty rows to describe a partition of them would
        bury the thing being described.
        """
        design = self["design"]
        params = self["parameters"]

        unit_note = ""
        if design["id"] is not None:
            unit_note = f"  ({design['n_units']} unit(s) of `{design['id']}`)"
        strata = design["stratified"] if design["stratified"] is not None else "none"

        lines = [
            "<SaSplit> train/test partition",
            f"  rows     : {design['n_rows']}{unit_note}",
            f"  stratify : {strata}",
        ]
        if design["strata_n"] is not None:
            counts = ", ".join(f"{name} {n}" for name, n in design["strata_n"].items())
            lines.append(f"             {counts}")

        seed_note = "" if params["seed"] is None else f", seed = {params['seed']}"
        lines.append(
            f"  settings : p_train = {params['p_train']}, times = {params['times']}{seed_note}"
        )
        lines.append("")
        lines.append("  splits")

        width = max(len(name) for name in self["datasets"])
        for name, dataset in self["datasets"].items():
            reached = params["achieved_p"][name]
            lines.append(
                f"    ${name:<{width}}  train {len(dataset['train_data'].index)}"
                f" / test {len(dataset['test_data'].index)}  (p = {reached:.3f})"
            )
        return "\n".join(lines)


class SaDiagnosis(SaResult):
    """The assumption checks a comparison rests on, on their own.

    Deliberately not a :class:`SaComparison`. Normality is a property of one
    sample and homogeneity a property of a set of them, so the two tables have
    different numbers of rows and would not fit a contract built around one row
    per feature.
    """

    def __repr__(self) -> str:
        """The summary ``print.sa_diagnosis`` prints, not the tables.

        How many features failed each check, rather than the checks themselves.
        The tables are in ``diagnosis.normality``, ``.variance`` and
        ``.outliers``.
        """
        alpha = self["parameters"]["alpha"]
        summary = self["summary"]
        design = self["design"]

        groups = ", ".join(design["group_lv"]) if design["grouped"] else "none, so no variance test"
        criterion = self["parameters"].get("criterion")

        def failing(column: str) -> int:
            flags = summary[column]
            return int((flags.notna() & ~flags.astype("boolean").fillna(False)).sum())

        lines = [
            f"<SaDiagnosis> {self['analysis']}",
            f"  features : {len(self['features'])}",
            f"  groups   : {groups}",
            f"  settings : alpha = {_fmt(alpha)}, outlier criterion = {criterion}",
            "",
            "  checks",
            f"    normality  {failing('normal_ok')} of {len(summary.index)} feature(s) "
            f"have a group failing Shapiro-Wilk at {_fmt(alpha)}",
        ]
        if len(self["variance"].index) > 0:
            lines.append(
                f"    variance   {failing('variance_ok')} of {len(summary.index)} "
                f"feature(s) fail Levene at {_fmt(alpha)}"
            )
        flagged = int((summary["n_outliers"] > 0).sum())
        lines.append(
            f"    outliers   {len(self['outliers'].index)} observation(s) flagged "
            f"across {flagged} feature(s)"
        )
        lines.append("")
        lines.append("  A failed check never changes which tests run. It changes which of")
        lines.append("  them deserves the most weight, and that judgement stays with you.")
        return "\n".join(lines)


def _fmt(value: float) -> str:
    """A number the way R's ``cat()`` writes it: no trailing zeros."""
    return f"{value:g}"


class SaComparison(SaResult):
    """The result of comparing groups on a set of features."""


class SaTwoGroup(SaComparison):
    """Two independent or paired groups."""


class SaMultiGroup(SaComparison):
    """Three or more groups, with a post-hoc stage."""


class SaOneSample(SaComparison):
    """One group against a reference value."""


class SaFactorial(SaComparison):
    """A crossed design, with a term axis and a cell axis."""


#: The subclass each scenario asks for, by the name R passes as ``subclass``.
_SUBCLASSES: dict[str, type[SaComparison]] = {
    "sa_two_group": SaTwoGroup,
    "sa_multi_group": SaMultiGroup,
    "sa_one_sample": SaOneSample,
    "sa_factorial": SaFactorial,
}


def _check_aligned(df: Any, what: str, features: Sequence[str]) -> None:
    """One row per feature, in the order ``features`` fixes."""
    if not isinstance(df, pd.DataFrame):
        raise SaInternalError(f"internal error: {what} must be a DataFrame.")
    present = list(df["features"]) if "features" in df.columns else None
    if present != list(features):
        raise SaInternalError(f"internal error: {what} is not aligned with `features`.")


def _check_columns(df: pd.DataFrame, what: str, contract: Sequence[str]) -> None:
    """Every column the contract names is present."""
    absent = [name for name in contract if name not in df.columns]
    if absent:
        raise SaInternalError(
            f"internal error: {what} is missing contract column(s): " + ", ".join(absent) + "."
        )


def _check_membership(df: Any, what: str, features: Sequence[str]) -> None:
    """Holds only features of this comparison, without being one row each."""
    if not isinstance(df, pd.DataFrame):
        raise SaInternalError(f"internal error: {what} must be a DataFrame.")
    known = set(features)
    unknown = list(dict.fromkeys(str(name) for name in df["features"] if name not in known))
    if unknown:
        raise SaInternalError(
            f"internal error: {what} holds feature(s) absent from the comparison: "
            + ", ".join(unknown)
            + "."
        )


def new_comparison(
    analysis: str,
    features: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    effect: pd.DataFrame,
    tests: Mapping[str, pd.DataFrame],
    test_info: Mapping[str, Any],
    terms: pd.DataFrame | None = None,
    cells: pd.DataFrame | None = None,
    posthoc: Mapping[str, pd.DataFrame] | None = None,
    pairwise: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
    diagnostics: Any = None,
    subclass: str | None = None,
) -> SaComparison:
    """Assemble a comparison result object.

    Port of ``sa_new_comparison()``. The checks here guard the contract rather
    than the user's input, so every one of them raises
    :class:`~statassist.core.errors.SaInternalError`: firing means a mistake
    inside the package, not a bad call from outside.

    Args:
        analysis: Scenario identifier, such as ``"two_group_comparison"``.
        features: Feature names, in the row order every table uses.
        design: How the data was laid out: group levels, whether the samples were
            paired, how pairs were formed, what was dropped.
        parameters: The analysis choices the caller made.
        effect: The effect estimates shared by all tests, one row per feature.
        tests: One table per test, one row per feature.
        test_info: One entry per element of ``tests``, describing what was run.
        terms: One row per feature and model term, or ``None`` for a scenario
            whose model has a single term, in which case the slot is left out.
        cells: One row per feature and cell of a crossed grid, or ``None`` for a
            scenario with no grid, in which case the slot is left out.
        posthoc: Post-hoc tables, named after the test each one follows. Rows are
            feature by pair. Left out when the scenario has no post-hoc stage.
        pairwise: Per test, one table per contrast. The same numbers as
            ``posthoc`` seen one contrast at a time, with every feature present.
        diagnostics: Assumption checks, or ``None`` when not requested.
        subclass: Which :class:`SaComparison` subclass to build, by the name R
            uses (``"sa_two_group"`` and so on).
    """
    names = [str(name) for name in features]

    if not tests:
        raise SaInternalError("internal error: `tests` must be a non-empty named mapping.")
    if set(tests) != set(test_info):
        raise SaInternalError(
            "internal error: `tests` and `test_info` name different tests: "
            + ", ".join(tests)
            + " vs "
            + ", ".join(test_info)
            + "."
        )

    posthoc = dict(posthoc or {})
    pairwise = dict(pairwise or {})

    for slot_name, slot in (("posthoc", posthoc), ("pairwise", pairwise)):
        stray = [name for name in slot if name not in tests]
        if stray:
            raise SaInternalError(
                f"internal error: `{slot_name}` names a test that was not run: "
                + ", ".join(stray)
                + "."
            )

    _check_aligned(effect, "`effect`", names)
    for name, table in tests.items():
        _check_aligned(table, f"`tests[{name!r}]`", names)
        _check_columns(table, f"`tests[{name!r}]`", test_table_columns())

    # A term table holds one row per feature and term, so it is checked the way a
    # post-hoc table is: the features it names have to be features of this
    # comparison, but there is no position to align them by.
    if terms is not None:
        _check_membership(terms, "`terms`", names)
        _check_columns(terms, "`terms`", term_table_columns())

    # A cell table is rectangular but not one row per feature, so it is checked
    # on membership and then on the block size the grid fixes.
    if cells is not None:
        _check_membership(cells, "`cells`", names)
        _check_columns(cells, "`cells`", cell_table_columns())
        if list(dict.fromkeys(str(name) for name in cells["features"])) != names:
            raise SaInternalError(
                "internal error: `cells` does not hold every feature of the "
                "comparison once, in order."
            )

    # Checked on membership rather than identity: a post-hoc table holds several
    # rows per feature and may legitimately skip one whose omnibus test was not
    # significant.
    for name, table in posthoc.items():
        what = f"`posthoc[{name!r}]`"
        _check_membership(table, what, names)
        _check_columns(table, what, posthoc_table_columns())

    # A pairwise table, unlike a post-hoc one, is rectangular, so it is held to
    # the same alignment every other table in the object is held to.
    for name, by_contrast in pairwise.items():
        if not isinstance(by_contrast, Mapping):
            raise SaInternalError(
                f"internal error: `pairwise[{name!r}]` must be a mapping keyed by contrast."
            )
        for contrast, table in by_contrast.items():
            what = f"`pairwise[{name!r}][{contrast!r}]`"
            _check_aligned(table, what, names)
            _check_columns(table, what, pairwise_table_columns())

    slots: dict[str, Any] = {
        "analysis": analysis,
        "features": names,
        "design": dict(design),
        "parameters": dict(parameters),
        "effect": effect,
        "tests": dict(tests),
        "terms": terms,
        "cells": cells,
        "posthoc": posthoc,
        "pairwise": pairwise,
        "test_info": dict(test_info),
        "diagnostics": diagnostics,
        "metadata": metadata(),
    }

    # A scenario whose model has one term says everything it has to say about
    # that term in `tests`, so the slot is dropped rather than left empty.
    # `cells` goes the same way: a scenario with no crossed grid has no cell to
    # report a mean for, and one level of a single factor is a group, not a cell.
    #
    # A scenario with no pairwise stage has nothing to say in the last two, and
    # an empty map reads as a result that is missing something rather than as one
    # for which the question does not arise. They are absent instead, so a reader
    # asks `"posthoc" in res`.
    #
    # `diagnostics` is deliberately not in this list: it stays as None, exactly
    # as R keeps it, because "not requested" is an answer about this analysis.
    if terms is None:
        del slots["terms"]
    if cells is None:
        del slots["cells"]
    if not posthoc:
        del slots["posthoc"]
    if not pairwise:
        del slots["pairwise"]

    cls = SaComparison if subclass is None else _SUBCLASSES.get(subclass, SaComparison)
    return cls(slots)


def pick_test(res: Any, test: Any, arg: str) -> pd.DataFrame:
    """Pull one test table out of a comparison result.

    Port of ``sa_pick_test()``. Shared by every function that lets the user name
    a test, so the message listing the valid choices is written once.
    """
    if not isinstance(res, SaComparison):
        raise SaValueError(
            f"`{arg}` must be a comparison result, as returned by compare_two_groups()."
        )
    if not isinstance(test, str):
        raise SaValueError("`test` must be a single test name.")
    if test not in res["tests"]:
        raise SaValueError(
            f"`test` must name one of the tests in `{arg}`: "
            + ", ".join(res["tests"])
            + f". Got {test}."
        )
    table: pd.DataFrame = res["tests"][test]
    return table
