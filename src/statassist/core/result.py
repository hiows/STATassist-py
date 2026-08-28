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

import math
import platform
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from .contracts import (
    assoc_scale,
    association_columns,
    categorical_cell_columns,
    categorical_nulls,
    categorical_test_columns,
    cell_table_columns,
    classification_comparison_columns,
    classification_metric_columns,
    cluster_assignment_columns,
    cluster_table_columns,
    curve_columns,
    model_coef_columns,
    model_inference_columns,
    null_label,
    pairwise_table_columns,
    posthoc_table_columns,
    prediction_columns,
    reduction_variance_columns,
    regression_comparison_columns,
    regression_metric_columns,
    selection_profile_columns,
    selection_ranking_columns,
    term_table_columns,
    test_table_columns,
)
from .errors import SaInternalError, SaValueError
from .validate import fmt_est

__all__ = [
    "CLUSTER_ANALYSES",
    "PERFORMANCE_ANALYSES",
    "REDUCTION_ANALYSES",
    "REPR_ALPHA",
    "REPR_PVAL_EPS",
    "SELECTION_ANALYSES",
    "SIGNIFICANCE_COLUMNS",
    "SaCategorical",
    "SaCategoricalSignificance",
    "SaCluster",
    "SaComparison",
    "SaDiagnosis",
    "SaModel",
    "SaPerformance",
    "SaReduction",
    "SaResult",
    "SaSelection",
    "SaSignificance",
    "SaSimulation",
    "SaSplit",
    "metadata",
    "new_categorical",
    "new_categorical_significance",
    "new_cluster",
    "new_comparison",
    "new_model",
    "new_performance",
    "new_reduction",
    "new_selection",
    "new_significance",
    "pick_test",
    "verdict_effect_col",
]

#: Columns of an omnibus or contrast verdict table, in order.
#:
#: A multi-group omnibus table carries ``extreme_level`` and a factorial one
#: ``extreme_cell`` after these, naming the level or cell whose centre produced
#: ``log2fc``. Those are per-scenario additions rather than part of the contract
#: every consumer may rely on, which is why they are not listed here. A term
#: table uses the same shape with ``log2_effect`` in place of ``log2fc``; see
#: :func:`verdict_effect_col`.
SIGNIFICANCE_COLUMNS = ("features", "log2fc", "pvalue", "adj_pvalue", "is_signif")


def verdict_effect_col(table: pd.DataFrame) -> str:
    """The effect-size column a verdict table carries.

    Port of ``sa_verdict_effect_col()``. Term readings store the ANOVA component
    under ``log2_effect``; every other reading keeps ``log2fc``.
    ``draw_volcano_plot()`` and :class:`SaSignificance` both ask here so the
    column they read cannot drift from the one the table was built with.

    A table that still carries its ``term`` attribute is treated as a term
    reading even after the effect column has been dropped, so the missing-
    column message names ``log2_effect`` rather than falling back to ``log2fc``.
    """
    if "log2_effect" in table.columns or table.attrs.get("term") is not None:
        return "log2_effect"
    return "log2fc"


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


#: Threshold ``repr`` counts significant features at.
#:
#: R's ``print.sa_comparison(x, alpha = 0.05)``. Python's ``__repr__`` takes no
#: argument, so the value R lets the caller override is fixed here; a reader who
#: wants another threshold counts it off ``res.tests[name]`` directly.
REPR_ALPHA = 0.05

#: Below this, ``repr`` reports a p-value as a bound rather than as a number.
#:
#: R's ``format.pval(..., eps = 1e-16)``. A p-value smaller than the double
#: precision the statistic was referred through is not a number to print to three
#: digits, so the bound is what is printed instead.
REPR_PVAL_EPS = 1e-16


def _fmt_pval(value: Any) -> str:
    """A p-value the way R's ``format.pval(digits = 3, eps = 1e-16)`` writes it."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    number = float(value)
    if number < REPR_PVAL_EPS:
        return f"<{REPR_PVAL_EPS:g}"
    return fmt_est(number)


class SaComparison(SaResult):
    """The result of comparing groups on a set of features."""

    def __repr__(self) -> str:
        """The summary ``print.sa_comparison`` prints, not the tables.

        Which tests were run and how many features each one called significant.
        The tables are in ``res.tests``, and the pairwise stage in
        ``res.posthoc``.
        """
        design = self["design"]
        params = self["parameters"]
        lines = [f"<{type(self).__name__}> {self['analysis']}"]
        lines.extend(self._header(design, params))
        lines.append(f"  features : {len(self['features'])}")
        lines.append(
            f"  settings : alternative = {params['alternative']}"
            f", conf_level = {_fmt(params['conf_level'])}"
            f", p_adjust = {params['p_adjust']}"
        )

        lines.append("")
        lines.append("  tests")
        lines.extend(self._test_lines())
        if "terms" in self and self["terms"] is not None:
            lines.append("")
            lines.append("  terms")
            lines.extend(self._term_lines())

        if self["diagnostics"] is not None:
            lines.append("")
            lines.append("  $diagnostics attached")
        unmatched = design.get("unmatched_ids") or []
        if len(unmatched) > 0:
            lines.append("")
            lines.append(f"  dropped  : {len(unmatched)} unpaired id(s)")
        if design.get("n_dropped"):
            lines.append(f"  dropped  : {design['n_dropped']} row(s) outside `group_lv`")
        return "\n".join(lines)

    def _header(self, design: Mapping[str, Any], params: Mapping[str, Any]) -> list[str]:
        """The line that says what was compared.

        Chosen from what the design holds rather than from the analysis name: a
        one-sample comparison has a hypothesised value where the others have
        group levels, and a crossed one has factors where they have a flat list.
        """
        if design.get("factor_lv") is not None:
            factors = design["factor_lv"]
            shape = " x ".join(f"{name} ({len(levels)})" for name, levels in factors.items())
            anova = str(design["anova_type"]).replace("_", "-", 1)
            return [
                f"  factors  : {shape}  ({len(design['group_lv'])} cells, independent)",
                f"  anova    : {anova}, Type {params['ss_type']} sums of squares",
            ]
        if design.get("group_lv") is None:
            return [f"  mu       : {_fmt(design['mu'])}"]
        how = (
            f"  (paired by {design['pairing']})"
            if design.get("paired") is True
            else "  (independent)"
        )
        return ["  groups   : " + " vs ".join(design["group_lv"]) + how]

    def _test_lines(self) -> list[str]:
        tests = self["tests"]
        posthoc = self["posthoc"] if "posthoc" in self else {}
        width = max(len(name) for name in tests)
        lines: list[str] = []
        for name, table in tests.items():
            adjusted = table["pval_adj"]
            n_signif = int((adjusted.notna() & (adjusted <= REPR_ALPHA)).sum())
            n_failed = int(table["pval"].isna().sum())
            failed = f"  ({n_failed} not computed)" if n_failed > 0 else ""
            info = self["test_info"][name]
            lines.append(
                f"    ${name:<{width}}  {n_signif} of {len(table.index)} "
                f"at pval_adj <= {_fmt(REPR_ALPHA)}{failed}"
            )
            lines.append(f"    {' ' * (width + 2)}{info['label']}")

            pairs = posthoc.get(name)
            if pairs is not None and len(pairs.index) > 0:
                pair_adj = pairs["pval_adj"]
                n_pairs = int((pair_adj.notna() & (pair_adj <= REPR_ALPHA)).sum())
                lines.append(
                    f"    {' ' * (width + 2)}post-hoc: {n_pairs} of {len(pairs.index)} "
                    f"contrast(s) over {pairs['features'].nunique()} feature(s), "
                    f"{info['posthoc_label']}"
                )
        return lines

    def _term_lines(self) -> list[str]:
        """Which part of the design a feature responds to.

        The whole-model row above says that it responds to the design at all,
        which is not the question a crossed model was fitted to answer.
        """
        terms = self["terms"]
        labels = list(dict.fromkeys(str(label) for label in terms["terms"]))
        width = max(len(label) for label in labels)
        lines: list[str] = []
        for label in labels:
            rows = terms[terms["terms"] == label]
            adjusted = rows["pval_adj"]
            n_signif = int((adjusted.notna() & (adjusted <= REPR_ALPHA)).sum())
            lines.append(
                f"    {label:<{width}}  {n_signif} of {len(rows.index)} "
                f"at pval_adj <= {_fmt(REPR_ALPHA)}"
            )
        return lines


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


# --------------------------------------------------------------------------- #
# The categorical result contract
#
# Port of ``R/categorical.R``, and the one place that says why it is a contract
# of its own rather than another comparison. Every comparison scenario answers
# one question per numeric feature, which is what lets :func:`new_comparison`
# hold each of its tables to one row per feature and what lets
# ``estimate_significance()`` read one ``log2fc`` per row. A contingency table
# has no such axis: the question is asked once, of the table as a whole, and at
# that level there is no signed effect to put beside the p-value - an association
# is not signed, and the odds ratio that is signed only exists when both
# variables are binary.
#
# What this contract keeps is the vocabulary - ``tests`` beside ``test_info``,
# ``design``, ``parameters``, ``metadata`` - so that a reader who knows one
# result object can read this one. What it adds is ``design["null"]``: a
# contingency table can be held against more than one null hypothesis, this
# scenario holds it against three, and ``expected`` means a different number
# under each.
# --------------------------------------------------------------------------- #


class SaCategorical(SaResult):
    """A contingency table, tested against the null its design names."""

    def __repr__(self) -> str:
        """The summary ``print.sa_categorical`` prints, not the cell table.

        The table, the null it was held against, the tests run on it and the
        association measures. ``as_table()`` gives the counts, ``res.cells`` the
        residuals and ``res.association`` the measures with their intervals.
        """
        design = self["design"]
        params = self["parameters"]
        n_row, n_col = design["dim"]
        how = f"matched by {design['pairing']}" if design.get("paired") is True else "independent"

        lines = [
            f"<{type(self).__name__}> {self['analysis']}",
            f"  table    : {design['row_var']} ({n_row}) x {design['col_var']} ({n_col})"
            f"  ({n_row * n_col} cells, {how})",
            f"  null     : {null_label(design['null'])}",
            f"  observed : {design['n_used']} row(s)",
        ]
        settings = (
            f"  settings : conf_level = {_fmt(params['conf_level'])}, correct = {params['correct']}"
        )
        if params.get("simulate_p_value") is True:
            settings += f", simulated on {params['n_resamples']} resample(s)"
        lines.append(settings)

        lines.append("")
        lines.append("  tests")
        lines.extend(self._test_lines())
        lines.append("")
        lines.append("  association")
        lines.extend(self._association_lines())

        if self["diagnostics"] is not None:
            met = "met" if self["diagnostics"]["approx_ok"] else "not met"
            lines.append("")
            lines.append(f"  $diagnostics attached, rule {self['diagnostics']['rule']}: {met}")
        if design.get("n_dropped"):
            lines.append("")
            lines.append(f"  dropped  : {design['n_dropped']} row(s) outside `category_lv`")
        if design.get("n_incomplete"):
            lines.append(
                f"  dropped  : {design['n_incomplete']} row(s) missing a value the table needs"
            )
        return "\n".join(lines)

    def _test_lines(self) -> list[str]:
        """One test per two lines: the verdict, then what was run."""
        tests = self["tests"]
        width = max((len(name) for name in tests), default=0)
        lines: list[str] = []
        for name, table in tests.items():
            pval = float(table["pval"].iloc[0])
            if math.isnan(pval):
                verdict = "not computed"
            elif pval <= REPR_ALPHA:
                verdict = f"null rejected at {_fmt(REPR_ALPHA)}"
            else:
                verdict = f"null retained at {_fmt(REPR_ALPHA)}"
            lines.append(f"    ${name:<{width}}  pval = {_fmt_pval(pval)}  ({verdict})")
            lines.append(f"    {' ' * (width + 2)}{self['test_info'][name]['label']}")
        return lines

    def _association_lines(self) -> list[str]:
        """One measure per line, with its interval where it has one."""
        association = self["association"]
        labels = [str(measure) for measure in association["measure"]]
        width = max((len(label) for label in labels), default=0)
        lines: list[str] = []
        for position, label in enumerate(labels):
            row = association.iloc[position]
            interval = ""
            if not pd.isna(row["lower_conf"]):
                interval = f"  [{fmt_est(row['lower_conf'])}, {fmt_est(row['upper_conf'])}]"
            lines.append(f"    {label:<{width}}  {fmt_est(row['estimate'])}{interval}")
        return lines

    def as_table(self) -> pd.DataFrame:
        """The contingency table this comparison was run on.

        The counterpart of R's ``as.table()`` method. ``cells`` is the canonical
        form, one row per cell, because that is the shape which survives being
        written out as JSON with its labels attached; a table is the shape to
        read it in, so it is built on request rather than stored twice and left
        to drift.

        The rows dropped for a missing value or for a level outside
        ``category_lv`` are already gone, so this is the table the tests were run
        on rather than a fresh crossing of the input.
        """
        from .contingency import categorical_table

        return categorical_table(
            self["cells"], self["design"]["row_var"], self["design"]["col_var"]
        )


def new_categorical(
    analysis: str,
    variables: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    cells: pd.DataFrame,
    tests: Mapping[str, pd.DataFrame],
    test_info: Mapping[str, Any],
    association: pd.DataFrame,
    diagnostics: Any = None,
) -> SaCategorical:
    """Assemble a categorical comparison result.

    Port of ``sa_new_categorical()``. The checks guard the contract rather than
    the user's input, so every one of them raises :class:`SaInternalError`.

    Args:
        analysis: Scenario identifier, ``"categorical_comparison"``.
        variables: The variable names, in the order ``category_lv`` fixed.
        design: How the table was laid out. Must carry ``null``, one of
            :func:`~statassist.core.contracts.categorical_nulls`, and ``dim``,
            the shape of the table the cells came from.
        parameters: The analysis choices as used.
        cells: One row per cell of the table.
        tests: One one-row table per test.
        test_info: One entry per element of ``tests``.
        association: One row per association measure.
        diagnostics: The approximation check, or ``None`` when not requested.
    """
    if design.get("null") not in categorical_nulls():
        raise SaInternalError(
            "internal error: `design['null']` must name one of "
            + ", ".join(categorical_nulls())
            + f", and holds {design.get('null')!r}."
        )
    dim = design.get("dim")
    if dim is None or len(dim) != 2 or any(int(size) < 1 for size in dim):
        raise SaInternalError(
            "internal error: `design['dim']` must be the two dimensions of the table."
        )

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

    for name, table in tests.items():
        what = f"`tests[{name!r}]`"
        if not isinstance(table, pd.DataFrame):
            raise SaInternalError(f"internal error: {what} must be a DataFrame.")
        if len(table.index) != 1:
            raise SaInternalError(
                f"internal error: {what} must hold exactly one row, one table being "
                f"one question, but holds {len(table.index)}."
            )
        _check_columns(table, what, categorical_test_columns())

    if not isinstance(cells, pd.DataFrame):
        raise SaInternalError("internal error: `cells` must be a DataFrame.")
    _check_columns(cells, "`cells`", categorical_cell_columns())
    expected_rows = int(dim[0]) * int(dim[1])
    if len(cells.index) != expected_rows:
        raise SaInternalError(
            f"internal error: `cells` holds {len(cells.index)} row(s) for a "
            + " x ".join(str(int(size)) for size in dim)
            + " table."
        )

    if not isinstance(association, pd.DataFrame):
        raise SaInternalError("internal error: `association` must be a DataFrame.")
    _check_columns(association, "`association`", association_columns())

    return SaCategorical(
        {
            "analysis": analysis,
            "variables": [str(name) for name in variables],
            "design": dict(design),
            "parameters": dict(parameters),
            "cells": cells,
            "tests": dict(tests),
            "test_info": dict(test_info),
            "association": association,
            "diagnostics": diagnostics,
            "metadata": metadata(),
        }
    )


class SaSignificance(SaResult):
    """A comparison reduced to one significance verdict per feature.

    Two slots, as R has: ``analysis_type``, the ``analysis`` of the comparison
    the verdict was read from, and ``significance``, either one verdict table or
    a mapping of them - one per contrast or per model term.

    The cutoffs, the test and the adjustment that produced a table are carried
    in that table's ``attrs`` rather than in a slot of their own, which is where
    ``draw_volcano_plot()`` reads them back from so that a plotted guide cannot
    disagree with the verdict beside it. R puts them in the same place, as
    attributes of the data.frame.
    """

    def __repr__(self) -> str:
        """The summary ``print.sa_significance`` prints, not the table.

        The rule that was applied and how many features cleared it. The table
        itself is in ``significance``.
        """
        held = self["significance"]
        # A contrast or term reading holds one table each, and every one of them
        # was judged by the same rule, so the header is read off whichever comes
        # first.
        tables = {"": held} if isinstance(held, pd.DataFrame) else dict(held)
        first = next(iter(tables.values()))
        attrs = first.attrs

        # Tukey's HSD and Games-Howell report family-wise p-values, so the
        # pairwise stage adjusted nothing and the entry is absent rather than
        # "none".
        adj_used = attrs.get("adj_type") or "none"
        effect_col = verdict_effect_col(first)
        lines = [
            f"<{type(self).__name__}> {self['analysis_type']}",
            f"  test     : {attrs.get('test')}  ({attrs.get('test_label')})",
            f"  cutoffs  : abs({effect_col}) >= {_fmt(attrs.get('log2fc_cutoff', float('nan')))}"
            f", adj_pvalue <= {_fmt(attrs.get('pval_cutoff', float('nan')))}  ({adj_used})",
        ]

        def count(table: pd.DataFrame) -> str:
            flags = table["is_signif"]
            n_signif = int((flags == True).sum())  # noqa: E712 - NA must not count
            n_undecided = int(flags.isna().sum())
            undecided = f"  ({n_undecided} undecided)" if n_undecided > 0 else ""
            return f"{n_signif} of {len(table.index)} significant{undecided}"

        if isinstance(held, pd.DataFrame):
            lines.append(f"  verdict  : {count(held)}")
            return "\n".join(lines)

        # Which axis the mapping runs along is a property of the tables, not of
        # the object, since the reading is not recorded anywhere else.
        axis = "contrast" if first.attrs.get("term") is None else "term"
        lines.append("")
        lines.append(f"  $significance, one table per {axis}")
        width = max((len(name) for name in tables), default=0)
        for name, table in tables.items():
            lines.append(f"    {name:<{width}}  {count(table)}")
        return "\n".join(lines)


def new_significance(
    analysis_type: str,
    significance: pd.DataFrame | Mapping[str, pd.DataFrame],
) -> SaSignificance:
    """Wrap a verdict table in the object the user sees.

    Port of ``sa_new_significance()``. The scenario name sits beside the table
    rather than only in its attributes, since a consumer reading the verdict has
    to know which question the ``log2fc`` column answers, and ``attrs`` is easy
    to lose and easy to miss.

    Args:
        analysis_type: The ``analysis`` of the comparison read from.
        significance: One verdict table, or a mapping of them keyed by contrast
            or by model term, in the order the comparison fixes.
    """
    if isinstance(significance, pd.DataFrame):
        tables = [significance]
        held: pd.DataFrame | dict[str, pd.DataFrame] = significance
    else:
        held = dict(significance)
        tables = list(held.values())
        if not tables:
            raise SaInternalError("internal error: `significance` must hold at least one table.")

    for table in tables:
        effect_col = verdict_effect_col(table)
        required = (
            "features",
            effect_col,
            "pvalue",
            "adj_pvalue",
            "is_signif",
        )
        absent = [name for name in required if name not in table.columns]
        if absent:
            raise SaInternalError(
                "internal error: a verdict table is missing contract column(s): "
                + ", ".join(absent)
                + "."
            )

    return SaSignificance({"analysis_type": analysis_type, "significance": held})


class SaCategoricalSignificance(SaResult):
    """A contingency table reduced to a verdict, per cell or for the table.

    Deliberately not a :class:`SaSignificance`. Its columns are cells rather
    than features and ``log2_lift`` rather than ``log2fc``, so
    ``draw_volcano_plot()`` refusing it is the point of the separate class rather
    than an omission; ``draw_mosaic_plot()`` is what draws this scenario.

    As there, the rule that produced a table is carried in that table's
    ``attrs`` rather than in a slot of its own.
    """

    def __repr__(self) -> str:
        """The summary ``print.sa_categorical_significance`` prints.

        The rule that was applied and what cleared it. The table itself is in
        ``significance``.
        """
        table = self["significance"]
        attrs = table.attrs
        by = attrs.get("by")
        dims = attrs.get("table_dim") or []

        lines = [
            f"<{type(self).__name__}> {self['analysis_type']}",
            f"  reading  : {by}  (" + " x ".join(str(int(size)) for size in dims) + " table)",
            f"  null     : {null_label(str(attrs.get('null')))}",
        ]
        if by == "cell":
            lines.extend(self._cell_lines(table))
            return "\n".join(lines)
        lines.extend(self._table_lines(table))
        return "\n".join(lines)

    def _cell_lines(self, table: pd.DataFrame) -> list[str]:
        """The cutoffs a cell reading applied, and the cells that cleared them."""
        attrs = table.attrs
        flags = table["is_signif"]
        n_signif = int((flags == True).sum())  # noqa: E712 - NA must not count
        n_undecided = int(flags.isna().sum())
        undecided = f"  ({n_undecided} undecided)" if n_undecided > 0 else ""

        lines = [
            f"  cutoffs  : abs(log2_lift) >= {_fmt(attrs.get('log2_lift_cutoff', float('nan')))}"
            f", adj_pvalue <= {_fmt(attrs.get('pval_cutoff', float('nan')))}"
            f"  ({attrs.get('adj_type')})",
            f"  verdict  : {n_signif} of {len(table.index)} cell(s) significant{undecided}",
        ]

        cleared = (flags == True).fillna(False)  # noqa: E712 - NA must not count
        hits = [position for position, keep in enumerate(cleared) if bool(keep)]
        if not hits:
            return lines

        labels = [f"{table['row_level'].iloc[at]} : {table['col_level'].iloc[at]}" for at in hits]
        width = max(len(label) for label in labels)
        lines.append("")
        lines.append("  cells")
        for label, at in zip(labels, hits, strict=True):
            lines.append(
                f"    {label:<{width}}  lift = {fmt_est(table['lift'].iloc[at])}"
                f", adj_pvalue = {_fmt_pval(table['adj_pvalue'].iloc[at])}"
            )
        return lines

    def _table_lines(self, table: pd.DataFrame) -> list[str]:
        """The test and the measure a table reading was judged on."""
        attrs = table.attrs
        measure = str(table["measure"].iloc[0])
        cutoff = attrs.get("effect_cutoff")
        if cutoff is None:
            rule = ""
        elif assoc_scale(measure) == "ratio":
            rule = f"{measure} >= {_fmt(cutoff)} or <= {_fmt(1 / cutoff)}, "
        else:
            rule = f"abs({measure}) >= {_fmt(cutoff)}, "

        flag = table["is_signif"].iloc[0]
        if pd.isna(flag):
            verdict = "undecided"
        elif bool(flag):
            verdict = "significant"
        else:
            verdict = "not significant"

        interval = ""
        if not pd.isna(table["lower_conf"].iloc[0]):
            interval = (
                f"  [{fmt_est(table['lower_conf'].iloc[0])}, "
                f"{fmt_est(table['upper_conf'].iloc[0])}]"
            )
        return [
            f"  test     : {attrs.get('test')}  ({attrs.get('test_label')})",
            f"  cutoffs  : {rule}pvalue <= {_fmt(attrs.get('pval_cutoff', float('nan')))}",
            f"  verdict  : {measure} = {fmt_est(table['estimate'].iloc[0])}{interval}  ({verdict})",
        ]


def new_categorical_significance(
    analysis_type: str,
    significance: pd.DataFrame,
) -> SaCategoricalSignificance:
    """Wrap a categorical verdict in the object the user sees.

    Port of ``sa_new_categorical_significance()``. Two slots, as R has: the
    ``analysis`` of the comparison the verdict was read from, and the verdict
    table itself.
    """
    if not isinstance(significance, pd.DataFrame):
        raise SaInternalError("internal error: `significance` must be a DataFrame.")
    return SaCategoricalSignificance({"analysis_type": analysis_type, "significance": significance})


# --------------------------------------------------------------------------- #
# The fitted model contract
#
# Port of ``R/model.R``. A comparison result is organised around a feature axis:
# every table repeats ``features`` in the same order. A model has no feature
# axis. It has one outcome and a set of terms, and the terms are not the columns
# that were passed in, since a factor predictor becomes several of them.
# ``terms`` is therefore what ``features`` is over there: the row order every
# table in the object follows.
#
# Where this port parts from R is the fitted engine object. R keeps it in a
# ``fit`` slot and says so as the one documented exception to the JSON rule.
# Here it is reached as ``model.fit``, an attribute rather than a slot, so
# ``"fit" in model`` is ``False`` and ``model.to_dict()`` is JSON-shaped without
# anything having to be dropped first. A model that cannot predict is still not
# much of a model, which is why the handle is kept at all.
# --------------------------------------------------------------------------- #


class SaModel(SaResult):
    """A model fitted to one outcome on a set of predictors.

    Ten slots, and the engine handle beside them as :attr:`fit`. The slots are
    the portable part: ``analysis``, ``terms``, ``design``, ``parameters``,
    ``coefficients``, ``fit_stats``, ``performance``, ``resampling``, ``engine``
    and ``metadata``.

    What ``coefficients`` holds depends on the model, and its columns say which
    kind it is. An unpenalized fit carries the standard error, the statistic, the
    p-value and the confidence limits, so ``"pval" in model.coefficients`` is the
    test for a fit that can be asked for inference. A penalized one has no
    standard error to report and carries ``selected`` instead. A forest and a
    machine have no coefficient of any kind and their table is the importance of
    each term.
    """

    __slots__ = ("_fit",)

    def __init__(self, slots: Mapping[str, Any], fit: Any = None) -> None:
        super().__init__(slots)
        object.__setattr__(self, "_fit", fit)

    @property
    def fit(self) -> Any:
        """The engine object, kept so that the model can predict.

        Deliberately not a slot. Every slot of a result object in this package is
        a scalar, a string, a list, a mapping or a DataFrame, which is what lets
        the object be written out as JSON; an estimator is none of those. R keeps
        it as ``$fit`` and documents the exception, and the exception is not
        needed here because an attribute is reached the same way and iterated
        over by nothing.
        """
        return self._fit

    def coef(self) -> pd.DataFrame:
        """The coefficient table, ``model.coefficients`` itself.

        The counterpart of ``coef.sa_model()``: one row per term, in the order of
        ``terms``, carrying everything the model estimated about each of them
        rather than the estimate alone. Every term keeps its row either way - a
        term a penalty dropped with an ``estimate`` of exactly 0, one the engine
        could not estimate with ``NaN`` - since a table shorter than the model
        would be a table that had lost terms.
        """
        table: pd.DataFrame = self["coefficients"]
        return table

    def predict(self, newdata: Any = None, type: str = "raw") -> Any:  # noqa: A002
        """Predict on rows the model was not fitted to.

        Port of ``predict.sa_model()``. Takes the data frame the fit took - the
        test half of a :func:`~statassist.split_data` result, say - and answers
        one prediction per row of it. The columns are read by name and coded the
        way the fit coded them, so the rows to predict can be handed over exactly
        as they came, outcome column and all.

        Args:
            newdata: Rows to predict, or ``None`` for the rows the model was
                fitted on.
            type: ``"raw"`` for the prediction itself, a fitted value for a
                regression and a class label for a classification;
                ``"response"`` for the prediction on the scale of the outcome,
                which for a classification is the probability of
                ``design["outcome_lv"][1]``, the class the coefficients describe;
                or ``"prob"`` for one column per class.

        Returns:
            One prediction per row of ``newdata``. Rows that are not complete
            across the predictors are missing throughout, which is the rule the
            fit already follows in reverse: those are the rows
            ``design["n_dropped"]`` counted.
        """
        from ..fit._shared import predict_model

        return predict_model(self, newdata, type)

    def __repr__(self) -> str:
        """The summary ``print.sa_model`` prints, not every table.

        What was fitted to what, and how it did. The coefficient table is in
        ``model.coefficients``, the resampled folds in ``model.resampling``.
        """
        design = self["design"]
        params = self["parameters"]
        lines = [
            f"<{type(self).__name__}> {self['analysis']}",
            f"  outcome  : {design['outcome']}  ({design['outcome_type']})",
        ]
        if design.get("outcome_lv") is not None:
            scale = "the odds of " if "odds_ratio" in self["coefficients"] else ""
            lines.append(
                f"             modelling {scale}{design['outcome_lv'][1]} against "
                f"{design['outcome_lv'][0]}, {design['n_events']} of "
                f"{design['n_used']} row(s)"
            )
        dropped = ""
        if design["n_dropped"] > 0:
            dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)"
        lines.append(f"  rows     : {design['n_used']} used{dropped}")
        lines.append(
            f"  terms    : {len(self['terms'])} over {len(design['predictors'])} predictor(s)"
        )
        lines.append(f"  settings : {self._settings_line(params)}")
        lines.extend(self._tuning_lines(params))
        lines.append("")
        lines.extend(self._coef_lines())
        lines.append("")
        lines.append("  fit      : " + self._stats_line())
        if self["performance"] is not None:
            lines.append("  resample : " + self._resample_line(params))
        if design["dropped_predictors"]:
            lines.append(
                "  dropped  : " + ", ".join(design["dropped_predictors"]) + " (single valued)"
            )
        return "\n".join(lines)

    def _settings_line(self, params: Mapping[str, Any]) -> str:
        """How the model was scored, and to what confidence."""
        if params["cv"]:
            scheme = str(params["cv_method"])
            if params["n_fold"] is not None:
                scheme += f", {params['n_fold']} fold(s)"
            if params["n_repeat"] is not None:
                scheme += f" x {params['n_repeat']} repeat(s)"
        else:
            scheme = "no resampling"
        # A model that reports no interval has no confidence level to report
        # either, so the label is left out rather than printed against nothing.
        if params.get("conf_level") is not None:
            scheme += f", conf_level = {_fmt(params['conf_level'])}"
        return scheme

    def _tuning_lines(self, params: Mapping[str, Any]) -> list[str]:
        """The hyperparameters the model was fitted at, where it has any.

        Read off ``parameters`` rather than off the engine, so the values printed
        are the ones the result records as chosen.
        """
        chosen = ""
        if params["cv"] and params.get("n_candidates", 1) > 1:
            chosen = f"  (chosen from {params['n_candidates']} candidate(s))"

        if params.get("penalty") is not None:
            return [
                f"  penalty  : {params['penalty']}, alpha = {fmt_est(params['alpha'])}"
                f", lambda = {fmt_est(params['lambda'])}{chosen}"
            ]
        if params.get("ntree") is not None:
            return [
                f"  forest   : {params['ntree']} tree(s), mtry = {params['mtry']}"
                f", nodesize = {params['nodesize']}{chosen}"
            ]
        if params.get("kernel") is not None:
            return [
                f"  kernel   : {params['kernel']}, C = {fmt_est(params['C'])}"
                f", sigma = {fmt_est(params['sigma'])}{chosen}"
            ]
        return []

    def _coef_lines(self) -> list[str]:
        """The head of the coefficient table, with the heading it deserves."""
        table = self["coefficients"]
        inference = "pval" in table.columns
        # A forest and a machine have no coefficients at all; what they answer
        # with is how much each term was worth to them, and `selected` is what
        # tells a penalized table from theirs.
        importance = not inference and "selected" not in table.columns
        lines = ["  " + ("importance  (permutation)" if importance else "coefficients")]

        shown = table.head(_REPR_MAX_TERMS)
        width = max((len(str(name)) for name in shown["terms"]), default=0)
        for position in range(len(shown.index)):
            row = shown.iloc[position]
            tail = ""
            if inference:
                tail = (
                    f"  [{fmt_est(row['lower_conf'])}, {fmt_est(row['upper_conf'])}]"
                    f"  p = {_fmt_pval(row['pval'])}"
                )
            elif "selected" in table.columns:
                tail = "  " + ("selected" if bool(row["selected"]) else "dropped")
            lines.append(f"    {str(row['terms']):<{width}}  {fmt_est(row['estimate']):>10}{tail}")
        if len(table.index) > len(shown.index):
            lines.append(
                f"    ... and {len(table.index) - len(shown.index)} more term(s) in $coefficients"
            )
        return lines

    def _stats_line(self) -> str:
        """The goodness-of-fit scalars, which are per-model rather than per-term."""
        return ", ".join(f"{name} = {fmt_est(value)}" for name, value in self["fit_stats"].items())

    def _resample_line(self, params: Mapping[str, Any]) -> str:
        """The resampled score at the hyperparameters that were chosen."""
        best = performance_row(self["performance"], params)
        scored = []
        for metric in self["engine"]["metrics"]:
            spread = ""
            sd_col = f"{metric}SD"
            if sd_col in best:
                spread = f" (SD {fmt_est(best[sd_col])})"
            scored.append(f"{metric} = {fmt_est(best[metric])}{spread}")
        over = ""
        if self["resampling"] is not None:
            over = f" over {len(self['resampling'].index)} resample(s)"
        return ", ".join(scored) + over


#: Coefficient rows ``repr`` shows before counting the rest.
#:
#: R's ``print.sa_model(x, n = 10L)``. Python's ``__repr__`` takes no argument,
#: so the value R lets the caller override is fixed here.
_REPR_MAX_TERMS = 10


def performance_row(performance: pd.DataFrame, chosen: Mapping[str, Any]) -> pd.Series:
    """The row of the performance table the model was actually fitted at.

    Port of ``sa_performance_row()``. ``performance`` holds one row per
    hyperparameter combination in the order they were scored rather than in the
    order they placed, so the chosen combination is the one ``parameters`` names
    and not the first row.

    The chosen values are read from the result's own ``parameters`` rather than
    from the engine, so this asks nothing of the fitted object. Falling back to
    the first row rather than failing is deliberate: this is only ever used to
    summarise for printing, and a table that cannot be matched is not a reason to
    refuse to print the rest of the object.
    """
    if len(performance.index) == 1:
        return performance.iloc[0]
    keep = pd.Series(True, index=performance.index)
    for name in performance.columns:
        if name in chosen and chosen[name] is not None:
            keep &= performance[name] == chosen[name]
    hits = performance.index[keep]
    return performance.loc[hits[0]] if len(hits) > 0 else performance.iloc[0]


def new_model(
    analysis: str,
    terms: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    coefficients: pd.DataFrame,
    fit_stats: Mapping[str, Any],
    engine: Mapping[str, Any],
    fit: Any,
    performance: pd.DataFrame | None = None,
    resampling: pd.DataFrame | None = None,
) -> SaModel:
    """Assemble a fitted model result object.

    Port of ``sa_new_model()``. The checks here guard the contract rather than
    the user's input, so every one of them raises :class:`SaInternalError`.

    Args:
        analysis: Model identifier, such as ``"linear_regression"``.
        terms: Coefficient term names, in the row order every table uses.
        design: What the model saw: the outcome, its type and levels, the
            predictors, and how many rows were usable.
        parameters: The fitting choices, with the resampling arguments as they
            were actually used rather than as they were passed.
        coefficients: One row per term.
        fit_stats: Goodness-of-fit scalars for the model as a whole, which are
            per-model rather than per-term and so do not fit the table.
        engine: What actually fitted the model. Must name ``package``,
            ``method``, ``label`` and ``metrics``.
        fit: The engine object, reached as :attr:`SaModel.fit`.
        performance: One row per hyperparameter combination, or ``None`` when
            nothing was resampled.
        resampling: One row per resample, or ``None``.
    """
    names = [str(name) for name in terms]
    if not names:
        raise SaInternalError("internal error: `terms` must be a non-empty sequence.")
    if not isinstance(coefficients, pd.DataFrame):
        raise SaInternalError("internal error: `coefficients` must be a DataFrame.")
    if [str(name) for name in coefficients["terms"]] != names:
        raise SaInternalError("internal error: `coefficients` is not aligned with `terms`.")
    _check_columns(coefficients, "`coefficients`", model_coef_columns())

    inference = model_inference_columns()
    present = [name for name in inference if name in coefficients.columns]
    if 0 < len(present) < len(inference):
        raise SaInternalError(
            "internal error: `coefficients` carries some inference column(s) and not "
            "others, so it is neither a table with inference nor one without: "
            + ", ".join(present)
            + "."
        )

    for name in ("package", "method", "label", "metrics"):
        if engine.get(name) is None:
            raise SaInternalError(f"internal error: `engine` is missing `{name}`.")
    for name, table in (("performance", performance), ("resampling", resampling)):
        if table is not None and not isinstance(table, pd.DataFrame):
            raise SaInternalError(f"internal error: `{name}` must be a DataFrame or None.")

    return SaModel(
        {
            "analysis": analysis,
            "terms": names,
            "design": dict(design),
            "parameters": dict(parameters),
            "coefficients": coefficients,
            "fit_stats": dict(fit_stats),
            "performance": performance,
            "resampling": resampling,
            "engine": dict(engine),
            "metadata": metadata(),
        },
        fit=fit,
    )


# --------------------------------------------------------------------------- #
# The evaluation contract
#
# Port of ``R/performance.R``. A comparison result is organised around a feature
# axis and a model around a term axis. An evaluation has neither: it holds a set
# of *models*, scored on rows none of them was fitted on, so ``models`` takes the
# place ``terms`` holds over there and every table repeats it in the same order.
#
# The exception :class:`SaModel` needed does not arise here. An evaluation has
# nothing to predict with - the calibration line of a regression is two numbers
# rather than the fit that produced them, and a ROC curve is a table of
# thresholds rather than whatever computed it - so every slot is JSON-shaped and
# there is no engine handle beside them.
# --------------------------------------------------------------------------- #

#: The analyses :func:`new_performance` will assemble, in the order R lists them.
PERFORMANCE_ANALYSES = ("regression_performance", "classification_performance")


class SaPerformance(SaResult):
    """A set of models scored on rows none of them was fitted on.

    Eight slots for a regression and nine for a classification: ``analysis``,
    ``models``, ``design``, ``parameters``, ``predictions``, ``metrics``,
    ``comparisons``, ``curves`` and ``metadata``. The last two of those are
    dropped rather than left empty where the question does not arise, so
    ``"comparisons" in res`` is what says whether anything was held against the
    baseline and ``"curves" in res`` tells a classification from a regression as
    surely as ``analysis`` does.
    """

    def __repr__(self) -> str:
        """The summary ``print.sa_performance`` prints, not every table.

        What was scored on what, and how each model did. The per-model scores are
        in ``res.metrics``, what each model did against the baseline in
        ``res.comparisons``, and the predictions themselves in
        ``res.predictions``.
        """
        design = self["design"]
        classify = self["analysis"] == "classification_performance"

        lines = [
            f"<{type(self).__name__}> {self['analysis']}",
            f"  outcome  : {design['outcome']}  ({design['outcome_type']})",
        ]
        if design.get("outcome_lv") is not None:
            lines.append(
                f"             scoring the probability of {design['outcome_lv'][1]}"
                f" against {design['outcome_lv'][0]}, {design['n_events']} of "
                f"{design['n_used']} row(s)"
            )
        dropped = f"  ({design['n_dropped']} row(s) dropped)" if design["n_dropped"] > 0 else ""
        lines.append(f"  rows     : {design['n_used']} scored{dropped}")
        lines.append(f"  models   : {len(self['models'])}, baseline = {design['baseline']}")
        if classify:
            lines.append(
                f"  threshold: {_fmt(self['parameters']['threshold'])}"
                "  (accuracy, sensitivity and specificity only)"
            )

        lines.append("")
        lines.extend(self._metric_lines(classify))
        if self._slots.get("comparisons") is not None:
            lines.append("")
            lines.extend(self._comparison_lines(classify, design["baseline"]))
        return "\n".join(lines)

    def _metric_lines(self, classify: bool) -> list[str]:
        """One line per model, and a count of the models that did not fit."""
        table = self["metrics"]
        shown = table.head(_REPR_MAX_MODELS)
        width = max((len(str(name)) for name in shown["model"]), default=0)
        lines = ["  metrics"]
        for position in range(len(shown.index)):
            row = shown.iloc[position]
            if classify:
                text = (
                    f"auc = {fmt_est(row['auc'])}"
                    f"  [{fmt_est(row['auc_lower_conf'])}, {fmt_est(row['auc_upper_conf'])}]"
                    f", brier = {fmt_est(row['brier'])}"
                    f", accuracy = {fmt_est(row['accuracy'])}"
                )
            else:
                text = (
                    f"cor = {fmt_est(row['cor'])}"
                    f", r_squared = {fmt_est(row['r_squared'])}"
                    f", rmse = {fmt_est(row['rmse'])}"
                    f", mae = {fmt_est(row['mae'])}"
                )
            lines.append(f"    {str(row['model']):<{width}}  {text}")
        if len(table.index) > len(shown.index):
            lines.append(
                f"    ... and {len(table.index) - len(shown.index)} more model(s) in $metrics"
            )
        return lines

    def _comparison_lines(self, classify: bool, baseline: str) -> list[str]:
        """What each model did against the baseline.

        A classification reports three estimates with an interval and a p-value
        apiece, which do not fit on a line beside a model name; folding them onto
        one would put the name of the model further from the numbers than the
        numbers are from each other, so the name gets a line of its own.
        """
        table = self["comparisons"]
        shown = table.head(_REPR_MAX_MODELS)
        width = max((len(str(name)) for name in shown["model"]), default=0)
        lines = [f"  comparisons  (against {baseline})"]
        for position in range(len(shown.index)):
            row = shown.iloc[position]
            if classify:
                lines.append(f"    {row['model']}")
                for label, stem in (("delta_auc", "delta_auc"), ("IDI", "idi"), ("NRI", "nri")):
                    lines.append(
                        f"      {label:<9}  = {fmt_est(row[stem])}"
                        f"  [{fmt_est(row[stem + '_lower_conf'])}, "
                        f"{fmt_est(row[stem + '_upper_conf'])}]"
                        f"  p = {_fmt_pval(row[stem + '_pval'])}"
                    )
            else:
                lines.append(
                    f"    {str(row['model']):<{width}}  "
                    f"delta_cor = {fmt_est(row['delta_cor'])}"
                    f", delta_r_squared = {fmt_est(row['delta_r_squared'])}"
                    f", delta_rmse = {fmt_est(row['delta_rmse'])}"
                    f", delta_mae = {fmt_est(row['delta_mae'])}"
                )
        if len(table.index) > len(shown.index):
            lines.append(
                f"    ... and {len(table.index) - len(shown.index)} more model(s) in $comparisons"
            )
        return lines


#: Model rows ``repr`` shows before counting the rest.
#:
#: R's ``print.sa_performance(x, n = 10L)``. Python's ``__repr__`` takes no
#: argument, so the value R lets the caller override is fixed here.
_REPR_MAX_MODELS = 10


def new_performance(
    analysis: str,
    models: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame | None = None,
    curves: pd.DataFrame | None = None,
) -> SaPerformance:
    """Assemble an evaluation result object.

    Port of ``sa_new_performance()``. The checks here guard the contract rather
    than the user's input, so every one of them raises :class:`SaInternalError`.

    Args:
        analysis: One of :data:`PERFORMANCE_ANALYSES`.
        models: Model names, the baseline first, in the row order every table
            uses.
        design: What was scored: the ``outcome`` label, its type, the
            ``baseline`` model's name, and the row counts.
        parameters: The scoring choices.
        predictions: One row per model and scored row.
        metrics: One row per model.
        comparisons: One row per model other than the baseline, or ``None`` when
            nothing was held against it, in which case the slot is left out.
        curves: One row per model and operating point, or ``None`` for a
            regression, in which case the slot is left out.
    """
    names = [str(name) for name in models]
    if not names:
        raise SaInternalError("internal error: `models` must be a non-empty sequence.")
    if len(set(names)) != len(names):
        raise SaInternalError("internal error: `models` must be unique.")
    if analysis not in PERFORMANCE_ANALYSES:
        raise SaInternalError(f"internal error: unknown analysis `{analysis}`.")
    if design.get("baseline") != names[0]:
        raise SaInternalError("internal error: `design['baseline']` is not the first of `models`.")

    regression = analysis == "regression_performance"
    metric_contract = regression_metric_columns() if regression else classification_metric_columns()
    comparison_contract = (
        regression_comparison_columns() if regression else classification_comparison_columns()
    )

    if not isinstance(metrics, pd.DataFrame):
        raise SaInternalError("internal error: `metrics` must be a DataFrame.")
    _check_columns(metrics, "`metrics`", metric_contract)
    if [str(name) for name in metrics["model"]] != names:
        raise SaInternalError("internal error: `metrics` is not aligned with `models`.")

    if not isinstance(predictions, pd.DataFrame):
        raise SaInternalError("internal error: `predictions` must be a DataFrame.")
    _check_columns(predictions, "`predictions`", prediction_columns())
    # Every model was scored on the same rows, which is the whole point of the
    # intersection the caller took, so the table holds each model once and in
    # order rather than merely holding known names.
    if list(dict.fromkeys(str(name) for name in predictions["model"])) != names:
        raise SaInternalError(
            "internal error: `predictions` does not hold every model once, in order."
        )

    # A comparison is against the baseline, so the baseline has no row of its own
    # and a name that is not a model of this evaluation has nowhere to come from.
    if comparisons is not None:
        if not isinstance(comparisons, pd.DataFrame):
            raise SaInternalError("internal error: `comparisons` must be a DataFrame or None.")
        _check_columns(comparisons, "`comparisons`", comparison_contract)
        if [str(name) for name in comparisons["model"]] != names[1:]:
            raise SaInternalError(
                "internal error: `comparisons` must hold every model other than the "
                "baseline, once and in order."
            )

    if curves is not None:
        if regression:
            raise SaInternalError("internal error: a regression evaluation has no ROC curve.")
        if not isinstance(curves, pd.DataFrame):
            raise SaInternalError("internal error: `curves` must be a DataFrame or None.")
        _check_columns(curves, "`curves`", curve_columns())
        unknown = list(dict.fromkeys(str(n) for n in curves["model"] if str(n) not in set(names)))
        if unknown:
            raise SaInternalError(
                "internal error: `curves` holds model(s) absent from the evaluation: "
                + ", ".join(unknown)
                + "."
            )

    slots: dict[str, Any] = {
        "analysis": analysis,
        "models": names,
        "design": dict(design),
        "parameters": dict(parameters),
        "predictions": predictions,
        "metrics": metrics,
        "comparisons": comparisons,
        "curves": curves,
        "metadata": metadata(),
    }
    # An evaluation of one model compared it to nothing, and a regression has no
    # curve to report. An empty table in either slot would read as a result that
    # lost its values rather than as one for which the question does not arise,
    # so the slot goes instead and ``"comparisons" in res`` is the test.
    if comparisons is None:
        del slots["comparisons"]
    if curves is None:
        del slots["curves"]
    return SaPerformance(slots)


#: The clusterings this contract covers.
#:
#: Port of ``sa_cluster_analyses()``. ``analysis`` names the method rather than the
#: family, as it does in :class:`SaModel`, so a result says which of the four it
#: came from without anything having to read ``engine``. ``"snn"`` is shared
#: nearest neighbour clustering on a k-nearest-neighbour graph.
CLUSTER_ANALYSES = ("hclust", "kmeans", "dbscan", "snn")

#: Clusters ``repr`` reports the size of before counting the rest.
#:
#: R's ``print.sa_cluster(x, n = 10L)``, fixed here for the reason
#: :data:`_REPR_MAX_MODELS` is.
_REPR_MAX_CLUSTERS = 10


class SaCluster(SaResult):
    """A label per point, and what the grouping came out as.

    Eight slots, and the engine handle beside them as :attr:`fit`: ``analysis``,
    ``points``, ``design``, ``parameters``, ``assignments``, ``clusters``,
    ``engine`` and ``metadata``.

    ``points`` is the row axis, the same name a reduction uses, so an assignment
    is aligned with a set of coordinates of the same rows without either of them
    being asked where its rows came from.

    The contract holds two shapes at once with one number rather than two slots.
    ``assignments["cluster"]`` is a whole number per point and ``0`` is noise,
    which is DBSCAN's own convention: a cluster number of zero is not a cluster.
    ``clusters`` then holds one row per real cluster and never a row for zero, so
    its length is the number of groups that were found and ``design["n_noise"]``
    is what did not join one. A partitioning method reports ``n_noise = 0`` and
    the two shapes read down the same columns.
    """

    __slots__ = ("_fit",)

    def __init__(self, slots: Mapping[str, Any], fit: Any = None) -> None:
        super().__init__(slots)
        object.__setattr__(self, "_fit", fit)

    @property
    def fit(self) -> Any:
        """The engine object, kept for the questions the tables cannot answer.

        A linkage is the tree, and the tree is most of what a hierarchical
        clustering has to say: a cut at another height is a question the caller
        should not have to rebuild the distance matrix to ask. Deliberately not a
        slot, for the reason :attr:`SaModel.fit` is not one.
        """
        return self._fit

    def __repr__(self) -> str:
        """The summary ``print.sa_cluster`` prints, not the label of every point.

        Those are in ``res.assignments``, and the engine object is ``res.fit``.
        """
        design = self["design"]
        params = self["parameters"]
        dropped = ""
        if design["n_dropped"] > 0:
            dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)"
        noise = ""
        if design["n_noise"] > 0:
            noise = f"  ({design['n_noise']} point(s) left as noise)"

        lines = [
            f"<{type(self).__name__}> {self['analysis']}",
            f"  data     : {design['n_used']} sample(s) x {design['n_feats']} feature(s){dropped}",
            f"  points   : {len(self['points'])} {design['point_type']}(s)",
            f"  scaling  : {_scaling_line(params['center'], params['scale'])}",
            # The count leads, because for two of the four methods it is the
            # answer rather than something that was asked for.
            f"  clusters : {design['n_clusters']}{noise}",
        ]
        if design["n_clusters"] > 0:
            lines.extend(self._cluster_lines(design, params))
        lines.extend(self._method_lines(params))
        if design["dropped_feats"]:
            lines.append("  dropped  : " + ", ".join(design["dropped_feats"]) + " (no variance)")
        return "\n".join(lines)

    def _cluster_lines(self, design: Mapping[str, Any], params: Mapping[str, Any]) -> list[str]:
        """The size and mean width of each cluster, and the overall width."""
        table = self["clusters"]
        shown = table.head(_REPR_MAX_CLUSTERS)
        sizes = "; ".join(
            f"#{int(row['cluster'])} n = {int(row['size'])}, s = {fmt_est(row['silhouette'])}"
            for _, row in shown.iterrows()
        )
        if len(shown.index) < design["n_clusters"]:
            sizes += f"  ({len(shown.index)} of {design['n_clusters']} shown)"
        widths = self["assignments"]["silhouette"]
        return [
            f"  sizes    : {sizes}",
            f"  silhouette: mean {fmt_est(widths.mean(skipna=True))} over the "
            f"{int(widths.notna().sum())} assigned {design['point_type']}(s), on the "
            f"{params['dist_method']} distance",
        ]

    def _method_lines(self, params: Mapping[str, Any]) -> list[str]:
        """The one line that only this method has to report."""
        analysis = self["analysis"]
        if analysis == "hclust":
            return [
                f"  linkage  : {params['hclust_method']}, cut at k = {params['n_clust']}",
            ]
        if analysis == "kmeans":
            seed = "" if params["seed"] is None else f"  (seed = {params['seed']})"
            return [
                f"  kmeans   : k = {params['n_clust']}, {params['n_start']} start(s), "
                f"{_fmt(params['tot_withinss'])} within-cluster ss{seed}",
            ]
        if analysis == "dbscan":
            return [
                f"  density  : eps = {_fmt(params['eps'])} ({params['eps_source']}), "
                f"min_pts = {params['min_pts']}",
            ]
        return [
            f"  density  : k = {params['k']}, eps = {params['eps']} shared neighbour(s), "
            f"min_pts = {params['min_pts']}",
        ]


def _scaling_line(center: bool, scale: bool) -> str:
    """How the features were standardised, in words.

    Shared by :class:`SaCluster` and the reduction contract, which report the same
    two flags.
    """
    if center and scale:
        return "centred and scaled"
    if center:
        return "centred"
    if scale:
        return "scaled"
    return "none, values as they arrived"


def new_cluster(
    analysis: str,
    points: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    assignments: pd.DataFrame,
    clusters: pd.DataFrame,
    engine: Mapping[str, Any],
    fit: Any = None,
) -> SaCluster:
    """Assemble a clustering result object.

    Port of ``sa_new_cluster()``. The checks here guard the contract rather than
    the user's input, so every one of them raises :class:`SaInternalError`. What
    they are guarding is the promise the object makes: ``assignments`` is aligned
    with ``points`` by position, and ``clusters`` accounts for every point that is
    not noise.

    Args:
        analysis: One of :data:`CLUSTER_ANALYSES`.
        points: Point labels, the row order every table follows.
        design: What was clustered.
        parameters: The choices as they were used.
        assignments: One row per point.
        clusters: One row per cluster found, noise excluded.
        engine: What computed the clustering.
        fit: The engine object, kept off the slots.
    """
    if analysis not in CLUSTER_ANALYSES:
        raise SaInternalError(
            "internal error: `analysis` must be one of " + ", ".join(CLUSTER_ANALYSES) + "."
        )
    labels = [str(value) for value in points]
    if not labels:
        raise SaInternalError("internal error: `points` must be a non-empty sequence.")
    if design.get("point_type") not in ("sample", "feature"):
        raise SaInternalError(
            'internal error: `design["point_type"]` must be "sample" or "feature".'
        )

    if not isinstance(assignments, pd.DataFrame):
        raise SaInternalError("internal error: `assignments` must be a DataFrame.")
    _check_columns(assignments, "`assignments`", cluster_assignment_columns())
    if [str(value) for value in assignments["points"]] != labels:
        raise SaInternalError("internal error: `assignments` is not aligned with `points`.")
    if not pd.api.types.is_integer_dtype(assignments["cluster"]):
        raise SaInternalError(
            "internal error: `assignments['cluster']` must be an integer column; noise is 0."
        )

    if not isinstance(clusters, pd.DataFrame):
        raise SaInternalError("internal error: `clusters` must be a DataFrame.")
    _check_columns(clusters, "`clusters`", cluster_table_columns())
    listed = [int(value) for value in clusters["cluster"]]
    if any(value == 0 for value in listed):
        raise SaInternalError(
            "internal error: `clusters` must be the clusters that were found, which "
            "never includes noise."
        )
    # The two tables are one fact counted twice, so they are made to agree here
    # rather than trusted to. Every point is either in a cluster `clusters` has a
    # row for or it is noise, and nothing is both.
    found = sorted({int(value) for value in assignments["cluster"] if int(value) > 0})
    if listed != found:
        raise SaInternalError(
            "internal error: `clusters` lists "
            + ", ".join(str(value) for value in listed)
            + " but the assignments hold "
            + ", ".join(str(value) for value in found)
            + "."
        )
    n_noise = int((assignments["cluster"] == 0).sum())
    if design.get("n_noise") != n_noise:
        raise SaInternalError(
            f"internal error: `design['n_noise']` is {design.get('n_noise')} but "
            f"{n_noise} point(s) were left unassigned."
        )
    if design.get("n_clusters") != len(clusters.index):
        raise SaInternalError(
            f"internal error: `design['n_clusters']` is {design.get('n_clusters')} but "
            f"`clusters` has {len(clusters.index)} row(s)."
        )
    for name in ("package", "method", "label", "overridden"):
        if engine.get(name) is None:
            raise SaInternalError(f"internal error: `engine` is missing `{name}`.")

    return SaCluster(
        {
            "analysis": analysis,
            "points": labels,
            "design": dict(design),
            "parameters": dict(parameters),
            "assignments": assignments,
            "clusters": clusters,
            "engine": dict(engine),
            "metadata": metadata(),
        },
        fit=fit,
    )


#: The reductions this contract covers.
#:
#: Port of ``sa_reduction_analyses()``. ``analysis`` names the method rather than
#: the family, as it does in :class:`SaModel`, so a result says which of the three
#: it came from without anything having to read ``engine``.
REDUCTION_ANALYSES = ("pca", "tsne", "umap")

#: Components ``repr`` reports the variance of before counting the rest.
#:
#: R's ``print.sa_reduction(x, n = 3L)``.
_REPR_MAX_COMPONENTS = 3


class SaReduction(SaResult):
    """Coordinates for every point, in fewer of them than the data arrived with.

    Seven slots for an embedding and nine for a rotation, and the engine handle
    beside them as :attr:`fit`: ``analysis``, ``points``, ``design``,
    ``parameters``, ``variance``, ``loadings``, ``scores``, ``engine`` and
    ``metadata``. The two middle ones are the rotation's other side, so they are
    present exactly when the analysis is a principal component analysis and
    ``"variance" in res`` is the test.

    A comparison result is organised around a feature axis and a model around a
    term axis. This repeats neither: what every table here repeats is ``points``.
    A point is whichever margin was embedded - a sample by default, a feature when
    asked - and ``design["point_type"]`` says which, because a reduction is the one
    result in this package whose row axis is the caller's choice rather than the
    question's.

    Nothing can be joined across the three methods except the points, and the
    points are what this contract fixes: every one of them lists its coordinates
    in the same row order, so two of the three can be plotted against each other
    and against anything else read from the same rows.
    """

    __slots__ = ("_fit",)

    def __init__(self, slots: Mapping[str, Any], fit: Any = None) -> None:
        super().__init__(slots)
        object.__setattr__(self, "_fit", fit)

    @property
    def fit(self) -> Any:
        """The engine object, kept so that the reduction can be read its own way.

        A fitted decomposition is what anyone who has done a principal component
        analysis already knows how to read, and hiding it behind these tables would
        make the familiar call the long way round. Deliberately not a slot, for the
        reason :attr:`SaModel.fit` is not one.
        """
        return self._fit

    def __repr__(self) -> str:
        """The summary ``print.sa_reduction`` prints, not the coordinates.

        Those are in ``res.scores``, and the engine object is ``res.fit``.
        """
        design = self["design"]
        params = self["parameters"]
        dropped = ""
        if design["n_dropped"] > 0:
            dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)"
        lines = [
            f"<{type(self).__name__}> {self['analysis']}",
            # `data` says what came in and `points` says which side of it was
            # embedded, so the two lines together read as the whole choice rather
            # than as a repetition.
            f"  data     : {design['n_used']} sample(s) x {design['n_feats']} feature(s){dropped}",
            f"  points   : {len(self['points'])} {design['point_type']}(s)",
            f"  scaling  : {_scaling_line(params['center'], params['scale'])}",
        ]
        if self._slots.get("variance") is not None:
            lines.extend(self._variance_lines())
        lines.extend(self._method_lines(params))
        if design["dropped_feats"]:
            lines.append("  dropped  : " + ", ".join(design["dropped_feats"]) + " (no variance)")
        return "\n".join(lines)

    def _variance_lines(self) -> list[str]:
        """What share of the variance the leading components carry."""
        table = self["variance"]
        shown = table.head(_REPR_MAX_COMPONENTS)
        if not len(shown.index):
            return []
        shares = ", ".join(
            f"{row['component']} {_fmt(row['prop_var'])}%" for _, row in shown.iterrows()
        )
        return [
            f"  variance : {shares}  ({len(shown.index)} of {len(table.index)} "
            f"component(s), {_fmt(shown['cum_var'].iloc[-1])}% cumulative)"
        ]

    def _method_lines(self, params: Mapping[str, Any]) -> list[str]:
        """The one line that only this method has to report."""
        analysis = self["analysis"]
        seed = "" if params.get("seed") is None else f"  (seed = {params['seed']})"
        if analysis == "tsne":
            return [
                f"  tsne     : {params['n_dim']} dimension(s), perplexity = "
                f"{_fmt(params['perplexity'])}, theta = {_fmt(params['theta'])}{seed}"
            ]
        if analysis == "umap":
            return [
                f"  umap     : {params['n_dim']} dimension(s), n_neighbors = "
                f"{params['n_neighbors']}, min_dist = {_fmt(params['min_dist'])}, "
                f"{params['metric']}{seed}"
            ]
        return []


def new_reduction(
    analysis: str,
    points: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    scores: pd.DataFrame,
    engine: Mapping[str, Any],
    variance: pd.DataFrame | None = None,
    loadings: pd.DataFrame | None = None,
    fit: Any = None,
) -> SaReduction:
    """Assemble a dimensionality reduction result object.

    Port of ``sa_new_reduction()``. The checks here guard the contract rather than
    the user's input, so every one of them raises :class:`SaInternalError`. What
    they are guarding is the one promise the object makes: ``scores`` is aligned
    with ``points`` by position, so a reduction can be plotted against anything
    else read from the same rows.

    Args:
        analysis: One of :data:`REDUCTION_ANALYSES`.
        points: Point labels, the row order every table follows.
        design: What was reduced.
        parameters: The choices as they were used.
        scores: Coordinates, one row per point.
        engine: What computed the reduction.
        variance: The component table, or ``None`` for an embedding.
        loadings: The margin that was not embedded, or ``None`` for an embedding.
        fit: The engine object, kept off the slots.
    """
    if analysis not in REDUCTION_ANALYSES:
        raise SaInternalError(
            "internal error: `analysis` must be one of " + ", ".join(REDUCTION_ANALYSES) + "."
        )
    labels = [str(value) for value in points]
    if not labels:
        raise SaInternalError("internal error: `points` must be a non-empty sequence.")
    if design.get("point_type") not in ("sample", "feature"):
        raise SaInternalError(
            'internal error: `design["point_type"]` must be "sample" or "feature".'
        )
    if not isinstance(scores, pd.DataFrame) or [str(v) for v in scores.get("points", [])] != labels:
        raise SaInternalError("internal error: `scores` is not a DataFrame aligned with `points`.")
    for name in ("package", "method", "label", "overridden"):
        if engine.get(name) is None:
            raise SaInternalError(f"internal error: `engine` is missing `{name}`.")

    # The two tables are the rotation's other side, so either both are here or
    # neither is, and which it is follows from the method rather than from what the
    # engine happened to return.
    rotation = analysis == "pca"
    if rotation != (variance is not None) or rotation != (loadings is not None):
        raise SaInternalError(
            "internal error: `variance` and `loadings` are present exactly when the "
            "analysis is a principal component analysis."
        )
    if variance is not None:
        _check_columns(variance, "`variance`", reduction_variance_columns())
    if loadings is not None:
        # `loadings` is the margin that was not embedded, so what it should have
        # one row of depends on which margin that is.
        n_vars = design["n_used"] if design["point_type"] == "feature" else design["n_feats"]
        if len(loadings.index) != n_vars:
            raise SaInternalError(
                f"internal error: `loadings` has {len(loadings.index)} row(s) for "
                f"{n_vars} variable(s)."
            )

    slots: dict[str, Any] = {
        "analysis": analysis,
        "points": labels,
        "design": dict(design),
        "parameters": dict(parameters),
        "variance": variance,
        "loadings": loadings,
        "scores": scores,
        "engine": dict(engine),
        "metadata": metadata(),
    }
    # A slot is present only when it has something to say, so an embedding does not
    # offer an empty version of the rotation's tables.
    if variance is None:
        del slots["variance"]
    if loadings is None:
        del slots["loadings"]
    return SaReduction(slots, fit=fit)


#: The searches this contract covers.
#:
#: Port of ``sa_selection_analyses()``. ``analysis`` names the procedure rather
#: than the family, the way :data:`REDUCTION_ANALYSES` names the three reductions,
#: so that a result says which search produced it without anything having to read
#: ``engine``.
SELECTION_ANALYSES = ("rfe", "stepwise")

#: Candidate rows ``repr`` shows before counting the rest.
#:
#: R's ``print.sa_selection(x, n = 10L)``. Python's ``__repr__`` takes no argument,
#: so the value R lets the caller override is fixed here.
_REPR_MAX_CANDIDATES = 10

#: Significant digits ``repr`` reports the chosen score at, by analysis.
#:
#: Three show all of a resampled metric, which is a small number. A criterion is
#: on the scale of the row count while the differences that decided its path are of
#: order one, so it needs the digits to show them.
_REPR_METRIC_DIGITS = 3
_REPR_CRITERION_DIGITS = 6

#: Significant digits the ranking column is reported at, R's ``sa_fmt_num()``
#: default. More than an estimate gets elsewhere, because two candidates can be
#: worth nearly the same and the point of the column is the order.
_REPR_RANKING_DIGITS = 4


class SaSelection(SaResult):
    """Which of the offered predictors are worth keeping, and how that was decided.

    Ten slots and the engine handle beside them as :attr:`fit`: ``analysis``,
    ``candidates``, ``design``, ``parameters``, ``selected``, ``ranking``,
    ``profile``, ``resampling``, ``engine`` and ``metadata``.

    This is the fourth row axis in the package. A comparison repeats ``features``,
    a model repeats ``terms`` and a reduction repeats ``points``; what a selection
    repeats is ``candidates``, the predictors that were offered, in the order the
    search ranked them rather than the order they arrived - so the first row is the
    one the procedure would drop last.

    A selection is not a model and is not stored as one. A model answers what each
    predictor is worth given all the others, and every table it holds is about the
    fit that is there. A selection answers a question one level up - how many
    predictors are worth keeping, and which - so it holds two tables a model has no
    counterpart for: ``ranking``, one row per candidate, and ``profile``, one row
    per model the search compared. The size is the answer, and a table with one row
    per candidate model is the only place it can be read.

    What ``profile`` repeats therefore depends on how the search moved.
    :func:`~statassist.perform_rfe` scores one model per subset size, so its rows
    are sizes and each appears once; :func:`~statassist.perform_stepwise` walks a
    path one term at a time, so its rows are steps and ``direction="both"`` can
    visit a size twice. What the two share is ``n_vars`` and the single ``chosen``
    row, which is all a reader needs to say how large the answer is and what it was
    compared against.

    ``resampling`` is what tells the two apart at a glance. An elimination has no
    score without holding rows out, so it always resamples; a criterion is a
    penalised likelihood computed on the rows the model was fitted to, so it never
    does, and the slot is ``None``.
    """

    __slots__ = ("_fit",)

    def __init__(self, slots: Mapping[str, Any], fit: Any = None) -> None:
        super().__init__(slots)
        object.__setattr__(self, "_fit", fit)

    @property
    def fit(self) -> Any:
        """The engine object, kept so that the search can be read its own way.

        Deliberately not a slot, for the reason :attr:`SaModel.fit` is not one:
        every other slot is a scalar, a string, a list or a DataFrame, so leaving
        this one out is what makes the object write out as JSON.
        """
        return self._fit

    def __repr__(self) -> str:
        """The summary ``print.sa_selection`` prints, not every table.

        The score of every model the search compared is in ``res["profile"]``,
        which is what says whether the answer won by much or by nothing, and the
        engine object is ``res.fit``.
        """
        design = self["design"]
        params = self["parameters"]
        lines = [
            f"<{type(self).__name__}> {self['analysis']}",
            f"  outcome  : {design['outcome']}  ({design['outcome_type']})",
        ]
        if design.get("outcome_lv") is not None:
            levels = design["outcome_lv"]
            lines.append(
                f"             modelling {levels[1]} against {levels[0]}, "
                f"{design['n_events']} of {design['n_used']} row(s)"
            )
        dropped = ""
        if design["n_dropped"] > 0:
            dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)"
        lines.append(f"  rows     : {design['n_used']} used{dropped}")
        lines.append(f"  search   : {self._search_line()}")
        lines.append(f"  settings : {self._settings_line(params)}")
        lines.append(f"  selected : {self._selected_line(params)}")
        lines.append("")
        lines.append(f"  ranking  ({self['engine']['importance']})")
        lines.extend(self._ranking_lines())
        if design["dropped_predictors"]:
            lines.append("")
            lines.append(
                "  dropped  : " + ", ".join(design["dropped_predictors"]) + " (single valued)"
            )
        return "\n".join(lines)

    def _chose_by(self, params: Mapping[str, Any]) -> str:
        """Which column of ``profile`` the answer was chosen by.

        ``metric`` for a search that resamples and ``criterion`` for one that
        charges a penalty. They are the same field under the name each search
        calls its own number by.
        """
        return str(params["metric"] if params.get("metric") is not None else params["criterion"])

    def _search_line(self) -> str:
        """What was compared, read off ``profile`` rather than out of ``parameters``.

        The table of the models the search visited is the record of what was asked
        for. An elimination scores one model per subset size and a stepwise search
        takes one move at a time, so the same table is counted two ways.
        """
        profile = self["profile"]
        opened = f"{self['engine']['label']} over {len(self['candidates'])} candidate(s), "
        if self["analysis"] == SELECTION_ANALYSES[0]:
            return opened + "size(s) " + ", ".join(str(int(n)) for n in profile["n_vars"])
        return opened + f"{len(profile.index) - 1} step(s)"

    def _settings_line(self, params: Mapping[str, Any]) -> str:
        """How the search was run, in the terms the search itself is set up in."""
        chose_by = self._chose_by(params)
        way = "maximised" if params["maximize"] else "minimised"
        if self["analysis"] == SELECTION_ANALYSES[1]:
            return (
                f"{params['direction']} search, {chose_by} {way} at "
                f"{fmt_est(params['k'])} per parameter"
            )
        scheme = str(params["cv_method"])
        if params.get("n_fold") is not None:
            scheme += f", {params['n_fold']} fold(s)"
        if params.get("n_repeat") is not None:
            scheme += f" x {params['n_repeat']} repeat(s)"
        return f"{scheme}, {chose_by} {way}"

    def _selected_line(self, params: Mapping[str, Any]) -> str:
        """The size that was kept and the one number that says whether it won."""
        chose_by = self._chose_by(params)
        best = self["profile"].loc[self["profile"]["chosen"].astype(bool)].iloc[0]
        digits = (
            _REPR_CRITERION_DIGITS
            if self["analysis"] == SELECTION_ANALYSES[1]
            else _REPR_METRIC_DIGITS
        )
        score = f"{chose_by} = {fmt_est(best[chose_by], digits)}"
        sd_col = f"{chose_by}SD"
        if sd_col in best.index:
            score += f" (SD {fmt_est(best[sd_col], 2)})"
        if self["resampling"] is not None:
            score += f" over {len(self['resampling'].index)} resample(s)"
        return f"{len(self['selected'])} of {len(self['candidates'])}  ({score})"

    def _ranking_lines(self) -> list[str]:
        """The head of the ranking table, one candidate per line."""
        table = self["ranking"]
        shown = table.head(_REPR_MAX_CANDIDATES)
        width = max((len(str(name)) for name in shown["candidates"]), default=0)
        lines = []
        for position in range(len(shown.index)):
            row = shown.iloc[position]
            verdict = "selected" if bool(row["selected"]) else "dropped"
            lines.append(
                f"    {str(row['candidates']):<{width}}  "
                f"{fmt_est(row['estimate'], _REPR_RANKING_DIGITS):>10}  {verdict}"
            )
        if len(table.index) > len(shown.index):
            lines.append(
                f"    ... and {len(table.index) - len(shown.index)} more candidate(s) in $ranking"
            )
        return lines


def new_selection(
    analysis: str,
    candidates: Sequence[str],
    design: Mapping[str, Any],
    parameters: Mapping[str, Any],
    selected: Sequence[str],
    ranking: pd.DataFrame,
    profile: pd.DataFrame,
    engine: Mapping[str, Any],
    resampling: pd.DataFrame | None = None,
    fit: Any = None,
) -> SaSelection:
    """Assemble a feature selection result object.

    Port of ``sa_new_selection()``. The checks here guard the contract rather than
    the user's input, so every one of them raises :class:`SaInternalError`. What
    they guard is the promise the object makes: ``ranking`` is aligned with
    ``candidates`` by position, ``selected`` is a subset of them, and exactly one
    row of ``profile`` is the size that was chosen, so the three tables cannot
    describe different searches.

    Args:
        analysis: One of :data:`SELECTION_ANALYSES`.
        candidates: Candidate predictor names, the row order ``ranking`` follows.
        design: What the search saw.
        parameters: The choices as they were used.
        selected: The candidates the search kept.
        ranking: One row per candidate.
        profile: One row per model the search compared: one subset size for an
            elimination, one step of the path for a stepwise search.
        engine: What ran the search. ``importance`` is required here and is not in
            a model's engine, because a ranking is only readable once it says what
            it ranked by.
        resampling: One row per resample at the chosen size, or ``None`` for a
            search that held nothing out.
        fit: The engine object, kept off the slots.
    """
    if analysis not in SELECTION_ANALYSES:
        raise SaInternalError(
            "internal error: `analysis` must be one of " + ", ".join(SELECTION_ANALYSES) + "."
        )
    offered = [str(name) for name in candidates]
    if not offered:
        raise SaInternalError("internal error: `candidates` must be a non-empty sequence.")
    if (
        not isinstance(ranking, pd.DataFrame)
        or [str(name) for name in ranking.get("candidates", [])] != offered
    ):
        raise SaInternalError(
            "internal error: `ranking` is not a DataFrame aligned with `candidates`."
        )
    _check_columns(ranking, "`ranking`", selection_ranking_columns())

    # A selection that kept nothing is not a selection, and one that kept something
    # it was never offered is a table that cannot be read against `ranking`.
    kept = [str(name) for name in selected]
    if not kept:
        raise SaInternalError("internal error: `selected` must be a non-empty sequence.")
    unknown = [name for name in kept if name not in set(offered)]
    if unknown:
        raise SaInternalError(
            "internal error: `selected` holds name(s) that are not candidates: "
            + ", ".join(unknown)
            + "."
        )
    flagged = [bool(value) for value in ranking["selected"]]
    if flagged != [name in set(kept) for name in offered]:
        raise SaInternalError("internal error: `ranking['selected']` disagrees with `selected`.")

    if not isinstance(profile, pd.DataFrame) or not len(profile.index):
        raise SaInternalError("internal error: `profile` must be a non-empty DataFrame.")
    _check_columns(profile, "`profile`", selection_profile_columns())
    chosen = profile["chosen"].astype(bool)
    if int(chosen.sum()) != 1:
        raise SaInternalError(
            "internal error: exactly one row of `profile` is the size that was chosen, "
            f"but {int(chosen.sum())} are marked."
        )
    at = int(profile.loc[chosen, "n_vars"].iloc[0])
    if at != len(kept):
        raise SaInternalError(
            f"internal error: the chosen row of `profile` is a subset of {at} "
            f"variable(s) and `selected` holds {len(kept)}."
        )
    for name in ("package", "method", "label", "metrics", "importance", "overridden"):
        if engine.get(name) is None:
            raise SaInternalError(f"internal error: `engine` is missing `{name}`.")
    if resampling is not None and not isinstance(resampling, pd.DataFrame):
        raise SaInternalError("internal error: `resampling` must be a DataFrame or None.")

    return SaSelection(
        {
            "analysis": analysis,
            "candidates": offered,
            "design": dict(design),
            "parameters": dict(parameters),
            "selected": kept,
            "ranking": ranking,
            "profile": profile,
            "resampling": resampling,
            "engine": dict(engine),
            "metadata": metadata(),
        },
        fit=fit,
    )


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
