"""Test a contingency table with every applicable test at once.

Port of ``R/compare_categorical_groups.R``.

This is the one scenario in the package with no feature axis. Every other
comparison asks its question once per numeric column and returns a table with one
row per column; a contingency table is asked about as a whole, so the result
carries the table itself in ``cells`` and one row per test in ``tests``. That is
also why the result is a :class:`~statassist.core.result.SaCategorical` rather
than a comparison, and why
:func:`~statassist.estimate_categorical_significance` is the counterpart that
reads it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.contingency import (
    MAX_CATEGORY_LEVELS,
    CategoricalInput,
    categorical_cells,
    categorical_condition_counts,
    categorical_counts,
    diagnose_discordance,
    diagnose_expected,
    diagnose_repeated,
    validate_categorical_input,
)
from ..core.contracts import categorical_test_columns
from ..core.errors import SaInternalError, SaValueError, notify, warn
from ..core.result import SaCategorical, new_categorical
from ..core.validate import check_count, check_flag, check_scalar_num
from ..kernel.categorical import (
    assoc_measures,
    assoc_measures_paired,
    assoc_measures_repeated,
    chisq,
    cochran_q,
    fisher,
    has_zero_cell,
    mcnemar,
)

__all__ = ["compare_categorical_groups"]

#: How many tables a Monte Carlo p-value is taken over unless told otherwise.
DEFAULT_RESAMPLES = 9999

#: Fewest tables a Monte Carlo p-value will be taken over.
#:
#: R's lower bound for ``n_resamples``. A simulated p-value cannot be finer than
#: ``1 / (n_resamples + 1)``, so below this the resolution of the answer is
#: coarser than the thresholds it would be read against.
MIN_RESAMPLES = 199

#: How many variables an independent categorical comparison crosses.
CROSSED_VARIABLES = 2

#: How many levels a matched design's shared level set may hold.
#:
#: The tests of symmetry that generalise McNemar's test past two levels,
#: Bowker's and Stuart-Maxwell's, are not implemented, so a wider level set is
#: refused rather than collapsed.
MATCHED_LEVELS = 2


class _Layout(NamedTuple):
    """Everything one design produced, ready for the result contract.

    The two designs differ in which table they build, which null they hold it
    against and which tests they run, and every one of those differences is
    reported rather than resolved here.
    """

    counts: pd.DataFrame
    cells: pd.DataFrame
    null: str
    tests: dict[str, pd.DataFrame]
    test_info: dict[str, dict[str, Any]]
    association: pd.DataFrame
    row_var: str
    col_var: str
    exact_used: bool | None
    enumerated: bool | None
    diagnostics: dict[str, Any]


def compare_categorical_groups(
    data: Any,
    category_lv: Mapping[str, Sequence[str]] | None = None,
    control_label: Any = None,
    paired: bool = False,
    conf_level: float = 0.95,
    correct: bool = True,
    exact: bool | None = None,
    simulate_p_value: bool = False,
    n_resamples: int = DEFAULT_RESAMPLES,
    max_levels: int = MAX_CATEGORY_LEVELS,
    seed: int | None = None,
    diagnose: bool = True,
) -> SaCategorical:
    """Test a contingency table with every applicable test at once.

    Crosses two categorical variables into a contingency table and returns an
    asymptotic and an exact test of it side by side, together with the measures
    of how strong the association is. Nothing is chosen on the caller's behalf:
    reporting both makes disagreement between them visible, which is the
    situation where the choice between an approximation and an exact enumeration
    actually matters.

    Which tests run is decided by ``paired`` and by how many variables
    ``category_lv`` names, so there is no argument naming a test:

    ====================================  =====================  ==============================
    design                                null hypothesis        tests
    ====================================  =====================  ==============================
    independent, two variables            independence           chi-square, Fisher's exact
    matched, two binary conditions        symmetry               McNemar's
    matched, three or more binary         marginal homogeneity   Cochran's Q
    ====================================  =====================  ==============================

    A matched design reads the columns as repeated measurements of one thing on
    the same row, so pairing is by row and there is no ``id`` argument.

    Args:
        data: Wide frame (or 2-D array) whose columns are the categorical
            variables. Unlike the other comparison scenarios there is no
            ``feats`` argument: the columns are what is tested rather than what
            is measured, so a categorical, a string, a boolean or a 0/1 coded
            column all read as categorical and are taken as their labels.
        category_lv: Mapping of one entry per variable giving its levels,
            reference first, or ``None`` to take every column of ``data`` with
            its levels in sorted order. Naming it does three things sorting
            cannot: it picks which columns take part, it fixes which level the
            odds ratio is read against, and it drops the rows belonging to any
            level it leaves out.
        control_label: The level to hold as the reference. For an independent
            design this is one name per variable it points at, as a mapping
            (``{"smoker": "n"}``); a variable it says nothing about is left as it
            arrived. A matched design has one level set shared by every
            condition, so there it is a single level name.
        paired: If ``True``, the columns are read as repeated conditions measured
            on the same row rather than as different variables.
        conf_level: Confidence level for the association intervals and for the
            conditional odds ratio of Fisher's exact test.
        correct: Whether to apply the continuity correction to the chi-square
            approximation. It applies to 2 x 2 tables only, so it changes nothing
            on a larger one, and the exact branch of McNemar's test needs none.
        exact: Read only by McNemar's test. ``True`` for the exact binomial test
            on the discordant pairs, ``False`` for the chi-square approximation,
            or ``None`` to take the exact test when there are fewer than 25
            discordant pairs. ``parameters["exact"]`` records which one ran.
        simulate_p_value: Read only by the tests of an independent design.
            Replaces the chi-square approximation with a Monte Carlo p-value, and
            is the way to get an answer out of Fisher's test on a large r x c
            table that cannot be enumerated.
        n_resamples: How many tables the Monte Carlo p-value is taken over.
        max_levels: How many levels a variable may take before it is refused as a
            category. Reading a continuous measurement as labels makes a table
            with a cell per observation and no test of association to run on it;
            this is the ceiling that catches that. Checked against the levels
            actually used, so naming three levels of a fifty-valued column in
            ``category_lv`` is a way through.
        seed: Seeds the Monte Carlo draw. R has one global stream and restores
            it on exit; this port seeds the draw per call instead, so a simulated
            p-value is reproducible within Python without the caller's random
            state being touched at all.
        diagnose: Whether to attach the rule the reported approximation rests on
            as ``diagnostics``.

    Returns:
        A :class:`~statassist.core.result.SaCategorical`. ``design`` carries
        ``null``, which is what ``expected`` and the residuals of ``cells`` are
        read under; ``cells`` is one row per cell of the table; ``tests`` holds
        one one-row table per test and carries no ``pval_adj``, there being a
        single question to answer; and ``association`` holds one row per measure,
        which measures existing at all depending on the design and on the size of
        the table.

    Raises:
        SaValueError: If an argument is unusable, if an independent design does
            not cross exactly two variables, or if a matched one is not binary.

    Notes:
        An association has no sign past a 2 x 2 table, so there is no signed
        effect to put beside the p-value and the result is deliberately not a
        comparison: a volcano plot needs one effect per feature and this scenario
        has no feature axis to carry one. A **cell** does have both axes, and
        :func:`~statassist.estimate_categorical_significance` is what reads that.

        Every measure is built from the **uncorrected** chi-square statistic,
        whatever ``correct`` was set to. Yates' correction is about referring a
        discrete statistic to a continuous distribution, which is a statement
        about a p-value; letting it into an effect size would make the reported
        strength of an association depend on a choice made about its tail
        probability.

        Fisher's exact test enumerates every table with the observed margins, and
        on a large r x c one there are more of those than the algorithm's
        workspace holds. That is a limit of the enumeration rather than a fault
        in the data, so its p-value comes back missing with a note saying so,
        instead of the whole call failing and taking the chi-square result with
        it.

    Examples:
        >>> import pandas as pd
        >>> smoking = pd.DataFrame(
        ...     {
        ...         "smoker": ["y"] * 60 + ["n"] * 60,
        ...         "grade": (
        ...             ["high"] * 10 + ["mid"] * 20 + ["low"] * 30
        ...             + ["high"] * 30 + ["mid"] * 20 + ["low"] * 10
        ...         ),
        ...     }
        ... )
        >>> res = compare_categorical_groups(smoking)
        >>> list(res.tests)
        ['chisq_test', 'fisher_test']
        >>> res.design["null"]
        'independence'

        The table the tests were run on, and how strong the association is.

        >>> res.as_table().to_numpy().tolist()
        [[30, 10, 20], [10, 30, 20]]
        >>> list(res.association["measure"])
        ['cramers_v', 'contingency_coefficient']

        A matched design is held against symmetry instead, so the diagonal is
        expected at exactly what it holds.

        >>> before_after = pd.DataFrame(
        ...     {
        ...         "before": ["pass"] * 20 + ["fail"] * 30,
        ...         "after": ["pass"] * 18 + ["fail"] * 2 + ["pass"] * 14 + ["fail"] * 16,
        ...     }
        ... )
        >>> matched = compare_categorical_groups(before_after, paired=True)
        >>> matched.design["null"]
        'symmetry'
        >>> list(matched.tests)
        ['mcnemar_test']
    """
    paired = check_flag(paired, "paired")
    correct = check_flag(correct, "correct")
    simulate_p_value = check_flag(simulate_p_value, "simulate_p_value")
    diagnose = check_flag(diagnose, "diagnose")
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    n_resamples = check_count(n_resamples, "n_resamples", MIN_RESAMPLES)
    max_levels = check_count(max_levels, "max_levels", MATCHED_LEVELS)
    if exact is not None:
        exact = check_flag(exact, "exact")

    if exact is not None and not paired:
        warn(
            "`exact` is only read by McNemar's test and is ignored by an "
            "independent design. `simulate_p_value` is what replaces the "
            "chi-square approximation there."
        )
    if simulate_p_value and paired:
        warn(
            "`simulate_p_value` is only read by the tests of an independent "
            "design and is ignored by a matched one. `exact` is what chooses "
            "the exact branch of McNemar's test."
        )

    validated = validate_categorical_input(data, category_lv, control_label, paired, max_levels)
    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `category_lv`.")
    if validated.n_incomplete > 0:
        notify(
            f"Dropped {validated.n_incomplete} row(s) missing a value in one of the "
            "variables; a contingency table needs the whole row."
        )

    if paired:
        layout = _matched(validated, correct, exact, conf_level)
    else:
        layout = _independent(validated, correct, simulate_p_value, n_resamples, conf_level, seed)

    # Raised here rather than inside the kernel, so that every message this
    # scenario can produce is raised in one place and the kernel stays a function
    # of its arguments.
    if has_zero_cell(layout.counts):
        notify(
            "A zero cell leaves the odds ratio undefined, so the "
            "Haldane-Anscombe correction of 0.5 per cell was applied to it. "
            "The tests read the table as it is."
        )
    if layout.enumerated is False:
        shape = " x ".join(str(size) for size in layout.counts.shape)
        total = int(layout.counts.to_numpy().sum())
        notify(
            f"Fisher's exact test could not enumerate a {shape} table of "
            f"{total} observation(s), so its p-value is NA. The chi-square test "
            "in the same result was computed; set `simulate_p_value = True` for "
            "the Monte Carlo variant of both."
        )
    if diagnose and not layout.diagnostics["approx_ok"]:
        notify(layout.diagnostics["note"])

    n_row, n_col = layout.counts.shape
    return new_categorical(
        analysis="categorical_comparison",
        variables=validated.variables,
        design={
            "category_lv": validated.category_lv,
            "null": layout.null,
            "paired": paired,
            "pairing": "row" if paired else None,
            "dim": [int(n_row), int(n_col)],
            "row_var": layout.row_var,
            "col_var": layout.col_var,
            "n_used": validated.n_used,
            "n_dropped": validated.n_dropped,
            "n_incomplete": validated.n_incomplete,
        },
        parameters={
            "conf_level": conf_level,
            "correct": correct,
            "exact": layout.exact_used,
            "simulate_p_value": simulate_p_value,
            "n_resamples": n_resamples if simulate_p_value else None,
            "max_levels": max_levels,
            "seed": seed,
        },
        cells=layout.cells,
        tests=layout.tests,
        test_info=layout.test_info,
        association=layout.association,
        diagnostics=layout.diagnostics if diagnose else None,
    )


def _independent(
    validated: CategoricalInput,
    correct: bool,
    simulate_p_value: bool,
    n_resamples: int,
    conf_level: float,
    seed: int | None,
) -> _Layout:
    """Run the independent design: chi-square beside Fisher's exact test.

    Port of ``sa_categorical_independent()``.
    """
    variables = validated.variables
    if len(variables) != CROSSED_VARIABLES:
        raise SaValueError(
            "an independent categorical comparison crosses exactly two variables, "
            f"and {len(variables)} were given: "
            + ", ".join(variables)
            + ". Name the two in `category_lv`, or set `paired = True` if the "
            "columns are repeated measurements of one thing."
        )

    counts = categorical_counts(validated.data, variables)
    cells = categorical_cells(counts, "independence")

    chisq_row = chisq(counts, correct, simulate_p_value, n_resamples, seed)
    fisher_row = fisher(counts, conf_level, simulate_p_value, n_resamples, seed)

    if simulate_p_value:
        chisq_label = f"Chi-square test of independence (Monte Carlo, {n_resamples} resamples)"
    elif correct and counts.shape == (CROSSED_VARIABLES, CROSSED_VARIABLES):
        chisq_label = "Chi-square test of independence (Yates' continuity correction)"
    else:
        chisq_label = "Chi-square test of independence"

    return _Layout(
        counts=counts,
        cells=cells,
        null="independence",
        tests={
            "chisq_test": _test_row(chisq_row),
            # `enumerated` is a fact about whether the test ran rather than a
            # finding about the table, and a missing p-value already says it in
            # the table itself.
            "fisher_test": _test_row(fisher_row, drop=("enumerated",)),
        },
        test_info={
            "chisq_test": {
                "id": "chisq_independence",
                "label": chisq_label,
                "paired": False,
            },
            "fisher_test": {
                "id": "fisher_exact",
                "label": "Fisher's exact test",
                "paired": False,
            },
        },
        association=assoc_measures(counts, conf_level),
        row_var=variables[0],
        col_var=variables[1],
        exact_used=None,
        enumerated=bool(fisher_row["enumerated"]),
        diagnostics=diagnose_expected(cells),
    )


def _matched(
    validated: CategoricalInput,
    correct: bool,
    exact: bool | None,
    conf_level: float,
) -> _Layout:
    """Run the matched design: McNemar's test or Cochran's Q.

    Port of ``sa_categorical_matched()``. The two branches read different tables
    against different nulls, and neither difference is incidental. McNemar's test
    is about a square table crossing two conditions against each other, held
    against symmetry, where the discordant cells are the whole of the evidence.
    Cochran's Q has no such table to be about, since three conditions do not
    cross into two dimensions; what it compares is the response rate of each
    condition, so the table is one row per condition and the null is that those
    rates agree.
    """
    variables = validated.variables
    levels = validated.category_lv[variables[0]]

    if len(levels) != MATCHED_LEVELS:
        raise SaValueError(
            f"`paired = True` needs binary conditions, and the shared level set "
            f"holds {len(levels)}: "
            + ", ".join(levels)
            + ". The tests of symmetry that generalise McNemar's test past two "
            "levels, Bowker's and Stuart-Maxwell's, are not implemented in this "
            "version. Name two levels in `category_lv` to reduce the design "
            "rather than have it collapsed here."
        )

    if len(variables) == CROSSED_VARIABLES:
        counts = categorical_counts(validated.data, variables)
        row = mcnemar(counts, correct, exact)
        exact_used = bool(row["exact_used"])

        if exact_used:
            label = "McNemar's test (exact binomial on the discordant pairs)"
        elif correct:
            label = "McNemar's test (continuity corrected)"
        else:
            label = "McNemar's test"

        return _Layout(
            counts=counts,
            cells=categorical_cells(counts, "symmetry"),
            null="symmetry",
            # `exact_used` is a setting rather than a finding, and
            # `parameters["exact"]` is where a setting is recorded, so it does
            # not go on into the table.
            tests={"mcnemar_test": _test_row(row, drop=("exact_used",))},
            test_info={"mcnemar_test": {"id": "mcnemar_test", "label": label, "paired": True}},
            association=assoc_measures_paired(counts, conf_level),
            row_var=variables[0],
            col_var=variables[1],
            exact_used=exact_used,
            enumerated=None,
            diagnostics=diagnose_discordance(int(row["n_discordant"])),
        )

    # The second level is the response, which is the direction rule every other
    # scenario reads: the first level of `category_lv` is the reference and what
    # is counted is departure from it.
    responses = np.column_stack(
        [
            np.asarray(validated.data[name].astype(object) == levels[1], dtype=np.int64)
            for name in variables
        ]
    )

    row = cochran_q(responses)
    counts = categorical_condition_counts(validated.data, variables, levels)

    return _Layout(
        counts=counts,
        cells=categorical_cells(counts, "marginal_homogeneity"),
        null="marginal_homogeneity",
        tests={"cochran_q": _test_row(row)},
        test_info={
            "cochran_q": {
                "id": "cochran_q",
                "label": f"Cochran's Q test over {len(variables)} repeated condition(s)",
                "paired": True,
            }
        },
        association=assoc_measures_repeated(row["statistic"], int(row["n_used"]), len(variables)),
        row_var=str(counts.index.name),
        col_var=str(counts.columns.name),
        exact_used=None,
        enumerated=None,
        diagnostics=diagnose_repeated(int(row["n_used"]), len(variables)),
    )


def _test_row(row: Mapping[str, float], drop: Sequence[str] = ()) -> pd.DataFrame:
    """Turn a kernel row into the one-row table ``tests`` holds.

    Port of ``sa_categorical_row()``. The contract columns come first and
    whatever else the test reported follows, so a consumer can read the columns
    every test carries without knowing which one ran.

    Args:
        row: The named row a kernel returned.
        drop: Names the kernel reported for the scenario to record elsewhere,
            which are not findings about the table and so do not belong in it.
    """
    kept = {name: value for name, value in row.items() if name not in drop}
    contract = categorical_test_columns()
    absent = [name for name in contract if name not in kept]
    if absent:
        raise SaInternalError(
            "internal error: kernel row is missing column(s): " + ", ".join(absent) + "."
        )
    order = contract + [name for name in kept if name not in contract]
    return pd.DataFrame({name: [kept[name]] for name in order})
