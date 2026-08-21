"""Resolving categorical input and laying it out as a table, against R's answers.

The expectation builders themselves are graded in ``tests/kernel/test_categorical.py``,
where the kernels that rest on them are. What is here is the rest of
``utils_categorical.R``: which rows survive, which levels they are read at, what
the table looks like, and the three diagnostics that report the approximation
each design rests on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from golden import assert_close, assert_frame_close, load_case

from statassist.core.contingency import (
    DISCORDANT_PAIR_MIN,
    EXPECTED_COUNT_MIN,
    MAX_CATEGORY_LEVELS,
    REPEATED_CELL_MIN,
    categorical_cells,
    categorical_condition_counts,
    categorical_counts,
    categorical_shared_lv,
    check_level_count,
    diagnose_discordance,
    diagnose_expected,
    diagnose_repeated,
    validate_categorical_input,
)
from statassist.core.contracts import (
    association_columns,
    categorical_cell_columns,
    categorical_nulls,
    categorical_test_columns,
)
from statassist.core.errors import SaInternalError, SaValueError
from statassist.core.validate import fmt_est
from statassist.kernel.categorical import ASSOC_COLUMNS, MCNEMAR_EXACT_MAX_DISCORDANT

#: The column R's ``write.csv`` spells the way R spells a logical.
#:
#: ``as.character(TRUE)`` is ``"TRUE"`` and ``str(True)`` is ``"True"``, so a
#: bool column read as bools would carry Python's spelling into the level names
#: and disagree with the fixture for a reason that has nothing to do with this
#: function. Read as text it is R's own column, which is what the case is about.
#: The Python spelling is asserted on its own further down.
_LOGICAL_AS_TEXT = {"flag": str}


def labelled_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rebuild each named table from the one long input frame, labels and all.

    The row and column order is the order of first appearance, which is the order
    ``expand.grid`` laid the frame out in and therefore the order R's own
    ``dimnames`` were in.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, block in frame.groupby("table", sort=False):
        rows = list(dict.fromkeys(block["row_level"]))
        cols = list(dict.fromkeys(block["col_level"]))
        wide = block.pivot(index="row_level", columns="col_level", values="count")
        out[str(name)] = wide.loc[rows, cols].astype(float)
    return out


@pytest.fixture(scope="module")
def cell_tables() -> dict[str, pd.DataFrame]:
    frame, _ = load_case("cat_cells")
    return labelled_tables(frame)


@pytest.fixture(scope="module")
def cat_input() -> pd.DataFrame:
    from golden import GOLDEN_ROOT

    return pd.read_csv(GOLDEN_ROOT / "cat_validate" / "input.csv", dtype=_LOGICAL_AS_TEXT)


#: The repeated conditions the paired cases of the fixture were built on. Written
#: here rather than loaded because they are a second input to the same case.
PAIRED_BEFORE = ["no", "no", "no", "yes", "no", "no", "yes", "no"]
PAIRED_AFTER = ["yes", "yes", "no", "yes", "maybe", "yes", "yes", "no"]


@pytest.fixture(scope="module")
def paired_input() -> pd.DataFrame:
    return pd.DataFrame({"before": PAIRED_BEFORE, "after": PAIRED_AFTER})


def as_labels(frame: pd.DataFrame) -> dict[str, list[str]]:
    """The resolved data as the labels R exported, rather than as factor codes."""
    return {str(name): [str(value) for value in frame[name]] for name in frame.columns}


# --------------------------------------------------------------------------- #
# The contract constants
# --------------------------------------------------------------------------- #


def test_the_contract_constants_are_the_lists_r_states() -> None:
    _, expected = load_case("cat_cells")

    assert categorical_cell_columns() == expected["columns"]
    assert categorical_nulls() == expected["nulls"]
    assert categorical_test_columns() == expected["test_columns"]
    assert association_columns() == expected["assoc_columns"]


def test_the_association_columns_are_one_list_and_not_two() -> None:
    """The kernel's tuple is the contract, not a second copy of it."""
    assert list(ASSOC_COLUMNS) == association_columns()


def test_the_discordant_pair_rule_is_one_number_and_not_two() -> None:
    """R writes 25 out twice; the branch and the check have to mean the same 25."""
    assert MCNEMAR_EXACT_MAX_DISCORDANT == DISCORDANT_PAIR_MIN


# --------------------------------------------------------------------------- #
# categorical_cells
# --------------------------------------------------------------------------- #


def test_the_cell_table_reproduces_r(cell_tables: dict[str, pd.DataFrame]) -> None:
    _, expected = load_case("cat_cells")

    assert_frame_close(categorical_cells(cell_tables["t2x2"]), expected["t2x2_independence"])
    assert_frame_close(
        categorical_cells(cell_tables["t2x2"], "symmetry"), expected["t2x2_symmetry"]
    )
    assert_frame_close(
        categorical_cells(cell_tables["t2x2"], "marginal_homogeneity"),
        expected["t2x2_marginal"],
    )
    assert_frame_close(categorical_cells(cell_tables["t3x4"]), expected["t3x4_independence"])
    assert_frame_close(
        categorical_cells(cell_tables["pair_small"], "symmetry"),
        expected["pair_small_symmetry"],
    )


def test_an_empty_margin_leaves_a_missing_proportion_and_not_a_zero(
    cell_tables: dict[str, pd.DataFrame],
) -> None:
    """The one table where the divisions have something to divide by nothing."""
    _, expected = load_case("cat_cells")

    cells = categorical_cells(cell_tables["hole"])
    assert_frame_close(cells, expected["hole_independence"])
    assert_frame_close(
        categorical_cells(cell_tables["hole"], "symmetry"), expected["hole_symmetry"]
    )

    empty_row = cells["row_level"] == "mid"
    assert cells.loc[empty_row, "prop_row"].isna().all()
    assert cells.loc[cells["col_level"] == "y", "prop_col"].isna().all()
    # And the cells with a margin are untouched by the ones without.
    assert cells.loc[cells["row_level"] == "low", "prop_row"].notna().all()


def test_marginal_homogeneity_is_the_independence_arithmetic(
    cell_tables: dict[str, pd.DataFrame],
) -> None:
    """R branches on ``symmetry`` alone, so the third null keeps its residuals.

    A port that reads the branch as "independence or not" makes
    ``marginal_homogeneity`` a table of missing standardized residuals, which is
    the claim Cochran's Q is about losing its cell-level reading entirely.
    """
    independence = categorical_cells(cell_tables["t2x2"])
    marginal = categorical_cells(cell_tables["t2x2"], "marginal_homogeneity")
    symmetry = categorical_cells(cell_tables["t2x2"], "symmetry")

    pd.testing.assert_frame_equal(independence, marginal)
    assert marginal["std_residual"].notna().all()
    assert symmetry["std_residual"].isna().all()
    assert symmetry["residual"].notna().any()


def test_the_cells_come_out_down_the_columns(cell_tables: dict[str, pd.DataFrame]) -> None:
    """The row order every ``truth_cell`` and every mosaic is merged onto."""
    cells = categorical_cells(cell_tables["t3x4"])
    table = cell_tables["t3x4"]

    assert list(cells.columns) == categorical_cell_columns()
    assert len(cells.index) == table.size
    # The first variable varies fastest: all three rows of `q1`, then of `q2`.
    assert list(cells["row_level"][:3]) == list(table.index)
    assert list(cells["col_level"][:3]) == [table.columns[0]] * 3
    assert list(cells["observed"][:3]) == list(table.iloc[:, 0])


def test_the_proportions_are_taken_along_the_axis_they_are_named_for(
    cell_tables: dict[str, pd.DataFrame],
) -> None:
    """``prop_row`` sums to one within a row, which a transposed port reverses."""
    cells = categorical_cells(cell_tables["t3x4"])

    by_row = cells.groupby("row_level", sort=False)["prop_row"].sum()
    by_col = cells.groupby("col_level", sort=False)["prop_col"].sum()
    assert np.allclose(by_row, 1.0)
    assert np.allclose(by_col, 1.0)
    assert cells["prop_total"].sum() == pytest.approx(1.0)


def test_a_null_the_scenario_does_not_state_is_an_internal_error(
    cell_tables: dict[str, pd.DataFrame],
) -> None:
    with pytest.raises(SaInternalError, match="must name one of"):
        categorical_cells(cell_tables["t2x2"], "homogeneity")


def test_a_table_with_no_labels_is_numbered_rather_than_refused() -> None:
    """Where this port is wider than R, said out loud rather than left implicit.

    R's ``expand.grid()`` on a table without dimnames builds a frame of no rows
    and the assembly fails. Every table the package itself passes here carries
    labels, so numbering the axes only changes what a caller handing over a bare
    array sees, and a numbered cell table is more use to them than a message
    about a grid.
    """
    cells = categorical_cells(np.array([[3.0, 5.0], [7.0, 2.0]]))
    assert list(cells["row_level"]) == ["1", "2", "1", "2"]
    assert list(cells["col_level"]) == ["1", "1", "2", "2"]


# --------------------------------------------------------------------------- #
# validate_categorical_input
# --------------------------------------------------------------------------- #


def test_the_resolved_input_reproduces_r(
    cat_input: pd.DataFrame, paired_input: pd.DataFrame
) -> None:
    _, expected = load_case("cat_validate")

    plain = validate_categorical_input(cat_input)
    assert plain.variables == expected["plain"]["variables"]
    assert plain.category_lv == expected["plain"]["category_lv"]
    assert plain.n_used == expected["plain"]["n_used"]
    assert plain.n_dropped == expected["plain"]["n_dropped"]
    assert plain.n_incomplete == expected["plain"]["n_incomplete"]
    assert as_labels(plain.data) == expected["plain"]["data"]

    named = validate_categorical_input(
        cat_input,
        category_lv={"answer": ["y", "n"], "grade": ["low", "mid"]},
        control_label={"answer": "n"},
    )
    assert named.variables == expected["named"]["variables"]
    assert named.category_lv == expected["named"]["category_lv"]
    assert named.n_used == expected["named"]["n_used"]
    assert named.n_dropped == expected["named"]["n_dropped"]
    assert named.n_incomplete == expected["named"]["n_incomplete"]
    assert as_labels(named.data) == expected["named"]["data"]

    paired = validate_categorical_input(paired_input, control_label="no", paired=True)
    assert paired.variables == expected["paired"]["variables"]
    assert paired.category_lv == expected["paired"]["category_lv"]
    assert paired.n_used == expected["paired"]["n_used"]
    assert as_labels(paired.data) == expected["paired"]["data"]


def test_a_missing_row_and_an_excluded_row_are_counted_apart(
    cat_input: pd.DataFrame,
) -> None:
    """Two different facts about a row, which one membership test would merge.

    ``answer`` holds one missing entry and one level the named ``category_lv``
    leaves out, and ``grade`` holds three rows at a level it leaves out. A row
    that was not measured and a row that was measured and excluded are counted
    separately, so a port that runs the membership test first reports four
    dropped rows and no incomplete one.
    """
    resolved = validate_categorical_input(
        cat_input, category_lv={"answer": ["y", "n"], "grade": ["low", "mid"]}
    )

    assert resolved.n_incomplete == 1
    assert resolved.n_dropped == 3
    assert resolved.n_used == len(cat_input.index) - 4
    assert len(resolved.data.index) == resolved.n_used
    assert list(resolved.data.index) == list(range(resolved.n_used))


def test_the_levels_are_the_labels_whatever_the_column_held(
    cat_input: pd.DataFrame,
) -> None:
    """A factor, a logical and a 0/1 code are all categorical here."""
    resolved = validate_categorical_input(cat_input)

    assert resolved.category_lv["coded"] == ["0", "1"]
    # A factor's own order is not kept: R reads the labels and sorts them, which
    # is what makes `category_lv` the only place an order is stated.
    assert resolved.category_lv["grade"] == ["high", "low", "mid"]
    for name in resolved.variables:
        assert list(resolved.data[name].cat.categories) == resolved.category_lv[name]


def test_a_python_logical_column_takes_pythons_spelling() -> None:
    """Where this port and R differ, said out loud rather than left to a fixture.

    ``as.character(TRUE)`` is ``"TRUE"`` and ``str(True)`` is ``"True"``. The
    levels are the labels the language gives them, so a caller filtering the
    frame with ``data["flag"] == True`` and a caller reading ``category_lv`` see
    the same two names.
    """
    frame = pd.DataFrame({"flag": [True, False, True, True], "y": ["a", "b", "a", "b"]})
    resolved = validate_categorical_input(frame)

    assert resolved.category_lv["flag"] == ["False", "True"]


def test_the_reference_moves_to_the_front_without_reordering_the_rest(
    cat_input: pd.DataFrame,
) -> None:
    resolved = validate_categorical_input(cat_input, control_label={"answer": "y"})

    assert resolved.category_lv["answer"] == ["y", "maybe", "n"]
    # Every other variable is left where it was.
    assert resolved.category_lv["grade"] == ["high", "low", "mid"]


def test_a_matched_design_shares_one_level_set(paired_input: pd.DataFrame) -> None:
    """The union, so a condition nobody answered a level under stays square."""
    resolved = validate_categorical_input(paired_input, paired=True)

    assert resolved.category_lv["before"] == ["maybe", "no", "yes"]
    assert resolved.category_lv["before"] == resolved.category_lv["after"]
    counts = categorical_counts(resolved.data, resolved.variables)
    assert counts.shape[0] == counts.shape[1]

    # Without the union `before` would take two levels and `after` three.
    independent = validate_categorical_input(paired_input)
    assert independent.category_lv["before"] == ["no", "yes"]


def test_the_shared_levels_reproduce_r() -> None:
    _, expected = load_case("cat_validate")
    given = {"before": ["no", "yes"], "after": ["maybe", "no", "yes"]}

    assert categorical_shared_lv(given, given, False, None) == expected["shared_lv_union"]
    assert categorical_shared_lv(given, given, False, "yes") == expected["shared_lv_control"]


def test_a_matched_design_takes_one_reference_and_not_one_per_variable(
    paired_input: pd.DataFrame,
) -> None:
    with pytest.raises(SaValueError, match="single level name rather than one per"):
        validate_categorical_input(
            paired_input, control_label={"before": "no", "after": "yes"}, paired=True
        )


def test_named_level_sets_that_disagree_are_not_a_matched_design(
    paired_input: pd.DataFrame,
) -> None:
    with pytest.raises(SaValueError, match="disagrees at"):
        validate_categorical_input(
            paired_input,
            category_lv={"before": ["no", "yes"], "after": ["no", "maybe"]},
            paired=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"data": pd.DataFrame({"a": [], "b": []})}, "zero rows"),
        ({"data": pd.DataFrame({"a": ["x", "y"]})}, "at least two variables"),
        (
            {"data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}), "category_lv": {"a": []}},
            "at least two variables",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y"], "b": [[1], [2]]}),
            },
            "must be an atomic column",
        ),
        (
            {"data": pd.DataFrame({"a": [None, None], "b": ["p", "q"]})},
            "no non-missing value",
        ),
        (
            {"data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}), "category_lv": []},
            "must be a named mapping",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}),
                "category_lv": {"a": ["x", "y"], "c": ["p", "q"]},
            },
            "absent from `data`",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}),
                "category_lv": {"a": "x", "b": ["p", "q"]},
            },
            "must hold that variable's levels",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}),
                "category_lv": {"a": ["x"], "b": ["p", "q"]},
            },
            "at least two distinct non-missing levels",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}),
                "category_lv": {"a": ["x", "x"], "b": ["p", "q"]},
            },
            "at least two distinct non-missing levels",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}),
                "category_lv": {"a": ["x", "z"], "b": ["p", "q"]},
            },
            r"absent from `data\$a`",
        ),
        (
            {
                "data": pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]}),
                "category_lv": {"a": ["x", "z"], "b": ["p", "q"]},
            },
            "only 1 row",
        ),
    ],
)
def test_an_input_that_is_not_a_table_says_which_way(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SaValueError, match=message):
        validate_categorical_input(**kwargs)


def test_a_matrix_is_read_the_way_r_reads_one() -> None:
    matrix = np.array([["x", "p"], ["y", "q"], ["x", "q"]], dtype=object)
    resolved = validate_categorical_input(matrix)

    assert resolved.variables == ["0", "1"]
    assert resolved.n_used == 3


# --------------------------------------------------------------------------- #
# check_level_count
# --------------------------------------------------------------------------- #


def test_a_measurement_read_as_a_category_is_refused() -> None:
    values = [f"v{index}" for index in range(MAX_CATEGORY_LEVELS + 1)]
    frame = pd.DataFrame({"measure": values, "group": ["a", "b"] * (len(values) // 2) + ["a"]})

    with pytest.raises(SaValueError, match="more levels than the 20 `max_levels` allows"):
        validate_categorical_input(frame)


def test_naming_the_levels_to_keep_is_the_way_through() -> None:
    """The ceiling is on the resolved levels, not on the column.

    Fifty labels in a column and three named in ``category_lv`` is a table of
    three levels and a lot of dropped rows, which is what the argument is for.
    """
    values = [f"v{index % 50}" for index in range(200)]
    frame = pd.DataFrame({"measure": values, "group": ["a", "b"] * 100})

    resolved = validate_categorical_input(
        frame, category_lv={"measure": ["v0", "v1", "v2"], "group": ["a", "b"]}
    )
    assert resolved.category_lv["measure"] == ["v0", "v1", "v2"]
    assert resolved.n_dropped == 200 - resolved.n_used


def test_the_way_through_is_only_offered_when_it_is_one() -> None:
    """Naming the levels cannot be suggested to a caller who already named them."""
    over = {"a": [f"v{index}" for index in range(21)], "b": ["p", "q"]}

    with pytest.raises(SaValueError, match=r"also drops the rows at the rest"):
        check_level_count(over, MAX_CATEGORY_LEVELS, named_lv=False)
    with pytest.raises(SaValueError) as caught:
        check_level_count(over, MAX_CATEGORY_LEVELS, named_lv=True)
    assert "also drops the rows" not in str(caught.value)


def test_the_ceiling_itself_is_checked() -> None:
    with pytest.raises(SaValueError, match="`max_levels`"):
        check_level_count({"a": ["p", "q"]}, 1, named_lv=False)


# --------------------------------------------------------------------------- #
# categorical_counts and categorical_condition_counts
# --------------------------------------------------------------------------- #


def test_the_tables_reproduce_r(cat_input: pd.DataFrame, paired_input: pd.DataFrame) -> None:
    _, expected = load_case("cat_counts")

    plain = validate_categorical_input(cat_input)
    counts = categorical_counts(plain.data, ["answer", "grade"])
    assert_frame_close(counts.reset_index(drop=True).astype(float), expected["plain"]["counts"])
    assert list(counts.index) == expected["plain"]["row_levels"]
    assert list(counts.columns) == expected["plain"]["col_levels"]
    assert [counts.index.name, counts.columns.name] == expected["plain"]["dim_names"]

    named = validate_categorical_input(
        cat_input,
        category_lv={"answer": ["y", "n"], "grade": ["low", "mid"]},
        control_label={"answer": "n"},
    )
    named_counts = categorical_counts(named.data, named.variables)
    assert_frame_close(
        named_counts.reset_index(drop=True).astype(float), expected["named"]["counts"]
    )
    assert list(named_counts.index) == expected["named"]["row_levels"]

    paired = validate_categorical_input(paired_input, control_label="no", paired=True)
    paired_counts = categorical_counts(paired.data, paired.variables)
    assert_frame_close(
        paired_counts.reset_index(drop=True).astype(float), expected["paired"]["counts"]
    )
    assert list(paired_counts.index) == expected["paired"]["row_levels"]
    assert_frame_close(categorical_cells(paired_counts, "symmetry"), expected["paired_cells"])


def test_the_condition_table_reproduces_r(paired_input: pd.DataFrame) -> None:
    _, expected = load_case("cat_counts")
    paired = validate_categorical_input(paired_input, control_label="no", paired=True)

    condition = categorical_condition_counts(
        paired.data, paired.variables, paired.category_lv[paired.variables[0]]
    )
    assert_frame_close(
        condition.reset_index(drop=True).astype(float), expected["condition"]["counts"]
    )
    assert list(condition.index) == expected["condition"]["row_levels"]
    assert list(condition.columns) == expected["condition"]["col_levels"]
    assert [condition.index.name, condition.columns.name] == expected["condition"]["dim_names"]
    assert_frame_close(
        categorical_cells(condition, "marginal_homogeneity"), expected["condition_cells"]
    )


def test_a_level_nobody_answered_is_still_a_row(paired_input: pd.DataFrame) -> None:
    """The shape is a property of the design, not of the draw."""
    paired = validate_categorical_input(paired_input, paired=True)
    counts = categorical_counts(paired.data, paired.variables)

    assert counts.shape == (3, 3)
    assert counts.loc["maybe"].sum() == 0
    assert int(counts.to_numpy().sum()) == paired.n_used


def test_crossing_anything_but_two_variables_is_an_internal_error(
    cat_input: pd.DataFrame,
) -> None:
    resolved = validate_categorical_input(cat_input)
    with pytest.raises(SaInternalError, match="crosses two variables"):
        categorical_counts(resolved.data, resolved.variables)


# --------------------------------------------------------------------------- #
# The three diagnostics
# --------------------------------------------------------------------------- #


def test_the_diagnostics_reproduce_r_down_to_the_sentence(
    cell_tables: dict[str, pd.DataFrame],
) -> None:
    frame, _ = load_case("cat_chisq")
    _, expected = load_case("cat_diagnose")
    from_chisq = labelled_tables(frame)

    assert_close(diagnose_expected(categorical_cells(cell_tables["t3x4"])), expected["expected_ok"])
    assert_close(
        diagnose_expected(categorical_cells(from_chisq["t3x3_small"])),
        expected["expected_sparse"],
    )
    assert_close(
        diagnose_expected(categorical_cells(from_chisq["t2x4"])),
        expected["expected_one_small"],
    )
    assert_close(
        diagnose_expected(categorical_cells(cell_tables["hole"])), expected["expected_hole"]
    )

    assert_close(diagnose_discordance(12), expected["discordance_below"])
    assert_close(diagnose_discordance(25), expected["discordance_at"])
    assert_close(diagnose_discordance(48), expected["discordance_above"])

    assert_close(diagnose_repeated(5, 3), expected["repeated_below"])
    assert_close(diagnose_repeated(8, 3), expected["repeated_at"])
    assert_close(diagnose_repeated(12, 3), expected["repeated_above"])


def test_the_note_is_written_whether_or_not_the_rule_holds() -> None:
    """R builds the sentence unconditionally, so a caller can always print it."""
    for check in (
        diagnose_expected(pd.DataFrame({"expected": [10.0, 12.0, 8.0, 9.0]})),
        diagnose_discordance(48),
        diagnose_repeated(12, 3),
    ):
        assert check["approx_ok"] is True
        assert isinstance(check["note"], str)
        assert check["note"]


def test_the_rules_turn_on_their_own_thresholds() -> None:
    assert diagnose_discordance(DISCORDANT_PAIR_MIN)["approx_ok"] is True
    assert diagnose_discordance(DISCORDANT_PAIR_MIN - 1)["approx_ok"] is False
    assert diagnose_repeated(REPEATED_CELL_MIN, 1)["approx_ok"] is True
    assert diagnose_repeated(REPEATED_CELL_MIN - 1, 1)["approx_ok"] is False


def test_a_fifth_of_the_cells_may_be_small_but_none_may_be_tiny() -> None:
    """The second clause of the expected-count rule, in both directions."""
    below = float(EXPECTED_COUNT_MIN) - 1.0
    ten = [10.0] * 9

    assert diagnose_expected(pd.DataFrame({"expected": [*ten, below]}))["approx_ok"] is True
    # Two of ten cells is exactly a fifth of them, and three is past it.
    assert (
        diagnose_expected(pd.DataFrame({"expected": [*ten[:8], below, below]}))["approx_ok"] is True
    )
    assert (
        diagnose_expected(pd.DataFrame({"expected": [*ten[:7], below, below, below]}))["approx_ok"]
        is False
    )
    # One cell below 1 fails whatever share of the table is small.
    assert diagnose_expected(pd.DataFrame({"expected": [*ten, 0.5]}))["approx_ok"] is False


# --------------------------------------------------------------------------- #
# fmt_est
# --------------------------------------------------------------------------- #


def test_the_estimate_format_is_significant_digits_and_not_decimals() -> None:
    """What R's ``format(digits = 3)`` does, which the notes above are written by."""
    assert fmt_est(12345.678) == "12346"
    assert fmt_est(4.16666) == "4.17"
    assert fmt_est(2.5) == "2.5"
    assert fmt_est(1e5) == "1e+05"
    assert fmt_est(0.000123456) == "0.000123"
    assert fmt_est(0.0000123456) == "1.23e-05"
    assert fmt_est(None) == "NA"
    assert fmt_est(float("nan")) == "NA"
    assert fmt_est(float("inf")) == "Inf"
