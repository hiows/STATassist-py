"""A contingency table with a known association, in two truth tables.

The draws are not R's, so this is graded on the contract and on the statistics
rather than against a frozen fixture. The contract is the larger half: which
columns exist is itself an answer here, because a measure a design does not
define is a column that is absent rather than a column of missing values, and a
port that filled it in would let a caller read a paired odds ratio off a design
that has no pairs.

What replaces the fixture is that the planted answer can be recomputed. The
simulator says what distribution it drew from; counting the drawn table says what
came out. Those are two independent statements about the same design, and the gap
between them is the thing the simulator exists to make visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from statassist import simulate_categorical_groups
from statassist.core.contingency import (
    categorical_cells,
    categorical_condition_counts,
    categorical_counts,
    validate_categorical_input,
)
from statassist.core.contracts import categorical_cell_columns
from statassist.core.errors import SaValueError, SaWarning
from statassist.kernel.categorical import (
    assoc_measures,
    assoc_measures_paired,
    chisq,
    cochran_q,
    fisher,
    mcnemar,
)
from statassist.simulate.categorical_groups import CAT_PATTERNS

#: Enough rows that a 2 x 3 table is filled in every cell at the default margins.
N_ROWS = 400

#: A 2 x 2 design, where the signed measures are defined.
TWO_BY_TWO = {"exposure": ["yes", "no"], "outcome": ["case", "control"]}

#: Three repeated conditions, where Cochran's Q is the test and the paired
#: measures are not defined.
THREE_CONDITIONS = {
    "week_0": ["fail", "pass"],
    "week_1": ["fail", "pass"],
    "week_2": ["fail", "pass"],
}

#: Seeds a statistical claim is checked over, so that no assertion rests on one
#: draw happening to fall the right way.
SEEDS = (1, 2, 3, 4, 5)


def drawn_table(sim: Any) -> pd.DataFrame:
    """The table the analysis will see, counted off the simulated data."""
    resolved = validate_categorical_input(
        sim.args["data"], sim.args["category_lv"], paired=sim.args["paired"]
    )
    return categorical_counts(resolved.data, resolved.variables)


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_the_three_the_result_promises() -> None:
    sim = simulate_categorical_groups(n_samples=N_ROWS, seed=1)
    assert list(sim) == ["args", "truth", "truth_cell"]


def test_args_is_named_after_the_comparison_that_consumes_it() -> None:
    sim = simulate_categorical_groups(n_samples=N_ROWS, seed=1)
    assert list(sim.args) == ["data", "category_lv", "paired"]
    assert list(sim.args["data"].columns) == list(sim.args["category_lv"])
    assert sim.args["paired"] is False


def test_the_default_design_differs_by_whether_it_is_matched() -> None:
    crossed = simulate_categorical_groups(n_samples=N_ROWS, seed=1)
    matched = simulate_categorical_groups(n_samples=N_ROWS, paired=True, seed=1)

    assert list(crossed.args["category_lv"]) == ["cat_1", "cat_2"]
    assert list(matched.args["category_lv"]) == ["before", "after"]
    assert matched.args["paired"] is True


def test_the_cells_are_laid_out_the_way_the_comparison_lays_them_out() -> None:
    """``truth_cell`` merges onto ``cells`` with neither side renamed."""
    sim = simulate_categorical_groups(n_samples=N_ROWS, seed=1)
    cells = categorical_cells(drawn_table(sim))
    truth = sim.truth_cell

    assert list(truth["row_level"]) == list(cells["row_level"])
    assert list(truth["col_level"]) == list(cells["col_level"])
    merged = cells.merge(truth, on=["row_level", "col_level"])
    assert len(merged.index) == len(cells.index)
    assert set(categorical_cell_columns()) <= set(merged.columns)


def test_the_signed_measures_are_absent_rather_than_missing_off_a_two_by_two() -> None:
    """Which columns exist is the answer, so a wider table has fewer of them."""
    square = simulate_categorical_groups(
        n_samples=N_ROWS, category_lv=TWO_BY_TWO, assoc=0.4, seed=1
    )
    wide = simulate_categorical_groups(n_samples=N_ROWS, assoc=0.4, seed=1)

    assert list(square.truth.columns) == [
        "n_samples",
        "pattern",
        "assoc",
        "cramers_v",
        "phi_coefficient",
        "odds_ratio",
    ]
    assert list(wide.truth.columns) == ["n_samples", "pattern", "assoc", "cramers_v"]
    assert "phi_coefficient" not in wide.truth.columns


def test_a_matched_design_reports_the_measures_its_number_of_conditions_defines() -> None:
    two = simulate_categorical_groups(n_samples=N_ROWS, paired=True, seed=1)
    three = simulate_categorical_groups(
        n_samples=N_ROWS, category_lv=THREE_CONDITIONS, paired=True, seed=1
    )

    shared = ["n_samples", "pattern", "n_conditions", "move_up", "move_down"]
    assert list(two.truth.columns) == [
        *shared,
        "odds_ratio_paired",
        "risk_difference_paired",
        "cohens_g",
    ]
    assert list(three.truth.columns) == [*shared, "rate_first", "rate_last", "rate_range"]
    assert two.truth["pattern"].iloc[0] == "transition"


def test_a_matched_pair_carries_what_symmetry_expects_and_three_do_not() -> None:
    """Symmetry is the null over two conditions, so it is what the cells are scored on."""
    base = ["row_level", "col_level", "p_independent", "p_planted", "lift", "expected_n"]
    two = simulate_categorical_groups(n_samples=N_ROWS, paired=True, seed=1)
    three = simulate_categorical_groups(
        n_samples=N_ROWS, category_lv=THREE_CONDITIONS, paired=True, seed=1
    )

    assert list(two.truth_cell.columns) == [*base, "p_symmetric", "expected_symmetry_n"]
    assert list(three.truth_cell.columns) == base

    # Over three the cells are condition by response, so the table is k x 2.
    assert len(three.truth_cell.index) == len(THREE_CONDITIONS) * 2
    assert list(dict.fromkeys(three.truth_cell["row_level"])) == list(THREE_CONDITIONS)


def test_the_diagonal_of_a_matched_pair_has_nothing_planted_in_it() -> None:
    sim = simulate_categorical_groups(n_samples=N_ROWS, paired=True, discordance=(0.3, 0.1), seed=1)
    cells = sim.truth_cell
    diagonal = cells["row_level"] == cells["col_level"]

    assert np.allclose(cells.loc[diagonal, "p_symmetric"], cells.loc[diagonal, "p_planted"])
    assert not np.allclose(cells.loc[~diagonal, "p_symmetric"], cells.loc[~diagonal, "p_planted"])


def test_the_expected_counts_of_a_repeated_table_count_every_measurement() -> None:
    """A subject is measured under every condition, so the table is n times k."""
    sim = simulate_categorical_groups(
        n_samples=N_ROWS, category_lv=THREE_CONDITIONS, paired=True, seed=1
    )
    assert sim.truth_cell["expected_n"].sum() == pytest.approx(N_ROWS * len(THREE_CONDITIONS))

    pair = simulate_categorical_groups(n_samples=N_ROWS, paired=True, seed=1)
    assert pair.truth_cell["expected_n"].sum() == pytest.approx(N_ROWS)


# --------------------------------------------------------------------------- #
# The two warnings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"paired": True, "assoc": 0.4}, "assoc"),
        ({"paired": True, "pattern": "single"}, "pattern"),
        ({"paired": True, "margins": {"before": [1, 1], "after": [1, 1]}}, "margins"),
        ({"discordance": (0.3, 0.1)}, "discordance"),
    ],
)
def test_an_argument_of_the_other_design_is_a_warning(kwargs: dict[str, Any], named: str) -> None:
    with pytest.warns(SaWarning, match=named):
        simulate_categorical_groups(n_samples=N_ROWS, seed=1, **kwargs)


def test_the_warning_names_both_lists_and_not_only_the_wrong_one() -> None:
    """A call that names the wrong knob is a call that expected the other design."""
    with pytest.warns(SaWarning) as caught:
        simulate_categorical_groups(n_samples=N_ROWS, paired=True, assoc=0.4, seed=1)

    message = str(caught[0].message)
    assert "a matched design reads discordance" in message
    assert "value(s) given for assoc" in message
    assert "`paired = FALSE`" in message


def test_passing_a_default_by_hand_is_still_passing_it() -> None:
    """The sentinel says what ``missing()`` said, and survives the default."""
    with pytest.warns(SaWarning, match="assoc"):
        simulate_categorical_groups(n_samples=N_ROWS, paired=True, assoc=0.3, seed=1)


def test_the_design_that_reads_the_argument_is_not_warned_at(
    recwarn: pytest.WarningsRecorder,
) -> None:
    simulate_categorical_groups(n_samples=N_ROWS, assoc=0.4, pattern="single", seed=1)
    simulate_categorical_groups(n_samples=N_ROWS, paired=True, discordance=(0.3, 0.1), seed=1)
    assert [w for w in recwarn if issubclass(w.category, SaWarning)] == []


def test_a_level_nobody_was_drawn_at_is_a_warning() -> None:
    """An empty row is not something a test of independence can be run on."""
    with pytest.warns(SaWarning, match="no row was drawn at level"):
        simulate_categorical_groups(
            n_samples=5,
            category_lv={"a": ["p", "q"], "b": [f"lv{i}" for i in range(12)]},
            assoc=0,
            seed=1,
        )


# --------------------------------------------------------------------------- #
# The failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"category_lv": ["a", "b"]}, "must be a named mapping"),
        ({"category_lv": {"a": ["x", "y"]}}, "must be a named mapping"),
        ({"category_lv": {"a": ["x", "y"], "b": "pq"}}, "at least two distinct"),
        ({"category_lv": {"a": ["x", "y"], "b": ["p"]}}, "at least two distinct"),
        ({"category_lv": {"a": ["x", "y"], "b": ["p", "p"]}}, "at least two distinct"),
        (
            {"category_lv": {"a": ["x", "y"], "b": ["p", "q"], "c": ["u", "v"]}},
            "exactly two variables",
        ),
        ({"pattern": "diagonal"}, "`pattern` must be one of"),
        ({"assoc": 1.5}, "`assoc`"),
        ({"paired": 1}, "`paired`"),
        ({"margins": {"cat_1": [1, 1]}}, "one entry per variable"),
        ({"margins": {"cat_1": [1, 1], "cat_2": [1, 1]}}, "one weight per level"),
        ({"margins": {"cat_1": [1, 0], "cat_2": [1, 1, 1]}}, "positive weight"),
        ({"margins": {"cat_1": [1, -1], "cat_2": [1, 1, 1]}}, "margins"),
    ],
)
def test_a_table_that_cannot_be_drawn_says_which_way(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SaValueError, match=message):
        simulate_categorical_groups(n_samples=N_ROWS, seed=1, **kwargs)


def test_a_table_needs_at_least_two_rows_to_be_one() -> None:
    with pytest.raises(SaValueError, match="`n_samples`"):
        simulate_categorical_groups(n_samples=1, seed=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"discordance": (0.3,)}, "two transition probabilities"),
        ({"discordance": (0.3, 0.1, 0.2)}, "two transition probabilities"),
        ({"discordance": "up"}, "two transition probabilities"),
        ({"discordance": (0.3, 1.4)}, "discordance"),
        (
            {"category_lv": {"before": ["fail", "pass"], "after": ["pass", "fail"]}},
            "same levels in the same order",
        ),
        (
            {"category_lv": {"before": ["a", "b", "c"], "after": ["a", "b", "c"]}},
            "needs binary conditions",
        ),
    ],
)
def test_a_matched_design_that_cannot_be_drawn_says_which_way(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(SaValueError, match=message):
        simulate_categorical_groups(n_samples=N_ROWS, paired=True, seed=1, **kwargs)


# --------------------------------------------------------------------------- #
# What was planted
# --------------------------------------------------------------------------- #


def test_no_association_is_the_product_of_the_margins_exactly() -> None:
    """``assoc = 0`` is null in the strict sense, which is what a type I rate needs."""
    sim = simulate_categorical_groups(
        n_samples=N_ROWS, margins={"cat_1": [3, 1], "cat_2": [2, 1, 1]}, assoc=0, seed=1
    )
    truth = sim.truth_cell

    assert (truth["lift"] == 1).all()
    assert np.array_equal(truth["p_planted"], truth["p_independent"])
    assert sim.truth["cramers_v"].iloc[0] == 0


@pytest.mark.parametrize("pattern", CAT_PATTERNS)
def test_every_pattern_leaves_the_margins_where_it_found_them(pattern: str) -> None:
    """That is what makes the null hypothesis the only thing that moves."""
    sim = simulate_categorical_groups(
        n_samples=N_ROWS,
        category_lv={"a": ["x", "y", "z"], "b": ["p", "q", "r"]},
        margins={"a": [3, 2, 1], "b": [1, 1, 2]},
        assoc=0.8,
        pattern=pattern,
        seed=1,
    )
    truth = sim.truth_cell
    planted = truth.pivot(index="row_level", columns="col_level", values="p_planted")
    independent = truth.pivot(index="row_level", columns="col_level", values="p_independent")

    assert np.allclose(planted.sum(axis=1), independent.sum(axis=1))
    assert np.allclose(planted.sum(axis=0), independent.sum(axis=0))
    assert sim.truth["cramers_v"].iloc[0] > 0


def test_the_largest_association_puts_a_structural_zero_in_the_table() -> None:
    """``assoc = 1`` is where the exact test and the approximation part company."""
    sim = simulate_categorical_groups(n_samples=N_ROWS, category_lv=TWO_BY_TWO, assoc=1, seed=1)
    truth = sim.truth_cell

    assert (truth["p_planted"] >= 0).all()
    assert (truth["p_planted"] == 0).any()
    assert truth["p_planted"].sum() == pytest.approx(1.0)
    # A zero in one cell of a 2 x 2 makes the odds ratio undefined rather than
    # very large, which is what the column records.
    assert np.isnan(sim.truth["odds_ratio"].iloc[0])


def test_the_corner_pattern_plants_an_odds_ratio_above_one() -> None:
    sim = simulate_categorical_groups(
        n_samples=N_ROWS, category_lv=TWO_BY_TWO, assoc=0.5, pattern="corner", seed=1
    )
    assert sim.truth["odds_ratio"].iloc[0] > 1
    assert sim.truth["phi_coefficient"].iloc[0] > 0


def test_the_gradient_pattern_is_monotone_over_the_levels_as_given() -> None:
    sim = simulate_categorical_groups(
        n_samples=N_ROWS,
        category_lv={"dose": ["low", "mid", "high"], "response": ["none", "partial", "full"]},
        assoc=0.6,
        pattern="gradient",
        seed=1,
    )
    lift = sim.truth_cell.pivot(index="row_level", columns="col_level", values="lift")
    lift = lift.loc[["low", "mid", "high"], ["none", "partial", "full"]]

    # The first level of each variable is lifted with the first of the other and
    # depressed with the last, and the middle level is untouched.
    assert lift.iloc[0, 0] > 1
    assert lift.iloc[0, -1] < 1
    assert lift.iloc[-1, 0] < 1
    assert lift.iloc[-1, -1] > 1
    assert np.allclose(lift.iloc[1, :], 1)


def test_the_matched_odds_ratio_is_the_ratio_of_the_transitions() -> None:
    sim = simulate_categorical_groups(n_samples=N_ROWS, paired=True, discordance=(0.3, 0.1), seed=1)
    assert sim.truth["odds_ratio_paired"].iloc[0] == pytest.approx(3.0)
    assert sim.truth["move_up"].iloc[0] == pytest.approx(0.3)


def test_equal_transitions_are_the_strict_null_of_both_matched_tests() -> None:
    """They cancel: the rate stays at one half and nothing is planted."""
    pair = simulate_categorical_groups(
        n_samples=N_ROWS, paired=True, discordance=(0.2, 0.2), seed=1
    )
    assert pair.truth["odds_ratio_paired"].iloc[0] == pytest.approx(1.0)
    assert pair.truth["risk_difference_paired"].iloc[0] == pytest.approx(0.0)
    assert pair.truth["cohens_g"].iloc[0] == pytest.approx(0.0)

    three = simulate_categorical_groups(
        n_samples=N_ROWS,
        category_lv=THREE_CONDITIONS,
        paired=True,
        discordance=(0.2, 0.2),
        seed=1,
    )
    assert three.truth["rate_range"].iloc[0] == pytest.approx(0.0)


def test_a_rising_response_rate_is_what_three_conditions_plant() -> None:
    sim = simulate_categorical_groups(
        n_samples=N_ROWS,
        category_lv=THREE_CONDITIONS,
        paired=True,
        discordance=(0.4, 0.1),
        seed=1,
    )
    rates = sim.truth_cell[sim.truth_cell["col_level"] == "pass"]
    by_condition = rates.set_index("row_level")["p_planted"]

    assert sim.truth["rate_last"].iloc[0] > sim.truth["rate_first"].iloc[0]
    assert list(by_condition.index) == list(THREE_CONDITIONS)
    assert by_condition.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# What comes back out, over several seeds
# --------------------------------------------------------------------------- #


def test_the_same_seed_draws_the_same_table_and_a_different_one_does_not() -> None:
    once = simulate_categorical_groups(n_samples=N_ROWS, assoc=0.4, seed=7)
    again = simulate_categorical_groups(n_samples=N_ROWS, assoc=0.4, seed=7)
    other = simulate_categorical_groups(n_samples=N_ROWS, assoc=0.4, seed=8)

    pd.testing.assert_frame_equal(once.args["data"], again.args["data"])
    assert not once.args["data"].equals(other.args["data"])
    # The planted answer is a function of the arguments alone.
    pd.testing.assert_frame_equal(once.truth, other.truth)


def test_the_estimated_association_sits_above_the_planted_one() -> None:
    """The bias the simulator exists to make visible.

    Every departure from independence counts towards Cramer's V whether it was
    planted or drawn, so the estimate is high on average - and at ``assoc = 0``,
    where nothing was planted at all, it is high in every draw.
    """
    high = 0
    for seed in SEEDS:
        sim = simulate_categorical_groups(n_samples=120, assoc=0, seed=seed)
        estimated = float(assoc_measures(drawn_table(sim))["estimate"].iloc[0])
        assert estimated > 0
        high += estimated > float(sim.truth["cramers_v"].iloc[0])
    assert high == len(SEEDS)


def test_a_planted_association_is_recovered_by_the_test_it_was_planted_for() -> None:
    for seed in SEEDS:
        sim = simulate_categorical_groups(
            n_samples=600, category_lv=TWO_BY_TWO, assoc=0.6, seed=seed
        )
        table = drawn_table(sim)
        assert chisq(table)["pval"] < 0.001
        assert fisher(table)["pval"] < 0.001
        # And the drawn odds ratio is on the side the corner pattern put it.
        assert float(assoc_measures(table)["estimate"].iloc[1]) > 0


def test_the_drawn_table_approaches_the_planted_distribution() -> None:
    """More rows, less gap. The two are not the same thing and converge."""
    planted = None
    gaps = []
    for n_samples in (200, 20000):
        sim = simulate_categorical_groups(n_samples=n_samples, assoc=0.5, seed=11)
        table = drawn_table(sim)
        share = table.to_numpy(dtype=float) / table.to_numpy(dtype=float).sum()
        planted = sim.truth_cell["p_planted"].to_numpy().reshape(share.shape, order="F")
        gaps.append(float(np.abs(share - planted).max()))

    assert gaps[1] < gaps[0]
    assert gaps[1] < 0.01


def test_the_matched_transitions_come_back_as_the_paired_odds_ratio() -> None:
    for seed in SEEDS:
        sim = simulate_categorical_groups(
            n_samples=4000, paired=True, discordance=(0.3, 0.1), seed=seed
        )
        table = drawn_table(sim)
        estimated = float(assoc_measures_paired(table)["estimate"].iloc[0])
        assert estimated == pytest.approx(3.0, rel=0.25)
        assert mcnemar(table)["pval"] < 0.001


def test_a_climb_in_the_response_rate_is_what_cochran_q_finds() -> None:
    sim = simulate_categorical_groups(
        n_samples=400,
        category_lv=THREE_CONDITIONS,
        paired=True,
        discordance=(0.4, 0.1),
        seed=1,
    )
    resolved = validate_categorical_input(sim.args["data"], sim.args["category_lv"], paired=True)
    condition = categorical_condition_counts(
        resolved.data, resolved.variables, resolved.category_lv[resolved.variables[0]]
    )
    assert condition["pass"].is_monotonic_increasing

    matrix = np.column_stack(
        [(resolved.data[name] == "pass").to_numpy(dtype=float) for name in resolved.variables]
    )
    assert cochran_q(matrix)["pval"] < 0.001


def test_equal_transitions_leave_the_matched_tests_with_nothing_to_find() -> None:
    """The other side of the same claim: the strict null is not rejected."""
    rejected = 0
    for seed in SEEDS:
        sim = simulate_categorical_groups(
            n_samples=400, paired=True, discordance=(0.25, 0.25), seed=seed
        )
        rejected += mcnemar(drawn_table(sim))["pval"] < 0.05
    assert rejected <= 1


# --------------------------------------------------------------------------- #
# The simulator against the kernels it feeds
# --------------------------------------------------------------------------- #


def test_the_simulator_and_the_kernels_read_the_same_labels() -> None:
    """Joined by hand, since the public comparison arrives in Phase 3.

    The chain is the one the scenario function will run: resolve the input, count
    the table, test it, and expand it into cells. What is checked is that the
    labels and the direction survive the whole of it - the level order the
    simulator fixed is the order the table is in, and the cell the simulator
    lifted is the cell whose residual is positive.
    """
    sim = simulate_categorical_groups(
        n_samples=800, category_lv=TWO_BY_TWO, assoc=0.6, pattern="corner", seed=3
    )
    resolved = validate_categorical_input(
        sim.args["data"], sim.args["category_lv"], paired=sim.args["paired"]
    )

    assert resolved.variables == list(TWO_BY_TWO)
    assert resolved.category_lv == TWO_BY_TWO
    assert resolved.n_used == 800
    assert resolved.n_dropped == 0
    assert resolved.n_incomplete == 0

    counts = categorical_counts(resolved.data, resolved.variables)
    assert list(counts.index) == TWO_BY_TWO["exposure"]
    assert list(counts.columns) == TWO_BY_TWO["outcome"]

    cells = categorical_cells(counts)
    scored = cells.merge(sim.truth_cell, on=["row_level", "col_level"])
    assert len(scored.index) == 4

    # The lift and the residual are two statements about the same cell, so they
    # agree in sign whenever the effect is large enough to survive the draw.
    lifted = scored["lift"] > 1
    assert (scored.loc[lifted, "std_residual"] > 0).all()
    assert (scored.loc[~lifted, "std_residual"] < 0).all()


def test_the_matched_chain_reads_the_same_square_table() -> None:
    sim = simulate_categorical_groups(n_samples=600, paired=True, discordance=(0.3, 0.1), seed=3)
    resolved = validate_categorical_input(sim.args["data"], sim.args["category_lv"], paired=True)
    counts = categorical_counts(resolved.data, resolved.variables)

    assert list(counts.index) == list(counts.columns) == ["fail", "pass"]
    assert counts.shape == (2, 2)

    cells = categorical_cells(counts, "symmetry")
    scored = cells.merge(sim.truth_cell, on=["row_level", "col_level"])
    off_diagonal = scored["row_level"] != scored["col_level"]

    # Symmetry is the null, so the expected count of a cell is what
    # `expected_symmetry_n` planted rather than what independence would say.
    assert np.corrcoef(scored["expected"], scored["expected_symmetry_n"])[0, 1] > 0.99
    assert scored.loc[off_diagonal, "std_residual"].isna().all()
    # More subjects moved up than down, so the upper discordant cell is the fuller.
    fail_to_pass = (scored["row_level"] == "fail") & (scored["col_level"] == "pass")
    pass_to_fail = (scored["row_level"] == "pass") & (scored["col_level"] == "fail")
    assert (
        scored.loc[fail_to_pass, "observed"].iloc[0] > scored.loc[pass_to_fail, "observed"].iloc[0]
    )
