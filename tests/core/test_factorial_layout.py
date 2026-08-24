"""``core/factorial.py`` against the numbers R produced.

Every index R wrote out is one-based, so the assertions add one back rather than
comparing against a shifted fixture: which cell is which is what is being graded,
not how it is counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_close, assert_frame_close, load_expected

from statassist.core.errors import SaValueError
from statassist.core.factorial import (
    FACT_TOL,
    _first_max_abs,
    fact_cell_index,
    fact_cell_labels,
    fact_collapse,
    fact_component,
    fact_contrast_skeleton,
    fact_control_first,
    fact_grid,
    fact_subsets,
    fact_term_effect,
    fact_term_labels,
    fact_terms,
)

#: The crossed design the fixtures were generated from.
FACT_LV = {"treatment": ["control", "treat_A", "treat_B"], "sex": ["male", "female"]}

#: A three-factor design, for the term order and column order of a three-way
#: interaction.
CUBE_LV = {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]}

#: The per-cell shift the decomposition fixture was taken on.
FACT_EFF = [0.0, 1.4, 2.1, 0.5, 0.4, 2.9]


def one_based(values) -> list[int]:
    """The port's zero-based indices as the one-based ones R wrote out."""
    return [int(value) + 1 for value in values]


class TestGrid:
    def test_the_cell_order_matches_r(self):
        expected = load_expected("fact_layout")
        cells = fact_grid(FACT_LV)
        assert list(cells.columns) == ["treatment", "sex"]
        for name, column in expected["cells"].items():
            assert one_based(cells[name]) == as_list(column)

    def test_the_first_factor_varies_fastest(self):
        cells = fact_grid(FACT_LV)
        # Three treatments cycle within one sex before the sex changes, which is
        # what R's `expand.grid()` does and the opposite of what
        # `itertools.product` and `numpy.meshgrid` do by default.
        assert list(cells["treatment"]) == [0, 1, 2, 0, 1, 2]
        assert list(cells["sex"]) == [0, 0, 0, 1, 1, 1]

    def test_a_three_factor_grid_matches_r(self):
        expected = load_expected("fact_layout")
        cells = fact_grid(CUBE_LV)
        for name, column in expected["cube_cells"].items():
            assert one_based(cells[name]) == as_list(column)

    def test_crossing_nothing_is_one_combination_holding_no_constraints(self):
        expected = load_expected("fact_layout")
        cells = fact_grid({})
        assert len(cells) == expected["empty_grid_rows"]
        assert len(cells.columns) == expected["empty_grid_cols"]


class TestCellLabels:
    def test_matches_r(self):
        expected = load_expected("fact_layout")
        assert fact_cell_labels(FACT_LV, fact_grid(FACT_LV)) == as_list(expected["labels"])

    def test_a_three_factor_label_joins_every_factor(self):
        expected = load_expected("fact_layout")
        assert fact_cell_labels(CUBE_LV, fact_grid(CUBE_LV)) == as_list(expected["cube_labels"])

    def test_a_design_of_no_factors_labels_its_single_cell_with_nothing(self):
        # R raises here, because `apply()` over a zero-column matrix has no
        # margin to walk. The answer `paste(character(0), collapse = ".")` gives
        # is the empty string, which is what the port returns, so the empty grid
        # survives every step built on it rather than all but this one.
        assert fact_cell_labels({}, fact_grid({})) == [""]


class TestCellIndex:
    def test_matches_r(self):
        expected = load_expected("fact_layout")
        cells = fact_grid(FACT_LV)
        produced = fact_cell_index(cells.to_numpy(), [3, 2])
        assert one_based(produced) == as_list(expected["cell_index"])

    def test_a_three_factor_index_matches_r(self):
        expected = load_expected("fact_layout")
        cells = fact_grid(CUBE_LV)
        produced = fact_cell_index(cells.to_numpy(), [2, 2, 2])
        assert one_based(produced) == as_list(expected["cube_cell_index"])

    def test_the_index_agrees_with_the_grid_it_came_from(self):
        for levels, dims in ((FACT_LV, [3, 2]), (CUBE_LV, [2, 2, 2])):
            cells = fact_grid(levels)
            produced = fact_cell_index(cells.to_numpy(), dims)
            assert list(produced) == list(range(len(cells)))

    def test_with_no_factors_every_row_belongs_to_the_single_cell(self):
        expected = load_expected("fact_layout")
        produced = fact_cell_index(np.zeros((3, 0), dtype=int), [])
        assert one_based(produced) == as_list(expected["empty_cell_index"])


class TestTerms:
    def test_matches_r(self):
        expected = load_expected("fact_layout")
        produced = fact_terms(list(FACT_LV))
        assert [list(term) for term in produced] == [as_list(term) for term in expected["terms"]]

    def test_main_effects_come_before_interactions(self):
        produced = fact_terms(list(CUBE_LV))
        assert [len(term) for term in produced] == [1, 1, 1, 2, 2, 2, 3]

    def test_a_three_factor_term_list_matches_r(self):
        expected = load_expected("fact_layout")
        produced = fact_terms(list(CUBE_LV))
        assert [list(term) for term in produced] == [
            as_list(term) for term in expected["cube_terms"]
        ]

    def test_the_labels_match_r(self):
        expected = load_expected("fact_layout")
        assert fact_term_labels(fact_terms(list(FACT_LV))) == as_list(expected["term_labels"])
        assert fact_term_labels(fact_terms(list(CUBE_LV))) == as_list(expected["cube_term_labels"])

    def test_an_interaction_is_labelled_the_way_r_writes_a_formula_term(self):
        assert fact_term_labels([("a", "b")]) == ["a:b"]


class TestSubsets:
    def test_matches_r_and_holds_the_empty_one(self):
        expected = load_expected("fact_layout")
        produced = fact_subsets(["treatment", "sex"])
        # R's empty subset is `character(0)`, which jsonlite writes as `[]`.
        assert [list(subset) for subset in produced] == [
            as_list(subset) if subset else [] for subset in expected["subsets"]
        ]
        assert produced[0] == ()


class TestTolerance:
    def test_matches_r(self):
        assert FACT_TOL == load_expected("fact_layout")["tol"]


class TestCollapse:
    def test_matches_r(self):
        expected = load_expected("fact_decompose")
        cells = fact_grid(FACT_LV)
        assert_close(
            list(fact_collapse(FACT_EFF, cells, [])),
            as_list(expected["collapse_none"]),
        )
        assert_close(
            list(fact_collapse(FACT_EFF, cells, ["treatment"])),
            as_list(expected["collapse_treatment"]),
        )
        assert_close(
            list(fact_collapse(FACT_EFF, cells, ["sex"])),
            as_list(expected["collapse_sex"]),
        )
        assert_close(
            list(fact_collapse(FACT_EFF, cells, ["treatment", "sex"])),
            as_list(expected["collapse_both"]),
        )

    def test_keeping_every_factor_leaves_the_values_alone(self):
        cells = fact_grid(FACT_LV)
        assert_close(
            list(fact_collapse(FACT_EFF, cells, list(FACT_LV))),
            FACT_EFF,
        )

    def test_the_average_is_unweighted(self):
        # Two cells of one level and one of the other: the level with two cells
        # still counts each of them once, which is what makes the decomposition
        # a statement about the levels rather than about the sample sizes.
        cells = pd.DataFrame({"f": [0, 0, 1]})
        assert list(fact_collapse([1.0, 3.0, 10.0], cells, ["f"])) == [2.0, 2.0, 10.0]


class TestComponent:
    def test_matches_r_term_by_term(self):
        expected = load_expected("fact_decompose")
        cells = fact_grid(FACT_LV)
        for term, wanted in zip(fact_terms(list(FACT_LV)), expected["component"], strict=True):
            assert_close(
                list(fact_component(FACT_EFF, cells, term)),
                as_list(wanted),
                path=f"component[{':'.join(term)}]",
            )

    def test_the_components_and_the_grand_mean_add_back_to_the_values(self):
        cells = fact_grid(FACT_LV)
        total = fact_collapse(FACT_EFF, cells, [])
        for term in fact_terms(list(FACT_LV)):
            total = total + fact_component(FACT_EFF, cells, term)
        assert_close(list(total), FACT_EFF)

    def test_a_component_of_a_flat_shift_is_exactly_zero(self):
        cells = fact_grid(FACT_LV)
        for term in fact_terms(list(FACT_LV)):
            assert list(fact_component([2.5] * 6, cells, term)) == [0.0] * 6


class TestTermEffect:
    def test_matches_r(self):
        # term_effect signs follow FACT_TOL near-ties (earlier cell), which is
        # the rule CRAN R's which.max does not yet apply; the golden was updated
        # for that contract. Collapse / component vectors still match R.
        expected = load_expected("fact_decompose")
        cells = fact_grid(FACT_LV)
        terms = fact_terms(list(FACT_LV))
        assert_close(
            list(fact_term_effect(FACT_EFF, cells, terms)),
            as_list(expected["term_effect"]),
        )

    def test_a_flat_shift_moves_no_term(self):
        expected = load_expected("fact_decompose")
        cells = fact_grid(FACT_LV)
        terms = fact_terms(list(FACT_LV))
        assert_close(
            list(fact_term_effect([2.5] * 6, cells, terms)),
            as_list(expected["flat_term_effect"]),
        )

    def test_a_tie_in_absolute_value_takes_the_earlier_cell(self):
        # A two-level factor shifted by d has components -d/2 and +d/2, which tie
        # in absolute value, so the earlier cell is the one reported.
        levels = {"f": ["lo", "hi"]}
        cells = fact_grid(levels)
        produced = fact_term_effect([0.0, 4.0], cells, fact_terms(list(levels)))
        assert produced[0] == pytest.approx(-2.0)

    def test_a_near_tie_broken_by_an_ulp_still_takes_the_earlier_cell(self):
        # Later magnitude wins under nanargmax; FACT_TOL keeps the earlier index.
        assert _first_max_abs(np.array([-0.5, 0.5 + 1e-15])) == 0
        produced = fact_term_effect(
            [0.0, 4.0 + np.finfo(float).eps],
            fact_grid({"f": ["lo", "hi"]}),
            fact_terms(["f"]),
        )
        assert produced[0] == pytest.approx(-2.0)

    def test_a_clear_winner_is_not_treated_as_a_tie(self):
        assert _first_max_abs(np.array([-0.3, 0.5])) == 1
        levels = {"f": ["a", "b", "c"]}
        cells = fact_grid(levels)
        produced = fact_term_effect([0.0, 1.0, 10.0], cells, fact_terms(list(levels)))
        # Grand mean 11/3; the last cell is furthest and positive.
        assert produced[0] == pytest.approx(10.0 - 11.0 / 3.0)

    def test_a_component_of_nothing_but_missing_values_is_missing(self):
        cells = fact_grid(FACT_LV)
        produced = fact_term_effect([np.nan] * 6, cells, fact_terms(list(FACT_LV)))
        assert np.isnan(produced).all()


class TestControlFirst:
    def test_matches_r(self):
        expected = load_expected("fact_layout")
        produced = fact_control_first(FACT_LV, {"sex": "female"})
        assert produced == {key: as_list(value) for key, value in expected["control_first"].items()}

    def test_a_factor_not_named_keeps_the_order_it_arrived_in(self):
        produced = fact_control_first(FACT_LV, {"sex": "female"})
        assert produced["treatment"] == FACT_LV["treatment"]

    def test_naming_nothing_leaves_every_factor_alone(self):
        assert fact_control_first(FACT_LV, None) == FACT_LV

    def test_the_input_is_not_modified(self):
        original = {name: list(levels) for name, levels in FACT_LV.items()}
        fact_control_first(FACT_LV, {"sex": "female"})
        assert FACT_LV == original

    def test_a_one_element_sequence_says_the_same_thing_as_a_bare_name(self):
        assert fact_control_first(FACT_LV, {"sex": ["female"]}) == fact_control_first(
            FACT_LV, {"sex": "female"}
        )

    def test_a_factor_the_design_does_not_hold_is_named(self):
        with pytest.raises(SaValueError, match="factor\\(s\\) the design does not hold: age"):
            fact_control_first(FACT_LV, {"age": "old"})

    def test_a_level_the_factor_does_not_hold_names_the_factor_it_is_missing_from(self):
        with pytest.raises(SaValueError, match=r"control_label\['sex'\].*factor_lv\['sex'\]"):
            fact_control_first(FACT_LV, {"sex": "other"})

    def test_the_source_of_the_levels_is_what_the_message_points_at(self):
        with pytest.raises(SaValueError, match=r"factors\['sex'\]"):
            fact_control_first(FACT_LV, {"sex": "other"}, lv_source="factors")

    def test_an_entry_holding_more_than_one_level_is_not_a_direction(self):
        with pytest.raises(SaValueError, match="one level name per factor"):
            fact_control_first(FACT_LV, {"sex": ["female", "male"]})

    def test_something_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(SaValueError, match="named list or named character vector"):
            fact_control_first(FACT_LV, ["female"])


class TestContrastSkeleton:
    def test_the_table_matches_r(self):
        expected = load_expected("fact_contrast_skeleton")
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        assert_frame_close(skeleton.table, expected["table"])

    def test_the_cell_selections_match_r(self):
        expected = load_expected("fact_contrast_skeleton")
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        for row, (first, second) in enumerate(zip(skeleton.sel1, skeleton.sel2, strict=True)):
            assert one_based(first) == as_list(expected["sel1"][row]), f"sel1[{row}]"
            assert one_based(second) == as_list(expected["sel2"][row]), f"sel2[{row}]"

    def test_the_marginal_block_of_a_factor_comes_before_its_simple_effects(self):
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        table = skeleton.table
        first_treatment = table.index[table["factor"] == "treatment"][0]
        assert table["stratum"].iloc[first_treatment] is None

    def test_a_marginal_selection_averages_the_other_factors_away(self):
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        marginal = skeleton.table["stratum"].isna()
        # Both sexes of one treatment level, rather than one cell.
        assert len(skeleton.sel1[int(np.flatnonzero(marginal)[0])]) == 2

    def test_a_simple_effect_holds_the_other_factors_at_one_combination(self):
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        simple = ~skeleton.table["stratum"].isna()
        assert len(skeleton.sel1[int(np.flatnonzero(simple)[0])]) == 1

    def test_the_direction_puts_the_reference_on_the_right(self):
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        rows = skeleton.table
        against_control = rows[rows["group2"] == "control"]
        assert len(against_control) > 0
        assert set(against_control["group1"]) == {"treat_A", "treat_B"}
