"""``compare_categorical_groups`` against its result contract and planted truths.

The numbers themselves are graded against R in ``tests/kernel/test_categorical.py``
and ``tests/core/test_contingency.py``, where the kernels and the cell table are.
What is here is the assembly: which design runs which tests, what the slots hold,
what the reference level does to the direction of a measure, and what the function
says to a caller who asked for something it cannot do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statassist import compare_categorical_groups, simulate_categorical_groups
from statassist.compare.categorical_groups import MIN_RESAMPLES
from statassist.core.contracts import (
    association_columns,
    categorical_cell_columns,
    categorical_test_columns,
)
from statassist.core.errors import SaValueError, SaWarning
from statassist.core.result import SaCategorical

#: Two variables crossed the corner way: the association is real and not extreme.
CROSSED = pd.DataFrame(
    {
        "smoker": ["y"] * 60 + ["n"] * 60,
        "grade": (
            ["high"] * 10
            + ["mid"] * 20
            + ["low"] * 30
            + ["high"] * 30
            + ["mid"] * 20
            + ["low"] * 10
        ),
    }
)

#: One thing measured twice on the same 50 rows.
MATCHED = pd.DataFrame(
    {
        "before": ["pass"] * 20 + ["fail"] * 30,
        "after": ["pass"] * 18 + ["fail"] * 2 + ["pass"] * 14 + ["fail"] * 16,
    }
)


def repeated(n_subjects: int = 40) -> pd.DataFrame:
    """Three binary conditions on the same subjects, with a rising response rate."""
    rng = np.random.default_rng(3)
    rates = (0.3, 0.5, 0.75)
    return pd.DataFrame(
        {
            f"t{index + 1}": np.where(rng.random(n_subjects) < rate, "y", "n")
            for index, rate in enumerate(rates)
        }
    )


@pytest.fixture(scope="module")
def res() -> SaCategorical:
    return compare_categorical_groups(CROSSED)


@pytest.fixture(scope="module")
def planted():
    sim = simulate_categorical_groups(n_samples=4000, assoc=0.5, seed=11)
    return sim, compare_categorical_groups(**sim.args)


class TestIndependentDesign:
    def test_returns_a_categorical_result_rather_than_a_comparison(self, res) -> None:
        assert isinstance(res, SaCategorical)
        assert res.analysis == "categorical_comparison"
        assert "features" not in res
        assert "effect" not in res

    def test_two_tests_run_and_the_null_is_independence(self, res) -> None:
        assert list(res.tests) == ["chisq_test", "fisher_test"]
        assert res.design["null"] == "independence"
        assert res.design["paired"] is False
        assert res.design["pairing"] is None

    def test_every_test_table_is_one_row_of_the_contract_columns(self, res) -> None:
        for name, table in res.tests.items():
            assert len(table.index) == 1, name
            assert list(table.columns)[: len(categorical_test_columns())] == (
                categorical_test_columns()
            )
            assert "pval_adj" not in table.columns

    def test_the_cells_carry_the_contract_columns_one_row_per_cell(self, res) -> None:
        assert list(res.cells.columns) == categorical_cell_columns()
        assert len(res.cells.index) == np.prod(res.design["dim"])

    def test_the_association_table_is_one_row_per_measure(self, res) -> None:
        assert list(res.association.columns) == association_columns()
        assert list(res.association["measure"]) == ["cramers_v", "contingency_coefficient"]

    def test_a_setting_the_kernel_recorded_is_a_parameter_and_not_a_column(self, res) -> None:
        """`enumerated` says whether Fisher's test ran, which the table already
        says by holding a p-value or not."""
        assert "enumerated" not in res.tests["fisher_test"].columns
        assert "odds_ratio_cond" in res.tests["fisher_test"].columns

    def test_as_table_folds_the_cells_back_by_level_name(self, res) -> None:
        table = res.as_table()
        assert list(table.index) == res.design["category_lv"]["smoker"]
        assert list(table.columns) == res.design["category_lv"]["grade"]
        assert table.index.name == "smoker"
        assert int(table.to_numpy().sum()) == res.design["n_used"]


class TestMatchedDesigns:
    def test_two_conditions_are_tested_for_symmetry(self) -> None:
        res = compare_categorical_groups(MATCHED, paired=True)
        assert list(res.tests) == ["mcnemar_test"]
        assert res.design["null"] == "symmetry"
        assert res.design["pairing"] == "row"
        assert list(res.association["measure"]) == [
            "odds_ratio_paired",
            "risk_difference_paired",
            "cohens_g",
        ]

    def test_the_branch_mcnemar_took_is_recorded_as_a_parameter(self) -> None:
        """Under `exact=None` the branch depends on the data, so the result says
        which one ran rather than leaving it to be re-derived."""
        exact = compare_categorical_groups(MATCHED, paired=True)
        approximate = compare_categorical_groups(MATCHED, paired=True, exact=False)
        assert exact.parameters["exact"] is True
        assert approximate.parameters["exact"] is False
        assert "exact_used" not in exact.tests["mcnemar_test"].columns
        assert np.isnan(exact.tests["mcnemar_test"]["statistic"].iloc[0])
        assert np.isfinite(approximate.tests["mcnemar_test"]["statistic"].iloc[0])

    def test_a_symmetric_table_carries_no_standardized_residual(self) -> None:
        res = compare_categorical_groups(MATCHED, paired=True)
        assert res.cells["std_residual"].isna().all()
        assert np.isfinite(res.cells["residual"]).any()

    def test_three_conditions_are_tested_for_marginal_homogeneity(self) -> None:
        res = compare_categorical_groups(repeated(), paired=True)
        assert list(res.tests) == ["cochran_q"]
        assert res.design["null"] == "marginal_homogeneity"
        assert list(res.association["measure"]) == ["kendalls_w"]

    def test_three_conditions_are_tabulated_by_condition_and_response(self) -> None:
        """Cochran's Q has no two-variable cross-classification, so the table it
        is drawn from is one row per condition."""
        res = compare_categorical_groups(repeated(), paired=True)
        assert res.design["dim"] == [3, 2]
        assert res.design["row_var"] == "condition"
        assert res.design["col_var"] == "response"
        assert list(res.as_table().index) == ["t1", "t2", "t3"]

    def test_kendalls_w_rescales_the_statistic_by_what_it_could_have_been(self) -> None:
        res = compare_categorical_groups(repeated(), paired=True)
        row = res.tests["cochran_q"].iloc[0]
        expected = row["statistic"] / (row["n_used"] * (len(res.variables) - 1))
        assert res.association["estimate"].iloc[0] == pytest.approx(expected)


class TestReferenceLevel:
    def test_pointing_the_reference_at_the_other_level_inverts_the_direction(self) -> None:
        """The odds ratio and phi are read against the first level of each
        variable, so re-pointing one of them turns both around."""
        binary = CROSSED.loc[CROSSED["grade"] != "mid"]
        forward = compare_categorical_groups(binary).association.set_index("measure")
        reversed_ = compare_categorical_groups(
            binary, control_label={"smoker": "y"}
        ).association.set_index("measure")

        assert forward.loc["odds_ratio", "estimate"] == pytest.approx(
            1 / reversed_.loc["odds_ratio", "estimate"]
        )
        assert forward.loc["phi_coefficient", "estimate"] == pytest.approx(
            -reversed_.loc["phi_coefficient", "estimate"]
        )

    def test_a_measure_the_shape_does_not_define_is_absent_rather_than_missing(self) -> None:
        wide = compare_categorical_groups(CROSSED)
        square = compare_categorical_groups(CROSSED.loc[CROSSED["grade"] != "mid"])
        assert "odds_ratio" not in list(wide.association["measure"])
        assert "odds_ratio" in list(square.association["measure"])

    def test_cramers_v_is_unchanged_by_which_level_is_the_reference(self) -> None:
        """It measures how far the table sits from independence, which is not a
        direction."""
        plain = compare_categorical_groups(CROSSED)
        repointed = compare_categorical_groups(CROSSED, control_label={"grade": "mid"})
        assert plain.association["estimate"].iloc[0] == pytest.approx(
            repointed.association["estimate"].iloc[0]
        )


class TestDroppedRows:
    def test_a_level_left_out_and_a_missing_value_are_counted_apart(self) -> None:
        held = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y", "z", None],
                "b": ["p", "q", "p", "q", "p", "q"],
            }
        )
        res = compare_categorical_groups(held, category_lv={"a": ["x", "y"], "b": ["p", "q"]})
        assert res.design["n_dropped"] == 1
        assert res.design["n_incomplete"] == 1
        assert res.design["n_used"] == 4

    def test_the_two_counts_are_reported_to_the_caller(self, caplog) -> None:
        held = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y", "z", None],
                "b": ["p", "q", "p", "q", "p", "q"],
            }
        )
        with caplog.at_level("INFO", logger="statassist"):
            compare_categorical_groups(held, category_lv={"a": ["x", "y"], "b": ["p", "q"]})
        messages = " ".join(record.message for record in caplog.records)
        assert "outside `category_lv`" in messages
        assert "missing a value" in messages


class TestPlantedTruth:
    def test_the_simulator_arguments_unpack_without_translation(self, planted) -> None:
        sim, res = planted
        assert res.variables == list(sim.args["category_lv"])
        assert res.design["category_lv"] == {
            name: list(levels) for name, levels in sim.args["category_lv"].items()
        }

    def test_the_estimated_cramers_v_lands_near_the_planted_one(self, planted) -> None:
        sim, res = planted
        estimated = res.association.set_index("measure").loc["cramers_v", "estimate"]
        assert estimated == pytest.approx(float(sim.truth["cramers_v"].iloc[0]), abs=0.05)

    def test_every_cell_moves_the_way_it_was_planted(self, planted) -> None:
        """`lift` above 1 was planted above independence, and the observed count
        should sit on the same side of `expected`."""
        sim, res = planted
        merged = res.cells.merge(sim.truth_cell, on=["row_level", "col_level"])
        assert len(merged.index) == len(res.cells.index)
        moved = merged.loc[merged["lift"] != 1]
        assert np.all(np.sign(moved["observed"] - moved["expected"]) == np.sign(moved["lift"] - 1))


class TestRefusals:
    def test_an_independent_design_crossing_three_variables_is_refused(self) -> None:
        held = CROSSED.assign(sex=["f", "m"] * 60)
        with pytest.raises(SaValueError, match="crosses exactly two variables"):
            compare_categorical_groups(held)

    def test_a_matched_design_past_two_levels_says_which_test_is_missing(self) -> None:
        with pytest.raises(SaValueError, match="Bowker's and Stuart-Maxwell's"):
            compare_categorical_groups(CROSSED.rename(columns={"grade": "after"}), paired=True)

    def test_a_matched_design_with_no_discordance_has_nothing_to_be_about(self) -> None:
        same = pd.DataFrame({"before": ["y", "n"] * 10, "after": ["y", "n"] * 10})
        with pytest.raises(SaValueError, match="no discordance"):
            compare_categorical_groups(same, paired=True)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"conf_level": 1}, "conf_level"),
            ({"n_resamples": MIN_RESAMPLES - 1}, "n_resamples"),
            ({"correct": 1}, "correct"),
            ({"exact": 1}, "exact"),
        ],
    )
    def test_an_unusable_argument_fails_at_the_boundary(self, kwargs, message) -> None:
        with pytest.raises(SaValueError, match=message):
            compare_categorical_groups(CROSSED, **kwargs)


class TestUnreadArguments:
    def test_exact_is_not_read_by_an_independent_design(self) -> None:
        with pytest.warns(SaWarning, match="only read by McNemar's test"):
            compare_categorical_groups(CROSSED, exact=True)

    def test_simulate_p_value_is_not_read_by_a_matched_one(self) -> None:
        with pytest.warns(SaWarning, match="only read by the tests of an independent"):
            compare_categorical_groups(MATCHED, paired=True, simulate_p_value=True)


class TestSimulatedPValue:
    def test_a_simulated_p_value_is_not_referred_to_a_chi_square(self) -> None:
        res = compare_categorical_groups(
            CROSSED, simulate_p_value=True, n_resamples=MIN_RESAMPLES, seed=1
        )
        row = res.tests["chisq_test"].iloc[0]
        assert np.isnan(row["df"])
        assert row["pval"] >= 1 / (MIN_RESAMPLES + 1)
        assert res.parameters["n_resamples"] == MIN_RESAMPLES

    def test_the_same_seed_draws_the_same_tables(self) -> None:
        first, second = (
            compare_categorical_groups(
                CROSSED, simulate_p_value=True, n_resamples=MIN_RESAMPLES, seed=4
            )
            for _ in range(2)
        )
        assert (
            first.tests["chisq_test"]["pval"].iloc[0] == second.tests["chisq_test"]["pval"].iloc[0]
        )

    def test_the_number_of_resamples_is_only_recorded_when_it_was_used(self) -> None:
        assert compare_categorical_groups(CROSSED).parameters["n_resamples"] is None


class TestDiagnostics:
    def test_the_rule_reported_is_the_one_the_design_rests_on(self) -> None:
        rules = {
            "expected_count_min": compare_categorical_groups(CROSSED),
            "discordant_pair_count": compare_categorical_groups(MATCHED, paired=True),
            "sample_size_repeated": compare_categorical_groups(repeated(), paired=True),
        }
        for rule, res in rules.items():
            assert res.diagnostics["rule"] == rule

    def test_diagnose_false_leaves_the_slot_empty(self) -> None:
        assert compare_categorical_groups(CROSSED, diagnose=False).diagnostics is None

    def test_a_failed_check_is_reported_to_the_caller(self, caplog) -> None:
        with caplog.at_level("INFO", logger="statassist"):
            compare_categorical_groups(MATCHED, paired=True)
        assert any("discordant pair(s)" in record.message for record in caplog.records)
