"""``estimate_categorical_significance`` on both of its readings.

The two axes are the point of this function, so most of what is checked here is
that they stay apart: a ratio that does not move with the sample size, a p-value
that does, and a verdict that is undecided rather than decided against wherever
one of them is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from crafted import crafted_categorical

from statassist import (
    compare_categorical_groups,
    compare_two_groups,
    estimate_categorical_significance,
    simulate_categorical_groups,
    simulate_two_groups,
)
from statassist.core.errors import SaValueError, SaWarning
from statassist.core.padjust import p_adjust
from statassist.core.result import SaCategorical, SaCategoricalSignificance

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

MATCHED = pd.DataFrame(
    {
        "before": ["pass"] * 20 + ["fail"] * 30,
        "after": ["pass"] * 18 + ["fail"] * 2 + ["pass"] * 14 + ["fail"] * 16,
    }
)

#: The columns a cell reading reports, in order.
CELL_COLUMNS = [
    "row_level",
    "col_level",
    "observed",
    "expected",
    "lift",
    "log2_lift",
    "std_residual",
    "pvalue",
    "adj_pvalue",
    "is_signif",
]


def repeated(n_subjects: int = 40) -> pd.DataFrame:
    """Three binary conditions on the same subjects, with a rising response rate."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            f"t{index + 1}": np.where(rng.random(n_subjects) < rate, "y", "n")
            for index, rate in enumerate((0.3, 0.5, 0.75))
        }
    )


@pytest.fixture(scope="module")
def res() -> SaCategorical:
    return compare_categorical_groups(CROSSED)


class TestCellReading:
    def test_the_columns_are_the_cell_axis_and_not_the_feature_axis(self, res) -> None:
        verdict = estimate_categorical_significance(res)
        assert isinstance(verdict, SaCategoricalSignificance)
        assert list(verdict.significance.columns) == CELL_COLUMNS
        assert "features" not in verdict.significance.columns

    def test_the_verdict_keys_on_the_same_pair_the_cell_table_does(self, res) -> None:
        verdict = estimate_categorical_significance(res)
        merged = verdict.significance.merge(res.cells, on=["row_level", "col_level"])
        assert len(merged.index) == len(res.cells.index)

    def test_lift_is_the_ratio_the_simulator_plants(self, res) -> None:
        verdict = estimate_categorical_significance(res).significance
        assert np.allclose(
            verdict["lift"], res.cells["observed"] / res.cells["expected"], equal_nan=True
        )
        assert np.allclose(verdict["log2_lift"], np.log2(verdict["lift"]))

    def test_the_p_value_is_the_cell_own_two_sided_normal_tail(self, res) -> None:
        from scipy import stats

        verdict = estimate_categorical_significance(res).significance
        assert np.allclose(
            verdict["pvalue"], 2 * stats.norm.cdf(-np.abs(res.cells["std_residual"]))
        )

    def test_the_adjustment_is_the_first_one_and_across_the_cells(self, res) -> None:
        """A categorical result carries no adjusted column, so this is where the
        family is chosen, and the family is the cells of the one table."""
        verdict = estimate_categorical_significance(res, adj_type="holm").significance
        assert np.allclose(verdict["adj_pvalue"], p_adjust(verdict["pvalue"], "holm"))
        assert verdict.attrs["adj_type"] == "holm"

    def test_naming_no_adjustment_tests_the_raw_p_values(self, res) -> None:
        verdict = estimate_categorical_significance(res, adj_type="none").significance
        assert np.allclose(verdict["adj_pvalue"], verdict["pvalue"])

    def test_a_stricter_cutoff_can_only_take_cells_away(self, res) -> None:
        loose = estimate_categorical_significance(res, log2_lift_cutoff=0).significance
        strict = estimate_categorical_significance(res, log2_lift_cutoff=2).significance
        assert int((strict["is_signif"] == True).sum()) <= int(  # noqa: E712
            (loose["is_signif"] == True).sum()  # noqa: E712
        )

    def test_the_rule_travels_with_the_table(self, res) -> None:
        verdict = estimate_categorical_significance(res, log2_lift_cutoff=0.5).significance
        assert verdict.attrs == {
            "analysis": "categorical_comparison",
            "null": "independence",
            "by": "cell",
            "table_dim": [2, 3],
            "pval_cutoff": 0.05,
            "log2_lift_cutoff": 0.5,
            "adj_type": "BH",
        }

    def test_three_or_more_matched_conditions_have_a_cell_reading(self) -> None:
        """Their null is marginal homogeneity, which is a statement about the
        margins, so the standardized residual exists there."""
        verdict = estimate_categorical_significance(
            compare_categorical_groups(repeated(), paired=True)
        ).significance
        assert np.isfinite(verdict["std_residual"]).all()


class TestUndecided:
    def test_an_empty_cell_is_an_infinite_shortfall_and_not_a_missing_one(self) -> None:
        """`lift` of exactly zero is a finding: nothing landed where something
        was expected, which clears any magnitude cutoff."""
        split = pd.DataFrame({"a": ["x"] * 10 + ["y"] * 10, "b": ["p"] * 10 + ["q"] * 10})
        verdict = estimate_categorical_significance(compare_categorical_groups(split)).significance
        empty = verdict.loc[verdict["observed"] == 0]
        assert len(empty.index) == 2
        assert (empty["lift"] == 0).all()
        assert np.isneginf(empty["log2_lift"]).all()
        assert (empty["is_signif"] == True).all()  # noqa: E712

    def test_a_cell_with_nothing_expected_has_no_ratio_at_all(self) -> None:
        """Which is a different fact from a lift of zero, and reported
        differently: dividing by an empty expectation is not a shortfall."""
        verdict = estimate_categorical_significance(crafted_categorical()).significance
        nothing_expected = verdict.loc[verdict["expected"] == 0]
        assert len(nothing_expected.index) == 1
        assert nothing_expected["lift"].isna().all()
        assert nothing_expected["log2_lift"].isna().all()

    def test_a_missing_residual_leaves_the_cell_undecided_rather_than_negative(self) -> None:
        verdict = estimate_categorical_significance(crafted_categorical()).significance
        undecided = verdict.loc[verdict["std_residual"].isna()]
        assert len(undecided.index) == 1
        assert undecided["is_signif"].isna().all()


class TestTableReading:
    @pytest.mark.parametrize(
        ("built", "measure"),
        [
            (lambda: compare_categorical_groups(CROSSED), "cramers_v"),
            (
                lambda: compare_categorical_groups(CROSSED.loc[CROSSED["grade"] != "mid"]),
                "odds_ratio",
            ),
            (lambda: compare_categorical_groups(MATCHED, paired=True), "odds_ratio_paired"),
            (lambda: compare_categorical_groups(repeated(), paired=True), "kendalls_w"),
        ],
    )
    def test_auto_takes_the_measure_the_design_defines(self, built, measure) -> None:
        verdict = estimate_categorical_significance(built(), by="table").significance
        assert str(verdict["measure"].iloc[0]) == measure

    def test_the_verdict_is_one_row_of_the_measure_and_its_test(self, res) -> None:
        verdict = estimate_categorical_significance(res, by="table").significance
        assert list(verdict.columns) == [
            "measure",
            "estimate",
            "lower_conf",
            "upper_conf",
            "pvalue",
            "is_signif",
        ]
        assert len(verdict.index) == 1
        assert verdict["pvalue"].iloc[0] == res.tests["chisq_test"]["pval"].iloc[0]

    def test_naming_a_test_reads_that_one_instead(self, res) -> None:
        verdict = estimate_categorical_significance(
            res, by="table", test="fisher_test"
        ).significance
        assert verdict["pvalue"].iloc[0] == res.tests["fisher_test"]["pval"].iloc[0]
        assert verdict.attrs["test"] == "fisher_test"
        assert verdict.attrs["test_label"] == res.test_info["fisher_test"]["label"]

    def test_without_a_cutoff_the_verdict_is_the_p_value_alone(self, res) -> None:
        verdict = estimate_categorical_significance(res, by="table").significance
        assert verdict.attrs["effect_cutoff"] is None
        assert bool(verdict["is_signif"].iloc[0])

    def test_a_cutoff_the_estimate_does_not_reach_takes_the_verdict_away(self, res) -> None:
        verdict = estimate_categorical_significance(res, by="table", effect_cutoff=0.9).significance
        assert verdict["estimate"].iloc[0] < 0.9
        assert not bool(verdict["is_signif"].iloc[0])

    def test_a_ratio_cutoff_is_read_as_a_fold_either_way(self) -> None:
        """The paired odds ratio here is above 1, and the same cutoff has to
        catch a table that moved the other way just as well."""
        matched = compare_categorical_groups(MATCHED, paired=True)
        reversed_ = compare_categorical_groups(
            MATCHED.rename(columns={"before": "after", "after": "before"})[["before", "after"]],
            paired=True,
        )
        for res in (matched, reversed_):
            verdict = estimate_categorical_significance(
                res, by="table", effect_cutoff=2
            ).significance
            assert bool(verdict["is_signif"].iloc[0])

    def test_the_rule_travels_with_the_table(self, res) -> None:
        verdict = estimate_categorical_significance(res, by="table", effect_cutoff=0.2).significance
        assert verdict.attrs["by"] == "table"
        assert verdict.attrs["measure"] == "cramers_v"
        assert verdict.attrs["effect_cutoff"] == 0.2
        assert "adj_type" not in verdict.attrs


class TestRefusals:
    def test_a_matched_pair_has_no_cell_reading(self) -> None:
        with pytest.raises(SaValueError, match="tested for symmetry"):
            estimate_categorical_significance(compare_categorical_groups(MATCHED, paired=True))

    def test_a_matched_pair_still_has_a_table_reading(self) -> None:
        verdict = estimate_categorical_significance(
            compare_categorical_groups(MATCHED, paired=True), by="table"
        ).significance
        assert str(verdict["measure"].iloc[0]) == "odds_ratio_paired"

    def test_a_numeric_comparison_is_pointed_at_the_other_function(self) -> None:
        sim = simulate_two_groups(n_feats=4, n_case=8, n_control=8, n_up=1, n_down=1, seed=2)
        comparison = compare_two_groups(**sim.args, diagnose=False)
        with pytest.raises(SaValueError, match="estimate_significance"):
            estimate_categorical_significance(comparison)

    def test_a_bare_frame_is_not_a_result(self) -> None:
        with pytest.raises(SaValueError, match="compare_categorical_groups"):
            estimate_categorical_significance(CROSSED)

    def test_an_unknown_reading_names_the_two_there_are(self, res) -> None:
        with pytest.raises(SaValueError, match="cell, table"):
            estimate_categorical_significance(res, by="feature")

    def test_a_measure_this_design_has_no_value_for_is_refused(self, res) -> None:
        with pytest.raises(SaValueError, match="odds_ratio"):
            estimate_categorical_significance(res, by="table", measure="odds_ratio")

    def test_a_ratio_cutoff_below_one_would_admit_everything(self, res) -> None:
        square = compare_categorical_groups(CROSSED.loc[CROSSED["grade"] != "mid"])
        with pytest.raises(SaValueError, match="at least 1"):
            estimate_categorical_significance(square, by="table", effect_cutoff=0.5)

    def test_a_test_this_result_did_not_run_is_refused(self, res) -> None:
        with pytest.raises(SaValueError, match="mcnemar_test"):
            estimate_categorical_significance(res, by="table", test="mcnemar_test")


class TestUnreadArguments:
    def test_a_cell_reading_says_so_about_the_table_arguments(self, res) -> None:
        with pytest.warns(SaWarning, match="`test` is not read"):
            estimate_categorical_significance(res, test="fisher_test")
        with pytest.warns(SaWarning, match="`measure` and `effect_cutoff` are not read"):
            estimate_categorical_significance(res, measure="cramers_v", effect_cutoff=0.2)

    def test_a_table_reading_says_so_about_the_cell_arguments(self, res) -> None:
        with pytest.warns(SaWarning, match="`log2_lift_cutoff` and `adj_type` are not read"):
            estimate_categorical_significance(res, by="table", log2_lift_cutoff=2, adj_type="holm")

    def test_a_default_left_alone_is_not_a_supplied_argument(self, res) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", SaWarning)
            estimate_categorical_significance(res)
            estimate_categorical_significance(res, by="table")


class TestPlantedTruth:
    def test_the_estimated_lift_agrees_with_the_planted_one(self) -> None:
        sim = simulate_categorical_groups(n_samples=6000, assoc=0.5, seed=13)
        verdict = estimate_categorical_significance(
            compare_categorical_groups(**sim.args)
        ).significance
        merged = verdict.merge(sim.truth_cell, on=["row_level", "col_level"], suffixes=("", "_t"))
        assert len(merged.index) == len(verdict.index)
        assert np.allclose(merged["lift"], merged["lift_t"], atol=0.1)
