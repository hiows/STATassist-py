"""The result contract: slot presence, alignment checks, both ways of reading."""

from __future__ import annotations

import pandas as pd
import pytest

from statassist.core import (
    SIGNIFICANCE_COLUMNS,
    SaCategorical,
    SaCategoricalSignificance,
    SaComparison,
    SaSignificance,
    metadata,
    new_categorical,
    new_categorical_significance,
    new_comparison,
    new_significance,
    pick_test,
    posthoc_table_columns,
)
from statassist.core.errors import SaInternalError, SaValueError

FEATS = ["gene_1", "gene_2"]


def _effect() -> pd.DataFrame:
    return pd.DataFrame({"features": FEATS, "log2fc": [1.0, -1.0], "fold_change": [2.0, 0.5]})


def _test_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "features": FEATS,
            "n_used": [10.0, 10.0],
            "pval": [0.01, 0.50],
            "pval_adj": [0.02, 0.50],
            "lower_conf": [0.1, -1.0],
            "upper_conf": [1.9, 1.0],
        }
    )


def _comparison(**overrides: object) -> SaComparison:
    kwargs: dict[str, object] = {
        "analysis": "two_group_comparison",
        "features": FEATS,
        "design": {"group_lv": ["case", "control"], "paired": False},
        "parameters": {"p_adjust": "BH"},
        "effect": _effect(),
        "tests": {"t_test": _test_table()},
        "test_info": {"t_test": {"name": "Welch t-test"}},
        "subclass": "sa_two_group",
    }
    kwargs.update(overrides)
    return new_comparison(**kwargs)  # type: ignore[arg-type]


class TestNewComparison:
    def test_an_axis_that_does_not_apply_is_absent(self) -> None:
        """A consumer asks `"posthoc" in res`, not whether it is empty."""
        res = _comparison()
        for slot in ("terms", "cells", "posthoc", "pairwise"):
            assert slot not in res

    def test_diagnostics_stays_even_when_not_requested(self) -> None:
        """Not requested is an answer about this analysis, so the slot is kept."""
        res = _comparison()
        assert "diagnostics" in res
        assert res["diagnostics"] is None

    def test_the_scenario_picks_the_class(self) -> None:
        assert type(_comparison()).__name__ == "SaTwoGroup"
        assert isinstance(_comparison(), SaComparison)

    def test_metadata_is_attached(self) -> None:
        assert set(_comparison()["metadata"]) == set(metadata())

    def test_tests_and_test_info_must_name_the_same_tests(self) -> None:
        with pytest.raises(SaInternalError, match="name different tests"):
            _comparison(test_info={"wilcoxon": {"name": "Wilcoxon"}})

    def test_an_empty_tests_mapping_is_a_contract_breach(self) -> None:
        with pytest.raises(SaInternalError, match="non-empty named mapping"):
            _comparison(tests={}, test_info={})

    def test_a_misaligned_test_table_is_a_contract_breach(self) -> None:
        reordered = _test_table().iloc[::-1].reset_index(drop=True)
        with pytest.raises(SaInternalError, match="not aligned with `features`"):
            _comparison(tests={"t_test": reordered})

    def test_a_test_table_missing_a_contract_column_is_a_contract_breach(self) -> None:
        stripped = _test_table().drop(columns=["upper_conf"])
        with pytest.raises(SaInternalError, match="contract column\\(s\\): upper_conf"):
            _comparison(tests={"t_test": stripped})

    def test_posthoc_may_skip_a_feature_but_not_invent_one(self) -> None:
        """A feature whose omnibus test was not significant is simply absent."""
        one = pd.DataFrame(
            {name: [0.0] for name in posthoc_table_columns()} | {"features": ["gene_1"]}
        )
        res = _comparison(posthoc={"t_test": one})
        assert "posthoc" in res

        stray = one.assign(features=["gene_9"])
        with pytest.raises(SaInternalError, match="absent from the comparison: gene_9"):
            _comparison(posthoc={"t_test": stray})

    def test_posthoc_cannot_name_a_test_that_was_not_run(self) -> None:
        one = pd.DataFrame(
            {name: [0.0] for name in posthoc_table_columns()} | {"features": ["gene_1"]}
        )
        with pytest.raises(SaInternalError, match="a test that was not run: wilcoxon"):
            _comparison(posthoc={"wilcoxon": one})


class TestSaResultAccess:
    def test_a_slot_reads_either_way(self) -> None:
        """`res.effect` is what a Python caller writes; `res["effect"]` is R's spelling."""
        res = _comparison()
        assert res.analysis == res["analysis"] == "two_group_comparison"
        assert res.effect is res["effect"]

    def test_an_unknown_slot_lists_the_present_ones(self) -> None:
        res = _comparison()
        with pytest.raises(AttributeError, match="Present: analysis"):
            _ = res.posthoc

    def test_it_behaves_as_a_mapping(self) -> None:
        res = _comparison()
        assert "effect" in res
        assert set(res.keys()) == set(res.to_dict())

    def test_to_dict_is_a_copy(self) -> None:
        res = _comparison()
        out = res.to_dict()
        out["analysis"] = "changed"
        assert res["analysis"] == "two_group_comparison"


def _verdict(**overrides: object) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "features": FEATS,
            "log2fc": [1.5, -0.2],
            "pvalue": [0.001, 0.40],
            "adj_pvalue": [0.002, 0.40],
            "is_signif": pd.array([True, False], dtype="boolean"),
        }
    )
    table.attrs = {
        "analysis": "two_group_comparison",
        "group_lv": ["control", "case"],
        "test": "t_test",
        "test_label": "Welch t-test",
        "adj_type": "BH",
        "log2fc_cutoff": 1.0,
        "pval_cutoff": 0.05,
    }
    for name, value in overrides.items():
        table[name] = value
    return table


def _term_verdict() -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "features": FEATS,
            "log2_effect": [0.8, -0.1],
            "pvalue": [0.001, 0.40],
            "adj_pvalue": [0.002, 0.40],
            "is_signif": pd.array([True, False], dtype="boolean"),
        }
    )
    table.attrs = {
        "analysis": "factorial_comparison",
        "group_lv": ["A.L", "A.M", "B.L", "B.M"],
        "test": "anova_test",
        "test_label": "Two-way ANOVA",
        "adj_type": "BH",
        "log2fc_cutoff": 1.0,
        "pval_cutoff": 0.05,
        "term": "dose",
        "term_order": 1,
    }
    return table


class TestNewSignificance:
    def test_the_two_slots_are_the_ones_r_has(self) -> None:
        res = new_significance("two_group_comparison", _verdict())
        assert list(res) == ["analysis_type", "significance"]
        assert isinstance(res, SaSignificance)
        assert res.analysis_type == "two_group_comparison"

    def test_a_single_table_is_kept_as_a_table(self) -> None:
        """Not wrapped in a one-element mapping: the two readings differ."""
        table = _verdict()
        res = new_significance("two_group_comparison", table)
        assert isinstance(res["significance"], pd.DataFrame)
        assert list(res["significance"].columns) == list(SIGNIFICANCE_COLUMNS)

    def test_a_mapping_is_kept_keyed_and_ordered(self) -> None:
        contrasts = {"case_vs_control": _verdict(), "case_vs_other": _verdict()}
        res = new_significance("multi_group_comparison", contrasts)
        assert list(res["significance"]) == ["case_vs_control", "case_vs_other"]

    def test_the_cutoffs_travel_with_the_table(self) -> None:
        """`draw_volcano_plot()` reads them back, so a plotted guide cannot drift."""
        res = new_significance("two_group_comparison", _verdict())
        attrs = res["significance"].attrs
        assert attrs["log2fc_cutoff"] == 1.0
        assert attrs["pval_cutoff"] == 0.05
        assert attrs["test"] == "t_test"

    def test_a_table_missing_a_contract_column_is_a_contract_breach(self) -> None:
        stripped = _verdict().drop(columns=["adj_pvalue"])
        with pytest.raises(SaInternalError, match="contract column\\(s\\): adj_pvalue"):
            new_significance("two_group_comparison", stripped)

    def test_every_table_of_a_mapping_is_checked(self) -> None:
        stripped = _verdict().drop(columns=["is_signif"])
        with pytest.raises(SaInternalError, match="contract column\\(s\\): is_signif"):
            new_significance("multi_group_comparison", {"a_vs_b": _verdict(), "a_vs_c": stripped})

    def test_an_empty_mapping_is_a_contract_breach(self) -> None:
        with pytest.raises(SaInternalError, match="at least one table"):
            new_significance("multi_group_comparison", {})

    def test_a_scenario_column_may_follow_the_contract_ones(self) -> None:
        table = _verdict(extreme_level="high")
        res = new_significance("multi_group_comparison", table)
        assert list(res["significance"].columns)[-1] == "extreme_level"

    def test_a_term_table_may_carry_log2_effect_instead_of_log2fc(self) -> None:
        table = _term_verdict()
        res = new_significance("factorial_comparison", {"wool": table})
        held = res["significance"]["wool"]
        assert list(held.columns) == [
            "features",
            "log2_effect",
            "pvalue",
            "adj_pvalue",
            "is_signif",
        ]
        assert "log2fc" not in held.columns


class TestSaSignificanceRepr:
    def test_it_reports_the_rule_and_the_count(self) -> None:
        text = repr(new_significance("two_group_comparison", _verdict()))
        assert "two_group_comparison" in text
        assert "t_test" in text
        assert "abs(log2fc) >= 1, adj_pvalue <= 0.05  (BH)" in text
        assert "1 of 2 significant" in text

    def test_an_undecided_verdict_is_counted_apart(self) -> None:
        """A feature with no defined log2fc is neither significant nor not."""
        table = _verdict()
        table.loc[1, "is_signif"] = pd.NA
        text = repr(new_significance("two_group_comparison", table))
        assert "1 of 2 significant  (1 undecided)" in text

    def test_a_family_wise_test_reports_no_adjustment(self) -> None:
        table = _verdict()
        table.attrs["adj_type"] = None
        assert "(none)" in repr(new_significance("multi_group_comparison", table))

    def test_a_mapping_names_the_axis_it_runs_along(self) -> None:
        by_contrast = {"case_vs_control": _verdict()}
        assert "one table per contrast" in repr(
            new_significance("multi_group_comparison", by_contrast)
        )

        by_term = {"dose": _term_verdict()}
        text = repr(new_significance("factorial_comparison", by_term))
        assert "one table per term" in text
        assert "abs(log2_effect) >= 1" in text


def _categorical(**overrides: object) -> SaCategorical:
    kwargs: dict[str, object] = {
        "analysis": "categorical_comparison",
        "variables": ["smoker", "grade"],
        "design": {
            "category_lv": {"smoker": ["y", "n"], "grade": ["high", "low"]},
            "null": "independence",
            "paired": False,
            "pairing": None,
            "dim": [2, 2],
            "row_var": "smoker",
            "col_var": "grade",
            "n_used": 80,
            "n_dropped": 0,
            "n_incomplete": 0,
        },
        "parameters": {"conf_level": 0.95, "correct": True},
        "cells": _cells(),
        "tests": {"chisq_test": _categorical_test()},
        "test_info": {"chisq_test": {"label": "Chi-square test of independence"}},
        "association": _association(),
    }
    kwargs.update(overrides)
    return new_categorical(**kwargs)  # type: ignore[arg-type]


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_level": ["y", "n", "y", "n"],
            "col_level": ["high", "high", "low", "low"],
            "observed": [10.0, 30.0, 30.0, 10.0],
            "expected": [20.0, 20.0, 20.0, 20.0],
            "residual": [-2.236, 2.236, 2.236, -2.236],
            "std_residual": [-3.162, 3.162, 3.162, -3.162],
            "prop_total": [0.125, 0.375, 0.375, 0.125],
            "prop_row": [0.25, 0.75, 0.75, 0.25],
            "prop_col": [0.25, 0.75, 0.75, 0.25],
        }
    )


def _categorical_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_used": [80.0],
            "statistic": [20.0],
            "df": [1.0],
            "pval": [7.7e-06],
            "lower_conf": [float("nan")],
            "upper_conf": [float("nan")],
        }
    )


def _association() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measure": ["cramers_v", "odds_ratio"],
            "estimate": [0.5, 0.111],
            "lower_conf": [float("nan"), 0.04],
            "upper_conf": [float("nan"), 0.29],
        }
    )


class TestNewCategorical:
    def test_the_slots_are_the_cell_axis_rather_than_the_feature_axis(self) -> None:
        res = _categorical()
        assert isinstance(res, SaCategorical)
        assert not isinstance(res, SaComparison)
        assert "cells" in res and "association" in res
        assert "features" not in res and "effect" not in res

    def test_diagnostics_stays_even_when_not_requested(self) -> None:
        res = _categorical()
        assert "diagnostics" in res
        assert res["diagnostics"] is None

    def test_metadata_is_attached(self) -> None:
        assert set(_categorical()["metadata"]) == set(metadata())

    def test_the_null_must_be_one_the_package_names(self) -> None:
        """A cell's `expected` means nothing without it, so an unnamed null is a
        table of numbers about nothing."""
        design = dict(_categorical()["design"]) | {"null": "no_association"}
        with pytest.raises(SaInternalError, match="must name one of independence"):
            _categorical(design=design)

    def test_the_dim_must_be_the_two_dimensions_of_the_table(self) -> None:
        for dim in ([2], [2, 3, 4], [0, 2]):
            design = dict(_categorical()["design"]) | {"dim": dim}
            with pytest.raises(SaInternalError, match="two dimensions of the table"):
                _categorical(design=design)

    def test_the_cells_must_account_for_every_cell_of_that_table(self) -> None:
        with pytest.raises(SaInternalError, match="3 row\\(s\\) for a 2 x 2 table"):
            _categorical(cells=_cells().iloc[:3])

    def test_a_cell_table_missing_a_contract_column_is_a_contract_breach(self) -> None:
        stripped = _cells().drop(columns=["std_residual"])
        with pytest.raises(SaInternalError, match="contract column\\(s\\): std_residual"):
            _categorical(cells=stripped)

    def test_a_test_holding_more_than_one_row_is_a_contract_breach(self) -> None:
        """One table is one question, so a test of it has one row and no feature
        axis to be aligned with."""
        two = pd.concat([_categorical_test()] * 2, ignore_index=True)
        with pytest.raises(SaInternalError, match="exactly one row"):
            _categorical(tests={"chisq_test": two})

    def test_a_test_table_missing_a_contract_column_is_a_contract_breach(self) -> None:
        stripped = _categorical_test().drop(columns=["n_used"])
        with pytest.raises(SaInternalError, match="contract column\\(s\\): n_used"):
            _categorical(tests={"chisq_test": stripped})

    def test_tests_and_test_info_must_name_the_same_tests(self) -> None:
        with pytest.raises(SaInternalError, match="name different tests"):
            _categorical(test_info={"fisher_test": {"label": "Fisher's exact test"}})

    def test_an_empty_tests_mapping_is_a_contract_breach(self) -> None:
        with pytest.raises(SaInternalError, match="non-empty named mapping"):
            _categorical(tests={}, test_info={})

    def test_an_association_table_missing_a_contract_column_is_a_contract_breach(self) -> None:
        stripped = _association().drop(columns=["measure"])
        with pytest.raises(SaInternalError, match="contract column\\(s\\): measure"):
            _categorical(association=stripped)


class TestSaCategoricalRepr:
    def test_it_reports_the_shape_the_null_and_the_verdict(self) -> None:
        text = repr(_categorical())
        assert "smoker (2) x grade (2)  (4 cells, independent)" in text
        assert "null     : independence" in text
        assert "observed : 80 row(s)" in text
        assert "$chisq_test  pval = " in text
        assert "null rejected at 0.05" in text

    def test_the_measures_are_printed_with_the_intervals_they_have(self) -> None:
        """And without the ones they do not: Cramer's V is reported here without
        an interval, and a blank pair of brackets would read as one."""
        text = repr(_categorical())
        assert "cramers_v   0.5\n" in text
        assert "odds_ratio  0.111  [0.04, 0.29]" in text

    def test_a_matched_design_says_what_it_was_matched_by(self) -> None:
        design = dict(_categorical()["design"]) | {
            "paired": True,
            "pairing": "before/after",
            "null": "symmetry",
        }
        text = repr(_categorical(design=design))
        assert "matched by before/after" in text
        assert "symmetry" in text

    def test_the_dropped_rows_are_reported_where_there_were_any(self) -> None:
        design = dict(_categorical()["design"]) | {"n_dropped": 3, "n_incomplete": 2}
        text = repr(_categorical(design=design))
        assert "3 row(s) outside `category_lv`" in text
        assert "2 row(s) missing a value the table needs" in text

    def test_the_cell_table_is_not_printed(self) -> None:
        """Four rows of a long table are how the contract stores a 2x2, and not
        how anyone reads one; `as_table()` is."""
        assert "prop_row" not in repr(_categorical())


class TestAsTable:
    def test_it_comes_back_as_the_table_that_was_crossed(self) -> None:
        table = _categorical().as_table()
        assert table.shape == (2, 2)
        assert list(table.index) == ["y", "n"]
        assert list(table.columns) == ["high", "low"]
        assert table.index.name == "smoker"
        assert table.columns.name == "grade"

    def test_a_count_stays_where_its_labels_put_it(self) -> None:
        """Which is the reason the cell table is long: a level's name travels
        with its count rather than with its position."""
        table = _categorical().as_table()
        assert float(table.loc["y", "high"]) == 10.0
        assert float(table.loc["n", "high"]) == 30.0


class TestNewCategoricalSignificance:
    def test_the_two_slots_are_the_ones_r_has(self) -> None:
        verdict = pd.DataFrame({"measure": ["cramers_v"], "is_signif": [True]})
        res = new_categorical_significance("categorical_comparison", verdict)
        assert list(res) == ["analysis_type", "significance"]
        assert res.analysis_type == "categorical_comparison"

    def test_it_is_not_the_numeric_verdict_object(self) -> None:
        """The two are kept apart so that a plot which needs a feature axis
        refuses a cell axis instead of drawing an empty panel."""
        verdict = pd.DataFrame({"measure": ["cramers_v"], "is_signif": [True]})
        res = new_categorical_significance("categorical_comparison", verdict)
        assert isinstance(res, SaCategoricalSignificance)
        assert not isinstance(res, SaSignificance)

    def test_a_verdict_that_is_not_a_table_is_a_contract_breach(self) -> None:
        with pytest.raises(SaInternalError, match="must be a DataFrame"):
            new_categorical_significance("categorical_comparison", {"a": 1})  # type: ignore[arg-type]


class TestPickTest:
    def test_returns_the_named_table(self) -> None:
        res = _comparison()
        assert pick_test(res, "t_test", "comparison_result") is res["tests"]["t_test"]

    def test_an_unknown_test_lists_the_available_ones(self) -> None:
        with pytest.raises(SaValueError, match="one of the tests in `comparison_result`: t_test"):
            pick_test(_comparison(), "wilcoxon", "comparison_result")

    def test_a_non_result_is_refused(self) -> None:
        with pytest.raises(SaValueError, match="must be a comparison result"):
            pick_test({"tests": {}}, "t_test", "comparison_result")
