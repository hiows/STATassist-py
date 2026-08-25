"""The result contract: slot presence, alignment checks, both ways of reading."""

from __future__ import annotations

import pandas as pd
import pytest

from statassist.core import (
    SIGNIFICANCE_COLUMNS,
    SaComparison,
    SaSignificance,
    metadata,
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
