"""The result contract: slot presence, alignment checks, both ways of reading."""

from __future__ import annotations

import pandas as pd
import pytest

from statassist.core import (
    SaComparison,
    metadata,
    new_comparison,
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
