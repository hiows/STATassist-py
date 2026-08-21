"""Table assembly: failure isolation, note aggregation, pair order."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from statassist.core import (
    feature_table,
    level_pairs,
    na_row,
    posthoc_table,
    posthoc_table_columns,
    stat_row,
)
from statassist.core.errors import SaInternalError, SaWarning

COLUMNS = ["n_used", "pval", "lower_conf", "upper_conf"]


class TestNaRowAndStatRow:
    def test_na_row_carries_every_expected_name(self) -> None:
        row = na_row(COLUMNS)
        assert list(row) == COLUMNS
        assert all(np.isnan(value) for value in row.values())

    def test_stat_row_keeps_the_keys_as_written(self) -> None:
        """An engine's own name for its statistic must not leak into the column name."""
        row = stat_row(n_used=10, pval=np.array([0.01]), statistic=np.float64(2.5))
        assert list(row) == ["n_used", "pval", "statistic"]
        assert row["pval"] == pytest.approx(0.01)

    def test_stat_row_maps_an_absent_value_to_nan(self) -> None:
        assert np.isnan(stat_row(df=None)["df"])


class TestLevelPairs:
    def test_the_reference_sits_on_the_right_of_every_contrast(self) -> None:
        """`group_lv[0]` is the reference, so it is group2 in every pair it joins."""
        pairs = level_pairs(["control", "t1", "t2"])
        assert pairs["contrast"].tolist() == ["t1 - control", "t2 - control", "t2 - t1"]
        assert pairs["group1"].tolist() == ["t1", "t2", "t2"]
        assert pairs["group2"].tolist() == ["control", "control", "t1"]

    def test_the_positions_are_zero_based(self) -> None:
        """R stores one-based positions here; these index `group_lv` in Python."""
        pairs = level_pairs(["control", "t1", "t2"])
        assert pairs["i"].tolist() == [1, 2, 2]
        assert pairs["j"].tolist() == [0, 0, 1]

    def test_two_levels_give_one_pair(self) -> None:
        assert len(level_pairs(["a", "b"]).index) == 1


class TestFeatureTable:
    def test_a_failing_feature_becomes_a_missing_row(self) -> None:
        def run(index: int) -> dict[str, float]:
            if index == 1:
                raise ValueError("not enough observations")
            return stat_row(n_used=10, pval=0.01, lower_conf=-1.0, upper_conf=1.0)

        with pytest.warns(SaWarning):
            out = feature_table(["g1", "g2", "g3"], COLUMNS, "Welch t-test", run, "BH")

        assert out["features"].tolist() == ["g1", "g2", "g3"]
        assert out.loc[1, COLUMNS].isna().all()
        assert out.loc[[0, 2], COLUMNS].notna().all().all()

    def test_a_scan_of_many_failures_warns_once(self) -> None:
        """The scanning functions rely on this: not one warning per feature."""

        def run(index: int) -> dict[str, float]:
            raise ValueError("constant within group")

        feats = [f"g{i}" for i in range(20)]
        with pytest.warns(SaWarning) as record:
            feature_table(feats, COLUMNS, "Welch t-test", run, "BH")

        assert len(record) == 1
        assert "20 of 20 feature(s)" in str(record[0].message)

    def test_engine_notes_are_aggregated_into_one_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tie warning for every feature is one note about all of them."""
        import warnings

        def run(index: int) -> dict[str, float]:
            warnings.warn("cannot compute exact p-value with ties", stacklevel=2)
            return stat_row(n_used=10, pval=0.01, lower_conf=-1.0, upper_conf=1.0)

        with caplog.at_level(logging.INFO, logger="statassist"):
            feature_table(["g1", "g2", "g3"], COLUMNS, "Wilcoxon test", run, "BH")

        assert len(caplog.records) == 1
        assert "[3 feature(s)] cannot compute exact p-value with ties" in caplog.text

    def test_the_feature_order_given_is_the_row_order(self) -> None:
        def run(index: int) -> dict[str, float]:
            return stat_row(n_used=index, pval=0.01, lower_conf=0.0, upper_conf=1.0)

        out = feature_table(["z", "a", "m"], COLUMNS, "test", run, "none")
        assert out["features"].tolist() == ["z", "a", "m"]
        assert out["n_used"].tolist() == [0.0, 1.0, 2.0]

    def test_pval_adj_follows_pval(self) -> None:
        def run(index: int) -> dict[str, float]:
            return stat_row(n_used=10, pval=0.01, lower_conf=0.0, upper_conf=1.0)

        out = feature_table(["g1", "g2"], COLUMNS, "test", run, "BH")
        assert list(out.columns) == [
            "features",
            "n_used",
            "pval",
            "pval_adj",
            "lower_conf",
            "upper_conf",
        ]

    def test_no_adjustment_means_no_column(self) -> None:
        """An effect table holds no p-value, so there is nothing to adjust."""

        def run(index: int) -> dict[str, float]:
            return stat_row(log2fc=1.0)

        out = feature_table(["g1"], ["log2fc"], "effect", run, p_adjust_method=None)
        assert list(out.columns) == ["features", "log2fc"]

    def test_a_row_missing_a_contract_column_is_a_contract_breach(self) -> None:
        def run(index: int) -> dict[str, float]:
            return stat_row(n_used=10, pval=0.01)

        with pytest.raises(SaInternalError, match="missing column\\(s\\): lower_conf"):
            feature_table(["g1"], COLUMNS, "test", run, "none")


PH_COLUMNS = [
    "n1",
    "n2",
    "estimate",
    "stderr",
    "statistic",
    "df",
    "pval",
    "lower_conf",
    "upper_conf",
]


def _pairs_frame(pvals: list[float]) -> pd.DataFrame:
    n = len(pvals)
    return pd.DataFrame(
        {
            "n1": [3.0] * n,
            "n2": [3.0] * n,
            "estimate": [1.0] * n,
            "stderr": [0.1] * n,
            "statistic": [5.0] * n,
            "df": [4.0] * n,
            "pval": pvals,
            "lower_conf": [0.0] * n,
            "upper_conf": [2.0] * n,
        }
    )


class TestPosthocTable:
    def test_no_features_still_returns_the_full_contract(self) -> None:
        """A scenario where nothing qualified must not need a special case downstream."""
        out = posthoc_table([], ["a", "b", "c"], PH_COLUMNS, "Tukey", _pairs_frame)
        assert list(out.columns) == posthoc_table_columns()
        assert len(out.index) == 0

    def test_columns_short_of_the_contract_is_a_contract_breach(self) -> None:
        with pytest.raises(SaInternalError, match="does not cover the post-hoc contract: df"):
            posthoc_table(
                ["g1"],
                ["a", "b"],
                [name for name in PH_COLUMNS if name != "df"],
                "Tukey",
                _pairs_frame,
            )

    def test_the_adjustment_is_within_a_feature(self) -> None:
        """The pairwise family is the contrasts of one feature, not of the whole scan."""
        out = posthoc_table(
            ["g1", "g2"],
            ["a", "b", "c"],
            PH_COLUMNS,
            "Tukey",
            lambda feature: _pairs_frame([0.01, 0.20, 0.30]),
        )
        # Holm across the 3 pairs of one feature, not across all 6 rows.
        assert out.loc[out["features"] == "g1", "pval_adj"].tolist() == pytest.approx(
            [0.03, 0.40, 0.40], rel=1e-12
        )

    def test_rows_follow_the_pair_order(self) -> None:
        out = posthoc_table(
            ["g1"], ["a", "b", "c"], PH_COLUMNS, "Tukey", lambda f: _pairs_frame([0.1] * 3)
        )
        assert out["contrast"].tolist() == ["b - a", "c - a", "c - b"]

    def test_a_failing_feature_becomes_missing_rows_and_warns_once(self) -> None:
        def run(feature: str) -> pd.DataFrame:
            if feature == "g2":
                raise ValueError("all values are identical")
            return _pairs_frame([0.1, 0.2, 0.3])

        with pytest.warns(SaWarning) as record:
            out = posthoc_table(["g1", "g2"], ["a", "b", "c"], PH_COLUMNS, "Tukey", run)

        assert len(record) == 1
        assert out.loc[out["features"] == "g2", "estimate"].isna().all()
        assert out.loc[out["features"] == "g1", "estimate"].notna().all()

    def test_a_wrong_row_count_is_a_contract_breach(self) -> None:
        with pytest.raises(SaInternalError, match="returned 2 row\\(s\\)"):
            posthoc_table(
                ["g1"], ["a", "b", "c"], PH_COLUMNS, "Tukey", lambda f: _pairs_frame([0.1, 0.2])
            )
