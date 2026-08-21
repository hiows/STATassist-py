"""Grading :mod:`statassist.diagnose.outliers` against R.

The interesting part of this stage is not arithmetic - the three rules are the
kernel's and are graded there - but bookkeeping. Which row of the frame the
caller passed in does a flag point at, once the rows outside ``group_lv`` have
been dropped? The ``row`` column answers that, and its answer is what these
tests are about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_frame_close, load_case, zero_based

from statassist.core.errors import SaValueError
from statassist.diagnose import screen_outliers, split_for_screening

CASE = "screen_outliers"

FEATS = ["gene_1", "gene_2"]


@pytest.fixture(scope="module")
def screened() -> tuple[pd.DataFrame, dict]:
    return load_case(CASE)


def expected_table(expected: dict, case: str) -> dict:
    """One frozen table with its ``row`` column translated to zero-based."""
    table = dict(expected[case]["table"])
    table["row"] = zero_based(table["row"])
    return table


class TestScreenOutliers:
    def test_matches_r_without_a_grouping(self, screened):
        frame, expected = screened
        got = screen_outliers(frame, FEATS)
        assert_frame_close(got, expected_table(expected, "ungrouped"))

    def test_matches_r_within_each_level(self, screened):
        frame, expected = screened
        got = screen_outliers(frame, FEATS, frame["group"])
        assert_frame_close(got, expected_table(expected, "grouped"))

    def test_the_row_column_indexes_the_frame_that_was_passed_in(self, screened):
        """Not the filtered one.

        Two of the three levels are kept, so a position in the screened data is
        no longer a row of ``frame``. The flag has to name the row the caller can
        look up, which is the whole reason ``row_id`` exists.
        """
        frame, expected = screened
        kept = ["treat_a", "treat_b"]
        got = screen_outliers(frame, "gene_1", frame["group"], kept)
        assert_frame_close(got, expected_table(expected, "subset"))

        flagged = int(got["row"].iloc[0])
        assert frame["group"].iloc[flagged] in kept
        assert got["value"].iloc[0] == pytest.approx(frame["gene_1"].iloc[flagged])

    def test_matches_r_under_the_robust_z_rule(self, screened):
        frame, expected = screened
        got = screen_outliers(frame, "gene_1", criterion="robust_z", z_threshold=2)
        assert_frame_close(got, expected_table(expected, "robust_z"))

    def test_matches_r_under_the_grubbs_rule(self, screened):
        frame, expected = screened
        got = screen_outliers(frame, "gene_1", criterion="grubbs")
        assert_frame_close(got, expected_table(expected, "grubbs"))

    def test_nothing_flagged_is_a_table_with_no_rows(self, screened):
        """Rather than an empty object of some other shape.

        A caller reading ``out["row"]`` should not have to check first whether
        anything was found.
        """
        frame, expected = screened
        got = screen_outliers(frame, "gene_2", iqr_multiplier=10)
        assert len(got.index) == 0
        assert list(got.columns) == list(expected["wide_fence"]["table"])

    def test_the_settings_come_back_attached(self, screened):
        frame, expected = screened
        got = screen_outliers(frame, "gene_1", criterion="robust_z", z_threshold=2)
        for name, value in expected["robust_z"]["settings"].items():
            assert got.attrs[name] == value

    def test_a_missing_value_is_not_a_flag(self, screened):
        """``gene_2`` has a hole in it, and a hole is not an outlier."""
        frame, _ = screened
        assert frame["gene_2"].isna().any()
        got = screen_outliers(frame, "gene_2")
        assert got["value"].notna().all()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"criterion": "eyeball"}, "criterion"),
            ({"iqr_multiplier": -1}, "iqr_multiplier"),
            ({"z_threshold": 0}, "z_threshold"),
            ({"alpha": 0}, "alpha"),
        ],
    )
    def test_a_setting_out_of_range_is_refused_by_name(self, screened, kwargs, message):
        frame, _ = screened
        with pytest.raises(SaValueError, match=message):
            screen_outliers(frame, "gene_1", **kwargs)


class TestSplitForScreening:
    def test_an_ungrouped_split_is_one_block_over_every_row(self, screened):
        frame, _ = screened
        split = split_for_screening(frame, FEATS)
        assert not split.grouped
        assert list(split.rows) == ["all"]
        assert np.array_equal(split.rows["all"], np.arange(len(frame.index)))

    def test_a_grouped_split_is_one_block_per_level_in_display_order(self, screened):
        frame, _ = screened
        levels = ["treat_b", "ctrl", "treat_a"]
        split = split_for_screening(frame, FEATS, frame["group"], levels)
        assert split.grouped
        assert list(split.rows) == levels

    def test_dropping_a_level_leaves_row_id_pointing_at_the_original(self, screened):
        frame, _ = screened
        kept = ["ctrl", "treat_a"]
        split = split_for_screening(frame, FEATS, frame["group"], kept)
        assert len(split.data.index) == len(split.row_id)
        assert list(frame["group"].iloc[split.row_id].unique()) == sorted(kept)

    @pytest.mark.parametrize(
        ("data", "feats", "message"),
        [
            (pd.DataFrame({"a": []}), "a", "zero rows"),
            (pd.DataFrame({"a": [1.0]}), "b", "not found"),
            (pd.DataFrame({"a": ["x"]}), "a", "numeric"),
            ([1.0, 2.0], "a", "data.frame or a matrix"),
        ],
    )
    def test_an_unusable_ungrouped_input_is_refused(self, data, feats, message):
        with pytest.raises(SaValueError, match=message):
            split_for_screening(data, feats)

    def test_a_matrix_is_read_as_a_frame_when_no_grouping_is_given(self):
        """R accepts one via ``as.data.frame()``, and its column names become the
        integers pandas gives an unnamed frame."""
        values = np.arange(12.0).reshape(6, 2)
        split = split_for_screening(pd.DataFrame(values, columns=["a", "b"]), ["a", "b"])
        assert len(split.data.index) == 6


class TestScreenOutliersOnSimulatedData:
    def test_a_simulated_experiment_can_be_screened_as_it_comes(self):
        """The first end-to-end pass: Phase 1's ``args`` feed this directly."""
        from statassist import simulate_two_groups

        sim = simulate_two_groups(n_feats=6, n_case=20, n_control=20, n_up=2, n_down=2, seed=7)
        args = sim.args
        found = screen_outliers(args["data"], args["feats"], args["group"], args["group_lv"])
        assert set(found["features"]) <= set(as_list(args["feats"]))
        assert set(found["group"]) <= set(as_list(args["group_lv"]))
        assert found["row"].between(0, len(args["data"].index) - 1).all()
