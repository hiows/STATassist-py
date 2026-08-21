"""``summarize/descriptive.py`` against the numbers R produced."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from golden import assert_close, assert_frame_close, load_case

from statassist.core.errors import SaValueError
from statassist.summarize.descriptive import (
    describe_columns,
    describe_vector,
    kurtosis,
    skewness,
    summarize_descriptive_stats,
)


@pytest.fixture
def described():
    """The ten vectors ``export_golden.R`` fed to ``sa_describe_vector``.

    They are spread over three cases because the fixtures share their inputs: the
    plain sample belongs to this case, the tie-heavy one is the same sample the
    rank tests use, and the three awkward ones are the screening fixtures.
    """
    frame, expected = load_case("describe_vector")
    plain = frame["value"].to_numpy(dtype=float)
    screening, _ = load_case("diag_flag_outliers")
    tied, _ = load_case("diag_shapiro")
    vectors = {
        "plain": plain,
        "outlier": screening["value"].to_numpy(dtype=float),
        "tied": tied["tied"].dropna().to_numpy(dtype=float),
        "four": plain[:4],
        "three": plain[:3],
        "single": plain[:1],
        "flat": screening["flat"].to_numpy(dtype=float),
        "holed": screening["holed"].to_numpy(dtype=float),
        "empty": np.array([], dtype=float),
        "all_missing": np.full(5, np.nan),
    }
    return vectors, expected


class TestDescribeColumns:
    def test_matches_the_column_contract_r_publishes(self, described):
        _, expected = described
        assert describe_columns() == expected["columns"]

    def test_there_are_eighteen_of_them(self, described):
        assert len(describe_columns()) == 18

    def test_the_row_and_the_fallback_cannot_disagree(self, described):
        vectors, _ = described
        assert list(describe_vector(vectors["plain"])) == describe_columns()
        assert list(describe_vector(vectors["empty"])) == describe_columns()


class TestDescribeVector:
    @pytest.mark.parametrize(
        "key",
        [
            "plain",
            "outlier",
            "tied",
            "four",
            "three",
            "single",
            "flat",
            "holed",
            "empty",
            "all_missing",
        ],
    )
    def test_matches_r(self, key, described):
        vectors, expected = described
        assert_close(describe_vector(vectors[key]), expected[key], path=key)

    def test_a_non_finite_value_is_counted_but_not_used(self, described):
        # `holed` is `outlier` with two values blanked and one set to infinity, so
        # the two rows differ by exactly those three observations - and the mean
        # stays finite, which is the point of dropping the infinity.
        vectors, _ = described
        holed = describe_vector(vectors["holed"])
        whole = describe_vector(vectors["outlier"])
        assert holed["n_missing"] == 3
        assert holed["n"] == whole["n"] - 3
        assert np.isfinite(holed["mean"])

    def test_an_empty_vector_still_reports_its_two_counts(self, described):
        vectors, _ = described
        row = describe_vector(vectors["empty"])
        assert row["n"] == 0
        assert row["n_missing"] == 0
        assert np.isnan(row["mean"])

    def test_an_all_missing_vector_counts_what_it_dropped(self, described):
        vectors, _ = described
        row = describe_vector(vectors["all_missing"])
        assert row["n"] == 0
        assert row["n_missing"] == vectors["all_missing"].size

    def test_the_fences_sit_an_iqr_and_a_half_outside_the_quartiles(self, described):
        vectors, _ = described
        row = describe_vector(vectors["plain"])
        assert row["out_lower_bound"] == pytest.approx(row["q1"] - 1.5 * row["iqr"])
        assert row["out_upper_bound"] == pytest.approx(row["q3"] + 1.5 * row["iqr"])

    def test_a_single_observation_has_no_spread_to_report(self, described):
        vectors, _ = described
        row = describe_vector(vectors["single"])
        assert row["n"] == 1
        for key in ("sd", "var", "se", "cv"):
            assert np.isnan(row[key])


class TestShapeEstimators:
    def test_matches_r(self, described):
        vectors, expected = described
        for key, values in vectors.items():
            finite = values[np.isfinite(values)]
            produced = float("nan") if finite.size == 0 else skewness(finite)
            assert_close(produced, expected["skewness"][key], path=f"skewness[{key}]")
            produced = float("nan") if finite.size == 0 else kurtosis(finite)
            assert_close(produced, expected["kurtosis"][key], path=f"kurtosis[{key}]")

    def test_they_are_the_bias_corrected_estimators_not_scipy_s_defaults(self):
        from scipy import stats

        values = np.array([1.0, 2.0, 2.0, 3.0, 9.0, 4.0, 5.0])
        # bias=False is the same G1/G2; the library default is the uncorrected
        # moment ratio, which is what this guards against.
        assert skewness(values) == pytest.approx(stats.skew(values, bias=False))
        assert kurtosis(values) == pytest.approx(stats.kurtosis(values, bias=False))
        assert skewness(values) != pytest.approx(stats.skew(values))

    def test_the_kurtosis_is_excess_so_a_normal_sample_sits_near_zero(self):
        rng = np.random.default_rng(11)
        assert abs(kurtosis(rng.normal(size=4000))) < 0.3

    def test_they_do_not_move_when_the_location_does(self):
        values = np.array([1.0, 2.0, 2.0, 3.0, 9.0, 4.0, 5.0])
        assert skewness(values + 1000) == pytest.approx(skewness(values))
        assert kurtosis(values + 1000) == pytest.approx(kurtosis(values))

    def test_too_few_observations_and_the_correction_is_undefined(self):
        assert np.isnan(skewness([1.0, 2.0]))
        assert not np.isnan(skewness([1.0, 2.0, 4.0]))
        assert np.isnan(kurtosis([1.0, 2.0, 4.0]))
        assert not np.isnan(kurtosis([1.0, 2.0, 4.0, 8.0]))

    def test_a_constant_sample_has_no_shape(self):
        assert np.isnan(skewness([3.0] * 6))
        assert np.isnan(kurtosis([3.0] * 6))


class TestSummarizeDescriptiveStats:
    @pytest.mark.parametrize(
        ("key", "feats"),
        [
            ("all", ["gene_1", "gene_2", "gene_3"]),
            ("one", ["gene_2"]),
            ("reordered", ["gene_3", "gene_1"]),
        ],
    )
    def test_matches_r_without_a_group(self, key, feats):
        frame, expected = load_case("descriptive_ungrouped")
        assert_frame_close(summarize_descriptive_stats(frame, feats), expected[key], path=key)

    @pytest.mark.parametrize(
        ("key", "feats", "group_lv"),
        [
            ("all", ["gene_1", "gene_2"], None),
            ("ordered", ["gene_2", "gene_1"], ["treat_b", "ctrl", "treat_a"]),
            ("subset", ["gene_1"], ["ctrl", "treat_a"]),
        ],
    )
    def test_matches_r_with_a_group(self, key, feats, group_lv):
        frame, expected = load_case("descriptive_grouped")
        produced = summarize_descriptive_stats(frame, feats, frame["group"], group_lv)
        assert_frame_close(produced, expected[key], path=key)

    def test_the_ungrouped_call_returns_no_group_column(self):
        frame, _ = load_case("descriptive_ungrouped")
        produced = summarize_descriptive_stats(frame, ["gene_1"])
        assert list(produced.columns) == ["features", *describe_columns()]

    def test_the_grouped_call_puts_the_group_second(self):
        frame, _ = load_case("descriptive_grouped")
        produced = summarize_descriptive_stats(frame, ["gene_1"], frame["group"])
        assert list(produced.columns) == ["features", "group", *describe_columns()]

    def test_the_feature_is_the_slow_axis_so_its_levels_stay_together(self):
        frame, _ = load_case("descriptive_grouped")
        produced = summarize_descriptive_stats(frame, ["gene_1", "gene_2"], frame["group"])
        assert produced["features"].tolist() == ["gene_1"] * 3 + ["gene_2"] * 3
        assert produced["group"].tolist()[:3] == produced["group"].tolist()[3:]

    def test_the_levels_of_an_unordered_group_come_out_sorted(self):
        frame, _ = load_case("descriptive_grouped")
        produced = summarize_descriptive_stats(frame, ["gene_1"], frame["group"])
        assert produced["group"].tolist() == ["ctrl", "treat_a", "treat_b"]

    def test_a_categorical_group_keeps_its_own_order(self):
        frame, _ = load_case("descriptive_grouped")
        categorical = pd.Categorical(frame["group"], categories=["treat_b", "treat_a", "ctrl"])
        produced = summarize_descriptive_stats(frame, ["gene_1"], categorical)
        assert produced["group"].tolist() == ["treat_b", "treat_a", "ctrl"]

    def test_a_category_no_row_uses_is_dropped_as_droplevels_would(self):
        frame, _ = load_case("descriptive_grouped")
        categorical = pd.Categorical(
            frame["group"], categories=["ctrl", "treat_a", "treat_b", "absent"]
        )
        produced = summarize_descriptive_stats(frame, ["gene_1"], categorical)
        assert "absent" not in produced["group"].tolist()

    def test_dropping_rows_is_reported_as_a_note_not_a_warning(self, caplog):
        frame, _ = load_case("descriptive_grouped")
        with caplog.at_level(logging.INFO, logger="statassist"):
            summarize_descriptive_stats(frame, ["gene_1"], frame["group"], ["ctrl"])
        assert "Dropped 16 row(s) belonging to a level outside `group_lv`." in caplog.text

    def test_a_single_level_is_enough_for_a_summary(self):
        # min_levels = 1 here, unlike every comparison, which needs two.
        frame, _ = load_case("descriptive_grouped")
        produced = summarize_descriptive_stats(frame, ["gene_1"], frame["group"], ["ctrl"])
        assert len(produced) == 1

    def test_a_feature_with_no_finite_value_gives_a_missing_row(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        produced = summarize_descriptive_stats(frame, ["a", "b"])
        assert produced.loc[1, "n"] == 0
        assert np.isnan(produced.loc[1, "mean"])

    def test_something_that_is_neither_a_frame_nor_a_matrix_is_refused(self):
        with pytest.raises(SaValueError, match="`data` must be a data.frame or a matrix"):
            summarize_descriptive_stats([1.0, 2.0, 3.0], ["a"])

    def test_a_feature_that_is_not_there_is_named(self):
        frame, _ = load_case("descriptive_ungrouped")
        with pytest.raises(SaValueError, match="not found in `data`: gene_9"):
            summarize_descriptive_stats(frame, ["gene_9"])

    def test_a_non_numeric_feature_is_named(self):
        frame, _ = load_case("descriptive_ungrouped")
        with pytest.raises(SaValueError, match="Not numeric: group"):
            summarize_descriptive_stats(frame, ["group"])

    def test_a_group_of_the_wrong_length_is_refused(self):
        frame, _ = load_case("descriptive_grouped")
        with pytest.raises(SaValueError, match="one entry per row of `data`"):
            summarize_descriptive_stats(frame, ["gene_1"], ["ctrl", "treat_a"])
