"""``kernel/diagnostic.py`` against the numbers R produced."""

from __future__ import annotations

import numpy as np
import pytest
from golden import assert_close, load_case, samples_from_long

from statassist.core.errors import SaValueError
from statassist.kernel.diagnostic import (
    LEVENE_CENTERS,
    MIN_SCREENED,
    OUTLIER_CRITERIA,
    bartlett,
    flag_outliers,
    grubbs,
    ks_normal,
    levene,
    shapiro,
)

GROUP_LV = ["ctrl", "low", "mid", "high"]


def vector(case: str, column: str) -> np.ndarray:
    """One padded fixture column as the finite vector R was given."""
    frame, expected = load_case(case)
    return frame[column].dropna().to_numpy(dtype=float), expected


class TestShapiro:
    @pytest.mark.parametrize(
        ("key", "column", "take"),
        [
            ("all", "value", None),
            ("small", "value", 8),
            ("tiny", "value", 3),
            ("outlier", "outlier", None),
            ("tied", "tied", None),
        ],
    )
    def test_matches_r(self, key, column, take):
        values, expected = vector("diag_shapiro", column)
        if take is not None:
            values = values[:take]
        produced = shapiro(values)
        # R and SciPy both implement Royston AS R94, so the statistic is a shared
        # calculation; the p-value goes through different polynomial fits and is
        # graded a little looser. Both were observed at 1e-8 first.
        assert_close(
            {"shapiro_stat": produced["shapiro_stat"]},
            {"shapiro_stat": expected[key]["shapiro_stat"]},
            path=key,
        )
        assert_close(
            {"shapiro_pval": produced["shapiro_pval"]},
            {"shapiro_pval": expected[key]["shapiro_pval"]},
            rtol=1e-6,
            path=key,
        )

    def test_the_two_ends_of_the_accepted_size_are_named_in_the_message(self):
        with pytest.raises(SaValueError, match="between 3 and 5000 observations, got 2"):
            shapiro([1.0, 2.0])

    def test_a_sample_of_three_is_the_smallest_it_will_take(self):
        assert "shapiro_stat" in shapiro([1.0, 2.0, 4.0])

    def test_a_constant_sample_is_refused_rather_than_called_perfectly_normal(self):
        """R's engine errors here. SciPy returns a statistic of 1 and a p-value of
        1 with a warning, which reads as a confident verdict on a sample that
        cannot support one."""
        with pytest.raises(SaValueError, match="identical"):
            shapiro([4.0] * 10)


class TestKsNormal:
    @pytest.mark.parametrize(
        ("key", "column", "take", "rtol"),
        [
            ("all", "value", None, 1e-8),
            ("small", "value", 8, 1e-8),
            ("outlier", "outlier", None, 1e-8),
            # The tied sample is the one that lands on the asymptotic p-value,
            # where R sums the Kolmogorov series to its own tolerance of 1e-6 and
            # SciPy to a different one. The exact branch above matches at 1e-8, so
            # the looseness is in that series and not in the port; 1e-8 was tried
            # first and missed by 2e-7 relative.
            ("tied", "tied", None, 1e-6),
        ],
    )
    def test_matches_r(self, key, column, take, rtol):
        values, expected = vector("diag_ks_normal", column)
        if take is not None:
            values = values[:take]
        assert_close(ks_normal(values), expected[key], rtol=rtol, path=key)

    def test_the_tied_sample_is_the_one_that_leaves_the_exact_p_value(self):
        # Two fixtures, one continuous and one tie-heavy, both matched above: that
        # is what pins the branch rather than either method on its own.
        continuous, expected = vector("diag_ks_normal", "value")
        tied, _ = vector("diag_ks_normal", "tied")
        assert np.unique(continuous).size == continuous.size
        assert np.unique(tied).size < tied.size
        assert ks_normal(tied)["ks_pval"] == pytest.approx(expected["tied"]["ks_pval"])

    def test_a_sample_of_one_is_refused(self):
        with pytest.raises(SaValueError, match="at least 2 observations"):
            ks_normal([1.0])

    def test_a_constant_sample_has_no_normal_to_be_compared_with(self):
        with pytest.raises(SaValueError, match="no normal reference distribution"):
            ks_normal([2.0, 2.0, 2.0, 2.0])


class TestLevene:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("median", {}),
            ("mean", {"center": "mean"}),
            ("trimmed", {"center": "trimmed"}),
            ("trimmed_25", {"center": "trimmed", "trim": 0.25}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("diag_levene")
        samples = samples_from_long(frame, GROUP_LV)
        assert_close(levene(samples, **kwargs), expected[key], path=key)

    def test_the_default_centre_is_the_brown_forsythe_median(self):
        frame, _ = load_case("diag_levene")
        samples = samples_from_long(frame, GROUP_LV)
        assert levene(samples) == levene(samples, center="median")

    def test_it_is_the_anova_of_the_absolute_deviations(self):
        from statassist.kernel.anova import oneway_anova

        frame, _ = load_case("diag_levene")
        samples = samples_from_long(frame, GROUP_LV)
        deviations = {name: np.abs(values - np.median(values)) for name, values in samples.items()}
        assert levene(samples)["levene_stat"] == pytest.approx(oneway_anova(deviations)["f_stat"])

    def test_an_unknown_centre_is_refused_by_name(self):
        frame, _ = load_case("diag_levene")
        samples = samples_from_long(frame, GROUP_LV)
        assert set(LEVENE_CENTERS) == {"median", "mean", "trimmed"}
        with pytest.raises(SaValueError, match="`center` must be one of"):
            levene(samples, center="mode")


class TestBartlett:
    def test_matches_r(self):
        frame, expected = load_case("diag_bartlett")
        samples = samples_from_long(frame, GROUP_LV)
        assert_close(bartlett(samples), expected)

    def test_the_degrees_of_freedom_count_the_groups_not_the_observations(self):
        frame, _ = load_case("diag_bartlett")
        samples = samples_from_long(frame, GROUP_LV)
        assert bartlett(samples)["bartlett_df"] == len(samples) - 1

    def test_a_group_of_one_is_refused(self):
        with pytest.raises(SaValueError, match="at least 2 observations in each group"):
            bartlett({"a": [1.0, 2.0, 3.0], "b": [4.0]})

    def test_a_group_with_no_variance_is_named(self):
        with pytest.raises(SaValueError, match="Bartlett statistic undefined: b"):
            bartlett({"a": [1.0, 2.0, 3.0], "b": [4.0, 4.0, 4.0]})


class TestGrubbs:
    @pytest.mark.parametrize(
        ("key", "column", "take"),
        [("outlier", "value", None), ("clean", "clean", None), ("tiny", "clean", 3)],
    )
    def test_matches_r(self, key, column, take):
        values, expected = vector("diag_grubbs", column)
        if take is not None:
            values = values[:take]
        produced = dict(produced_grubbs := grubbs(values))
        # R reports the position one-based; the port reports it zero-based, so the
        # one column that cannot be compared as-is is shifted back for the check.
        produced["grubbs_index"] = produced_grubbs["grubbs_index"] + 1
        assert_close(produced, expected[key], path=key)

    def test_the_index_is_zero_based_where_r_counts_from_one(self):
        values, expected = vector("diag_grubbs", "value")
        assert grubbs(values)["grubbs_index"] == expected["outlier"]["grubbs_index"] - 1

    def test_it_points_at_whichever_tail_is_further_out(self):
        assert grubbs([1.0, 2.0, 3.0, 4.0, 40.0])["grubbs_index"] == 4
        assert grubbs([-40.0, 1.0, 2.0, 3.0, 4.0])["grubbs_index"] == 0

    def test_the_p_value_is_capped_at_one_by_the_bonferroni_factor(self):
        # Multiplying a two-sided tail by n is what makes the cap necessary; a
        # sample with nothing out of place is where it bites.
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        assert grubbs(values)["grubbs_pval"] == 1.0

    def test_a_sample_of_two_is_refused(self):
        with pytest.raises(SaValueError, match="at least 3 observations, got 2"):
            grubbs([1.0, 2.0])

    def test_a_constant_sample_has_nothing_extreme_in_it(self):
        with pytest.raises(SaValueError, match="no observation can be called extreme"):
            grubbs([5.0, 5.0, 5.0, 5.0])


class TestFlagOutliers:
    @pytest.mark.parametrize(
        ("key", "column", "kwargs"),
        [
            ("iqr", "value", {}),
            ("iqr_3", "value", {"iqr_multiplier": 3}),
            ("robust_z", "value", {"criterion": "robust_z"}),
            ("robust_z_2", "value", {"criterion": "robust_z", "z_threshold": 2}),
            ("grubbs", "value", {"criterion": "grubbs"}),
            ("grubbs_strict", "value", {"criterion": "grubbs", "alpha": 1e-6}),
            ("holed", "holed", {}),
            ("holed_z", "holed", {"criterion": "robust_z"}),
            ("holed_grubbs", "holed", {"criterion": "grubbs"}),
            ("short", "short", {}),
            ("flat", "flat", {}),
            ("flat_z", "flat", {"criterion": "robust_z"}),
            ("flat_grubbs", "flat", {"criterion": "grubbs"}),
        ],
    )
    def test_matches_r(self, key, column, kwargs):
        frame, expected = load_case("diag_flag_outliers")
        produced = flag_outliers(frame[column], **kwargs)
        assert_close(
            {
                "flag": [bool(value) for value in produced["flag"]],
                "score": list(produced["score"]),
            },
            expected[key],
            path=key,
        )

    def test_the_flag_and_the_score_are_as_long_as_the_input_was(self):
        frame, _ = load_case("diag_flag_outliers")
        column = frame["holed"]
        produced = flag_outliers(column)
        assert produced["flag"].size == len(column)
        assert produced["score"].size == len(column)

    def test_a_missing_or_infinite_value_is_never_flagged_and_never_scored(self):
        frame, _ = load_case("diag_flag_outliers")
        column = frame["holed"].to_numpy(dtype=float)
        produced = flag_outliers(column)
        unusable = ~np.isfinite(column)
        assert unusable.any()
        assert not produced["flag"][unusable].any()
        assert np.isnan(produced["score"][unusable]).all()

    def test_too_few_usable_observations_and_no_rule_runs(self):
        frame, _ = load_case("diag_flag_outliers")
        usable = np.isfinite(frame["short"].to_numpy(dtype=float)).sum()
        assert usable < MIN_SCREENED
        for criterion in OUTLIER_CRITERIA:
            produced = flag_outliers(frame["short"], criterion=criterion)
            assert not produced["flag"].any()
            assert np.isnan(produced["score"]).all()

    def test_the_iqr_score_does_not_move_when_the_fence_does(self):
        # The score is in IQR units past the nearer quartile, so widening the
        # fence changes which values are flagged and none of the scores.
        frame, _ = load_case("diag_flag_outliers")
        narrow = flag_outliers(frame["value"])
        wide = flag_outliers(frame["value"], iqr_multiplier=3)
        assert np.array_equal(narrow["score"], wide["score"], equal_nan=True)
        assert narrow["flag"].sum() >= wide["flag"].sum()

    def test_only_the_grubbs_rule_scores_a_single_observation(self):
        frame, _ = load_case("diag_flag_outliers")
        produced = flag_outliers(frame["value"], criterion="grubbs")
        assert np.isfinite(produced["score"]).sum() == 1

    def test_an_unknown_criterion_is_refused_by_name(self):
        with pytest.raises(SaValueError, match="`criterion` must be one of"):
            flag_outliers([1.0, 2.0, 3.0, 90.0], criterion="sigma")
