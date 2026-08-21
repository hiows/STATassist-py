"""``kernel/anova.py`` against the numbers R produced."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from golden import assert_close, load_case, samples_from_long

from statassist.core.errors import SaValueError
from statassist.kernel.anova import (
    friedman,
    kruskal,
    oneway_anova,
    rm_anova,
    sphericity,
    split_groups,
    welch_anova,
    yuen_anova,
)

#: The four levels of the shared long fixture, in the order R split them.
GROUP_LV = ["ctrl", "low", "mid", "high"]

#: The three conditions of the shared repeated-measures fixture.
CONDITIONS = ["t1", "t2", "t3"]


def long_samples(case: str = "anova_oneway", **kwargs):
    frame, expected = load_case(case)
    return samples_from_long(frame, GROUP_LV, **kwargs), expected


def rm_matrix(case: str = "anova_rm"):
    frame, expected = load_case(case)
    return frame[CONDITIONS], expected


class TestSplitGroups:
    def test_matches_r_including_which_values_were_dropped(self):
        frame, expected = load_case("anova_split_groups")
        group = pd.Categorical(frame["group"], categories=GROUP_LV)
        produced = split_groups(frame["value"], group)
        assert list(produced) == GROUP_LV
        assert_close({name: list(values) for name, values in produced.items()}, expected)

    def test_the_level_order_comes_from_the_categories_not_the_data(self):
        values = [1.0, 2.0, 3.0, 4.0]
        group = pd.Categorical(["b", "a", "b", "a"], categories=["b", "a"])
        assert list(split_groups(values, group)) == ["b", "a"]

    def test_a_plain_vector_is_read_as_r_would_read_a_factor(self):
        produced = split_groups([1.0, 2.0, 3.0, 4.0], ["b", "a", "b", "a"])
        assert list(produced) == ["a", "b"]

    def test_a_level_left_too_short_is_named(self):
        group = pd.Categorical(["a", "a", "b"], categories=["a", "b"])
        with pytest.raises(SaValueError, match=r"per group; b = 1\."):
            split_groups([1.0, 2.0, 3.0], group)

    def test_the_shortfall_is_measured_after_the_missing_values_go(self):
        group = pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"])
        with pytest.raises(SaValueError, match="b = 1"):
            split_groups([1.0, 2.0, 3.0, float("nan")], group)

    def test_a_mismatched_group_length_is_refused(self):
        with pytest.raises(SaValueError, match="one entry per value"):
            split_groups([1.0, 2.0, 3.0], ["a", "b"])


class TestOnewayAnova:
    def test_matches_r(self):
        samples, expected = long_samples()
        assert_close(oneway_anova(samples), expected)

    def test_the_omnibus_row_carries_the_interval_columns_as_missing(self):
        samples, _ = long_samples()
        produced = oneway_anova(samples)
        assert np.isnan(produced["lower_conf"])
        assert np.isnan(produced["upper_conf"])

    def test_omega_squared_is_allowed_to_go_negative(self):
        # Three groups that say nothing apart from each other: the grouping
        # explains less than chance alone would, and the estimate says so.
        samples = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.1, 2.1, 3.1, 4.1, 5.1],
            "c": [0.9, 1.9, 2.9, 3.9, 4.9],
        }
        assert oneway_anova(samples)["omega_sq"] < 0

    def test_groups_with_no_residual_degrees_of_freedom_are_refused(self):
        with pytest.raises(SaValueError, match="residual degrees of freedom"):
            oneway_anova({"a": [1.0], "b": [2.0]})

    def test_groups_with_no_variance_at_all_are_refused(self):
        with pytest.raises(SaValueError, match="every group has zero variance"):
            oneway_anova({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})

    def test_a_single_group_is_not_an_analysis_of_variance(self):
        with pytest.raises(SaValueError, match="at least 2 groups"):
            oneway_anova({"a": [1.0, 2.0, 3.0]})


class TestWelchAnova:
    def test_matches_r(self):
        samples, expected = long_samples("anova_welch")
        assert_close(welch_anova(samples), expected)

    def test_the_effect_sizes_are_the_pooled_ones(self):
        samples, _ = long_samples("anova_welch")
        assert welch_anova(samples)["eta_sq"] == oneway_anova(samples)["eta_sq"]

    def test_a_group_of_one_leaves_no_variance_to_weight_by(self):
        with pytest.raises(SaValueError, match="at least 2 observations per group"):
            welch_anova({"a": [1.0, 2.0, 3.0], "b": [4.0]})

    def test_a_group_with_no_variance_is_named(self):
        with pytest.raises(SaValueError, match="Welch weight infinite: b"):
            welch_anova({"a": [1.0, 2.0, 3.0], "b": [4.0, 4.0, 4.0]})


class TestYuenAnova:
    @pytest.mark.parametrize(("key", "tr"), [("tr_20", 0.2), ("tr_10", 0.1), ("tr_0", 0.0)])
    def test_matches_r(self, key, tr):
        samples, expected = long_samples("anova_yuen")
        assert_close(yuen_anova(samples, tr=tr), expected[key], path=key)

    def test_the_robust_effect_size_does_not_move_when_the_scale_does(self):
        samples, _ = long_samples("anova_yuen")
        scaled = {name: values * 7.0 for name, values in samples.items()}
        assert yuen_anova(scaled)["robust_eta_sq"] == pytest.approx(
            yuen_anova(samples)["robust_eta_sq"]
        )

    def test_a_group_too_small_to_survive_trimming_is_named(self):
        # floor(0.4 * 3) = 1 from each tail leaves one observation in b.
        samples = {"a": [float(i) for i in range(10)], "b": [1.0, 2.0, 3.0]}
        with pytest.raises(SaValueError, match=r"group\(s\): b\."):
            yuen_anova(samples, tr=0.4)

    def test_a_group_whose_winsorised_values_are_constant_is_named(self):
        samples = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [0.0, 1.0, 1.0, 1.0, 1.0, 9.0],
        }
        with pytest.raises(SaValueError, match="trimmed weight infinite: b"):
            yuen_anova(samples, tr=0.2)


class TestKruskal:
    def test_matches_r_on_continuous_data(self):
        samples, expected = long_samples("anova_kruskal")
        assert_close(kruskal(samples), expected["continuous"], path="continuous")

    def test_matches_r_when_the_tie_correction_does_the_work(self):
        frame, expected = load_case("anova_kruskal")
        tied = samples_from_long(frame, ["a", "b", "c"], block="tied")
        assert_close(kruskal(tied), expected["tied"], path="tied")

    def test_a_pooled_sample_with_one_distinct_value_is_refused(self):
        with pytest.raises(SaValueError, match="tie correction is undefined"):
            kruskal({"a": [2.0, 2.0, 2.0], "b": [2.0, 2.0]})


class TestSphericity:
    def test_matches_r(self):
        matrix, expected = rm_matrix("anova_sphericity")
        assert_close(sphericity(matrix), expected["full"], path="full")

    def test_a_singular_condition_covariance_falls_back_to_the_lower_bound(self):
        matrix, expected = rm_matrix("anova_sphericity")
        assert_close(sphericity(matrix.iloc[:3]), expected["singular"], path="singular")
        assert expected["singular"]["gg_eps"] == pytest.approx(1 / 2)

    def test_both_epsilons_stay_inside_their_range(self):
        matrix, _ = rm_matrix("anova_sphericity")
        produced = sphericity(matrix)
        for key in ("gg_eps", "hf_eps"):
            assert 0.5 <= produced[key] <= 1.0

    def test_the_contrast_basis_does_not_have_to_be_r_s_own(self):
        # Every quantity is a function of the eigenvalues of C' S C, and swapping
        # C for another orthonormal basis of the same subspace only conjugates
        # that matrix, so a rotated matrix must give the same answer.
        matrix, _ = rm_matrix("anova_sphericity")
        values = matrix.to_numpy(dtype=float)
        assert_close(sphericity(values + 100.0), sphericity(values))

    def test_a_matrix_with_a_hole_is_a_caller_error(self):
        values = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        with pytest.raises(SaValueError, match="complete subjects only"):
            sphericity(values)


class TestRmAnova:
    def test_matches_r(self):
        matrix, expected = rm_matrix()
        assert_close(rm_anova(matrix), expected)

    def test_it_reports_the_uncorrected_p_value_and_both_corrections(self):
        matrix, _ = rm_matrix()
        produced = rm_anova(matrix)
        for key in ("pval", "pval_gg", "pval_hf"):
            assert 0.0 <= produced[key] <= 1.0
        # A correction shrinks the degrees of freedom, so it cannot make the
        # result look stronger than the uncorrected test did.
        assert produced["pval_gg"] >= produced["pval"]
        assert produced["pval_hf"] >= produced["pval"]

    def test_fewer_than_two_subjects_is_refused(self):
        with pytest.raises(SaValueError, match="at least 2 complete subjects"):
            rm_anova(np.array([[1.0, 2.0, 3.0]]))

    def test_a_matrix_with_no_residual_at_all_is_refused(self):
        # Every subject shifted by a constant, so the subject-by-condition
        # interaction is exactly zero.
        values = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        with pytest.raises(SaValueError, match="residuals are all zero"):
            rm_anova(values)


class TestFriedman:
    def test_matches_r(self):
        matrix, expected = rm_matrix("anova_friedman")
        assert_close(friedman(matrix), expected)

    def test_kendalls_w_is_one_when_every_subject_ranks_alike(self):
        values = np.array([[1.0, 2.0, 3.0], [4.0, 9.0, 20.0], [0.0, 0.5, 0.7]])
        assert friedman(values)["kendalls_w"] == pytest.approx(1.0)

    def test_a_design_no_subject_can_rank_is_refused(self):
        values = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        with pytest.raises(SaValueError, match="no subject distinguishes"):
            friedman(values)

    def test_the_tie_correction_is_per_subject_not_pooled(self):
        # The two subjects share every value, so a pooled tie term would fire
        # while a within-subject one does not.
        values = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        assert friedman(values)["chi_sq"] == pytest.approx(4.0)
