"""``kernel/robust.py`` against the numbers R produced."""

from __future__ import annotations

import math

import numpy as np
import pytest
from golden import assert_close, load_case

from statassist.core.errors import SaValueError
from statassist.kernel.robust import (
    brunner_munzel,
    t_ci,
    t_pval,
    trimmed_mean,
    winsorize,
    winsorized_normal_var,
    yuen_paired,
)


def _column(frame, name):
    values = frame[name].to_numpy(dtype=float)
    return values[np.isfinite(values)]


class TestTPval:
    def test_matches_r_on_every_tail(self):
        params, expected = load_case("robust_t_pval")
        produced = [t_pval(row.stat, row.df, row.alternative) for row in params.itertuples()]
        assert_close(produced, expected["pval"])

    def test_an_unknown_alternative_is_refused_by_name(self):
        with pytest.raises(SaValueError, match="`alternative` must be one of"):
            t_pval(1.0, 5, "bigger")


class TestTCi:
    def test_matches_r_including_the_open_ends(self):
        params, expected = load_case("robust_t_ci")
        lower = []
        upper = []
        for row in params.itertuples():
            low, high = t_ci(
                row.est,
                row.se,
                row.df,
                row.alternative,
                row.conf_level,
                bounds=(row.lower_bound, row.upper_bound),
            )
            lower.append(low)
            upper.append(high)
        assert_close(lower, expected["lower_conf"])
        assert_close(upper, expected["upper_conf"])

    def test_a_one_sided_interval_leaves_the_untested_side_at_the_bound(self):
        assert t_ci(1.0, 0.1, 10, "greater", 0.95)[1] == math.inf
        assert t_ci(1.0, 0.1, 10, "less", 0.95)[0] == -math.inf
        assert t_ci(0.5, 0.1, 10, "greater", 0.95, bounds=(0.0, 1.0))[1] == 1.0

    def test_an_unknown_alternative_is_refused_by_name(self):
        with pytest.raises(SaValueError, match="`alternative` must be one of"):
            t_ci(1.0, 0.1, 5, "bigger", 0.95)


class TestWinsorize:
    def test_matches_r_at_every_trimming_proportion(self):
        frame, expected = load_case("robust_winsorize")
        values = frame["value"].to_numpy(dtype=float)
        for key, tr in (("tr_0", 0.0), ("tr_10", 0.1), ("tr_20", 0.2), ("tr_45", 0.45)):
            assert_close(list(winsorize(values, tr)), expected[key], path=key)

    def test_the_length_and_the_order_survive(self):
        values = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        out = winsorize(values, 0.2)
        assert out.size == values.size
        # Pulling a tail in can tie two values together but never reorders them,
        # which is what lets the result be used for a covariance.
        assert (np.diff(out[np.argsort(values)]) >= 0).all()

    def test_the_input_is_not_modified(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        winsorize(values, 0.2)
        assert values[-1] == 100.0


class TestTrimmedMean:
    def test_it_drops_a_count_from_each_tail_rather_than_a_quantile(self):
        # floor(0.2 * 5) = 1 from each end of the sorted vector, so the 100 and
        # the 1 both go and the mean is of 2, 3, 4.
        assert trimmed_mean([1.0, 2.0, 3.0, 4.0, 100.0], 0.2) == 3.0

    def test_no_trimming_is_the_plain_mean(self):
        values = [1.0, 2.0, 4.0]
        assert trimmed_mean(values, 0.0) == pytest.approx(float(np.mean(values)))


class TestWinsorizedNormalVar:
    def test_matches_r(self):
        frame, expected = load_case("robust_winsorized_normal_var")
        produced = [winsorized_normal_var(tr) for tr in frame["tr"]]
        assert_close(produced, expected["value"])

    def test_no_trimming_leaves_the_variance_alone(self):
        assert winsorized_normal_var(0.0) == 1.0
        assert winsorized_normal_var(-0.1) == 1.0


class TestBrunnerMunzel:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("conf_90", {"conf_level": 0.90}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("robust_brunner_munzel")
        produced = brunner_munzel(_column(frame, "x"), _column(frame, "y"), **kwargs)
        assert_close(produced, expected[key], path=key)

    def test_matches_r_on_a_tied_sample(self):
        frame, expected = load_case("anova_kruskal")
        tied = frame[frame["block"] == "tied"]
        a = tied.loc[tied["group"] == "a", "value"].to_numpy(dtype=float)
        c = tied.loc[tied["group"] == "c", "value"].to_numpy(dtype=float)
        _, bm_expected = load_case("robust_brunner_munzel")
        assert_close(brunner_munzel(a, c), bm_expected["tied"], path="tied")

    def test_the_relative_effect_points_at_the_first_sample(self):
        # Overlapping, since two samples that share no value leave the variance
        # estimate at zero and the test refuses them outright.
        low = [1.0, 2.0, 3.0, 4.0, 5.0]
        high = [3.0, 4.0, 5.0, 6.0, 7.0]
        assert brunner_munzel(high, low)["relative_effect"] > 0.5
        assert brunner_munzel(low, high)["relative_effect"] < 0.5
        assert brunner_munzel(high, low)["bm_stat"] > 0

    def test_the_interval_stays_inside_the_unit_interval_when_one_sided(self):
        low = [1.0, 2.0, 3.0, 4.0, 5.0]
        high = [3.0, 4.0, 5.0, 6.0, 7.0]
        assert brunner_munzel(high, low, "greater")["upper_conf"] == 1.0
        assert brunner_munzel(high, low, "less")["lower_conf"] == 0.0

    def test_samples_that_do_not_overlap_at_all_are_refused(self):
        with pytest.raises(SaValueError, match="do not overlap"):
            brunner_munzel([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

    def test_a_missing_value_reaching_a_kernel_is_a_caller_error(self):
        with pytest.raises(SaValueError, match="missing or infinite"):
            brunner_munzel([1.0, float("nan"), 3.0], [2.0, 4.0, 6.0])


class TestYuenPaired:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("tr_20", {}),
            ("tr_10", {"tr": 0.1}),
            ("tr_0", {"tr": 0.0}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("conf_90", {"conf_level": 0.90}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("robust_yuen_paired")
        produced = yuen_paired(frame["x"], frame["y"], **kwargs)
        assert_close(produced, expected[key], path=key)

    def test_the_robust_effect_size_is_signed(self):
        frame, _ = load_case("robust_yuen_paired")
        forwards = yuen_paired(frame["x"], frame["y"])
        backwards = yuen_paired(frame["y"], frame["x"])
        assert forwards["robust_dz"] == pytest.approx(-backwards["robust_dz"])

    def test_unequal_lengths_are_refused(self):
        with pytest.raises(SaValueError, match="complete pairs of the same length"):
            yuen_paired([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_constant_winsorised_differences_are_refused(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        shifted = [value + 1 for value in values]
        with pytest.raises(SaValueError, match="zero variance"):
            yuen_paired(values, shifted)
