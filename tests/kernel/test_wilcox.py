"""``kernel/wilcox.py`` against the numbers ``stats::wilcox.test()`` produced.

The four paths the fixtures cover - exact or asymptotic, tied or not - are
different pieces of code in R and different pieces of code here, so each one is
graded separately rather than through one parametrised sweep.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from golden import assert_close, load_case

from statassist.core.errors import SaValueError
from statassist.kernel.wilcox import (
    psignrank,
    pwilcox,
    qsignrank,
    qwilcox,
    rank_sum,
    signed_rank,
)


def _column(frame, name):
    values = frame[name].to_numpy(dtype=float)
    return values[np.isfinite(values)]


class TestExactDistributions:
    def test_the_untied_signed_rank_support_is_a_distribution(self):
        # 2 ** n sign patterns over the ranks 1..n, exactly one of which puts
        # every rank on the negative side.
        assert psignrank(0, 6) == pytest.approx(1 / 2**6)
        assert psignrank(21, 6) == pytest.approx(1.0)
        # The upper tail is the strict one, so the two partition the support.
        assert psignrank(10, 6) + psignrank(10, 6, lower=False) == pytest.approx(1.0)

    def test_the_signed_rank_support_is_symmetric_about_its_centre(self):
        n = 8
        total = n * (n + 1) / 2
        for q in (3, 7, 12):
            assert psignrank(q, n) == pytest.approx(psignrank(total - q - 1, n, lower=False))

    def test_the_untied_rank_sum_support_is_a_distribution(self):
        assert pwilcox(0, 4, 5) == pytest.approx(1 / math.comb(9, 4))
        assert pwilcox(20, 4, 5) == pytest.approx(1.0)

    def test_the_quantiles_invert_the_distribution_functions(self):
        for p in (0.01, 0.025, 0.1, 0.5):
            q = qsignrank(p, 9)
            assert psignrank(q, 9) >= p - 1e-12
            assert psignrank(q - 1, 9) < p

            w = qwilcox(p, 6, 7)
            assert pwilcox(w, 6, 7) >= p - 1e-12
            assert pwilcox(w - 1, 6, 7) < p


class TestRankSumExact:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("conf_90", {"conf_level": 0.90}),
            ("shifted", {"mu": 0.5}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("wilcox_rank_sum_exact")
        produced = rank_sum(_column(frame, "x"), _column(frame, "y"), **kwargs)
        assert_close(produced, expected[key], path=key)

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("conf_99", {"conf_level": 0.99}),
        ],
    )
    def test_matches_r_on_a_tied_sample(self, key, kwargs):
        frame, expected = load_case("wilcox_rank_sum_exact_tied")
        produced = rank_sum(_column(frame, "x"), _column(frame, "y"), **kwargs)
        assert_close(produced, expected[key], path=key)


class TestRankSumAsymptotic:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("no_correction", {"correct": False}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("wilcox_rank_sum_asymptotic")
        produced = rank_sum(_column(frame, "x"), _column(frame, "y"), **kwargs)
        assert_close(produced, expected[key], path=key)

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
        ],
    )
    def test_matches_r_on_a_tied_sample(self, key, kwargs):
        frame, expected = load_case("wilcox_rank_sum_asymptotic_tied")
        produced = rank_sum(_column(frame, "x"), _column(frame, "y"), **kwargs)
        assert_close(produced, expected[key], path=key)


class TestSignedRankExact:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("conf_90", {"conf_level": 0.90}),
        ],
    )
    def test_matches_r_on_the_within_pair_differences(self, key, kwargs):
        frame, expected = load_case("wilcox_signed_rank_exact")
        differences = frame["x"].to_numpy(dtype=float) - frame["y"].to_numpy(dtype=float)
        assert_close(signed_rank(differences, **kwargs), expected[key], path=key)

    def test_matches_r_against_a_stated_location(self):
        frame, expected = load_case("wilcox_signed_rank_exact")
        produced = signed_rank(frame["x"], mu=5.2)
        assert_close(produced, expected["against_mu"], path="against_mu")

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("with_zeros", {"mu": 3.0}),
            ("greater", {"mu": 3.0, "alternative": "greater"}),
            ("off_centre", {"mu": 2.5}),
        ],
    )
    def test_matches_r_on_a_tied_sample(self, key, kwargs):
        frame, expected = load_case("wilcox_signed_rank_exact_tied")
        assert_close(signed_rank(frame["value"], **kwargs), expected[key], path=key)


class TestSignedRankAsymptotic:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("two_sided", {}),
            ("greater", {"alternative": "greater"}),
            ("less", {"alternative": "less"}),
            ("no_correction", {"correct": False}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("wilcox_signed_rank_asymptotic")
        assert_close(signed_rank(_column(frame, "value"), **kwargs), expected[key], path=key)

    def test_matches_r_on_a_tied_sample(self):
        frame, expected = load_case("wilcox_signed_rank_asymptotic")
        produced = signed_rank(_column(frame, "tied"), mu=3.0)
        assert_close(produced, expected["tied"], path="tied")


class TestContract:
    def test_a_one_sided_interval_leaves_the_untested_side_open(self):
        x = [4.1, 5.2, 6.3, 3.8, 7.1, 5.9, 6.6]
        y = [1.2, 2.4, 0.9, 3.1, 2.2]
        assert rank_sum(x, y, alternative="greater")["upper_conf"] == math.inf
        assert rank_sum(x, y, alternative="less")["lower_conf"] == -math.inf
        assert signed_rank(x, alternative="greater")["upper_conf"] == math.inf
        assert signed_rank(x, alternative="less")["lower_conf"] == -math.inf

    def test_swapping_the_samples_flips_the_shift(self):
        x = [4.1, 5.2, 6.3, 3.8, 7.1, 5.9, 6.6]
        y = [1.2, 2.4, 0.9, 3.1, 2.2]
        forwards = rank_sum(x, y)
        backwards = rank_sum(y, x)
        assert forwards["hl_shift"] == pytest.approx(-backwards["hl_shift"])
        assert forwards["pval"] == pytest.approx(backwards["pval"])

    def test_the_exact_path_can_be_forced_on_and_off(self):
        x = [4.1, 5.2, 6.3, 3.8, 7.1, 5.9, 6.6]
        y = [1.2, 2.4, 0.9, 3.1, 2.2]
        # Same statistic either way; the p-value comes from a different null.
        assert rank_sum(x, y, exact=True)["w_stat"] == rank_sum(x, y, exact=False)["w_stat"]
        assert rank_sum(x, y, exact=True)["pval"] != rank_sum(x, y, exact=False)["pval"]

    def test_an_unknown_alternative_is_refused_by_name(self):
        with pytest.raises(SaValueError, match="`alternative` must be one of"):
            rank_sum([1.0, 2.0], [3.0, 4.0], alternative="bigger")
        with pytest.raises(SaValueError, match="`alternative` must be one of"):
            signed_rank([1.0, 2.0], alternative="bigger")

    def test_an_empty_sample_is_refused(self):
        with pytest.raises(SaValueError, match="no observation"):
            rank_sum([], [1.0, 2.0])
        with pytest.raises(SaValueError, match="no observation"):
            signed_rank([])

    def test_a_missing_value_reaching_a_kernel_is_a_caller_error(self):
        with pytest.raises(SaValueError, match="missing or infinite"):
            rank_sum([1.0, float("nan"), 3.0], [2.0, 4.0])
        with pytest.raises(SaValueError, match="missing or infinite"):
            signed_rank([1.0, float("nan"), 3.0])
