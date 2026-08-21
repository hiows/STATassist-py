"""``kernel/posthoc.py`` against the numbers R produced.

The kernels return the nine unlabelled columns of ``sa_posthoc_columns()``; the
level names are attached later by :func:`~statassist.core.posthoc_table`, so
these tests read the pair order off :func:`~statassist.core.level_pairs` rather
than off the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from golden import assert_close, assert_frame_close, load_case, samples_from_long

from statassist.core.errors import SaValueError
from statassist.core.tables import level_pairs
from statassist.kernel.posthoc import (
    conover,
    dunn,
    games_howell,
    pair_matrix,
    pairwise_paired_t,
    pairwise_yuen,
    posthoc_columns,
    tukey,
    yuen_independent,
)

GROUP_LV = ["ctrl", "low", "mid", "high"]

TIED_LV = ["a", "b", "c"]

CONDITIONS = ["t1", "t2", "t3"]

#: Tolerance for anything that passes through the studentised range.
#:
#: R's ``ptukey``/``qtukey`` integrate the distribution by the Copenhaver-Holland
#: algorithm, :class:`scipy.stats.studentized_range` by its own quadrature. The
#: two agree to about 7e-8 relative on this fixture, which is the accuracy of the
#: quadratures rather than of the port, so the default ``1e-8`` was observed to
#: fail on ``lower_conf`` and ``pval`` before being relaxed to here. Every other
#: column of the same tables is still graded at ``1e-8``.
QTUKEY_RTOL = 1e-6


def long_samples(case: str, **kwargs):
    frame, expected = load_case(case)
    return samples_from_long(frame, GROUP_LV, **kwargs), expected


def rm_matrix(case: str):
    frame, expected = load_case(case)
    return frame[CONDITIONS], expected


def assert_studentised_range_close(actual, expected, *, path):
    """Grade the columns the studentised range touches apart from the rest.

    ``pval`` comes from its survival function and the two interval bounds from
    its quantile, so those three carry the quadrature difference and nothing else
    does.
    """
    touched = {"pval", "lower_conf", "upper_conf"}
    exact = {name: values for name, values in expected.items() if name not in touched}
    approximate = {name: values for name, values in expected.items() if name in touched}
    assert_frame_close(actual[list(exact)], exact, path=path)
    assert_frame_close(actual[list(approximate)], approximate, rtol=QTUKEY_RTOL, path=path)


class TestPosthocColumns:
    def test_matches_the_column_contract_r_publishes(self):
        _, expected = load_case("posthoc_columns")
        assert posthoc_columns() == expected["columns"]

    def test_every_pairwise_table_is_built_on_it(self):
        samples, _ = long_samples("posthoc_tukey")
        matrix, _ = rm_matrix("posthoc_pairwise_paired_t")
        for table in (
            tukey(samples),
            games_howell(samples),
            dunn(samples),
            pairwise_yuen(samples),
            pairwise_paired_t(matrix),
            conover(matrix),
        ):
            assert list(table.columns) == posthoc_columns()

    def test_the_labels_live_outside_the_kernel(self):
        # posthoc_table pairs these rows with level_pairs, so a kernel that
        # carried its own labels would give that function two sources to trust.
        assert "group1" not in posthoc_columns()
        assert "group2" not in posthoc_columns()


class TestPairMatrix:
    def test_the_rows_follow_level_pairs_with_the_later_level_first(self):
        table = pair_matrix(GROUP_LV, lambda i, j: dict.fromkeys(posthoc_columns(), 0.0))
        assert len(table) == len(level_pairs(GROUP_LV))

    def test_the_helper_receives_the_zero_based_indices_r_passes_one_based(self):
        seen: list[tuple[int, int]] = []

        def one_pair(i: int, j: int) -> dict[str, float]:
            seen.append((i, j))
            return dict.fromkeys(posthoc_columns(), float(i - j))

        table = pair_matrix(["a", "b", "c"], one_pair)
        assert seen == [(1, 0), (2, 0), (2, 1)]
        assert table["estimate"].tolist() == [1.0, 2.0, 1.0]

    def test_a_column_the_helper_forgot_is_named(self):
        wanted = posthoc_columns()

        def one_pair(i: int, j: int) -> dict[str, float]:
            row = dict.fromkeys(wanted, 0.0)
            del row["stderr"]
            return row

        with pytest.raises(SaValueError, match="missing column"):
            pair_matrix(["a", "b"], one_pair)

    def test_fewer_than_two_levels_leaves_no_pair(self):
        table = pair_matrix(["a"], lambda i, j: dict.fromkeys(posthoc_columns(), 0.0))
        assert len(table) == 0
        assert list(table.columns) == posthoc_columns()


class TestTukey:
    @pytest.mark.parametrize(("key", "conf_level"), [("conf_95", 0.95), ("conf_99", 0.99)])
    def test_matches_r(self, key, conf_level):
        samples, expected = long_samples("posthoc_tukey")
        assert_studentised_range_close(
            tukey(samples, conf_level=conf_level), expected[key], path=key
        )

    def test_the_direction_is_the_later_level_minus_the_earlier(self):
        samples = {"a": [1.0, 2.0, 3.0], "b": [11.0, 12.0, 13.0]}
        assert tukey(samples).loc[0, "estimate"] == pytest.approx(10.0)

    def test_every_pair_shares_the_pooled_denominator(self):
        samples, _ = long_samples("posthoc_tukey")
        table = tukey(samples)
        # One pooled mean square error and one residual df for the whole family:
        # that is the equal-variance assumption it inherits from the one-way ANOVA.
        assert table["df"].nunique() == 1
        assert table["df"].iloc[0] == sum(v.size for v in samples.values()) - len(samples)

    def test_a_pooled_mean_square_of_zero_is_refused(self):
        with pytest.raises(SaValueError, match="pooled mean square error is zero"):
            tukey({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})


class TestGamesHowell:
    @pytest.mark.parametrize(("key", "conf_level"), [("conf_95", 0.95), ("conf_90", 0.90)])
    def test_matches_r(self, key, conf_level):
        samples, expected = long_samples("posthoc_games_howell")
        assert_studentised_range_close(
            games_howell(samples, conf_level=conf_level), expected[key], path=key
        )

    def test_each_pair_carries_its_own_degrees_of_freedom(self):
        samples, _ = long_samples("posthoc_games_howell")
        assert games_howell(samples)["df"].nunique() > 1

    def test_it_parts_from_tukey_when_the_spreads_differ(self):
        # One narrow group and one wide one: pooling would understate the pair's
        # standard error, so the two tests cannot agree.
        samples = {
            "a": [1.0, 1.1, 0.9, 1.05, 0.95],
            "b": [1.0, 9.0, -7.0, 4.0, -2.0],
        }
        # The standard errors coincide when the two groups are the same size, so
        # the tests can only part on the degrees of freedom - which is the point.
        assert games_howell(samples).loc[0, "df"] != pytest.approx(tukey(samples).loc[0, "df"])

    def test_a_pair_with_no_spread_on_either_side_is_named(self):
        samples = {"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]}
        with pytest.raises(SaValueError, match="the pair b - a have zero variance"):
            games_howell(samples)


class TestDunn:
    @pytest.mark.parametrize(("key", "conf_level"), [("continuous", 0.95), ("continuous_99", 0.99)])
    def test_matches_r(self, key, conf_level):
        samples, expected = long_samples("posthoc_dunn", block="continuous")
        assert_frame_close(dunn(samples, conf_level=conf_level), expected[key], path=key)

    def test_matches_r_when_the_tie_correction_does_the_work(self):
        frame, expected = load_case("posthoc_dunn")
        tied = samples_from_long(frame, TIED_LV, block="tied")
        assert_frame_close(dunn(tied), expected["tied"], path="tied")

    def test_the_degrees_of_freedom_are_missing_because_the_statistic_is_normal(self):
        samples, _ = long_samples("posthoc_dunn", block="continuous")
        assert dunn(samples)["df"].isna().all()

    def test_the_ranks_are_pooled_not_taken_pair_by_pair(self):
        # Adding a level changes the pooled ranking, so a pair's statistic moves
        # even though neither of its two samples did.
        two = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        three = dict(two, c=[100.0, 200.0, 300.0])
        assert dunn(two).loc[0, "statistic"] != pytest.approx(dunn(three).loc[0, "statistic"])

    def test_a_pooled_sample_of_one_distinct_value_is_refused(self):
        with pytest.raises(SaValueError, match="every observation is tied"):
            dunn({"a": [2.0, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})


class TestYuenIndependent:
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
        frame, expected = load_case("posthoc_yuen_independent")
        x = frame["x"].dropna().to_numpy()
        y = frame["y"].dropna().to_numpy()
        assert_close(yuen_independent(x, y, **kwargs), expected[key], path=key)

    def test_the_one_sided_interval_is_left_open_at_the_far_end(self):
        frame, _ = load_case("posthoc_yuen_independent")
        x = frame["x"].dropna().to_numpy()
        y = frame["y"].dropna().to_numpy()
        assert yuen_independent(x, y, alternative="greater")["upper_conf"] == np.inf
        assert yuen_independent(x, y, alternative="less")["lower_conf"] == -np.inf

    def test_the_difference_is_signed_by_the_order_of_the_arguments(self):
        low = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        high = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        assert yuen_independent(high, low)["trim_diff"] > 0
        assert yuen_independent(low, high)["trim_diff"] < 0

    def test_a_sample_too_small_to_survive_trimming_is_refused(self):
        with pytest.raises(SaValueError, match="fewer than 2 observations survive"):
            yuen_independent([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], tr=0.4)

    def test_two_constant_samples_leave_no_standard_error(self):
        with pytest.raises(SaValueError, match="both winsorised samples are constant"):
            yuen_independent([1.0] * 6, [2.0] * 6)

    def test_an_unknown_alternative_is_refused(self):
        with pytest.raises(SaValueError, match="`alternative` must be one of"):
            yuen_independent([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], alternative="both")


class TestPairwiseYuen:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [("tr_20", {}), ("tr_10", {"tr": 0.1, "conf_level": 0.99})],
    )
    def test_matches_r(self, key, kwargs):
        samples, expected = long_samples("posthoc_pairwise_yuen")
        assert_frame_close(pairwise_yuen(samples, **kwargs), expected[key], path=key)

    def test_each_row_is_the_two_sample_test_on_that_pair(self):
        samples, _ = long_samples("posthoc_pairwise_yuen")
        arrays = list(samples.values())
        table = pairwise_yuen(samples)
        pairs = level_pairs(GROUP_LV)
        for row, i, j in zip(table.itertuples(), pairs["i"], pairs["j"], strict=True):
            direct = yuen_independent(arrays[i], arrays[j])
            assert row.estimate == pytest.approx(direct["trim_diff"])
            assert row.pval == pytest.approx(direct["pval"])

    def test_every_pair_is_tested_two_sided(self):
        samples, _ = long_samples("posthoc_pairwise_yuen")
        table = pairwise_yuen(samples)
        assert np.isfinite(table[["lower_conf", "upper_conf"]].to_numpy()).all()


class TestPairwisePairedT:
    @pytest.mark.parametrize(("key", "conf_level"), [("conf_95", 0.95), ("conf_90", 0.90)])
    def test_matches_r(self, key, conf_level):
        matrix, expected = rm_matrix("posthoc_pairwise_paired_t")
        assert_frame_close(
            pairwise_paired_t(matrix, conf_level=conf_level), expected[key], path=key
        )

    def test_the_estimate_is_the_difference_of_the_condition_means(self):
        matrix, _ = rm_matrix("posthoc_pairwise_paired_t")
        means = matrix.mean().to_numpy()
        table = pairwise_paired_t(matrix)
        pairs = level_pairs(CONDITIONS)
        for row, i, j in zip(table.itertuples(), pairs["i"], pairs["j"], strict=False):
            assert row.estimate == pytest.approx(means[i] - means[j])

    def test_both_sizes_are_the_subject_count(self):
        matrix, _ = rm_matrix("posthoc_pairwise_paired_t")
        table = pairwise_paired_t(matrix)
        assert (table["n1"] == len(matrix)).all()
        assert (table["n2"] == len(matrix)).all()

    def test_a_pair_whose_differences_never_move_is_refused(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with pytest.raises(SaValueError, match="the differences are constant"):
            pairwise_paired_t(values)


class TestConover:
    @pytest.mark.parametrize(("key", "conf_level"), [("conf_95", 0.95), ("conf_99", 0.99)])
    def test_matches_r(self, key, conf_level):
        matrix, expected = rm_matrix("posthoc_conover")
        assert_frame_close(conover(matrix, conf_level=conf_level), expected[key], path=key)

    def test_the_ranks_are_taken_within_each_subject(self):
        # Shifting one subject changes every raw value it holds but no rank it
        # produces, so the test must not notice.
        matrix, _ = rm_matrix("posthoc_conover")
        shifted = matrix.copy()
        shifted.iloc[0] = shifted.iloc[0] + 1000.0
        assert_frame_close(conover(shifted), conover(matrix).to_dict("list"))

    def test_every_pair_shares_the_residual_scale(self):
        matrix, _ = rm_matrix("posthoc_conover")
        table = conover(matrix)
        assert table["stderr"].nunique() == 1
        assert table["df"].iloc[0] == (len(matrix) - 1) * (matrix.shape[1] - 1)

    def test_a_perfectly_consistent_design_leaves_no_residual(self):
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        with pytest.raises(SaValueError, match="ranks the conditions identically"):
            conover(values)


class TestTheTypesTheTablesCarry:
    def test_every_column_is_numeric_because_the_labels_come_later(self):
        samples, _ = long_samples("posthoc_tukey")
        table = tukey(samples)
        for column in posthoc_columns():
            assert pd.api.types.is_float_dtype(table[column])
