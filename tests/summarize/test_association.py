"""``summarize/association.py`` and the two exact distributions under it."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_close, load_case

from statassist.core.errors import SaValueError
from statassist.summarize._correlation import (
    KENDALL_EXACT_MAX_N,
    RHO_SMALL_N,
    cor_test_pvalue,
    kendall_tau,
    p_kendall,
    p_rho,
    spearman_rho,
)
from statassist.summarize.association import (
    association_matrices,
    pairwise_n,
    summarize_association_stats,
)

FEATS = ["gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]

#: Absolute tolerance for a p-value that cancelled to nothing.
#:
#: R reads the upper tail of the exact Kendall distribution as
#: ``1 - pkendall(q - 1, n)``, and on the near-perfectly correlated pair the
#: cumulative sum overshoots 1 by a rounding step, so R itself reports
#: -1.33e-15. Where the truth is zero there is no relative tolerance to speak
#: of: this side sums the same counts in a different order and lands on 0.0
#: exactly. Only the cells at that scale need it, and the rest are graded at the
#: default 1e-8.
CANCELLED_ATOL = 1e-12


def assoc_matrix(frame, feats=FEATS):
    """The matrix ``export_golden.R`` handed the kernel: non-finite as missing."""
    values = frame[feats].to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


class TestPRho:
    def test_matches_r_across_the_whole_grid(self):
        frame, expected = load_case("association_p_rho")
        for index, (n, s) in enumerate(zip(frame["n"], frame["s"], strict=False)):
            for key, lower_tail in (("lower", True), ("upper", False)):
                offset = 2 if lower_tail else 0
                assert_close(
                    p_rho(round(s) + offset, int(n), lower_tail),
                    expected[key][index],
                    path=f"{key}[n={int(n)}, s={s}]",
                )

    def test_the_exact_branch_returns_multiples_of_one_over_n_factorial(self):
        # Enumeration, so every probability is a count of permutations. Above
        # RHO_SMALL_N the Edgeworth series takes over and this stops holding.
        import math

        for n in range(2, RHO_SMALL_N + 1):
            scaled = p_rho(4, n, False) * math.factorial(n)
            assert scaled == pytest.approx(round(scaled))

    def test_a_statistic_at_or_below_zero_is_certain(self):
        assert p_rho(0, 8, False) == 1.0
        assert p_rho(-5, 8, False) == 1.0
        assert p_rho(0, 8, True) == 0.0

    def test_a_statistic_past_the_maximum_is_impossible(self):
        top = (8**3 - 8) / 3
        assert p_rho(top + 2, 8, False) == 0.0
        assert p_rho(top + 2, 8, True) == 1.0

    def test_only_the_reversed_ranking_reaches_the_maximum(self):
        import math

        for n in range(3, 8):
            top = (n**3 - n) / 3
            assert p_rho(top, n, False) == pytest.approx(1 / math.factorial(n))

    def test_a_sample_of_one_has_no_distribution(self):
        with pytest.raises(ValueError, match="at least 2"):
            p_rho(2, 1, False)


class TestPKendall:
    def test_matches_r_across_the_whole_grid(self):
        frame, expected = load_case("association_p_kendall")
        for index, (n, q) in enumerate(zip(frame["n"], frame["q"], strict=False)):
            assert_close(
                p_kendall(q, int(n)),
                expected["value"][index],
                path=f"value[n={int(n)}, q={q}]",
            )

    def test_it_runs_from_zero_to_one_over_the_range_of_t(self):
        n = 7
        top = n * (n - 1) // 2
        assert p_kendall(-1, n) == 0.0
        assert p_kendall(top, n) == pytest.approx(1.0)
        assert p_kendall(top + 1, n) == 1.0

    def test_the_distribution_is_symmetric_about_its_middle(self):
        n = 8
        top = n * (n - 1) // 2
        for q in range(top + 1):
            assert p_kendall(q, n) == pytest.approx(1 - p_kendall(top - q - 1, n))

    def test_a_value_just_short_of_an_integer_still_counts_it(self):
        # R adds 1e-7 before flooring because q arrives as a double the caller
        # computed, so a value meant to be 14 can turn up as 13.999999999.
        assert p_kendall(14 - 1e-9, 8) == p_kendall(14, 8)


class TestCorTestPvalue:
    @pytest.mark.parametrize(
        ("key", "columns", "method"),
        [
            ("pearson_ab", ("a", "b"), "pearson"),
            ("spearman_ab", ("a", "b"), "spearman"),
            ("kendall_ab", ("a", "b"), "kendall"),
            ("spearman_ac", ("a", "c"), "spearman"),
            ("kendall_ac", ("a", "c"), "kendall"),
        ],
    )
    def test_matches_r_on_the_short_pair(self, key, columns, method):
        frame, expected = load_case("association_cor_test_pvalue")
        first, second = (frame[name].to_numpy(dtype=float) for name in columns)
        assert_close(cor_test_pvalue(first, second, method), expected[key], path=key)

    @pytest.mark.parametrize(
        ("key", "columns", "method"),
        [
            ("pearson_long", ("gene_1", "gene_2"), "pearson"),
            ("spearman_long", ("gene_1", "gene_2"), "spearman"),
            ("kendall_long", ("gene_1", "gene_2"), "kendall"),
            ("pearson_corr", ("gene_1", "gene_3"), "pearson"),
            ("spearman_corr", ("gene_1", "gene_3"), "spearman"),
            ("kendall_corr", ("gene_1", "gene_3"), "kendall"),
        ],
    )
    def test_matches_r_on_the_long_pair(self, key, columns, method):
        # The expectations belong to the short-pair case; the columns they were
        # computed on are the wide frame the public functions share, since n = 24
        # is what takes Spearman off the enumerated range and onto the series.
        _, expected = load_case("association_cor_test_pvalue")
        frame, _ = load_case("association_public")
        first, second = (frame[name].to_numpy(dtype=float) for name in columns)
        assert_close(
            cor_test_pvalue(first, second, method),
            expected[key],
            atol=CANCELLED_ATOL,
            path=key,
        )

    def test_a_feature_with_no_variance_has_nothing_to_correlate(self):
        frame, expected = load_case("association_cor_test_pvalue")
        flat = np.full(len(frame), 7.0)
        produced = cor_test_pvalue(frame["a"], flat, "pearson")
        assert np.isnan(produced)
        assert expected["flat"] is None

    def test_two_observations_are_too_few_for_pearson(self):
        _, expected = load_case("association_cor_test_pvalue")
        assert np.isnan(cor_test_pvalue([1.0, 2.0], [3.0, 5.0], "pearson"))
        assert expected["tiny"] is None

    def test_the_pair_is_reduced_to_its_complete_cases_first(self):
        frame, _ = load_case("association_cor_test_pvalue")
        holed = frame["b"].to_numpy(dtype=float).copy()
        holed[2] = np.nan
        direct = cor_test_pvalue(
            np.delete(frame["a"].to_numpy(dtype=float), 2), np.delete(holed, 2), "pearson"
        )
        assert cor_test_pvalue(frame["a"], holed, "pearson") == pytest.approx(direct)

    def test_an_infinity_is_dropped_like_a_blank(self):
        frame, _ = load_case("association_cor_test_pvalue")
        holed = frame["b"].to_numpy(dtype=float).copy()
        blanked = holed.copy()
        holed[2] = np.inf
        blanked[2] = np.nan
        assert cor_test_pvalue(frame["a"], holed, "pearson") == pytest.approx(
            cor_test_pvalue(frame["a"], blanked, "pearson")
        )

    def test_ties_take_spearman_off_the_exact_branch(self):
        # `c` is tied and `b` is not, and R's exact and asymptotic p-values on a
        # sample of eight differ in the second decimal, so the branch is not a
        # tolerance question.
        frame, expected = load_case("association_cor_test_pvalue")
        assert expected["spearman_ab"] != pytest.approx(expected["spearman_ac"], rel=1e-2)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="`method` must be one of"):
            cor_test_pvalue([1.0, 2.0, 3.0], [1.0, 3.0, 2.0], "polychoric")

    def test_the_exact_kendall_branch_is_the_one_below_fifty(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=KENDALL_EXACT_MAX_N - 1)
        y = rng.normal(size=KENDALL_EXACT_MAX_N - 1)
        # No ties in continuous data, so the exact permutation p-value applies
        # and it is a multiple of 2 / n!, unlike a normal tail.
        assert 0 < cor_test_pvalue(x, y, "kendall") <= 1


class TestCoefficients:
    def test_spearman_is_pearson_on_the_ranks(self):
        frame, _ = load_case("association_public")
        x = frame["gene_1"].to_numpy(dtype=float)
        y = frame["gene_3"].to_numpy(dtype=float)
        assert spearman_rho(x, y) == pytest.approx(spearman_rho(np.argsort(np.argsort(x)) + 1.0, y))

    def test_spearman_is_one_on_a_monotone_transformation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spearman_rho(x, np.exp(x)) == pytest.approx(1.0)
        assert kendall_tau(x, np.exp(x)) == pytest.approx(1.0)

    def test_a_flat_vector_has_no_coefficient(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(spearman_rho(x, np.full(4, 5.0)))
        assert np.isnan(kendall_tau(x, np.full(4, 5.0)))

    def test_kendall_is_tau_b_so_ties_shrink_the_denominator(self):
        from scipy import stats

        x = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 3.0, 2.0, 4.0, 4.0, 6.0])
        assert kendall_tau(x, y) == pytest.approx(stats.kendalltau(x, y).statistic)


class TestPairwiseN:
    def test_matches_r(self):
        frame, expected = load_case("association_matrices")
        produced = pairwise_n(assoc_matrix(frame))
        assert_close(as_rows(produced), expected["pairwise_n"], path="pairwise_n")

    def test_the_diagonal_counts_one_feature_on_its_own(self):
        frame, _ = load_case("association_matrices")
        values = assoc_matrix(frame)
        produced = pairwise_n(values)
        for index in range(values.shape[1]):
            assert produced[index, index] == int(np.isfinite(values[:, index]).sum())

    def test_it_is_symmetric(self):
        frame, _ = load_case("association_matrices")
        produced = pairwise_n(assoc_matrix(frame))
        assert np.array_equal(produced, produced.T)


class TestAssociationMatrices:
    @pytest.mark.parametrize(
        ("method", "adj_type"),
        [("pearson", "BH"), ("spearman", "holm"), ("kendall", "bonferroni")],
    )
    def test_matches_r(self, method, adj_type):
        frame, expected = load_case("association_matrices")
        produced = association_matrices(assoc_matrix(frame), method, adj_type, FEATS)
        assert list(produced) == list(expected[method])
        for name, matrix in produced.items():
            assert_close(
                as_rows(matrix),
                expected[method][name],
                atol=CANCELLED_ATOL,
                path=f"{method}[{name}]",
            )

    def test_the_diagonal_is_a_convention_not_an_estimate(self):
        frame, _ = load_case("association_matrices")
        produced = association_matrices(assoc_matrix(frame), "pearson", "BH", FEATS)
        # gene_5 has no variance, so its off-diagonal correlations are missing and
        # its correlation with itself is still 1.
        assert produced["corr"].loc["gene_5", "gene_5"] == 1.0
        assert np.isnan(produced["corr"].loc["gene_5", "gene_1"])
        assert produced["pvalue"].to_numpy().diagonal().tolist() == [np.nan] * 5 or all(
            np.isnan(produced["pvalue"].to_numpy().diagonal())
        )

    def test_every_matrix_is_symmetric(self):
        frame, _ = load_case("association_matrices")
        produced = association_matrices(assoc_matrix(frame), "spearman", "BH", FEATS)
        for matrix in produced.values():
            values = matrix.to_numpy(dtype=float)
            assert np.allclose(values, values.T, equal_nan=True)

    def test_a_refused_pair_is_left_out_of_the_adjustment_family(self):
        frame, _ = load_case("association_matrices")
        produced = association_matrices(assoc_matrix(frame), "pearson", "bonferroni", FEATS)
        pvalue = produced["pvalue"].to_numpy(dtype=float)
        adjusted = produced["adj_pvalue"].to_numpy(dtype=float)
        upper = np.triu(np.ones_like(pvalue, dtype=bool), 1)
        tested = upper & np.isfinite(pvalue)
        # Four features carry variance, so six of the ten pairs were tested and
        # the Bonferroni factor is six rather than ten.
        assert tested.sum() == 6
        assert adjusted[tested] == pytest.approx(np.minimum(1.0, 6 * pvalue[tested]))
        # And the pair that was refused stays missing rather than joining at 1.
        assert np.isnan(adjusted[upper & ~np.isfinite(pvalue)]).all()


class TestSummarizeAssociationStats:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("pairwise", {"feats": FEATS}),
            (
                "complete",
                {
                    "feats": ["gene_1", "gene_2", "gene_4"],
                    "use": "complete.obs",
                    "adj_type": "holm",
                },
            ),
            (
                "spearman_only",
                {
                    "feats": ["gene_1", "gene_3", "gene_4"],
                    "methods": "spearman",
                    "adj_type": "bonferroni",
                },
            ),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("association_public")
        produced = summarize_association_stats(frame, **kwargs)
        case = expected[key]
        assert list(produced) == list(case)
        for method in produced["design"]["methods"]:
            for name, matrix in produced[method].items():
                assert_close(
                    as_rows(matrix),
                    case[method][name],
                    atol=CANCELLED_ATOL,
                    path=f"{key}[{method}][{name}]",
                )
        design = dict(case["design"])
        # jsonlite unboxes a length-one vector, so a single method or feature
        # arrives as a bare string where the port keeps a list.
        for slot in ("feats", "methods"):
            design[slot] = as_list(design[slot])
        assert_close(produced["design"], design, path=f"{key}[design]")

    def test_only_the_methods_asked_for_get_a_key(self):
        frame, _ = load_case("association_public")
        produced = summarize_association_stats(frame, FEATS, methods=["kendall"])
        assert list(produced) == ["kendall", "design"]

    def test_the_methods_keep_the_order_they_were_given_in(self):
        frame, _ = load_case("association_public")
        produced = summarize_association_stats(frame, FEATS, methods=["kendall", "pearson"])
        assert list(produced) == ["kendall", "pearson", "design"]

    def test_no_feats_means_every_numeric_column(self):
        frame, _ = load_case("association_public")
        produced = summarize_association_stats(frame, methods=["pearson"])
        assert produced["design"]["feats"] == FEATS

    def test_a_feature_with_no_variance_is_named_in_a_note(self, caplog):
        frame, _ = load_case("association_public")
        with caplog.at_level(logging.INFO, logger="statassist"):
            summarize_association_stats(frame, FEATS, methods=["pearson"])
        assert (
            "1 feature(s) have no variance to correlate and come back as NA: gene_5." in caplog.text
        )

    def test_dropping_rows_for_complete_obs_is_reported(self, caplog):
        frame, _ = load_case("association_public")
        with caplog.at_level(logging.INFO, logger="statassist"):
            summarize_association_stats(
                frame, ["gene_1", "gene_4"], methods=["pearson"], use="complete.obs"
            )
        assert (
            'Dropped 4 row(s) with a missing value, as `use = "complete.obs"` asks.' in caplog.text
        )

    def test_complete_obs_reads_every_pair_on_one_set_of_rows(self):
        frame, _ = load_case("association_public")
        feats = ["gene_1", "gene_2", "gene_4"]
        produced = summarize_association_stats(
            frame, feats, methods=["pearson"], use="complete.obs"
        )
        counts = produced["pearson"]["n"].to_numpy()
        assert (counts == produced["design"]["n_obs"]).all()

    def test_pairwise_complete_obs_reads_each_pair_on_what_it_shares(self):
        frame, _ = load_case("association_public")
        produced = summarize_association_stats(
            frame, ["gene_1", "gene_2", "gene_4"], methods=["pearson"]
        )
        counts = produced["pearson"]["n"].to_numpy()
        assert counts.min() < counts.max()

    def test_one_feature_is_not_a_pair(self):
        frame, _ = load_case("association_public")
        with pytest.raises(SaValueError, match="at least 2 features to correlate"):
            summarize_association_stats(frame, ["gene_1"])

    def test_complete_obs_that_leaves_nothing_is_refused(self):
        import pandas as pd

        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        with pytest.raises(SaValueError, match="leaves no row"):
            summarize_association_stats(frame, ["a", "b"], use="complete.obs")

    def test_an_unknown_method_is_named(self):
        frame, _ = load_case("association_public")
        with pytest.raises(SaValueError, match="Not recognised: polychoric"):
            summarize_association_stats(frame, FEATS, methods=["pearson", "polychoric"])

    def test_a_repeated_method_is_named(self):
        frame, _ = load_case("association_public")
        with pytest.raises(SaValueError, match="duplicated names: pearson"):
            summarize_association_stats(frame, FEATS, methods=["pearson", "pearson"])

    def test_an_empty_methods_vector_is_refused(self):
        frame, _ = load_case("association_public")
        with pytest.raises(SaValueError, match="non-empty character vector"):
            summarize_association_stats(frame, FEATS, methods=[])

    def test_an_unknown_missing_policy_is_refused(self):
        frame, _ = load_case("association_public")
        with pytest.raises(SaValueError, match="`use` must be one of"):
            summarize_association_stats(frame, FEATS, use="all.obs")

    def test_an_unknown_adjustment_is_refused(self):
        frame, _ = load_case("association_public")
        with pytest.raises(SaValueError, match="adj_type"):
            summarize_association_stats(frame, FEATS, adj_type="sidak")

    def test_something_that_is_neither_a_frame_nor_a_matrix_is_refused(self):
        with pytest.raises(SaValueError, match="`data` must be a data.frame or a matrix"):
            summarize_association_stats([1.0, 2.0, 3.0])


def as_rows(values) -> list[list[float]]:
    """A square matrix as the row-major nesting ``jsonlite`` wrote for it."""
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    return [list(row) for row in np.asarray(values, dtype=float)]
