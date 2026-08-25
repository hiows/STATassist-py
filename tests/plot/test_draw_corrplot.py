"""What the corrplot reordered, blanked and handed on to the heatmap.

The three decisions this function owns are the three things checked: the matrix
goes out unscaled, one clustering is applied to both axes, and the blanking
happens after the clustering rather than before it.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
import pytest

from statassist import draw_corrplot, summarize_association_stats
from statassist.core.errors import SaValueError
from statassist.plot.corrplot import CORR_LIMITS

#: How many features the fixture holds, and how many observations of each.
N_FEATS, N_OBS = 7, 40


@functools.lru_cache(maxsize=1)
def _blocked():
    """Blocks of features that move together, interleaved so an order is needed.

    Finding the blocks is the only reason to reorder the axes at all, and having
    them interleaved on the way in means a clustering that reordered nothing could
    not pass for one that found them.
    """
    rng = np.random.default_rng(13)
    drivers = (rng.normal(size=N_OBS), rng.normal(size=N_OBS), None)
    feats = [f"feat_{index + 1}" for index in range(N_FEATS)]
    columns = {}
    for index, name in enumerate(feats):
        driver = drivers[index % len(drivers)]
        noise = rng.normal(scale=0.3, size=N_OBS)
        columns[name] = noise if driver is None else driver + noise
    return feats, pd.DataFrame(columns)


@functools.lru_cache(maxsize=1)
def _result():
    feats, frame = _blocked()
    return summarize_association_stats(frame, feats, methods=["pearson", "spearman"])


class TestReadingTheInput:
    def test_a_result_is_drawn_from_its_first_method_unless_told_otherwise(self):
        drawn = draw_corrplot(_result())
        expected = _result()["pearson"]["corr"]
        one = drawn["corr"].index[0]
        assert drawn["corr"].loc[one, one] == pytest.approx(expected.loc[one, one])

    def test_a_named_method_reads_that_slot(self):
        drawn = draw_corrplot(_result(), method="spearman", sig_level=1.0)
        expected = _result()["spearman"]["corr"]
        order = list(drawn["corr"].index)
        assert np.allclose(drawn["corr"].to_numpy(), expected.loc[order, order].to_numpy())

    def test_a_method_the_result_does_not_hold_is_refused(self):
        with pytest.raises(SaValueError, match="`method` must name one of the methods"):
            draw_corrplot(_result(), method="kendall")

    def test_a_matrix_has_no_slot_to_name(self):
        with pytest.raises(SaValueError, match="`method` names a slot"):
            draw_corrplot(_result()["pearson"]["corr"], method="pearson")

    def test_the_p_values_of_a_result_come_from_the_same_slot_as_the_coefficients(self):
        with pytest.raises(SaValueError, match="`pvalue` cannot be given"):
            draw_corrplot(_result(), pvalue=_result()["pearson"]["pvalue"])

    def test_a_bare_matrix_is_drawn_with_no_p_values_and_nothing_blanked(self):
        drawn = draw_corrplot(_result()["pearson"]["corr"])
        assert drawn["pvalue"] is None
        assert drawn["n_masked"] == 0

    def test_an_unlabelled_matrix_has_its_features_named_v1_upwards(self):
        drawn = draw_corrplot(_result()["pearson"]["corr"].to_numpy(), cluster=False)
        assert drawn["feats"][0] == "V1"

    def test_the_features_come_back_beside_the_order_that_indexes_them(self):
        drawn = draw_corrplot(_result())
        assert [drawn["feats"][index] for index in drawn["order"]] == list(drawn["corr"].index)

    @pytest.mark.parametrize(
        ("matrix", "match"),
        [
            (np.array([[1.0, 0.5]]), "must be square"),
            (np.array([[1.0]]), "at least 2 features"),
            (np.array([[1.0, 0.5], [0.4, 1.0]]), "must be symmetric"),
            (np.array([[1.0, 1.5], [1.5, 1.0]]), "outside"),
            (np.array([["a", "b"], ["b", "a"]]), "numeric correlation matrix"),
        ],
    )
    def test_something_that_is_not_a_correlation_matrix_is_refused(self, matrix, match):
        with pytest.raises(SaValueError, match=match):
            draw_corrplot(matrix)

    def test_p_values_that_do_not_line_up_with_the_matrix_are_refused(self):
        corr = _result()["pearson"]["corr"]
        with pytest.raises(SaValueError, match="`pvalue` must be a numeric matrix"):
            draw_corrplot(corr, pvalue=np.zeros((2, 2)))

    def test_p_values_naming_other_features_are_refused(self):
        corr = _result()["pearson"]["corr"]
        renamed = corr.copy()
        renamed.columns = [f"other_{name}" for name in corr.columns]
        with pytest.raises(SaValueError, match="must name the same features"):
            draw_corrplot(corr, pvalue=renamed)


class TestOneOrderForBothAxes:
    def test_the_rows_and_the_columns_are_in_the_same_order(self):
        drawn = draw_corrplot(_result())
        assert list(drawn["corr"].index) == list(drawn["corr"].columns)

    def test_the_diagonal_stays_on_the_diagonal(self):
        drawn = draw_corrplot(_result())
        assert np.allclose(np.diag(drawn["corr"].to_numpy()), 1.0)

    def test_the_clustering_puts_the_features_that_move_together_beside_each_other(self):
        feats, _ = _blocked()
        drawn = draw_corrplot(_result(), sig_level=1.0)
        order = [feats.index(name) for name in drawn["corr"].index]
        block = [position % 3 for position in order]  # the fixture's three drivers
        # Each driven block lands in consecutive places. The features driven by
        # nothing are not asserted about: nothing correlates them, so there is no
        # order for them to be in.
        for driven in (0, 1):
            places = [place for place, held in enumerate(block) if held == driven]
            assert places == list(range(places[0], places[0] + len(places))), driven

    def test_not_clustering_keeps_the_order_the_features_arrived_in(self):
        drawn = draw_corrplot(_result(), cluster=False)
        assert list(drawn["order"]) == list(range(len(_blocked()[0])))
        assert drawn["hclust"] is None

    def test_the_tree_says_which_linkage_and_distance_produced_it(self):
        drawn = draw_corrplot(_result(), hclust_method="ward.D2")
        assert drawn["hclust"].method == "ward.D2"
        assert drawn["hclust"].dist_method == "correlation"
        assert drawn["hclust"].labels == _blocked()[0]

    def test_a_pair_with_no_distance_leaves_the_order_alone_and_says_so(self, caplog):
        corr = _result()["pearson"]["corr"].copy()
        first, second = corr.index[0], corr.index[1]
        corr.loc[first, second] = np.nan
        corr.loc[second, first] = np.nan
        with caplog.at_level(logging.INFO, logger="statassist"):
            drawn = draw_corrplot(corr)
        assert "the order they arrived" in caplog.text
        assert drawn["hclust"] is None
        assert list(drawn["order"]) == list(range(len(corr.columns)))

    def test_a_linkage_the_package_does_not_offer_is_refused(self):
        with pytest.raises(SaValueError, match="`hclust_method` must be one of"):
            draw_corrplot(_result(), hclust_method="nope")


class TestBlanking:
    def test_a_cell_above_the_level_is_blanked_and_counted(self):
        drawn = draw_corrplot(_result(), sig_level=0.05)
        assert drawn["n_masked"] == int(drawn["corr"].isna().to_numpy().sum())

    def test_a_stricter_level_blanks_at_least_as_many_cells(self):
        loose = draw_corrplot(_result(), sig_level=0.5)
        strict = draw_corrplot(_result(), sig_level=0.001)
        assert strict["n_masked"] >= loose["n_masked"]

    def test_nothing_is_blanked_when_every_p_value_clears_the_level(self):
        drawn = draw_corrplot(_result(), sig_level=1.0)
        assert drawn["n_masked"] == 0
        assert not drawn["corr"].isna().to_numpy().any()

    def test_the_diagonal_is_never_blanked_a_feature_not_being_tested_on_itself(self):
        drawn = draw_corrplot(_result(), sig_level=1e-12)
        assert np.isfinite(np.diag(drawn["corr"].to_numpy())).all()

    def test_blanking_happens_after_the_clustering_so_the_order_is_of_the_matrix(self):
        loose = draw_corrplot(_result(), sig_level=1.0)
        strict = draw_corrplot(_result(), sig_level=0.001)
        assert list(strict["corr"].index) == list(loose["corr"].index)

    def test_the_unadjusted_p_values_blank_no_more_than_the_adjusted_ones(self):
        adjusted = draw_corrplot(_result(), use_adjusted=True)
        raw = draw_corrplot(_result(), use_adjusted=False)
        assert raw["n_masked"] <= adjusted["n_masked"]

    def test_a_pair_that_could_not_be_tested_is_left_as_it_arrived(self):
        corr = _result()["pearson"]["corr"]
        pvalue = _result()["pearson"]["pvalue"].copy()
        first, second = pvalue.index[0], pvalue.index[1]
        pvalue.loc[first, second] = np.nan
        pvalue.loc[second, first] = np.nan
        drawn = draw_corrplot(corr, pvalue=pvalue, sig_level=1e-12, cluster=False)
        assert np.isfinite(drawn["corr"].loc[first, second])


class TestDelegation:
    def test_the_colours_span_the_range_a_correlation_can_take(self):
        assert draw_corrplot(_result())["zlim"] == CORR_LIMITS

    def test_a_range_given_is_passed_on_as_given(self):
        assert draw_corrplot(_result(), zlim=(-0.5, 0.5))["zlim"] == (-0.5, 0.5)

    def test_the_matrix_is_drawn_unscaled_so_the_cells_are_the_coefficients(self):
        drawn = draw_corrplot(_result(), sig_level=1.0)
        order = list(drawn["corr"].index)
        expected = _result()["pearson"]["corr"].loc[order, order]
        assert np.allclose(drawn["matrix"].to_numpy(), expected.to_numpy())

    def test_the_heatmap_slots_come_back_beside_the_correlation_ones(self):
        drawn = draw_corrplot(_result())
        for slot in ("matrix", "zlim", "corr", "pvalue", "order", "hclust", "n_masked", "feats"):
            assert slot in drawn

    def test_an_argument_this_function_decides_cannot_be_given_again(self):
        for taken in ("data", "group", "group_lv", "scale", "cluster_feats", "cluster_samples"):
            with pytest.raises(SaValueError, match="cannot be given again"):
                draw_corrplot(_result(), **{taken: "whatever"})

    def test_the_character_expansions_are_this_functions_own_and_are_forwarded(self):
        drawn = draw_corrplot(_result(), cex_anno=0.5, cex_axis=0.5, cex_legend=0.5)
        assert drawn["n_masked"] == draw_corrplot(_result())["n_masked"]

    def test_a_remaining_heatmap_argument_is_passed_through(self):
        drawn = draw_corrplot(_result(), show_feat_names=False)
        assert drawn["corr"].shape == (len(_blocked()[0]),) * 2

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sig_level": 0}, "`sig_level`"),
            ({"cex_anno": 0}, "`cex_anno`"),
            ({"cex_main": 0}, "`cex_main`"),
            ({"zlim": (1.0, 2.0, 3.0)}, "`zlim`"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        with pytest.raises(SaValueError, match=match):
            draw_corrplot(_result(), **kwargs)
