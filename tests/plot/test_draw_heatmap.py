"""What the heatmap scaled, clustered and coloured.

The clustering is on the result rather than only on the picture, so what the plot
shows can be checked instead of eyeballed: the matrix comes back in the order it
was drawn, and the trees that put it in that order come back with it.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from statassist import draw_heatmap, simulate_two_groups
from statassist.core.errors import SaValueError


@functools.lru_cache(maxsize=1)
def _simulated():
    return simulate_two_groups(n_feats=6, n_case=8, n_control=8, n_up=2, n_down=2, seed=21)


def _call(**kwargs):
    args = _simulated().args
    kwargs.setdefault("group", args["group"])
    kwargs.setdefault("group_lv", args["group_lv"])
    return draw_heatmap(args["data"], **kwargs)


class TestScaling:
    def test_a_feature_is_z_scored_across_the_samples_by_default(self):
        drawn = _call()["matrix"]
        assert np.allclose(drawn.mean(axis=1), 0, atol=1e-9)
        assert np.allclose(drawn.std(axis=1, ddof=1), 1)

    def test_scale_sample_z_scores_the_other_margin(self):
        drawn = _call(scale="sample")["matrix"]
        assert np.allclose(drawn.mean(axis=0), 0, atol=1e-9)

    def test_scale_none_draws_the_values_as_they_arrived(self):
        out = _call(scale="none", cluster_feats=False, cluster_samples=False)
        args = _simulated().args
        expected = args["data"][list(args["feats"])].to_numpy(dtype=float).T
        assert np.allclose(out["matrix"].to_numpy(), expected)

    def test_a_flat_feature_is_centred_rather_than_divided_by_zero(self):
        args = _simulated().args
        data = args["data"].copy()
        data[args["feats"][0]] = 5.0
        out = draw_heatmap(data, args["group"], args["group_lv"], cluster_feats=False)
        row = out["matrix"].loc[args["feats"][0]]
        assert np.isfinite(row).all()
        assert np.allclose(row, 0)


class TestClustering:
    def test_the_returned_order_is_the_order_the_matrix_came_back_in(self):
        out = _call()
        args = _simulated().args
        expected = [list(args["feats"])[i] for i in out["feat_order"]]
        assert list(out["matrix"].index) == expected

    def test_an_axis_that_was_not_clustered_keeps_its_input_order(self):
        out = _call(cluster_feats=False)
        assert list(out["feat_order"]) == list(range(len(_simulated().args["feats"])))
        assert out["feat_hclust"] is None

    def test_the_tree_says_which_linkage_and_distance_produced_it(self):
        out = _call(hclust_method="ward.D2", dist_method="correlation")
        assert out["sample_hclust"].method == "ward.D2"
        assert out["sample_hclust"].dist_method == "correlation"
        assert out["sample_hclust"].linkage.shape[1] == 4

    def test_an_undefined_distance_leaves_that_axis_alone_rather_than_failing(self):
        """A pair that shares no observation has no distance to be clustered on."""
        args = _simulated().args
        data = args["data"].copy()
        feats = list(args["feats"])
        data.loc[data.index[:4], feats[0]] = np.nan
        data.loc[data.index[4:], feats[1]] = np.nan
        out = draw_heatmap(data, args["group"], args["group_lv"])
        assert out["feat_hclust"] is None or out["sample_hclust"] is None


class TestColour:
    def test_a_derived_range_is_symmetric_about_zero_when_both_signs_are_drawn(self):
        low, high = _call()["zlim"]
        assert pytest.approx(-low) == high

    def test_a_supplied_range_is_used_as_given_and_the_values_are_not_clamped(self):
        out = _call(zlim=(-1.0, 1.0))
        assert out["zlim"] == (-1.0, 1.0)
        assert out["matrix"].to_numpy().max() > 1.0

    def test_one_colour_per_group_level_in_the_order_they_were_named(self):
        out = _call()
        assert list(out["group_colors"]) == list(_simulated().args["group_lv"])

    def test_no_group_means_no_strip_and_no_legend(self):
        out = draw_heatmap(_simulated().args["data"])
        assert out["group_colors"] is None


class TestInput:
    def test_a_bare_array_has_its_columns_named_v1_upwards(self):
        args = _simulated().args
        out = draw_heatmap(args["data"].to_numpy())
        assert list(out["matrix"].index)[0].startswith("V")

    def test_feats_selects_the_rows_and_their_unclustered_order(self):
        args = _simulated().args
        wanted = [args["feats"][2], args["feats"][0]]
        out = _call(feats=wanted, cluster_feats=False)
        assert list(out["matrix"].index) == wanted

    def test_labels_replace_the_names_they_stand_for(self):
        args = _simulated().args
        out = _call(
            feat_labels=[f"f{i}" for i in range(len(args["feats"]))],
            cluster_feats=False,
        )
        assert list(out["matrix"].index)[0] == "f0"

    def test_hiding_an_axis_changes_nothing_but_the_drawing(self):
        shown = _call()
        hidden = _call(show_feat_names=False, show_sample_names=False)
        assert list(shown["matrix"].index) == list(hidden["matrix"].index)
        assert np.allclose(shown["matrix"].to_numpy(), hidden["matrix"].to_numpy())

    def test_missing_cells_survive_into_the_returned_matrix(self):
        args = _simulated().args
        data = args["data"].copy()
        data.loc[data.index[0], args["feats"][0]] = np.nan
        out = draw_heatmap(data, args["group"], args["group_lv"], cluster_feats=False)
        assert out["matrix"].isna().to_numpy().sum() == 1


class TestArgumentChecks:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"scale": "nope"}, "`scale` must be one of"),
            ({"dist_method": "nope"}, "`dist_method` must be one of"),
            ({"hclust_method": "nope"}, "`hclust_method` must be one of"),
            ({"n_colors": 2}, "`n_colors`"),
            ({"zlim": (1.0, 1.0)}, "two different ends"),
            ({"cex_axis": 0}, "`cex_axis`"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        with pytest.raises(SaValueError, match=match):
            _call(**kwargs)

    def test_a_group_without_its_levels_is_refused(self):
        args = _simulated().args
        with pytest.raises(SaValueError, match="both be supplied"):
            draw_heatmap(args["data"], group=args["group"])

    def test_labels_that_do_not_match_what_they_label_are_refused(self):
        with pytest.raises(SaValueError, match="`feat_labels` must have one entry"):
            _call(feat_labels=["only one"])
        with pytest.raises(SaValueError, match="`sample_labels` must have one entry"):
            _call(sample_labels=["only one"])

    def test_one_feature_is_not_enough_to_cluster_and_draw(self):
        args = _simulated().args
        with pytest.raises(SaValueError, match="at least 2 features"):
            _call(feats=[args["feats"][0]])

    def test_a_frame_with_nothing_finite_in_it_is_refused(self):
        args = _simulated().args
        empty = pd.DataFrame(np.nan, index=args["data"].index, columns=list(args["feats"]))
        with pytest.raises(SaValueError, match="no finite value"):
            draw_heatmap(empty, args["group"], args["group_lv"])
