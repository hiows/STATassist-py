"""The picture the unsupervised family ends at.

Two channels carry two readings, and which one carries which is decided from the
arguments that arrived rather than from anything about the data. That decision is
most of what is checked here, along with the alignment the two result objects
already promise: a clustering of different points is refused rather than lined up by
position.

What is not checked is where a point landed. The reduction decides that and its own
tests grade it; what this one grades is that the point drawn at those coordinates is
the point the returned table says it is, in the colour and the marker the table says
it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from statassist import (
    cluster_dbscan,
    cluster_kmeans,
    draw_dim_reduction_plot,
    perform_pca,
    perform_tsne,
    simulate_two_groups,
)
from statassist.core import SaValueError
from statassist.plot import SCATTER_VIEWS
from statassist.plot._scatter import MAX_SCATTER_LEVELS, SCATTER_MARKERS

#: A wide table with a grouping the reduction is never shown.
_N_FEATS = 20
_N_PER_GROUP = 14

#: How many clusters the partitioning is asked for, which is what the grouping has.
_N_CLUST = 2


@pytest.fixture(scope="module")
def planted():
    return simulate_two_groups(
        n_feats=_N_FEATS,
        n_case=_N_PER_GROUP,
        n_control=_N_PER_GROUP,
        n_up=5,
        n_down=5,
        seed=9,
    )


@pytest.fixture(scope="module")
def rotated(planted):
    return perform_pca(planted.args["data"])


@pytest.fixture(scope="module")
def grouped(rotated, planted):
    return cluster_kmeans(planted.args["data"], n_clust=_N_CLUST, seed=1)


def _axes():
    """The scatter panel, which is the first axes the figure was given."""
    import matplotlib.pyplot as plt

    return plt.gcf().axes[0]


def _drawn_points(ax) -> np.ndarray:
    """Every point on an axes, in no particular order, as coordinates."""
    clouds = [artist for artist in ax.collections if isinstance(artist, PathCollection)]
    return np.vstack([np.asarray(cloud.get_offsets()) for cloud in clouds])


def _legend_labels(ax) -> list[str]:
    """The labels of every legend block on an axes, in the order they were added."""
    legends = [*ax.get_children()]
    labels: list[str] = []
    for artist in legends:
        if hasattr(artist, "get_texts") and hasattr(artist, "get_title"):
            labels.extend(text.get_text() for text in artist.get_texts())
    return labels


def _legend_titles(ax) -> list[str]:
    titles: list[str] = []
    for artist in ax.get_children():
        if hasattr(artist, "get_texts") and hasattr(artist, "get_title"):
            titles.append(artist.get_title().get_text())
    return titles


class TestReturnedTable:
    """What comes back describes what was drawn."""

    def test_the_points_and_coordinates_are_the_reductions_own(self, rotated):
        drawn = draw_dim_reduction_plot(rotated)
        assert drawn["points"].tolist() == rotated["points"]
        assert drawn["x"].to_numpy() == pytest.approx(rotated["scores"]["PC1"].to_numpy())
        assert drawn["y"].to_numpy() == pytest.approx(rotated["scores"]["PC2"].to_numpy())

    def test_every_point_of_the_table_reaches_the_figure(self, rotated, planted, grouped):
        drawn = draw_dim_reduction_plot(
            rotated, group=planted.args["group"], cluster_result=grouped
        )
        placed = _drawn_points(_axes())
        assert len(placed) == len(drawn.index)
        # Drawn one call per marker, so the order is by shape rather than by row;
        # what has to match is the set of coordinates.
        wanted = np.sort(drawn[["x", "y"]].to_numpy(), axis=0)
        assert np.sort(placed, axis=0) == pytest.approx(wanted)

    def test_dims_choose_which_two_coordinates_are_drawn(self, rotated):
        drawn = draw_dim_reduction_plot(rotated, dims=(3, 1))
        assert drawn["x"].to_numpy() == pytest.approx(rotated["scores"]["PC3"].to_numpy())
        assert drawn["y"].to_numpy() == pytest.approx(rotated["scores"]["PC1"].to_numpy())

    def test_the_columns_follow_the_channels_that_were_used(self, rotated, planted, grouped):
        plain = draw_dim_reduction_plot(rotated)
        assert list(plain.columns) == ["points", "x", "y", "col", "marker"]
        both = draw_dim_reduction_plot(rotated, group=planted.args["group"], cluster_result=grouped)
        assert list(both.columns) == [
            "points",
            "x",
            "y",
            "cluster",
            "group",
            "col",
            "marker",
        ]


class TestChannels:
    """Which reading takes the colours and which takes the markers."""

    def test_the_four_views_are_decided_by_the_arguments(self, rotated, planted, grouped):
        group = planted.args["group"]
        views = {
            SCATTER_VIEWS[0]: {"group": group, "cluster_result": grouped},
            SCATTER_VIEWS[1]: {"cluster_result": grouped},
            SCATTER_VIEWS[2]: {"group": group},
            SCATTER_VIEWS[3]: {},
        }
        for view, kwargs in views.items():
            assert draw_dim_reduction_plot(rotated, **kwargs).attrs["view"] == view

    def test_a_clustering_takes_the_colours_and_a_group_the_markers(
        self, rotated, planted, grouped
    ):
        drawn = draw_dim_reduction_plot(
            rotated, group=planted.args["group"], cluster_result=grouped
        )
        # One colour per cluster and one marker per group, which is what makes the
        # agreement between the two readable off the picture.
        assert drawn.groupby("cluster", observed=True)["col"].nunique().eq(1).all()
        assert drawn.groupby("group", observed=True)["marker"].nunique().eq(1).all()
        assert drawn["col"].nunique() == _N_CLUST
        assert drawn["marker"].nunique() == 2

    def test_a_lone_group_is_shaped_and_coloured_only_when_col_is_named(self, rotated, planted):
        group = planted.args["group"]
        shaped = draw_dim_reduction_plot(rotated, group=group)
        assert shaped["marker"].nunique() == 2
        assert shaped["col"].nunique() == 1

        coloured = draw_dim_reduction_plot(rotated, group=group, col=["#E69F00", "#56B4E9"])
        assert coloured["col"].nunique() == 2
        assert coloured.groupby("group", observed=True)["col"].nunique().eq(1).all()

    def test_a_lone_clustering_is_coloured_and_left_one_marker(self, rotated, grouped):
        drawn = draw_dim_reduction_plot(rotated, cluster_result=grouped)
        assert drawn["col"].nunique() == _N_CLUST
        assert drawn["marker"].unique().tolist() == [SCATTER_MARKERS[0]]

    def test_with_neither_channel_col_is_one_colour_for_every_point(self, rotated):
        drawn = draw_dim_reduction_plot(rotated, col="#009E73")
        assert drawn["col"].unique().tolist() == ["#009E73"]

    def test_a_col_that_is_neither_one_nor_one_per_point_is_refused(self, rotated):
        with pytest.raises(SaValueError, match="one per point"):
            draw_dim_reduction_plot(rotated, col=["red", "blue"])

    def test_marker_can_be_named_per_group_level(self, rotated, planted):
        drawn = draw_dim_reduction_plot(rotated, group=planted.args["group"], marker=["P", "X"])
        assert set(drawn["marker"]) == {"P", "X"}

    def test_a_factors_own_order_reaches_the_legend_and_a_lists_does_not(self, rotated, planted):
        levels = list(planted.args["group_lv"])
        # A plain list carries no order, so the levels are sorted, which is what R
        # does with a character vector.
        plain = draw_dim_reduction_plot(rotated, group=planted.args["group"])
        assert list(plain["group"].cat.categories) == sorted(levels)
        # A factor does carry one, so the simulator's `group_lv` reaches the legend
        # without being named again.
        factor = pd.Categorical(planted.args["group"], categories=levels)
        declared = draw_dim_reduction_plot(rotated, group=factor)
        assert list(declared["group"].cat.categories) == levels

    def test_group_lv_reorders_and_does_not_select(self, rotated, planted):
        levels = list(planted.args["group_lv"])[::-1]
        drawn = draw_dim_reduction_plot(rotated, group=planted.args["group"], group_lv=levels)
        assert list(drawn["group"].cat.categories) == levels
        assert len(drawn.index) == len(rotated["points"])

        with pytest.raises(SaValueError, match="level left out"):
            draw_dim_reduction_plot(rotated, group=planted.args["group"], group_lv=levels[:1])

    def test_more_levels_than_there_are_markers_is_refused(self, rotated):
        many = [f"g{index % (MAX_SCATTER_LEVELS + 1)}" for index in range(len(rotated["points"]))]
        with pytest.raises(SaValueError, match="markers to tell them apart"):
            draw_dim_reduction_plot(rotated, group=many)


class TestLegend:
    """Two readings need two blocks, and noise needs a line of its own."""

    def test_both_channels_get_a_titled_block(self, rotated, planted, grouped):
        draw_dim_reduction_plot(rotated, group=planted.args["group"], cluster_result=grouped)
        import matplotlib.pyplot as plt

        legend_ax = plt.gcf().axes[1]
        assert sorted(_legend_titles(legend_ax)) == ["cluster", "group"]
        labels = _legend_labels(legend_ax)
        assert [f"#{index + 1}" for index in range(_N_CLUST)] == labels[:_N_CLUST]
        assert set(planted.args["group_lv"]) <= set(labels)

    def test_no_channel_means_no_legend_panel(self, rotated):
        import matplotlib.pyplot as plt

        draw_dim_reduction_plot(rotated)
        assert len(plt.gcf().axes) == 1

    def test_cluster_lv_names_what_the_legend_would_number(self, rotated, grouped):
        import matplotlib.pyplot as plt

        names = [f"cluster {index}" for index in range(_N_CLUST)]
        draw_dim_reduction_plot(rotated, cluster_result=grouped, cluster_lv=names)
        assert _legend_labels(plt.gcf().axes[1]) == names

    def test_a_cluster_lv_of_the_wrong_length_or_with_a_repeat_is_refused(self, rotated, grouped):
        with pytest.raises(SaValueError, match="one label per cluster"):
            draw_dim_reduction_plot(rotated, cluster_result=grouped, cluster_lv=["only"])
        with pytest.raises(SaValueError, match="repeat a level"):
            draw_dim_reduction_plot(rotated, cluster_result=grouped, cluster_lv=["same"] * _N_CLUST)

    def test_naming_levels_of_a_channel_that_was_not_given_is_refused(self, rotated):
        with pytest.raises(SaValueError, match="`group_lv`"):
            draw_dim_reduction_plot(rotated, group_lv=["a", "b"])
        with pytest.raises(SaValueError, match="`cluster_lv`"):
            draw_dim_reduction_plot(rotated, cluster_lv=["a"])


@pytest.fixture(scope="module")
def with_noise(planted):
    """The planted table with two rows moved far enough out to be left alone."""
    frame = planted.args["data"].copy()
    frame.iloc[:2] = frame.to_numpy().max() * 10
    return frame


class TestNoise:
    """A point in no cluster is the absence of a cluster, not one of them."""

    def test_noise_is_grey_and_counted_on_a_line_of_its_own(self, with_noise):
        import matplotlib.pyplot as plt

        density = cluster_dbscan(with_noise)
        # Asserted rather than skipped over: a table that stopped producing noise
        # would otherwise leave this test passing without testing anything.
        assert density["design"]["n_noise"] > 0
        res = perform_pca(with_noise)
        drawn = draw_dim_reduction_plot(res, cluster_result=density)
        noise_col = set(drawn.loc[drawn["cluster"] == 0, "col"])
        assert len(noise_col) == 1
        assert noise_col.isdisjoint(set(drawn.loc[drawn["cluster"] > 0, "col"]))
        labels = _legend_labels(plt.gcf().axes[1])
        assert f"noise ({density['design']['n_noise']})" in labels


class TestRefusals:
    """The two objects the plot is given, and what is not one of them."""

    def test_a_clustering_in_the_first_slot_is_turned_away_by_name(self, grouped):
        with pytest.raises(SaValueError, match="is a clustering"):
            draw_dim_reduction_plot(grouped)

    def test_something_that_is_neither_is_refused(self, planted):
        with pytest.raises(SaValueError, match="must be a reduction"):
            draw_dim_reduction_plot(planted.args["data"])

    def test_a_reduction_in_the_cluster_slot_is_refused(self, rotated):
        with pytest.raises(SaValueError, match="must be a clustering"):
            draw_dim_reduction_plot(rotated, cluster_result=rotated)

    def test_a_clustering_of_different_points_is_refused(self, rotated, planted):
        other = cluster_kmeans(planted.args["data"].iloc[:-1], n_clust=_N_CLUST, seed=1)
        with pytest.raises(SaValueError, match="different points"):
            draw_dim_reduction_plot(rotated, cluster_result=other)

    def test_a_group_of_the_wrong_length_says_what_it_is_missing(self, planted):
        frame = planted.args["data"].copy()
        frame.iloc[0, 0] = np.nan
        res = perform_pca(frame)
        with pytest.raises(SaValueError, match="dropped 1 of the"):
            draw_dim_reduction_plot(res, group=planted.args["group"])

    def test_a_group_on_the_feature_scale_says_the_points_are_features(self, planted):
        by_feat = perform_pca(planted.args["data"], embedding_scale="features")
        with pytest.raises(SaValueError, match="features rather than samples"):
            draw_dim_reduction_plot(by_feat, group=planted.args["group"])

    def test_a_group_with_a_missing_label_is_refused(self, rotated, planted):
        group = pd.Series(planted.args["group"], dtype=object)
        group.iloc[0] = None
        with pytest.raises(SaValueError, match="missing value"):
            draw_dim_reduction_plot(rotated, group=group)

    def test_dims_must_name_two_different_coordinates_the_reduction_has(self, rotated):
        with pytest.raises(SaValueError, match="two different coordinates"):
            draw_dim_reduction_plot(rotated, dims=(2, 2))
        with pytest.raises(SaValueError, match="two numbers"):
            draw_dim_reduction_plot(rotated, dims=1)
        with pytest.raises(SaValueError, match="asks for coordinate"):
            draw_dim_reduction_plot(rotated, dims=(1, len(rotated["points"]) + 99))


class TestAxes:
    """What the panel says about the coordinates on it."""

    def test_a_rotation_labels_its_axes_with_the_share_of_the_variance(self, rotated):
        draw_dim_reduction_plot(rotated)
        ax = _axes()
        assert ax.get_xlabel().startswith("PC1 (")
        assert ax.get_xlabel().endswith("%)")
        assert ax.get_title() == rotated["engine"]["label"]

    def test_an_embedding_labels_its_axes_with_the_names_alone(self, planted):
        embedded = perform_tsne(planted.args["data"], seed=2)
        draw_dim_reduction_plot(embedded)
        assert _axes().get_xlabel() == "tSNE1"

    def test_the_labels_and_title_can_be_replaced(self, rotated):
        draw_dim_reduction_plot(rotated, xlab="left", ylab="up", main="mine")
        ax = _axes()
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("left", "up", "mine")

    def test_the_ranges_can_be_set_and_asp_makes_the_units_comparable(self, rotated):
        draw_dim_reduction_plot(rotated, xlim=(-9, 9), ylim=(-4, 4), asp=1)
        ax = _axes()
        assert ax.get_xlim() == (-9, 9)
        assert ax.get_aspect() == 1

    def test_anno_points_writes_one_label_per_point(self, rotated):
        draw_dim_reduction_plot(rotated, anno_points=True)
        annotations = [text.get_text() for text in _axes().texts]
        assert annotations == rotated["points"]

    def test_a_size_that_is_not_a_positive_number_is_refused(self, rotated):
        with pytest.raises(SaValueError, match="cex"):
            draw_dim_reduction_plot(rotated, cex=0)
        with pytest.raises(SaValueError, match="asp"):
            draw_dim_reduction_plot(rotated, asp=-1)
