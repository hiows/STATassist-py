"""Grouping points without being told what the groups are.

The four functions share the whole input side and the whole result contract, so
what is checked here in one place is that shared part: which rows and columns reach
an engine, what ``design`` says about them, and the invariants that hold between
``assignments`` and ``clusters`` whichever method filled them.

What is specific to each is checked separately, and it is a statistical property
rather than a number. The two partitioning methods are told how many groups to find
and must find that many; the two density methods derive the count, so what has to
hold is that they find the structure a synthetic table was built with and call the
points outside it noise.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from statassist import (
    cluster_dbscan,
    cluster_hclust,
    cluster_kmeans,
    cluster_snn,
    simulate_two_groups,
)
from statassist.core import SaValueError
from statassist.core.contracts import cluster_assignment_columns, cluster_table_columns
from statassist.kernel.cluster import NOISE_LABEL

#: How far apart the planted blobs sit, in standard deviations of their spread.
#:
#: Far enough that every method finds them and none of the assertions below is
#: about a borderline case. A test that grades a clustering has to be run on data
#: whose grouping is not in question.
_BLOB_GAP = 8.0

#: Spread of one blob, and how many points it holds.
_BLOB_SD = 1.0
_BLOB_N = 25


@pytest.fixture(scope="module")
def blobs():
    """Two clearly separated blobs in two dimensions, plus two far-off strays.

    The strays are what a density method is allowed to call noise and a
    partitioning method is not, so one table grades both behaviours.
    """
    rng = np.random.default_rng(11)
    points = np.vstack(
        [
            rng.normal(-_BLOB_GAP / 2, _BLOB_SD, (_BLOB_N, 2)),
            rng.normal(_BLOB_GAP / 2, _BLOB_SD, (_BLOB_N, 2)),
            np.array([[-40.0, 40.0], [40.0, -40.0]]),
        ]
    )
    frame = pd.DataFrame(points, columns=["x", "y"])
    truth = np.array([0] * _BLOB_N + [1] * _BLOB_N + [-1, -1])
    return {
        "data": frame,
        # The same table without the strays. A partitioning method has to place
        # them, and the cheapest way to place two points 80 units from everything
        # is to give them a group of their own, so on the full table the count it
        # was asked for goes on the strays rather than on the structure. Grading
        # the structure needs a table that only has structure in it.
        "pure": frame.iloc[: 2 * _BLOB_N].reset_index(drop=True),
        "truth": truth,
        "n_stray": 2,
    }


@pytest.fixture(scope="module")
def planted():
    """A wide simulation, which is the shape this package is usually given."""
    return simulate_two_groups(n_feats=12, n_case=15, n_control=15, n_up=6, n_down=6, seed=3)


@pytest.fixture(scope="module")
def results(blobs):
    """Every method run on the same table, keyed by ``analysis``."""
    frame = blobs["data"]
    return {
        "hclust": cluster_hclust(frame, n_clust=2, scale=False),
        "kmeans": cluster_kmeans(frame, n_clust=2, scale=False, seed=1),
        "dbscan": cluster_dbscan(frame, scale=False),
        "snn": cluster_snn(frame, scale=False),
    }


class TestSharedContract:
    """What holds of a clustering whichever of the four produced it."""

    def test_the_analysis_names_the_method_that_was_called(self, results):
        for name, res in results.items():
            assert res["analysis"] == name

    def test_every_table_carries_its_columns_in_order(self, results):
        for res in results.values():
            assert list(res["assignments"].columns) == cluster_assignment_columns()
            assert list(res["clusters"].columns) == cluster_table_columns()

    def test_the_assignments_are_aligned_with_the_points_by_position(self, results, blobs):
        for res in results.values():
            assert list(res["assignments"]["points"]) == res["points"]
            assert len(res["points"]) == len(blobs["data"].index)

    def test_the_two_tables_are_one_fact_counted_twice(self, results):
        """`new_cluster` refuses a mismatch, so this is checking it was reached."""
        for res in results.values():
            labels = res["assignments"]["cluster"]
            listed = list(res["clusters"]["cluster"])
            assert listed == sorted(set(labels[labels > NOISE_LABEL]))
            assert list(res["clusters"]["size"]) == [
                int((labels == value).sum()) for value in listed
            ]
            assert res["design"]["n_clusters"] == len(listed)
            assert res["design"]["n_noise"] == int((labels == NOISE_LABEL).sum())

    def test_the_labels_are_numbered_from_one_by_first_appearance(self, results):
        """Which is what makes two methods' answers comparable at all."""
        for res in results.values():
            seen = []
            for value in res["assignments"]["cluster"]:
                if value != NOISE_LABEL and value not in seen:
                    seen.append(int(value))
            assert seen == list(range(1, len(seen) + 1))

    def test_noise_has_no_silhouette_and_nothing_else_is_missing_one(self, results):
        for res in results.values():
            table = res["assignments"]
            noise = table["cluster"] == NOISE_LABEL
            assert table.loc[noise, "silhouette"].isna().all()
            if res["design"]["n_clusters"] > 1:
                assert table.loc[~noise, "silhouette"].notna().all()

    def test_the_design_describes_the_input_and_does_not_turn_with_the_margin(self, blobs):
        by_sample = cluster_hclust(blobs["data"], n_clust=2, scale=False)
        by_feature = cluster_hclust(blobs["data"], cluster_scale="features", n_clust=2, scale=False)
        for res in (by_sample, by_feature):
            assert res["design"]["n_samples"] == len(blobs["data"].index)
            assert res["design"]["n_feats"] == len(blobs["data"].columns)
            assert res["design"]["feats"] == list(blobs["data"].columns)
        assert by_sample["design"]["point_type"] == "sample"
        assert by_feature["design"]["point_type"] == "feature"
        assert by_feature["points"] == list(blobs["data"].columns)

    def test_every_slot_survives_a_round_trip_through_json(self, results):
        """The engine object is `res.fit` rather than a slot, which is the point."""
        for res in results.values():
            payload = {
                key: (value.to_dict(orient="list") if isinstance(value, pd.DataFrame) else value)
                for key, value in res.items()
            }
            assert json.loads(json.dumps(payload)) is not None
            assert res.fit is not None

    def test_the_repr_summarises_rather_than_listing_every_point(self, results):
        for name, res in results.items():
            text = repr(res)
            assert text.startswith(f"<SaCluster> {name}")
            assert str(res["design"]["n_clusters"]) in text
            assert str(res["points"][0]) not in text.split("\n")[0]


class TestSharedInput:
    """What is dropped before any engine is called, and what is refused."""

    def test_a_row_with_a_hole_is_dropped_and_counted(self, blobs, caplog):
        frame = blobs["data"].copy()
        frame.iloc[0, 0] = np.nan
        with caplog.at_level(logging.INFO):
            res = cluster_hclust(frame, n_clust=2, scale=False)
        assert res["design"]["n_samples"] == len(frame.index)
        assert res["design"]["n_dropped"] == 1
        assert len(res["points"]) == len(frame.index) - 1
        assert "not complete and finite" in caplog.text

    def test_an_infinite_value_is_dropped_with_the_missing_ones(self, blobs):
        frame = blobs["data"].copy()
        frame.iloc[1, 1] = np.inf
        res = cluster_hclust(frame, n_clust=2, scale=False)
        assert res["design"]["n_dropped"] == 1

    def test_a_feature_of_no_variance_is_left_out_when_it_would_be_scaled(self, blobs, caplog):
        frame = blobs["data"].copy()
        frame["flat"] = 1.0
        with caplog.at_level(logging.INFO):
            res = cluster_kmeans(frame, n_clust=2, seed=1)
        assert res["design"]["dropped_feats"] == ["flat"]
        assert res["design"]["n_feats"] == len(blobs["data"].columns)
        assert "no variance" in caplog.text

    def test_the_same_feature_is_kept_when_nothing_will_divide_by_it(self, blobs):
        frame = blobs["data"].copy()
        frame["flat"] = 1.0
        res = cluster_kmeans(frame, n_clust=2, scale=False, seed=1)
        assert res["design"]["dropped_feats"] == []
        assert res["design"]["n_feats"] == len(frame.columns)

    def test_a_non_numeric_column_is_left_out_rather_than_refused(self, blobs, caplog):
        frame = blobs["data"].copy()
        frame["label"] = "a"
        with caplog.at_level(logging.INFO):
            res = cluster_hclust(frame, n_clust=2, scale=False)
        assert res["design"]["feats"] == list(blobs["data"].columns)
        assert "non-numeric" in caplog.text

    def test_the_index_is_what_labels_a_point(self, blobs):
        frame = blobs["data"].copy()
        frame.index = [f"s{i}" for i in range(len(frame.index))]
        res = cluster_hclust(frame, n_clust=2, scale=False)
        assert res["points"][:2] == ["s0", "s1"]

    def test_a_repeated_label_is_a_naming_choice_rather_than_an_error(self, blobs):
        frame = blobs["data"].copy()
        frame.index = ["same"] * len(frame.index)
        res = cluster_hclust(frame, n_clust=2, scale=False)
        assert set(res["points"]) == {"same"}

    def test_a_table_too_small_to_cluster_is_refused_by_name(self):
        with pytest.raises(SaValueError, match="cluster_hclust"):
            cluster_hclust(pd.DataFrame({"x": [1.0], "y": [2.0]}), n_clust=2)

    def test_one_feature_is_not_a_table_either(self):
        with pytest.raises(SaValueError, match="at least 2 samples and 2 features"):
            cluster_kmeans(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), n_clust=2)

    @pytest.mark.parametrize(
        "call",
        [cluster_hclust, cluster_kmeans, cluster_dbscan, cluster_snn],
    )
    def test_an_unknown_margin_is_refused_before_anything_is_read(self, call, blobs):
        with pytest.raises(SaValueError, match="cluster_scale"):
            call(blobs["data"], cluster_scale="rows")

    @pytest.mark.parametrize(
        "call",
        [cluster_hclust, cluster_kmeans, cluster_dbscan, cluster_snn],
    )
    def test_a_flag_that_is_not_a_flag_is_refused(self, call, blobs):
        with pytest.raises(SaValueError, match="`center` must be TRUE or FALSE"):
            call(blobs["data"], center=1)

    def test_data_that_is_not_a_table_is_refused(self):
        with pytest.raises(SaValueError, match="DataFrame or a 2-d array"):
            cluster_kmeans([1, 2, 3], n_clust=2)


class TestPartitioning:
    """The two that are told the count and always return it."""

    @pytest.mark.parametrize("call", [cluster_hclust, cluster_kmeans])
    def test_every_point_is_placed_including_the_strays(self, call, blobs):
        res = call(blobs["data"], n_clust=2, scale=False)
        assert res["design"]["n_noise"] == 0
        assert (res["assignments"]["cluster"] > NOISE_LABEL).all()

    @pytest.mark.parametrize("call", [cluster_hclust, cluster_kmeans])
    @pytest.mark.parametrize("n_clust", [2, 3, 4])
    def test_the_count_asked_for_is_the_count_returned(self, call, n_clust, planted):
        res = call(planted.args["data"], n_clust=n_clust)
        assert res["design"]["n_clusters"] == n_clust

    @pytest.mark.parametrize("call", [cluster_hclust, cluster_kmeans])
    def test_one_cluster_is_not_a_clustering(self, call, blobs):
        with pytest.raises(SaValueError, match="`n_clust`"):
            call(blobs["data"], n_clust=1)

    @pytest.mark.parametrize("call", [cluster_hclust, cluster_kmeans])
    def test_there_cannot_be_more_groups_than_points(self, call, blobs):
        with pytest.raises(SaValueError, match="more groups than there are things"):
            call(blobs["data"], n_clust=len(blobs["data"].index) + 1, scale=False)

    @pytest.mark.parametrize("call", [cluster_hclust, cluster_kmeans])
    def test_the_two_blobs_come_apart(self, call, blobs):
        res = call(blobs["pure"], n_clust=2, scale=False)
        labels = res["assignments"]["cluster"].to_numpy()
        left = set(labels[:_BLOB_N])
        right = set(labels[_BLOB_N:])
        assert len(left) == 1
        assert len(right) == 1
        assert left != right

    def test_the_same_seed_gives_kmeans_the_same_answer(self, planted):
        first = cluster_kmeans(planted.args["data"], n_clust=3, seed=7)
        again = cluster_kmeans(planted.args["data"], n_clust=3, seed=7)
        assert list(first["assignments"]["cluster"]) == list(again["assignments"]["cluster"])
        assert first["parameters"]["tot_withinss"] == pytest.approx(
            again["parameters"]["tot_withinss"]
        )

    def test_more_clusters_never_leave_more_within_cluster_spread(self, planted):
        two = cluster_kmeans(planted.args["data"], n_clust=2, seed=7)
        four = cluster_kmeans(planted.args["data"], n_clust=4, seed=7)
        assert four["parameters"]["tot_withinss"] <= two["parameters"]["tot_withinss"]

    def test_a_centre_cannot_be_placed_where_another_one_already_is(self):
        frame = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0], "y": [2.0, 2.0, 3.0, 3.0]})
        with pytest.raises(SaValueError, match="are distinct"):
            cluster_kmeans(frame, n_clust=3, scale=False)

    def test_hclust_keeps_the_tree_so_another_cut_costs_nothing(self, blobs):
        from scipy.cluster.hierarchy import fcluster

        res = cluster_hclust(blobs["data"], n_clust=2, scale=False)
        again = fcluster(res.fit, t=3, criterion="maxclust")
        assert len(set(again)) == 3

    @pytest.mark.parametrize("hclust_method", ["average", "complete", "ward.D2"])
    def test_every_linkage_on_offer_runs_and_is_reported(self, hclust_method, blobs):
        res = cluster_hclust(blobs["data"], n_clust=2, scale=False, hclust_method=hclust_method)
        assert res["parameters"]["hclust_method"] == hclust_method

    @pytest.mark.parametrize("dist_method", ["euclidean", "manhattan", "correlation"])
    def test_every_distance_on_offer_runs_and_is_reported(self, dist_method, planted):
        res = cluster_hclust(
            planted.args["data"], cluster_scale="features", n_clust=2, dist_method=dist_method
        )
        assert res["parameters"]["dist_method"] == dist_method

    def test_an_unknown_linkage_or_distance_is_refused(self, blobs):
        with pytest.raises(SaValueError, match="hclust_method"):
            cluster_hclust(blobs["data"], hclust_method="single")
        with pytest.raises(SaValueError, match="dist_method"):
            cluster_hclust(blobs["data"], dist_method="cosine")

    def test_a_distance_that_is_undefined_somewhere_stops_the_tree(self):
        """A correlation needs a point to vary, and a flat row does not."""
        frame = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 5.0], "y": [2.0, 4.0, 6.0, 5.0], "z": [3.0, 1.0, 2.0, 5.0]}
        )
        with pytest.raises(SaValueError, match="some of them are undefined"):
            cluster_hclust(frame, n_clust=2, center=False, scale=False, dist_method="correlation")


class TestDensity:
    """The two that derive the count and may refuse to place a point."""

    def test_dbscan_finds_the_two_blobs_and_calls_the_strays_noise(self, blobs):
        res = cluster_dbscan(blobs["data"], scale=False)
        assert res["design"]["n_clusters"] == 2
        assert res["design"]["n_noise"] == blobs["n_stray"]
        labels = res["assignments"]["cluster"].to_numpy()
        assert (labels[-blobs["n_stray"] :] == NOISE_LABEL).all()

    def test_snn_finds_the_two_blobs_as_well(self, blobs):
        res = cluster_snn(blobs["data"], scale=False, k=8)
        labels = res["assignments"]["cluster"].to_numpy()
        assert res["design"]["n_clusters"] == 2
        assert len(set(labels[:_BLOB_N])) == 1
        assert len(set(labels[_BLOB_N : 2 * _BLOB_N])) == 1

    def test_snn_finds_two_blobs_of_different_spread_which_is_its_reason_to_exist(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame(
            np.vstack([rng.normal(-6, 0.4, (40, 2)), rng.normal(6, 2.5, (40, 2))]),
            columns=["x", "y"],
        )
        res = cluster_snn(frame, scale=False, k=10)
        labels = res["assignments"]["cluster"].to_numpy()
        assert res["design"]["n_clusters"] == 2
        assert len(set(labels[:40])) == 1
        assert len(set(labels[40:])) == 1

    def test_a_derived_eps_is_reported_as_derived_and_said_out_loud(self, blobs, caplog):
        with caplog.at_level(logging.INFO):
            res = cluster_dbscan(blobs["data"], scale=False)
        assert res["parameters"]["eps_source"] == "derived"
        assert res["parameters"]["eps"] > 0
        assert "Using eps" in caplog.text

    def test_a_supplied_eps_is_used_as_it_stands(self, blobs):
        res = cluster_dbscan(blobs["data"], scale=False, eps=2.0, min_pts=4)
        assert res["parameters"]["eps_source"] == "supplied"
        assert res["parameters"]["eps"] == pytest.approx(2.0)

    def test_a_derived_min_pts_is_said_out_loud(self, blobs, caplog):
        with caplog.at_level(logging.INFO):
            cluster_dbscan(blobs["data"], scale=False)
        assert "Using min_pts" in caplog.text

    def test_a_radius_too_small_makes_everything_noise_and_says_so(self, blobs, caplog):
        with caplog.at_level(logging.INFO):
            res = cluster_dbscan(blobs["data"], scale=False, eps=1e-6, min_pts=4)
        assert res["design"]["n_clusters"] == 0
        assert res["design"]["n_noise"] == len(blobs["data"].index)
        assert res["clusters"].empty
        assert "every one of them is noise" in caplog.text

    def test_a_radius_has_to_be_positive(self, blobs):
        with pytest.raises(SaValueError, match="`eps`"):
            cluster_dbscan(blobs["data"], eps=0)

    def test_no_neighbourhood_can_hold_more_points_than_there_are(self, blobs):
        with pytest.raises(SaValueError, match="No neighbourhood can hold"):
            cluster_dbscan(blobs["data"], scale=False, min_pts=len(blobs["data"].index) + 1)

    def test_a_derived_k_is_said_out_loud_and_the_two_thresholds_follow_it(self, blobs, caplog):
        with caplog.at_level(logging.INFO):
            res = cluster_snn(blobs["data"], scale=False)
        k = res["parameters"]["k"]
        assert "Using k =" in caplog.text
        assert res["parameters"]["eps"] == max(1, k // 2)
        assert res["parameters"]["min_pts"] == max(2, k // 2)

    def test_a_point_is_not_its_own_neighbour(self, blobs):
        with pytest.raises(SaValueError, match="is not its own neighbour"):
            cluster_snn(blobs["data"], scale=False, k=len(blobs["data"].index))

    def test_two_points_cannot_share_more_neighbours_than_they_keep(self, blobs):
        with pytest.raises(SaValueError, match="cannot share more neighbours"):
            cluster_snn(blobs["data"], scale=False, k=5, eps=6)

    def test_asking_for_more_overlap_than_exists_makes_everything_noise(self, blobs, caplog):
        with caplog.at_level(logging.INFO):
            res = cluster_snn(blobs["data"], scale=False, k=5, eps=5, min_pts=20)
        assert res["design"]["n_clusters"] == 0
        assert "asks for less overlap" in caplog.text

    def test_the_graph_is_what_snn_keeps_instead_of_an_engine_object(self, blobs):
        res = cluster_snn(blobs["data"], scale=False, k=8)
        graph = res.fit
        assert graph.k == 8
        assert len(graph.neighbours) == len(res["points"])
        assert all(len(row) <= 8 for row in graph.neighbours)
        assert all(index not in row for index, row in enumerate(graph.neighbours))
        assert graph.core.sum() > 0

    def test_the_engine_says_the_graph_was_not_a_third_party_one(self, blobs):
        res = cluster_snn(blobs["data"], scale=False, k=8)
        assert res["engine"]["overridden"]


class TestSilhouette:
    """The one number that compares across the four."""

    def test_a_clean_separation_scores_near_one(self, blobs):
        res = cluster_kmeans(blobs["pure"], n_clust=2, scale=False, seed=1)
        assert res["assignments"]["silhouette"].mean() > 0.5

    def test_asking_for_groups_that_are_not_there_shows_up_in_the_width(self, blobs):
        two = cluster_kmeans(blobs["pure"], n_clust=2, scale=False, seed=1)
        many = cluster_kmeans(blobs["pure"], n_clust=6, scale=False, seed=1)
        assert many["assignments"]["silhouette"].mean() < two["assignments"]["silhouette"].mean()

    def test_a_single_cluster_has_no_other_cluster_to_be_far_from(self, blobs):
        res = cluster_dbscan(blobs["data"], scale=False, eps=200.0, min_pts=2)
        assert res["design"]["n_clusters"] == 1
        assert res["assignments"]["silhouette"].isna().all()
        assert res["clusters"]["silhouette"].isna().all()

    def test_the_cluster_table_holds_the_mean_of_its_members(self, blobs):
        res = cluster_kmeans(blobs["data"], n_clust=3, scale=False, seed=1)
        table = res["assignments"]
        for _, row in res["clusters"].iterrows():
            members = table.loc[table["cluster"] == row["cluster"], "silhouette"]
            assert float(row["silhouette"]) == pytest.approx(members.mean())
