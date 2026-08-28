"""Placing the same points in fewer coordinates, three ways.

The three functions share their whole input side with the four ``cluster_*`` ones,
so what is checked here is what is specific to a reduction: that ``scores`` is
aligned with ``points`` whichever margin was embedded, that the rotation's two extra
tables say what they claim about the rotation, and that an embedding does not carry
them at all.

What cannot be checked is a coordinate. A principal component is signed
arbitrarily and an embedding has no fixed orientation at all, so every assertion
below is about something that survives those freedoms: a share of the variance, a
distance between two points, a count, or a reconstruction.
"""

from __future__ import annotations

import importlib.util
import json
import logging

import numpy as np
import pandas as pd
import pytest

from statassist import perform_pca, perform_tsne, perform_umap, simulate_two_groups
from statassist.core import SaValueError
from statassist.core.contracts import reduction_variance_columns
from statassist.reduce import COMPONENT_PREFIX, MAX_TSNE_DIM, UMAP_METRICS
from statassist.reduce._shared import EMBEDDING_SCALES

#: A reduction is only worth reading when the features outnumber the components,
#: so the tables here are wide rather than square.
_N_FEATS = 24
_N_PER_GROUP = 15

#: How much of a share of the variance may be lost to floating point before a
#: total is called something other than a whole.
_SHARE_TOL = 1e-8


@pytest.fixture(scope="module")
def planted():
    """A wide simulation with a grouping the reduction is never shown."""
    return simulate_two_groups(
        n_feats=_N_FEATS,
        n_case=_N_PER_GROUP,
        n_control=_N_PER_GROUP,
        n_up=6,
        n_down=6,
        seed=5,
    )


@pytest.fixture(scope="module")
def rotated(planted):
    return perform_pca(planted.args["data"])


@pytest.fixture(scope="module")
def embedded(planted):
    return perform_tsne(planted.args["data"], seed=7)


@pytest.fixture(scope="module")
def both(rotated, embedded):
    """The two methods that need no optional engine, by the name each reports."""
    return {"pca": rotated, "tsne": embedded}


class TestSharedContract:
    """What holds for a reduction whichever method produced it."""

    def test_the_analysis_names_the_method_that_ran(self, both):
        for name, res in both.items():
            assert res["analysis"] == name

    def test_every_table_repeats_the_points_in_one_order(self, both):
        for res in both.values():
            assert res["scores"]["points"].tolist() == res["points"]
            assert len(res["points"]) == res["design"]["n_used"]

    def test_the_points_are_the_samples_unless_asked_otherwise(self, both, planted):
        index = [str(value) for value in planted.args["data"].index]
        for res in both.values():
            assert res["design"]["point_type"] == "sample"
            assert res["points"] == index

    def test_the_coordinates_are_finite_and_numbered_from_one(self, both):
        for res in both.values():
            coords = res["scores"].drop(columns="points")
            assert np.isfinite(coords.to_numpy(dtype=float)).all()
            assert list(coords.columns)[0].endswith("1")

    def test_design_describes_the_input_and_not_the_output(self, both, planted):
        for res in both.values():
            design = res["design"]
            assert design["n_samples"] == len(planted.args["data"].index)
            assert design["n_feats"] == _N_FEATS
            assert design["feats"] == list(planted.args["data"].columns)
            assert design["n_dropped"] == 0
            assert design["dropped_feats"] == []

    def test_the_public_slots_survive_a_json_round_trip(self, both):
        for res in both.values():
            payload = {
                name: (value.to_dict(orient="list") if isinstance(value, pd.DataFrame) else value)
                for name, value in res.items()
            }
            assert json.loads(json.dumps(payload))["analysis"] == res["analysis"]

    def test_the_engine_object_is_reachable_but_is_not_a_slot(self, both):
        for res in both.values():
            assert res.fit is not None
            assert "fit" not in res

    def test_repr_summarises_rather_than_printing_the_coordinates(self, both):
        for res in both.values():
            text = repr(res)
            assert res["analysis"] in text
            assert "points" in text
            assert str(len(res["points"])) in text


class TestRotation:
    """What only a principal component analysis can be asked."""

    def test_the_shares_of_the_variance_are_a_whole(self, rotated):
        variance = rotated["variance"]
        assert list(variance.columns) == reduction_variance_columns()
        assert variance["prop_var"].sum() == pytest.approx(100.0, abs=_SHARE_TOL)
        assert variance["cum_var"].iloc[-1] == pytest.approx(100.0, abs=_SHARE_TOL)

    def test_the_components_are_ordered_by_the_variance_they_carry(self, rotated):
        sdev = rotated["variance"]["sdev"].to_numpy(dtype=float)
        assert (np.diff(sdev) <= 1e-9).all()

    def test_the_component_names_number_from_one_and_match_the_scores(self, rotated):
        names = rotated["variance"]["component"].tolist()
        assert names[0] == f"{COMPONENT_PREFIX}1"
        assert names == [column for column in rotated["scores"].columns if column != "points"]

    def test_the_loadings_describe_the_margin_that_was_not_embedded(self, rotated, planted):
        loadings = rotated["loadings"]
        assert loadings["variables"].tolist() == list(planted.args["data"].columns)
        assert len(loadings.index) == _N_FEATS

    def test_the_loadings_are_directions_of_unit_length(self, rotated):
        directions = rotated["loadings"].drop(columns="variables").to_numpy(dtype=float)
        lengths = np.sqrt((directions**2).sum(axis=0))
        assert lengths == pytest.approx(np.ones(directions.shape[1]))

    def test_the_scores_reconstruct_the_standardised_matrix(self, rotated, planted):
        # The one check that pins the rotation rather than a summary of it, and it
        # holds whichever sign each component happened to be given.
        x = planted.args["data"].to_numpy(dtype=float)
        standardised = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        scores = rotated["scores"].drop(columns="points").to_numpy(dtype=float)
        directions = rotated["loadings"].drop(columns="variables").to_numpy(dtype=float)
        assert scores @ directions.T == pytest.approx(standardised, abs=1e-8)

    def test_the_variance_is_the_spread_of_the_scores(self, rotated):
        scores = rotated["scores"].drop(columns="points").to_numpy(dtype=float)
        assert scores.std(axis=0, ddof=1) == pytest.approx(
            rotated["variance"]["sdev"].to_numpy(dtype=float)
        )

    def test_the_feature_scale_reads_the_same_fit_from_the_other_end(self, planted):
        by_feat = perform_pca(planted.args["data"], embedding_scale=EMBEDDING_SCALES[1])
        by_sample = perform_pca(planted.args["data"])
        assert by_feat["design"]["point_type"] == "feature"
        assert by_feat["points"] == list(planted.args["data"].columns)
        assert len(by_feat["loadings"].index) == 2 * _N_PER_GROUP
        # One decomposition answers on both margins, so the shares of the variance
        # are the same numbers on either scale rather than a second analysis.
        assert by_feat["variance"]["prop_var"].tolist() == pytest.approx(
            by_sample["variance"]["prop_var"].tolist()
        )

    def test_not_scaling_leaves_the_widest_feature_in_charge(self, planted):
        frame = planted.args["data"].copy()
        widest = frame.columns[0]
        frame[widest] = frame[widest] * 1000
        unscaled = perform_pca(frame, scale=False)
        first = unscaled["loadings"].set_index("variables")[f"{COMPONENT_PREFIX}1"]
        assert abs(first.loc[widest]) == pytest.approx(abs(first).max())
        assert perform_pca(frame)["parameters"]["scale"] is True

    def test_a_feature_of_no_variance_is_dropped_only_when_it_would_be_scaled(
        self, planted, caplog
    ):
        frame = planted.args["data"].copy()
        flat = frame.columns[0]
        frame[flat] = 1.0
        with caplog.at_level(logging.INFO):
            scaled = perform_pca(frame)
        assert scaled["design"]["dropped_feats"] == [flat]
        assert scaled["design"]["n_feats"] == _N_FEATS - 1
        assert flat in caplog.text

        kept = perform_pca(frame, scale=False)
        assert kept["design"]["dropped_feats"] == []
        assert kept["variance"]["sdev"].iloc[-1] == pytest.approx(0.0, abs=1e-8)


class TestEmbedding:
    """What an embedding is and is not."""

    def test_an_embedding_carries_no_components(self, embedded):
        assert "variance" not in embedded
        assert "loadings" not in embedded
        assert embedded.get("variance") is None

    def test_the_parameters_hold_the_choices_as_they_were_used(self, embedded):
        params = embedded["parameters"]
        assert params["n_dim"] == 2
        assert params["seed"] == 7
        # Not passed, so this is the value that was derived from the point count.
        assert params["perplexity"] > 0
        assert 3 * params["perplexity"] <= len(embedded["points"]) - 1

    def test_a_seed_makes_the_picture_repeat(self, planted):
        first = perform_tsne(planted.args["data"], seed=3)
        again = perform_tsne(planted.args["data"], seed=3)
        assert first["scores"].drop(columns="points").to_numpy() == pytest.approx(
            again["scores"].drop(columns="points").to_numpy()
        )

    def test_the_planted_groups_end_up_closer_to_themselves_than_to_each_other(
        self, embedded, planted
    ):
        # The only statistical claim an embedding supports here: it was never shown
        # the grouping, so two points of one group ending up closer on average than
        # two points of different groups is its own finding rather than an
        # arrangement of the input. Read across pairs rather than between the two
        # centres, since t-SNE has no reason to place a group compactly and a group
        # drawn as two arcs has a centre that is nowhere near either of them.
        coords = embedded["scores"].drop(columns="points").to_numpy(dtype=float)
        group = np.asarray(planted.args["group"], dtype=object)
        within, between = _pair_distances(coords, group)
        assert np.mean(within) < np.mean(between)

    def test_the_number_of_dimensions_is_the_number_of_columns(self, planted):
        res = perform_tsne(planted.args["data"], n_dim=1, seed=1)
        assert list(res["scores"].columns) == ["points", "tSNE1"]

    def test_more_dimensions_than_the_gradient_has_are_refused(self, planted):
        with pytest.raises(SaValueError, match=str(MAX_TSNE_DIM)):
            perform_tsne(planted.args["data"], n_dim=MAX_TSNE_DIM + 1)

    def test_a_perplexity_the_points_cannot_carry_is_refused(self, planted):
        with pytest.raises(SaValueError):
            perform_tsne(planted.args["data"], perplexity=len(planted.args["data"].index))

    def test_the_feature_scale_really_transposes(self, planted, caplog):
        with caplog.at_level(logging.INFO):
            by_feat = perform_tsne(
                planted.args["data"], embedding_scale=EMBEDDING_SCALES[1], seed=1
            )
        assert by_feat["design"]["point_type"] == "feature"
        assert by_feat["points"] == list(planted.args["data"].columns)
        assert len(by_feat["scores"].index) == _N_FEATS
        # `design` describes the input either way, so its counts do not turn.
        assert by_feat["design"]["n_feats"] == _N_FEATS


class TestOptionalEngine:
    """The one public function whose engine is not installed by default."""

    @pytest.mark.skipif(
        importlib.util.find_spec("umap") is not None, reason="umap-learn is installed"
    )
    def test_a_missing_engine_says_what_to_install(self, planted):
        # An optional extra is only optional if not having it is answered rather
        # than raised, so the message names the extra and the two methods that need
        # nothing beyond the core dependencies.
        with pytest.raises(SaValueError, match="umap-learn"):
            perform_umap(planted.args["data"])

    @pytest.mark.skipif(
        importlib.util.find_spec("umap") is None, reason="umap-learn is not installed"
    )
    def test_an_installed_engine_answers_on_the_same_points(self, planted):
        res = perform_umap(planted.args["data"], scale=True, seed=1)
        assert res["analysis"] == "umap"
        assert res["scores"]["points"].tolist() == res["points"]
        assert "variance" not in res
        assert res["parameters"]["metric"] == UMAP_METRICS[0]

    def test_an_unknown_metric_is_refused_before_the_engine_is_reached(self, planted):
        with pytest.raises(SaValueError, match="metric"):
            perform_umap(planted.args["data"], metric="mahalanobis")


class TestSharedInput:
    """The input side, which the ``cluster_*`` four read through the same helper."""

    def test_an_unknown_scale_is_refused_by_name(self, planted):
        with pytest.raises(SaValueError, match="embedding_scale"):
            perform_pca(planted.args["data"], embedding_scale="rows")

    def test_a_flag_that_is_not_one_is_refused(self, planted):
        with pytest.raises(SaValueError, match="center"):
            perform_pca(planted.args["data"], center=1)

    def test_an_incomplete_row_is_dropped_and_counted(self, planted):
        frame = planted.args["data"].copy()
        frame.iloc[0, 0] = np.nan
        res = perform_pca(frame)
        assert res["design"]["n_dropped"] == 1
        assert res["design"]["n_used"] == len(frame.index) - 1
        assert len(res["points"]) == len(frame.index) - 1

    def test_a_non_numeric_column_is_left_out_with_a_message(self, planted, caplog):
        frame = planted.args["data"].copy()
        frame["group"] = list(planted.args["group"])
        with caplog.at_level(logging.INFO):
            res = perform_pca(frame)
        assert res["design"]["feats"] == list(planted.args["data"].columns)
        assert "group" in caplog.text

    def test_a_table_too_small_to_reduce_is_refused(self):
        with pytest.raises(SaValueError):
            perform_pca(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}).iloc[:1])


def _pair_distances(coords: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pairwise distance, split by whether the pair shares a group."""
    gaps = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    same = group[:, None] == group[None, :]
    upper = np.triu(np.ones_like(gaps, dtype=bool), k=1)
    return gaps[upper & same], gaps[upper & ~same]
