"""Control-versus-treatments data with a known answer in three tables.

The three tables are the point: the omnibus stage and the post-hoc stage fail
separately, so the answer has to be readable on the feature axis, the level axis
and the contrast axis. The direction of the contrast axis is the claim most worth
pinning, since a sign error there would score every post-hoc table backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from statassist import simulate_multiple_groups
from statassist.core.errors import SaValueError

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_the_four_the_result_promises() -> None:
    sim = simulate_multiple_groups(n_feats=10, seed=1)
    assert list(sim) == ["args", "truth", "truth_group", "truth_contrast"]


def test_args_is_named_after_the_comparison_that_consumes_it() -> None:
    sim = simulate_multiple_groups(n_feats=10, seed=1)
    assert list(sim.args) == ["data", "feats", "group", "group_lv", "input_scale"]


def test_the_three_tables_hold_their_columns_in_order() -> None:
    sim = simulate_multiple_groups(n_feats=10, seed=1)
    assert sim.truth.columns.tolist() == [
        "features",
        "pattern",
        "direction",
        "extreme_level",
        "extreme_tied",
        "log2fc",
        "baseline",
        "sd_subject",
    ]
    assert sim.truth_group.columns.tolist() == [
        "features",
        "group",
        "is_ref",
        "delta",
        "center",
        "sd",
        "n",
    ]
    assert sim.truth_contrast.columns.tolist() == [
        "features",
        "contrast",
        "group1",
        "group2",
        "delta",
        "is_diff",
    ]


def test_the_default_levels_are_the_control_and_one_per_treatment_size() -> None:
    sim = simulate_multiple_groups(n_feats=5, n_treat=[30, 20], seed=1)
    assert sim.args["group_lv"] == ["control", "treat_1", "treat_2"]


def test_the_group_sizes_are_the_ones_asked_for() -> None:
    sim = simulate_multiple_groups(n_feats=5, n_control=40, n_treat=[30, 25, 20], seed=1)
    counts = [sim.args["group"].count(level) for level in sim.args["group_lv"]]
    assert counts == [40, 30, 25, 20]
    assert len(sim.args["data"].index) == 115


def test_labels_alone_say_how_many_groups_there_are() -> None:
    sim = simulate_multiple_groups(
        n_feats=5, n_treat=25, group_lv=["dmso", "low", "mid", "high"], seed=1
    )
    assert [sim.args["group"].count(level) for level in sim.args["group_lv"]] == [50, 25, 25, 25]


def test_a_default_size_is_spread_over_the_labels_that_were_given() -> None:
    sim = simulate_multiple_groups(
        n_feats=5, group_lv=["a", "b", "c", "d", "e"], n_control=10, seed=1
    )
    assert [sim.args["group"].count(level) for level in sim.args["group_lv"]] == [10] + [50] * 4


def test_the_long_table_is_one_row_per_feature_and_level() -> None:
    sim = simulate_multiple_groups(n_feats=6, n_treat=[10, 10], seed=1)
    assert len(sim.truth_group.index) == 6 * 3
    assert sim.truth_group["group"].tolist()[:3] == ["control", "treat_1", "treat_2"]
    assert sim.truth_group["is_ref"].tolist()[:3] == [True, False, False]


def test_the_contrast_table_is_one_row_per_feature_and_pair() -> None:
    sim = simulate_multiple_groups(n_feats=4, n_treat=[10, 10], seed=1)
    assert len(sim.truth_contrast.index) == 4 * 3
    assert sim.truth_contrast["contrast"].tolist()[:3] == [
        "treat_1 - control",
        "treat_2 - control",
        "treat_2 - treat_1",
    ]


def test_the_contrast_delta_reads_group1_minus_group2() -> None:
    """A feature the treatment raised is positive here, as in the post-hoc table."""
    sim = simulate_multiple_groups(n_feats=20, n_treat=[10, 10], pattern_mix={"all": 1}, seed=1)
    rows = sim.truth_contrast
    against_control = rows[rows["group2"] == "control"].merge(
        sim.truth[["features", "direction"]], on="features"
    )
    up = against_control[against_control["direction"] == "up"]
    down = against_control[against_control["direction"] == "down"]
    assert (up["delta"] > 0).all()
    assert (down["delta"] < 0).all()


# --------------------------------------------------------------------------- #
# What was planted
# --------------------------------------------------------------------------- #


def test_the_shapes_are_handed_out_in_the_counts_the_weights_ask_for() -> None:
    for seed in range(3):
        truth = simulate_multiple_groups(n_feats=60, n_up=9, n_down=9, seed=seed).truth
        counts = truth["pattern"].value_counts()
        assert int(counts["all"]) == 6
        assert int(counts["gradient"]) == 6
        assert int(counts["single"]) == 6


def test_a_weight_of_zero_leaves_the_shape_out() -> None:
    truth = simulate_multiple_groups(
        n_feats=30, n_up=5, n_down=5, pattern_mix={"all": 1, "single": 1}, seed=1
    ).truth
    assert set(truth["pattern"]) == {"none", "all", "single"}


def test_the_planted_counts_default_to_a_share_of_the_features() -> None:
    truth = simulate_multiple_groups(n_feats=40, seed=1).truth
    assert int((truth["direction"] == "up").sum()) == 6
    assert int((truth["direction"] == "down").sum()) == 6


def test_the_control_carries_no_effect() -> None:
    sim = simulate_multiple_groups(n_feats=20, seed=1)
    control = sim.truth_group[sim.truth_group["group"] == "control"]
    assert (control["delta"] == 0).all()


def test_all_moves_every_treatment_group_by_the_same_amount() -> None:
    sim = simulate_multiple_groups(n_feats=20, n_treat=[10, 10, 10], pattern_mix={"all": 1}, seed=1)
    planted = sim.truth["pattern"] == "all"
    wide = sim.truth_group.pivot(index="features", columns="group", values="delta")
    treated = wide[["treat_1", "treat_2", "treat_3"]]
    moved = treated.loc[sim.truth.loc[planted, "features"]]
    assert (moved.nunique(axis=1) == 1).all()


def test_gradient_reaches_the_full_effect_only_at_the_last_level() -> None:
    sim = simulate_multiple_groups(
        n_feats=20, n_treat=[10, 10, 10], pattern_mix={"gradient": 1}, seed=1
    )
    wide = sim.truth_group.pivot(index="features", columns="group", values="delta")
    planted = sim.truth.loc[sim.truth["pattern"] == "gradient", "features"]
    rows = wide.loc[planted]
    assert np.allclose(rows["treat_1"] * 3, rows["treat_3"])
    assert np.allclose(rows["treat_2"] * 3, rows["treat_3"] * 2)


def test_single_leaves_every_other_level_at_exactly_zero() -> None:
    sim = simulate_multiple_groups(
        n_feats=20, n_treat=[10, 10, 10], pattern_mix={"single": 1}, seed=1
    )
    wide = sim.truth_group.pivot(index="features", columns="group", values="delta")
    planted = sim.truth.loc[sim.truth["pattern"] == "single", "features"]
    moved = (wide.loc[planted, ["treat_1", "treat_2", "treat_3"]] != 0).sum(axis=1)
    assert (moved == 1).all()


def test_a_tie_is_reported_rather_than_broken_silently() -> None:
    sim = simulate_multiple_groups(n_feats=20, n_treat=[10, 10], pattern_mix={"all": 1}, seed=1)
    planted = sim.truth["direction"] != "none"
    assert sim.truth.loc[planted, "extreme_tied"].all()
    assert sim.truth.loc[planted, "extreme_level"].tolist() == ["treat_1"] * int(planted.sum())


def test_an_unplanted_feature_has_no_extreme_level_and_counts_as_tied() -> None:
    sim = simulate_multiple_groups(n_feats=20, seed=1)
    null = sim.truth["direction"] == "none"
    assert sim.truth.loc[null, "extreme_level"].isna().all()
    assert sim.truth.loc[null, "extreme_tied"].all()
    assert (sim.truth.loc[null, "log2fc"] == 0).all()


def test_the_gradient_names_the_last_level_as_the_extreme_one() -> None:
    sim = simulate_multiple_groups(
        n_feats=20, n_treat=[10, 10, 10], pattern_mix={"gradient": 1}, seed=1
    )
    planted = sim.truth["direction"] != "none"
    assert (sim.truth.loc[planted, "extreme_level"] == "treat_3").all()
    assert not sim.truth.loc[planted, "extreme_tied"].any()


def test_is_diff_marks_exactly_the_non_zero_contrasts() -> None:
    sim = simulate_multiple_groups(n_feats=20, seed=1)
    rows = sim.truth_contrast
    assert (rows["is_diff"] == (rows["delta"] != 0)).all()


def test_the_planted_effect_is_recovered_from_the_data() -> None:
    sim = simulate_multiple_groups(
        n_feats=60, n_control=300, n_treat=[300, 300], control_sd=(1, 1.2), seed=1
    )
    data = sim.args["data"].to_numpy()
    group = np.asarray(sim.args["group"])
    observed = data[group == "treat_1"].mean(axis=0) - data[group == "control"].mean(axis=0)
    wide = sim.truth_group.pivot(index="features", columns="group", values="delta")
    planted = wide.loc[sim.args["feats"], "treat_1"].to_numpy()
    assert np.abs(observed - planted).mean() < 0.15


# --------------------------------------------------------------------------- #
# Repeated conditions
# --------------------------------------------------------------------------- #


def test_a_paired_design_adds_the_two_keys_the_within_subject_tests_need() -> None:
    sim = simulate_multiple_groups(n_feats=10, n_control=12, n_treat=[12, 12], paired=True, seed=1)
    assert list(sim.args) == ["data", "feats", "group", "group_lv", "id", "paired", "input_scale"]
    assert sim.args["paired"] is True


def test_an_unpaired_design_leaves_the_keys_out_rather_than_setting_them_false() -> None:
    sim = simulate_multiple_groups(n_feats=10, seed=1)
    assert "id" not in sim.args
    assert "paired" not in sim.args


def test_every_subject_appears_under_every_condition() -> None:
    sim = simulate_multiple_groups(n_feats=5, n_control=12, n_treat=[12, 12], paired=True, seed=1)
    ids = np.asarray(sim.args["id"])
    group = np.asarray(sim.args["group"])
    for level in sim.args["group_lv"]:
        assert sorted(ids[group == level]) == sorted(set(ids))


def test_the_subject_spread_is_reported_only_when_it_was_used() -> None:
    paired = simulate_multiple_groups(
        n_feats=10, n_control=12, n_treat=[12, 12], paired=True, subject_sd=(2, 3), seed=1
    )
    assert paired.truth["sd_subject"].between(2, 3).all()
    assert simulate_multiple_groups(n_feats=10, seed=1).truth["sd_subject"].isna().all()


def test_a_subject_offset_is_shared_across_the_conditions() -> None:
    """That shared offset is what a within-subject test exists to remove."""
    sim = simulate_multiple_groups(
        n_feats=1,
        n_control=200,
        n_treat=[200, 200],
        n_up=0,
        n_down=0,
        control_sd=(0.5, 0.5),
        treat_sd=(0.5, 0.5),
        subject_sd=(4, 4),
        paired=True,
        seed=1,
    )
    values = sim.args["data"]["prot_1"].to_numpy()
    group = np.asarray(sim.args["group"])
    control = values[group == "control"]
    treat = values[group == "treat_1"]
    assert np.corrcoef(control, treat)[0, 1] > 0.9


def test_unequal_group_sizes_are_refused_for_a_paired_design() -> None:
    with pytest.raises(SaValueError, match="every group holds the same number"):
        simulate_multiple_groups(n_feats=5, n_control=12, n_treat=[10, 12], paired=True)


# --------------------------------------------------------------------------- #
# Reproducibility and refusals
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_data() -> None:
    first = simulate_multiple_groups(n_feats=8, seed=5)
    second = simulate_multiple_groups(n_feats=8, seed=5)
    assert first.args["data"].equals(second.args["data"])
    assert first.truth_contrast.equals(second.truth_contrast)


def test_a_different_seed_gives_different_data() -> None:
    first = simulate_multiple_groups(n_feats=8, seed=5)
    second = simulate_multiple_groups(n_feats=8, seed=6)
    assert not first.args["data"].equals(second.args["data"])


def test_one_treatment_group_is_refused_as_a_two_group_design() -> None:
    with pytest.raises(SaValueError, match="at least two of them"):
        simulate_multiple_groups(n_feats=5, n_treat=[30])


def test_labels_and_sizes_that_count_differently_are_refused() -> None:
    with pytest.raises(SaValueError, match="but `n_treat` gives 3 size"):
        simulate_multiple_groups(n_feats=5, n_treat=[10, 10, 10], group_lv=["a", "b", "c"])


def test_fewer_than_three_labels_is_refused() -> None:
    with pytest.raises(SaValueError, match="at least three distinct"):
        simulate_multiple_groups(n_feats=5, group_lv=["a", "b"])


def test_more_planted_features_than_features_is_refused() -> None:
    with pytest.raises(SaValueError, match="more features than the 10"):
        simulate_multiple_groups(n_feats=10, n_up=6, n_down=5)


def test_an_unknown_shape_is_refused() -> None:
    with pytest.raises(SaValueError, match="names unknown shape"):
        simulate_multiple_groups(n_feats=10, pattern_mix={"all": 1, "sigmoid": 1})


def test_an_empty_feature_prefix_is_refused() -> None:
    with pytest.raises(SaValueError, match="`feat_prefix` must be a single non-empty string"):
        simulate_multiple_groups(n_feats=5, feat_prefix="")


def test_the_feature_prefix_names_the_columns() -> None:
    sim = simulate_multiple_groups(n_feats=3, feat_prefix="gene", seed=1)
    assert sim.args["feats"] == ["gene_1", "gene_2", "gene_3"]
