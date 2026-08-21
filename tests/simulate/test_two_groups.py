"""Two-group expression data with a known answer.

The draw is not R's, so the numbers are not compared with R's. Three other things
are: the contract of what comes back, the statistical properties of what was
planted, and reproducibility within this package.
"""

from __future__ import annotations

import numpy as np
import pytest

from statassist import simulate_two_groups
from statassist.core.errors import SaValueError

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_args_and_truth() -> None:
    assert list(simulate_two_groups(seed=1)) == ["args", "truth"]


def test_args_is_named_after_the_comparison_that_consumes_it() -> None:
    sim = simulate_two_groups(seed=1)
    assert list(sim.args) == ["data", "feats", "group", "group_lv", "input_scale"]
    assert sim.args["input_scale"] == "log2"


def test_the_slots_are_readable_both_ways() -> None:
    sim = simulate_two_groups(seed=1)
    assert sim.truth is sim["truth"]


def test_truth_holds_the_six_columns_in_order() -> None:
    truth = simulate_two_groups(seed=1).truth
    assert truth.columns.tolist() == [
        "features",
        "direction",
        "log2fc",
        "baseline",
        "sd_case",
        "sd_control",
    ]


def test_truth_is_aligned_with_feats() -> None:
    sim = simulate_two_groups(n_feats=20, n_up=3, n_down=3, seed=1)
    assert sim.truth["features"].tolist() == sim.args["feats"]


def test_the_features_are_the_columns_of_data_in_order() -> None:
    sim = simulate_two_groups(n_feats=20, n_up=3, n_down=3, seed=1)
    assert sim.args["data"].columns.tolist() == sim.args["feats"]
    assert sim.args["feats"][:2] == ["gene_1", "gene_2"]


def test_the_rows_are_the_control_group_first() -> None:
    """`group_lv` names the control first, so the rows have to follow."""
    sim = simulate_two_groups(n_case=7, n_control=5, seed=1)
    assert sim.args["group"] == ["control"] * 5 + ["case"] * 7
    assert len(sim.args["data"].index) == 12


def test_the_group_labels_are_carried_through_as_given() -> None:
    sim = simulate_two_groups(group_lv=["untreated", "treated"], seed=1)
    assert sim.args["group_lv"] == ["untreated", "treated"]
    assert set(sim.args["group"]) == {"untreated", "treated"}


# --------------------------------------------------------------------------- #
# What was planted
# --------------------------------------------------------------------------- #


def test_the_planted_counts_are_a_function_of_the_arguments() -> None:
    for seed in range(5):
        truth = simulate_two_groups(n_feats=50, n_up=7, n_down=4, seed=seed).truth
        counts = truth["direction"].value_counts()
        assert int(counts["up"]) == 7
        assert int(counts["down"]) == 4
        assert int(counts["none"]) == 39


def test_an_unplanted_feature_is_null_in_the_strict_sense() -> None:
    truth = simulate_two_groups(seed=1).truth
    assert (truth.loc[truth["direction"] == "none", "log2fc"] == 0).all()


def test_the_planted_effect_points_the_way_the_direction_says() -> None:
    truth = simulate_two_groups(seed=1).truth
    assert (truth.loc[truth["direction"] == "up", "log2fc"] > 0).all()
    assert (truth.loc[truth["direction"] == "down", "log2fc"] < 0).all()


def test_the_planted_effect_lies_in_the_range_it_was_drawn_from() -> None:
    truth = simulate_two_groups(deg_log2fc=(1.5, 2.0), seed=1).truth
    planted = truth.loc[truth["direction"] != "none", "log2fc"].abs()
    assert planted.between(1.5, 2.0).all()


def test_the_baseline_lies_in_the_range_it_was_drawn_from() -> None:
    truth = simulate_two_groups(expr_range=(4, 6), seed=1).truth
    assert truth["baseline"].between(4, 6).all()


def test_the_two_groups_get_their_own_spreads() -> None:
    truth = simulate_two_groups(case_sd=(3, 4), control_sd=(1, 2), seed=1).truth
    assert truth["sd_case"].between(3, 4).all()
    assert truth["sd_control"].between(1, 2).all()


def test_the_planted_difference_is_recovered_from_the_data() -> None:
    """The observed difference of means estimates `log2fc`, which is the whole point."""
    sim = simulate_two_groups(n_feats=200, n_case=400, n_control=400, seed=1)
    data = sim.args["data"].to_numpy()
    group = np.asarray(sim.args["group"])
    observed = data[group == "case"].mean(axis=0) - data[group == "control"].mean(axis=0)
    assert np.corrcoef(observed, sim.truth["log2fc"])[0, 1] > 0.9
    assert np.abs(observed - sim.truth["log2fc"]).mean() < 0.3


def test_the_baseline_is_shared_so_a_null_feature_differs_by_nothing() -> None:
    sim = simulate_two_groups(n_feats=200, n_case=400, n_control=400, seed=1)
    data = sim.args["data"].to_numpy()
    group = np.asarray(sim.args["group"])
    observed = data[group == "case"].mean(axis=0) - data[group == "control"].mean(axis=0)
    null = sim.truth["direction"].to_numpy() == "none"
    assert abs(observed[null].mean()) < 0.1


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_data() -> None:
    first = simulate_two_groups(n_feats=10, n_up=2, n_down=2, seed=3)
    second = simulate_two_groups(n_feats=10, n_up=2, n_down=2, seed=3)
    assert first.args["data"].equals(second.args["data"])
    assert first.truth.equals(second.truth)


def test_a_different_seed_gives_different_data() -> None:
    first = simulate_two_groups(n_feats=10, n_up=2, n_down=2, seed=3)
    second = simulate_two_groups(n_feats=10, n_up=2, n_down=2, seed=4)
    assert not first.args["data"].equals(second.args["data"])


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_more_planted_features_than_features_is_refused() -> None:
    with pytest.raises(SaValueError, match="more features than the 10"):
        simulate_two_groups(n_feats=10, n_up=6, n_down=5)


def test_planting_nothing_is_allowed() -> None:
    truth = simulate_two_groups(n_feats=10, n_up=0, n_down=0, seed=1).truth
    assert (truth["direction"] == "none").all()
    assert (truth["log2fc"] == 0).all()


def test_a_group_of_one_is_refused() -> None:
    with pytest.raises(SaValueError, match="`n_case` must be in"):
        simulate_two_groups(n_case=1)


def test_two_labels_that_are_the_same_are_refused() -> None:
    with pytest.raises(SaValueError, match="two distinct non-missing group labels"):
        simulate_two_groups(group_lv=["case", "case"])


def test_a_reversed_range_is_refused() -> None:
    with pytest.raises(SaValueError, match="`expr_range` must be increasing"):
        simulate_two_groups(expr_range=(12, 2))


def test_a_negative_spread_is_refused() -> None:
    with pytest.raises(SaValueError, match="`case_sd` must not go below 0"):
        simulate_two_groups(case_sd=(-1, 2))
