"""Train/test partitions.

R draws these with ``caret::createDataPartition()`` and this port draws them
itself, so **which** rows are chosen cannot be compared across the two languages.
What can be, and is checked here, is the arithmetic: ``ceil(n_k * p)`` from each
stratum, a stratum of one kept for training, every row of a unit on one side, and
the row proportion that comes out reported rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statassist import split_data
from statassist.core.errors import SaValueError, SaWarning


def frame(n: int = 40, levels: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    """A wide frame with a balanced label and a numeric column."""
    return pd.DataFrame(
        {
            "label": [levels[i % len(levels)] for i in range(n)],
            "value": np.arange(n, dtype=float),
        }
    )


def repeated(n_units: int = 20, per_unit: int = 3) -> pd.DataFrame:
    """One subject per unit, measured ``per_unit`` times, in two arms."""
    return pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(n_units) for _ in range(per_unit)],
            "arm": [
                "control" if i < n_units // 2 else "treated"
                for i in range(n_units)
                for _ in range(per_unit)
            ],
            "value": np.arange(n_units * per_unit, dtype=float),
        }
    )


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_the_ones_the_result_promises() -> None:
    split = split_data(frame(), stratified="label", seed=1)
    assert list(split) == [
        "full_data",
        "datasets",
        "train_idx",
        "design",
        "parameters",
        "metadata",
    ]


def test_one_repeat_still_comes_back_as_a_mapping_of_one() -> None:
    split = split_data(frame(), seed=1)
    assert list(split.datasets) == ["Resample1"]


def test_the_repeats_are_named_from_one_upwards() -> None:
    split = split_data(frame(), times=3, seed=1)
    assert list(split.datasets) == ["Resample1", "Resample2", "Resample3"]
    assert list(split.train_idx) == list(split.datasets)


def test_each_repeat_holds_the_two_frames_and_the_rows_they_came_from() -> None:
    dataset = split_data(frame(), seed=1).datasets["Resample1"]
    assert list(dataset) == ["train_data", "test_data", "train_rows", "test_rows"]


def test_the_two_halves_partition_the_rows() -> None:
    data = frame()
    dataset = split_data(data, stratified="label", seed=1).datasets["Resample1"]
    train = dataset["train_rows"].tolist()
    test = dataset["test_rows"].tolist()
    assert sorted(train + test) == list(range(len(data.index)))
    assert set(train) & set(test) == set()


def test_the_row_positions_are_zero_based() -> None:
    """R stores one-based indices here; these index the frame that was passed in."""
    split = split_data(frame(n=4), p_train=0.5, seed=1)
    positions = np.concatenate(
        [split.datasets["Resample1"]["train_rows"], split.datasets["Resample1"]["test_rows"]]
    )
    assert positions.min() == 0
    assert positions.max() == 3


def test_the_frames_are_reindexed_from_zero_and_the_positions_kept_beside_them() -> None:
    dataset = split_data(frame(), seed=1).datasets["Resample1"]
    assert dataset["train_data"].index.tolist() == list(range(len(dataset["train_rows"])))


def test_the_input_is_held_once_rather_than_once_per_repeat() -> None:
    data = frame()
    split = split_data(data, times=3, seed=1)
    assert split.full_data is data


# --------------------------------------------------------------------------- #
# Stratification, by count
# --------------------------------------------------------------------------- #


def test_each_stratum_gives_up_the_ceiling_of_its_share() -> None:
    train = split_data(frame(n=40), stratified="label", p_train=0.75, seed=1).datasets["Resample1"][
        "train_data"
    ]
    assert [int((train["label"] == level).sum()) for level in ("a", "b")] == [15, 15]


def test_an_uneven_stratum_rounds_its_share_up() -> None:
    data = pd.DataFrame({"label": ["a"] * 7 + ["b"] * 3, "value": np.arange(10.0)})
    train = split_data(data, stratified="label", p_train=0.5, seed=1).datasets["Resample1"][
        "train_data"
    ]
    assert [int((train["label"] == level).sum()) for level in ("a", "b")] == [
        math.ceil(7 * 0.5),
        math.ceil(3 * 0.5),
    ]


def test_the_unit_count_per_stratum_is_reported_for_discrete_strata() -> None:
    split = split_data(frame(n=40), stratified="label", seed=1)
    assert split.design["strata_n"] == {"a": 20, "b": 20}


def test_a_numeric_stratifier_reports_no_counts_because_it_was_binned() -> None:
    split = split_data(frame(), stratified="value", seed=1)
    assert split.design["strata_n"] is None


def test_a_numeric_stratifier_is_cut_into_bins_rather_than_matched() -> None:
    """Up to five quantile bins, so both halves span the range of the outcome."""
    data = pd.DataFrame({"value": np.arange(100.0)})
    dataset = split_data(data, stratified="value", p_train=0.5, seed=1).datasets["Resample1"]
    train = dataset["train_data"]["value"]
    assert train.min() < 25
    assert train.max() > 75


def test_a_constant_numeric_stratifier_defines_no_strata() -> None:
    data = pd.DataFrame({"value": np.ones(10)})
    with pytest.raises(SaValueError, match="numeric and constant"):
        split_data(data, stratified="value", seed=1)


def test_a_stratum_of_one_unit_goes_to_training_and_says_so() -> None:
    data = pd.DataFrame({"label": ["a"] * 9 + ["rare"], "value": np.arange(10.0)})
    with pytest.warns(SaWarning, match="single unit"):
        split = split_data(data, stratified="label", seed=1)
    train = split.datasets["Resample1"]["train_data"]
    assert int((train["label"] == "rare").sum()) == 1


def test_no_stratifier_draws_a_simple_random_sample() -> None:
    split = split_data(frame(n=40), p_train=0.75, seed=1)
    assert split.design["stratified"] is None
    assert split.datasets["Resample1"]["train_rows"].size == 30


def test_a_stratifier_given_as_a_vector_is_labelled_as_one() -> None:
    data = frame()
    split = split_data(data, stratified=list(data["label"]), seed=1)
    assert split.design["stratified"] == "<vector>"


# --------------------------------------------------------------------------- #
# Sampling units
# --------------------------------------------------------------------------- #


def test_no_subject_appears_on_both_sides_of_a_unit_split() -> None:
    dataset = split_data(repeated(), stratified="arm", id="subject", seed=1).datasets["Resample1"]
    shared = set(dataset["train_data"]["subject"]) & set(dataset["test_data"]["subject"])
    assert shared == set()


def test_every_row_of_a_chosen_unit_comes_with_it() -> None:
    data = repeated(n_units=20, per_unit=3)
    dataset = split_data(data, id="subject", seed=1).datasets["Resample1"]
    counts = dataset["train_data"]["subject"].value_counts()
    assert set(counts.tolist()) == {3}


def test_p_train_is_a_proportion_of_units_and_the_row_share_is_reported() -> None:
    split = split_data(repeated(), stratified="arm", id="subject", p_train=0.75, seed=1)
    assert split.design["n_units"] == 20
    assert split.design["n_rows"] == 60
    # Two strata of ten units, each giving up ceil(7.5) = 8, so 16 units of 20.
    assert split.parameters["achieved_p"]["Resample1"] == pytest.approx(48 / 60)


def test_units_of_unequal_size_make_the_row_share_differ_from_p_train() -> None:
    data = pd.DataFrame(
        {
            "subject": ["big"] * 10 + ["a", "b", "c", "d"],
            "value": np.arange(14.0),
        }
    )
    split = split_data(data, id="subject", p_train=0.5, seed=1)
    reached = split.parameters["achieved_p"]["Resample1"]
    assert split.design["n_units"] == 5
    assert reached != 0.5


def test_a_stratifier_that_is_not_constant_within_a_unit_is_refused() -> None:
    data = repeated(n_units=4, per_unit=2)
    data.loc[0, "arm"] = "treated"
    with pytest.raises(SaValueError, match="must be constant within each `id`"):
        split_data(data, stratified="arm", id="subject", seed=1)


def test_fewer_than_two_units_cannot_be_split() -> None:
    data = pd.DataFrame({"subject": ["s1", "s1"], "value": [1.0, 2.0]})
    with pytest.raises(SaValueError, match="at least 2 sampling units"):
        split_data(data, id="subject", seed=1)


# --------------------------------------------------------------------------- #
# Reproducibility and refusals
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_partition() -> None:
    data = frame()
    first = split_data(data, stratified="label", seed=7).train_idx["Resample1"]
    second = split_data(data, stratified="label", seed=7).train_idx["Resample1"]
    assert first.tolist() == second.tolist()


def test_a_different_seed_gives_a_different_partition() -> None:
    data = frame()
    first = split_data(data, stratified="label", seed=1).train_idx["Resample1"]
    second = split_data(data, stratified="label", seed=2).train_idx["Resample1"]
    assert first.tolist() != second.tolist()


def test_the_repeats_of_one_call_are_independent_draws() -> None:
    split = split_data(frame(), stratified="label", times=3, seed=1)
    drawn = {tuple(rows.tolist()) for rows in split.train_idx.values()}
    assert len(drawn) == 3


def test_a_p_train_that_empties_the_test_set_is_refused() -> None:
    data = pd.DataFrame({"label": ["a", "b"], "value": [1.0, 2.0]})
    # Both strata hold one unit, so both are kept for training and warned about,
    # and between them they leave nothing to test on.
    with (
        pytest.warns(SaWarning, match="single unit"),
        pytest.raises(SaValueError, match="leaves the test set empty"),
    ):
        split_data(data, stratified="label", p_train=0.75, seed=1)


def test_p_train_at_the_ends_is_refused() -> None:
    for bad in (0, 1):
        with pytest.raises(SaValueError, match="`p_train` must be in"):
            split_data(frame(), p_train=bad, seed=1)


def test_zero_rows_is_refused() -> None:
    with pytest.raises(SaValueError, match="zero rows"):
        split_data(pd.DataFrame({"value": []}), seed=1)


def test_something_that_is_not_a_frame_is_refused() -> None:
    with pytest.raises(SaValueError, match="must be a data.frame or a matrix"):
        split_data([1, 2, 3], seed=1)


def test_a_matrix_is_read_as_a_frame() -> None:
    split = split_data(np.arange(20.0).reshape(10, 2), seed=1)
    assert split.design["n_rows"] == 10


def test_the_summary_says_what_the_split_was_made_on() -> None:
    text = repr(split_data(repeated(), stratified="arm", id="subject", seed=1))
    assert "20 unit(s) of `subject`" in text
    assert "stratify : arm" in text
    assert "control 10, treated 10" in text
    assert "Resample1" in text
