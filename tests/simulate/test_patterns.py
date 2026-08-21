"""Shape weights, the split between them, and the deltas they imply.

Two of the three helpers here deliberately do not draw, so they are checked for
exact values. That is the property worth pinning: how many features take each
shape is a function of the arguments, and a test that only checked the total
would not notice it becoming a multinomial draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from statassist.core.errors import SaInternalError, SaValueError
from statassist.simulate._patterns import (
    allocate,
    pattern_delta,
    pattern_mix,
    pick_up_down,
)

# --------------------------------------------------------------------------- #
# pattern_mix
# --------------------------------------------------------------------------- #


def test_the_weights_come_back_in_the_order_they_were_given() -> None:
    mix = pattern_mix({"single": 2, "all": 1})
    assert list(mix) == ["single", "all"]


def test_a_zero_weight_leaves_the_shape_out() -> None:
    assert list(pattern_mix({"all": 1, "gradient": 0, "single": 3})) == ["all", "single"]


def test_an_unknown_shape_is_refused() -> None:
    with pytest.raises(SaValueError, match="names unknown shape"):
        pattern_mix({"all": 1, "linear": 1})


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(SaValueError, match="must not be negative"):
        pattern_mix({"all": -1, "single": 1})


def test_all_weights_at_zero_leaves_no_shape_to_plant_in() -> None:
    with pytest.raises(SaValueError, match="at least one positive weight"):
        pattern_mix({"all": 0, "single": 0})


def test_a_non_mapping_is_refused() -> None:
    with pytest.raises(SaValueError, match="must be a mapping from shape name"):
        pattern_mix([1, 1, 1])


# --------------------------------------------------------------------------- #
# allocate
# --------------------------------------------------------------------------- #


def test_an_even_mix_splits_evenly() -> None:
    assert allocate(12, {"all": 1, "gradient": 1, "single": 1}) == {
        "all": 4,
        "gradient": 4,
        "single": 4,
    }


def test_the_counts_sum_to_n_whatever_the_weights() -> None:
    for n in range(0, 30):
        counts = allocate(n, {"all": 1, "gradient": 2, "single": 5})
        assert sum(counts.values()) == n


def test_the_remainder_goes_to_the_earlier_shapes_when_the_mix_is_even() -> None:
    """Ties in the remainder are left in the order the weights were given."""
    assert allocate(10, {"all": 1, "gradient": 1, "single": 1}) == {
        "all": 4,
        "gradient": 3,
        "single": 3,
    }


def test_weights_are_relative_not_proportions() -> None:
    assert allocate(10, {"all": 30, "single": 70}) == {"all": 3, "single": 7}


def test_nothing_to_allocate_gives_every_shape_zero() -> None:
    assert allocate(0, {"all": 1, "single": 1}) == {"all": 0, "single": 0}


# --------------------------------------------------------------------------- #
# pattern_delta
# --------------------------------------------------------------------------- #


def test_all_moves_every_treatment_group_by_the_same_amount() -> None:
    rng = np.random.default_rng(1)
    assert pattern_delta(2.0, "all", 3, rng).tolist() == [2.0, 2.0, 2.0]


def test_gradient_reaches_the_full_effect_at_the_last_group() -> None:
    rng = np.random.default_rng(1)
    assert pattern_delta(3.0, "gradient", 3, rng).tolist() == [1.0, 2.0, 3.0]


def test_single_moves_one_group_and_leaves_the_rest_at_exactly_zero() -> None:
    rng = np.random.default_rng(1)
    delta = pattern_delta(2.0, "single", 4, rng)
    assert sorted(delta.tolist()) == [0.0, 0.0, 0.0, 2.0]


def test_only_single_consumes_the_stream() -> None:
    """Which group is moved is drawn; the other two shapes are arithmetic."""
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    pattern_delta(2.0, "gradient", 3, rng)
    assert rng.bit_generator.state == before


def test_an_unknown_shape_is_an_internal_error() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(SaInternalError, match="unknown effect shape"):
        pattern_delta(2.0, "sigmoid", 3, rng)


# --------------------------------------------------------------------------- #
# pick_up_down
# --------------------------------------------------------------------------- #


def test_the_two_sets_are_the_sizes_asked_for_and_do_not_overlap() -> None:
    rng = np.random.default_rng(1)
    up, down = pick_up_down(20, 5, 3, rng)
    assert up.size == 5
    assert down.size == 3
    assert set(up.tolist()) & set(down.tolist()) == set()


def test_an_empty_up_set_leaves_the_down_set_the_size_it_asked_for() -> None:
    """The trap the head-and-tail draw exists to avoid: the complement would be all 20."""
    rng = np.random.default_rng(1)
    up, down = pick_up_down(20, 0, 3, rng)
    assert up.size == 0
    assert down.size == 3
