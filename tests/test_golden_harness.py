"""The harness itself, checked before anything is graded with it.

A loader that silently returned nothing, or a comparison that passed whatever it
was given, would make every golden test in the phase vacuous. These check that
the fixtures are all readable and that the comparison fails when it should.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from golden import (
    as_list,
    assert_close,
    assert_frame_close,
    case_names,
    load_case,
    samples_from_long,
    zero_based,
)


def test_there_are_frozen_cases_to_grade_against():
    names = case_names()
    assert len(names) > 30
    assert "anova_oneway" in names


@pytest.mark.parametrize("name", case_names())
def test_every_case_loads(name):
    frame, expected = load_case(name)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame.index) > 0
    assert expected is not None


def test_a_named_vector_is_compared_by_name_and_by_number():
    _, expected = load_case("anova_oneway")
    assert_close(
        {
            "n_used": 30,
            "n_groups": 4,
            "f_stat": expected["f_stat"],
            "df1": 3,
            "df2": 26,
            "eta_sq": expected["eta_sq"],
            "omega_sq": expected["omega_sq"],
            "pval": expected["pval"],
            "lower_conf": float("nan"),
            "upper_conf": float("nan"),
        },
        expected,
    )


def test_a_key_in_the_wrong_place_is_a_failure():
    with pytest.raises(AssertionError, match="key order differs"):
        assert_close({"b": 1, "a": 2}, {"a": 2, "b": 1})


def test_a_number_outside_the_tolerance_is_a_failure():
    with pytest.raises(AssertionError, match="off by"):
        assert_close(1.0 + 1e-6, 1.0)
    assert_close(1.0 + 1e-12, 1.0)


def test_r_null_has_to_line_up_with_a_missing_value():
    assert_close(float("nan"), None)
    assert_close(None, None)
    with pytest.raises(AssertionError, match="expected a missing value"):
        assert_close(0.0, None)
    with pytest.raises(AssertionError, match="got a missing value"):
        assert_close(float("nan"), 1.0)


def test_a_boolean_is_not_satisfied_by_a_number():
    assert_close(True, True)
    with pytest.raises(AssertionError, match="expected a boolean"):
        assert_close(1, True)


def test_an_infinity_has_to_be_an_infinity():
    assert_close(float("inf"), math.inf)
    with pytest.raises(AssertionError):
        assert_close(1e308, math.inf)


def test_a_table_is_compared_on_its_columns_and_their_order():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert_frame_close(frame, {"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(AssertionError, match="column order differs"):
        assert_frame_close(frame, {"b": ["x", "y"], "a": [1.0, 2.0]})
    with pytest.raises(AssertionError, match="row"):
        assert_frame_close(frame, {"a": [1.0], "b": ["x"]})


def test_the_long_input_rebuilds_the_samples_a_kernel_takes():
    frame, _ = load_case("anova_oneway")
    samples = samples_from_long(frame, ["ctrl", "low", "mid", "high"])
    assert list(samples) == ["ctrl", "low", "mid", "high"]
    assert [len(v) for v in samples.values()] == [9, 7, 8, 6]


def test_a_block_column_selects_one_of_two_inputs():
    frame, _ = load_case("anova_kruskal")
    tied = samples_from_long(frame, ["a", "b", "c"], block="tied")
    assert [len(v) for v in tied.values()] == [6, 6, 6]


def test_row_numbers_come_back_zero_based():
    assert zero_based(2) == [1]
    assert as_list("x") == ["x"]
