"""Scalar and vector argument checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statassist.core import (
    check_count,
    check_feat_names,
    check_flag,
    check_lim,
    check_margin,
    check_num_vector,
    check_p_adjust,
    check_pvalues,
    check_range,
    check_scalar_num,
)
from statassist.core.errors import SaValueError


class TestCheckFlag:
    def test_accepts_a_bool(self) -> None:
        assert check_flag(True, "paired") is True
        assert check_flag(False, "paired") is False

    def test_refuses_a_number(self) -> None:
        """``is.logical(1)`` is FALSE in R, and 1 would otherwise pass as True."""
        with pytest.raises(SaValueError, match="must be TRUE or FALSE"):
            check_flag(1, "paired")

    def test_refuses_none(self) -> None:
        with pytest.raises(SaValueError):
            check_flag(None, "paired")


class TestCheckScalarNum:
    def test_returns_the_value(self) -> None:
        assert check_scalar_num(0.05, "significance_level", 0, 1) == 0.05

    def test_unwraps_a_one_element_vector(self) -> None:
        """A length-one vector is a scalar in R, so it is one here too."""
        assert check_scalar_num(np.array([0.05]), "significance_level", 0, 1) == 0.05

    def test_refuses_a_bool(self) -> None:
        """``is.numeric(TRUE)`` is FALSE in R; Python's bool is an int subclass."""
        with pytest.raises(SaValueError, match="single non-missing number"):
            check_scalar_num(True, "significance_level")

    def test_refuses_nan(self) -> None:
        with pytest.raises(SaValueError, match="single non-missing number"):
            check_scalar_num(float("nan"), "significance_level")

    def test_refuses_a_longer_vector(self) -> None:
        with pytest.raises(SaValueError):
            check_scalar_num([0.05, 0.1], "significance_level")

    def test_an_open_lower_bound_excludes_it(self) -> None:
        assert check_scalar_num(0.0, "x", 0, 1) == 0.0
        with pytest.raises(SaValueError, match=r"must be in \(0, 1\], but is 0\."):
            check_scalar_num(0.0, "x", 0, 1, lower_open=True)

    def test_an_open_upper_bound_excludes_it(self) -> None:
        assert check_scalar_num(1.0, "x", 0, 1) == 1.0
        with pytest.raises(SaValueError, match=r"must be in \[0, 1\), but is 1\."):
            check_scalar_num(1.0, "x", 0, 1, upper_open=True)

    def test_infinity_passes_an_unbounded_check(self) -> None:
        """R tests ``is.na`` here, not ``is.finite``; ruling Inf out is check_count's job."""
        assert check_scalar_num(math.inf, "x") == math.inf


class TestCheckCount:
    def test_a_whole_double_becomes_an_int(self) -> None:
        """``n_feats = 100`` is a double in R and an int is what indexing wants."""
        result = check_count(100.0, "n_feats")
        assert result == 100
        assert isinstance(result, int)

    def test_refuses_a_fraction(self) -> None:
        with pytest.raises(SaValueError, match="finite whole number, but is 2.5"):
            check_count(2.5, "n_feats")

    def test_refuses_infinity(self) -> None:
        with pytest.raises(SaValueError, match="finite whole number"):
            check_count(math.inf, "n_feats")

    def test_refuses_a_bool(self) -> None:
        with pytest.raises(SaValueError):
            check_count(True, "n_feats")

    def test_honours_the_lower_bound(self) -> None:
        assert check_count(0, "n_dropped") == 0
        with pytest.raises(SaValueError, match=r"must be in \[1, Inf\]"):
            check_count(0, "n_feats", lower=1)


class TestCheckNumVector:
    def test_returns_a_writable_float_array(self) -> None:
        out = check_num_vector([0.1, 0.5], "grid", 0, 1)
        assert out.tolist() == [0.1, 0.5]
        out[0] = 0.9  # must not be a read-only view of the caller's data

    def test_names_every_offending_value_once(self) -> None:
        with pytest.raises(SaValueError, match=r"must be in \[0, 1\], but holds 1.5, 2.5\."):
            check_num_vector([0.1, 1.5, 2.5, 1.5], "grid", 0, 1)

    def test_refuses_an_empty_vector(self) -> None:
        with pytest.raises(SaValueError, match="non-empty numeric vector"):
            check_num_vector([], "grid")

    def test_refuses_a_missing_value(self) -> None:
        with pytest.raises(SaValueError, match="finite values"):
            check_num_vector([0.1, float("nan")], "grid")


class TestCheckRange:
    def test_returns_the_two_ends(self) -> None:
        assert check_range([2, 12], "expr_range") == (2.0, 12.0)

    def test_refuses_a_reversed_range(self) -> None:
        """A uniform draw accepts a reversed range silently, so this is the only guard."""
        with pytest.raises(SaValueError, match=r"must be increasing, but is c\(12, 2\)\."):
            check_range([12, 2], "expr_range")

    def test_honours_the_lower_bound(self) -> None:
        with pytest.raises(SaValueError, match="must not go below 0, but starts at -1"):
            check_range([-1, 5], "expr_range", lower=0)

    def test_refuses_a_wrong_length(self) -> None:
        with pytest.raises(SaValueError, match="length 2"):
            check_range([1, 2, 3], "expr_range")


class TestCheckMargin:
    def test_accepts_four_non_negative_values(self) -> None:
        assert check_margin([5, 4, 2, 1]) == (5.0, 4.0, 2.0, 1.0)

    def test_refuses_a_negative_value(self) -> None:
        with pytest.raises(SaValueError, match="4 non-negative values"):
            check_margin([5, 4, 2, -1])

    def test_refuses_a_wrong_length(self) -> None:
        with pytest.raises(SaValueError):
            check_margin([5, 4, 2])


class TestCheckLim:
    def test_none_means_derive_from_the_data(self) -> None:
        assert check_lim(None, "xlim") is None

    def test_accepts_a_finite_pair(self) -> None:
        assert check_lim([-2, 2], "xlim") == (-2.0, 2.0)

    def test_refuses_an_infinite_end(self) -> None:
        with pytest.raises(SaValueError, match="NULL or a finite numeric vector"):
            check_lim([0, math.inf], "xlim")


class TestCheckPvalues:
    def test_a_missing_p_value_is_allowed(self) -> None:
        """A feature the test could not be run on has one, and it is not an error."""
        out = check_pvalues([0.01, float("nan"), 1.0])
        assert np.isnan(out[1])

    def test_none_in_a_plain_list_reads_as_missing(self) -> None:
        """What R's ``NA`` in a numeric vector looks like coming from Python."""
        out = check_pvalues([0.01, None, 1.0])
        assert np.isnan(out[1])

    def test_a_non_numeric_entry_is_still_refused(self) -> None:
        with pytest.raises(SaValueError, match="must be a numeric vector"):
            check_pvalues([0.01, "high"])

    def test_refuses_an_out_of_range_value_and_reports_its_position(self) -> None:
        with pytest.raises(SaValueError, match=r"Offending position\(s\): 1, 3\."):
            check_pvalues([0.5, 1.2, 0.3, -0.1])

    def test_refuses_infinity(self) -> None:
        with pytest.raises(SaValueError, match=r"must lie in \[0, 1\]"):
            check_pvalues([0.5, math.inf])

    def test_reports_at_most_five_positions(self) -> None:
        with pytest.raises(SaValueError, match=r"0, 1, 2, 3, 4, \.\.\.\."):
            check_pvalues([2.0] * 6)


class TestCheckPAdjust:
    def test_accepts_a_known_method(self) -> None:
        assert check_p_adjust("BH", "p_adjust") == "BH"

    def test_refuses_a_misspelling(self) -> None:
        with pytest.raises(SaValueError, match="must be one of: holm"):
            check_p_adjust("bh", "p_adjust")


class TestCheckFeatNames:
    def test_returns_the_names_in_order(self) -> None:
        assert check_feat_names(["g2", "g1"]) == ["g2", "g1"]

    def test_a_bare_string_is_one_feature(self) -> None:
        """A length-one character vector in R, not three characters."""
        assert check_feat_names("gene_1") == ["gene_1"]

    def test_refuses_an_empty_vector(self) -> None:
        with pytest.raises(SaValueError, match="non-empty character vector"):
            check_feat_names([])

    def test_refuses_a_missing_name(self) -> None:
        with pytest.raises(SaValueError, match="must not contain NA"):
            check_feat_names(["g1", None])

    def test_names_the_duplicates(self) -> None:
        with pytest.raises(SaValueError, match="duplicated names: g1"):
            check_feat_names(["g1", "g2", "g1"])

    def test_refuses_a_non_string_name(self) -> None:
        with pytest.raises(SaValueError, match="character vector"):
            check_feat_names([1, 2])
