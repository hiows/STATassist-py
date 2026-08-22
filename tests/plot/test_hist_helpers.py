"""Breaks, bins and a bandwidth, against the definitions they were ported from.

These are the numbers :func:`draw_butterfly_hist` reports, so they are R's
functions rather than numpy's: ``pretty()`` picks the break points, ``hist()``
closes its bins on the right, and ``density()`` uses Silverman's rule of thumb on
a grid three bandwidths past each end of the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from statassist.plot._hist import (
    DENSITY_CUT,
    DENSITY_N,
    bw_nrd0,
    density,
    histogram,
    nclass,
    pretty,
)


class TestPretty:
    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            (0, 10, [0, 2, 4, 6, 8, 10]),
            (0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            (1, 7, [1, 2, 3, 4, 5, 6, 7]),
            (0, 100, [0, 20, 40, 60, 80, 100]),
        ],
    )
    def test_it_picks_the_round_numbers_r_picks(self, low, high, expected):
        assert np.allclose(pretty(low, high), expected)

    def test_the_range_is_always_covered_however_odd_its_ends(self):
        for low, high in [(-3.2, 4.7), (0.013, 0.048), (-1e5, 3e5), (7, 7.5)]:
            breaks = pretty(low, high)
            assert breaks[0] <= low
            assert breaks[-1] >= high

    def test_the_steps_are_equal(self):
        steps = np.diff(pretty(-3.2, 4.7))
        assert np.allclose(steps, steps[0])

    def test_a_range_of_no_width_still_gives_something_to_draw_on(self):
        breaks = pretty(3.0, 3.0)
        assert breaks.size >= 2
        assert breaks[0] <= 3.0 <= breaks[-1]


class TestBins:
    def test_a_bin_is_closed_on_the_right_and_the_lowest_break_is_included(self):
        breaks = np.asarray([0.0, 1.0, 2.0])
        binned = histogram(np.asarray([0.0, 1.0, 1.5, 2.0]), breaks, xname="x")
        assert list(binned["counts"]) == [2, 2]

    def test_the_counts_add_up_and_the_density_integrates_to_one(self):
        values = np.linspace(0, 9, 40)
        breaks = np.asarray([0.0, 3.0, 6.0, 9.0])
        binned = histogram(values, breaks, xname="x")
        assert binned["counts"].sum() == values.size
        widths = np.diff(breaks)
        assert pytest.approx(1.0) == float((binned["density"] * widths).sum())

    def test_a_value_outside_the_breaks_is_refused_rather_than_dropped(self):
        from statassist.core.errors import SaValueError

        with pytest.raises(SaValueError, match="do not span the range"):
            histogram(np.asarray([5.0]), np.asarray([0.0, 1.0]), xname="x")

    @pytest.mark.parametrize("rule", ["Sturges", "Scott", "FD"])
    def test_every_rule_asks_for_at_least_one_bin(self, rule):
        values = np.random.default_rng(3).normal(size=50)
        assert nclass(values, rule) >= 1

    def test_sturges_is_the_binary_logarithm_of_the_sample_size(self):
        assert nclass(np.arange(64.0), "Sturges") == 7


class TestDensity:
    def test_the_bandwidth_is_silvermans_rule_of_thumb(self):
        """``bw.nrd0(1:10)`` is 1.7190..., which is what the constants give."""
        assert pytest.approx(1.719, abs=1e-3) == bw_nrd0(np.arange(1.0, 11.0))

    def test_a_sample_with_no_spread_still_gets_a_bandwidth(self):
        assert bw_nrd0(np.asarray([2.0, 2.0, 2.0])) > 0

    def test_the_grid_runs_past_the_data_by_three_bandwidths(self):
        values = np.arange(1.0, 11.0)
        curve = density(values)
        assert curve["x"].size == DENSITY_N
        assert pytest.approx(values.min() - DENSITY_CUT * curve["bw"]) == curve["x"][0]
        assert pytest.approx(values.max() + DENSITY_CUT * curve["bw"]) == curve["x"][-1]

    def test_the_curve_integrates_to_about_one(self):
        values = np.random.default_rng(5).normal(size=200)
        curve = density(values)
        area = np.trapezoid(curve["y"], curve["x"])
        assert pytest.approx(1.0, abs=0.01) == area

    def test_adjust_widens_the_bandwidth_it_multiplies(self):
        values = np.random.default_rng(5).normal(size=50)
        assert pytest.approx(2 * density(values)["bw"]) == density(values, adjust=2)["bw"]
