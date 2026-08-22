"""The numbers behind the butterfly, and the rule that keeps them comparable.

Both groups are binned on shared breaks, which is what makes bin *i* the same
interval on either side of the centre line, and a curve is only ever drawn on the
density scale, since a count per bin scales with the bin width and a curve does
not. Those two are what the returned tables are checked for here.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from statassist import draw_butterfly_hist, simulate_two_groups
from statassist.core.errors import SaValueError


@functools.lru_cache(maxsize=1)
def _simulated():
    return simulate_two_groups(n_feats=4, n_case=30, n_control=30, n_up=1, n_down=1, seed=17)


def _call(**kwargs):
    sim = _simulated()
    args = sim.args
    return draw_butterfly_hist(
        args["data"], args["feats"][0], args["group"], args["group_lv"], **kwargs
    )


class TestReturnedNumbers:
    def test_the_bins_are_shared_and_the_counts_add_up_to_each_group(self):
        out = _call()
        bins = out["bin_summary_stats"]
        levels = list(_simulated().args["group_lv"])
        assert list(bins.columns) == ["bin_start", "bin_end", "bin_mid", *levels]
        for level in levels:
            assert bins[level].sum() == out["group_summary_stats"].loc["n", level]

    def test_the_bin_mid_is_the_middle_of_the_bin_it_belongs_to(self):
        bins = _call()["bin_summary_stats"]
        middle = (bins["bin_start"] + bins["bin_end"]) / 2
        assert np.allclose(bins["bin_mid"], middle)

    def test_a_proportion_is_the_count_divided_by_the_group(self):
        counts = _call(scale="count")["bin_summary_stats"]
        shares = _call(scale="proportion")["bin_summary_stats"]
        for level in _simulated().args["group_lv"]:
            assert np.allclose(shares[level], counts[level] / counts[level].sum())

    def test_a_density_integrates_to_one_over_the_bins(self):
        out = _call(scale="density")
        bins = out["bin_summary_stats"]
        widths = bins["bin_end"] - bins["bin_start"]
        for level in _simulated().args["group_lv"]:
            assert pytest.approx(1.0) == float((bins[level] * widths).sum())

    def test_every_group_is_reported_with_what_was_used_and_what_was_left_out(self):
        stats = _call()["group_summary_stats"]
        assert list(stats.index) == ["n", "n_dropped", "min", "max"]
        assert list(stats.columns) == list(_simulated().args["group_lv"])
        assert (stats.loc["n"] > 0).all()

    def test_out_statistics_false_returns_nothing_but_still_draws(self):
        import matplotlib.pyplot as plt

        assert _call(out_statistics=False) is None
        assert plt.gcf().axes

    def test_a_curve_is_returned_only_when_one_was_drawn(self):
        assert "group_densities" not in _call()
        both = _call(type="both")
        assert set(both["group_densities"]) == set(_simulated().args["group_lv"])
        assert both["bin_summary_stats"].shape[0] > 0


class TestScaleRule:
    def test_a_curve_moves_the_bars_onto_the_density_scale(self):
        import matplotlib.pyplot as plt

        _call(type="dens")
        assert plt.gcf().axes[0].get_xlabel() == "Density"

    @pytest.mark.parametrize("type_", ["dens", "both"])
    def test_asking_for_counts_beside_a_curve_is_refused(self, type_):
        with pytest.raises(SaValueError, match="shares an axis with the bars"):
            _call(type=type_, scale="count")

    def test_asking_for_a_density_beside_a_curve_is_accepted(self):
        out = _call(type="both", scale="density")
        assert "group_densities" in out


class TestAxis:
    def test_the_left_group_is_drawn_at_negative_coordinates(self):
        import matplotlib.pyplot as plt

        _call()
        widths = [patch.get_width() for patch in plt.gcf().axes[0].patches]
        assert min(widths) < 0
        assert max(widths) > 0

    def test_the_tick_labels_are_absolute_so_a_bar_reads_the_same_either_side(self):
        import matplotlib.pyplot as plt

        _call()
        ax = plt.gcf().axes[0]
        labels = [text.get_text() for text in ax.get_xticklabels()]
        assert not any(label.startswith("-") or label.startswith("\u2212") for label in labels)

    def test_a_derived_value_range_covers_the_breaks(self):
        import matplotlib.pyplot as plt

        out = _call()
        low, high = plt.gcf().axes[0].get_ylim()
        bins = out["bin_summary_stats"]
        assert low <= bins["bin_start"].min()
        assert high >= bins["bin_end"].max()

    def test_a_derived_range_covers_the_tails_of_the_curve_as_well(self):
        import matplotlib.pyplot as plt

        out = _call(type="dens")
        low, high = plt.gcf().axes[0].get_ylim()
        for curve in out["group_densities"].values():
            assert low <= curve["x"].min()
            assert high >= curve["x"].max()


class TestBreaks:
    def test_a_bin_count_is_read_as_roughly_that_many_bins(self):
        out = _call(breaks=6)
        assert 3 <= out["bin_summary_stats"].shape[0] <= 12

    def test_break_points_given_outright_are_the_bins(self):
        pooled = _simulated().args["data"][_simulated().args["feats"][0]]
        edges = np.linspace(float(pooled.min()) - 1, float(pooled.max()) + 1, 5)
        out = _call(breaks=edges)
        assert np.allclose(out["bin_summary_stats"]["bin_start"], edges[:-1])

    @pytest.mark.parametrize("breaks", [[1.0], "nope", 0, [3.0, 1.0, 2.0]])
    def test_breaks_that_are_neither_a_rule_a_count_nor_increasing_are_refused(self, breaks):
        with pytest.raises(SaValueError, match="`breaks`"):
            _call(breaks=breaks)


class TestArgumentChecks:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"type": "nope"}, "`type` must be one of"),
            ({"scale": "nope"}, "`scale` must be one of"),
            ({"dens_alpha": 2}, "`dens_alpha`"),
            ({"col": "red"}, "`col` must contain exactly two"),
            ({"dens_col": ["a", "b", "c"]}, "`dens_col`"),
            ({"margin": (1, 2)}, "`margin`"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        with pytest.raises(SaValueError, match=match):
            _call(**kwargs)

    def test_more_than_one_feature_at_a_time_is_refused(self):
        sim = _simulated()
        with pytest.raises(SaValueError, match="one feature at a time"):
            draw_butterfly_hist(
                sim.args["data"],
                list(sim.args["feats"][:2]),
                sim.args["group"],
                sim.args["group_lv"],
            )

    def test_a_level_with_no_finite_value_names_itself(self):
        sim = _simulated()
        data = sim.args["data"].copy()
        feat = sim.args["feats"][0]
        reference = sim.args["group_lv"][0]
        data.loc[np.asarray(sim.args["group"]) == reference, feat] = np.nan
        with pytest.raises(SaValueError, match="no finite value in group level"):
            draw_butterfly_hist(data, feat, sim.args["group"], sim.args["group_lv"])

    def test_a_curve_needs_two_distinct_values_and_says_which_group_has_not(self):
        sim = _simulated()
        data = sim.args["data"].copy()
        feat = sim.args["feats"][0]
        reference = sim.args["group_lv"][0]
        data.loc[np.asarray(sim.args["group"]) == reference, feat] = 1.0
        with pytest.raises(SaValueError, match="two distinct finite values"):
            draw_butterfly_hist(data, feat, sim.args["group"], sim.args["group_lv"], type="dens")
