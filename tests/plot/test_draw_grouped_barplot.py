"""What the bars were, and which of them had an interval.

The heights are one column of the descriptive summary, so the check is that the
figure and the table agree rather than that the figure looks right: the returned
bars are compared against
:func:`~statassist.summarize_descriptive_stats` on the same input.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statassist import (
    draw_grouped_barplot,
    simulate_multiple_groups,
    summarize_descriptive_stats,
)
from statassist.core.errors import SaValueError
from statassist.plot.barplot import BAR_HEIGHTS, NOTCH_WIDTH


@functools.lru_cache(maxsize=1)
def _simulated():
    return simulate_multiple_groups(n_feats=4, n_control=10, n_up=1, n_down=1, seed=17)


def _call(**kwargs):
    args = _simulated().args
    kwargs.setdefault("group", args["group"])
    kwargs.setdefault("group_lv", args["group_lv"])
    return draw_grouped_barplot(args["data"], args["feats"], **kwargs)


def _summary():
    args = _simulated().args
    return summarize_descriptive_stats(args["data"], args["feats"], args["group"], args["group_lv"])


def _panel():
    """The axis holding the bars, which is the one without the legend on it."""
    import matplotlib.pyplot as plt

    return next(ax for ax in plt.gcf().axes if ax.get_legend() is None)


class TestTheTableBehindTheBars:
    def test_the_columns_are_the_ones_documented(self):
        assert list(_call().columns) == ["features", "group", "n", "value", "lower", "upper"]

    def test_one_row_per_bar_with_a_feature_levels_kept_together(self):
        args = _simulated().args
        bars = _call()
        assert len(bars.index) == len(args["feats"]) * len(args["group_lv"])
        assert list(bars["group"])[: len(args["group_lv"])] == list(args["group_lv"])
        assert bars["features"].iloc[0] == args["feats"][0]

    def test_the_heights_are_the_summary_column_they_name(self):
        summary = _summary()
        for mainbar in ("mean", "median", "n", "sd", "skewness"):
            drawn = _call(mainbar=mainbar)
            assert np.allclose(drawn["value"], summary[mainbar], equal_nan=True), mainbar

    def test_which_height_and_interval_were_drawn_come_back_with_the_table(self):
        bars = _call(mainbar="median", errorbar="ci")
        assert bars.attrs["mainbar"] == "median"
        assert bars.attrs["errorbar"] == "ci"

    def test_out_statistics_false_draws_and_returns_nothing(self):
        assert _call(out_statistics=False) is None

    def test_the_bars_are_ordered_the_way_the_summary_is(self):
        bars = _call()
        summary = _summary()
        assert list(bars["features"]) == list(summary["features"])
        assert list(bars["group"]) == list(summary["group"])


class TestIntervals:
    def test_no_errorbar_leaves_both_ends_missing(self):
        bars = _call()
        assert bars["lower"].isna().all()
        assert bars["upper"].isna().all()

    def test_se_is_the_summary_standard_error_either_side_of_the_mean(self):
        bars = _call(errorbar="se")
        summary = _summary()
        assert np.allclose(bars["upper"] - bars["value"], summary["se"])
        assert np.allclose(bars["value"] - bars["lower"], summary["se"])

    def test_sd_is_the_spread_of_the_observations_rather_than_of_the_mean(self):
        bars = _call(errorbar="sd")
        summary = _summary()
        assert np.allclose(bars["upper"] - bars["value"], summary["sd"])

    def test_ci_about_a_mean_is_a_student_interval_at_the_level_given(self):
        level = 0.9
        bars = _call(errorbar="ci", conf_level=level)
        summary = _summary()
        half = stats.t.ppf(1 - (1 - level) / 2, summary["n"] - 1) * summary["se"]
        assert np.allclose(bars["upper"] - bars["value"], half)

    def test_a_wider_level_gives_a_wider_interval(self):
        narrow = _call(errorbar="ci", conf_level=0.8)
        wide = _call(errorbar="ci", conf_level=0.99)
        assert ((wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])).all()

    def test_ci_about_a_median_is_the_notch_and_reads_no_level(self):
        bars = _call(mainbar="median", errorbar="ci")
        summary = _summary()
        half = NOTCH_WIDTH * summary["iqr"] / np.sqrt(summary["n"])
        assert np.allclose(bars["upper"] - bars["value"], half)
        at_other_level = _call(mainbar="median", errorbar="ci", conf_level=0.8)
        assert np.allclose(at_other_level["upper"], bars["upper"])

    def test_a_single_observation_has_no_interval_rather_than_a_zero_wide_one(self):
        args = _simulated().args
        group = pd.Series(args["group"]).astype(str)
        alone = group.index[group == args["group_lv"][1]][0]
        thinned = group.copy()
        thinned[group == args["group_lv"][1]] = args["group_lv"][0]
        thinned[alone] = args["group_lv"][1]
        bars = draw_grouped_barplot(
            args["data"], args["feats"], thinned, args["group_lv"], errorbar="ci"
        )
        of_one = bars[bars["group"] == args["group_lv"][1]]
        assert (of_one["n"] == 1).all()
        assert of_one["lower"].isna().all()


class TestRefusedCombinations:
    def test_a_median_takes_the_notch_but_not_a_spread_of_the_mean(self):
        for errorbar in ("se", "sd"):
            with pytest.raises(SaValueError, match="not a width to draw either side"):
                _call(mainbar="median", errorbar=errorbar)

    @pytest.mark.parametrize(
        "mainbar", [name for name in BAR_HEIGHTS if name not in ("mean", "median")]
    )
    def test_a_height_that_is_not_a_location_takes_no_interval(self, mainbar):
        with pytest.raises(SaValueError, match="itself a spread, a count or a shape"):
            _call(mainbar=mainbar, errorbar="se")

    def test_every_height_accepts_no_interval_at_all(self):
        for mainbar in BAR_HEIGHTS:
            assert _call(mainbar=mainbar) is not None

    def test_the_pair_is_checked_before_the_data_is_read(self):
        with pytest.raises(SaValueError, match="itself a spread"):
            draw_grouped_barplot("not a frame", ["nope"], mainbar="sd", errorbar="se")


class TestGroupLevels:
    def test_group_is_required_because_a_summary_of_everything_has_no_clusters(self):
        args = _simulated().args
        with pytest.raises(SaValueError, match="`group` says which bars there are"):
            draw_grouped_barplot(args["data"], args["feats"])

    def test_levels_left_unnamed_are_taken_the_way_the_summary_takes_them(self):
        args = _simulated().args
        bars = draw_grouped_barplot(args["data"], args["feats"], args["group"])
        assert list(bars["group"])[: len(args["group_lv"])] == sorted(args["group_lv"])

    def test_control_label_moves_that_level_to_the_front_of_every_cluster(self):
        args = _simulated().args
        last = args["group_lv"][-1]
        bars = _call(control_label=last)
        assert bars["group"].iloc[0] == last

    def test_a_level_outside_the_ones_named_is_dropped_and_reported(self, caplog):
        args = _simulated().args
        kept = list(args["group_lv"])[:-1]
        with caplog.at_level(logging.INFO, logger="statassist"):
            bars = draw_grouped_barplot(args["data"], args["feats"], args["group"], kept)
        assert "belonging to a level outside `group_lv`" in caplog.text
        assert sorted(bars["group"].unique()) == sorted(kept)


class TestWhatWasDrawn:
    def test_one_bar_per_row_of_the_returned_table(self):
        bars = _call()
        assert len(_panel().patches) == len(bars.index)

    def test_the_bar_heights_are_the_values_that_came_back(self):
        bars = _call()
        drawn = [patch.get_height() for patch in _panel().patches]
        assert np.allclose(drawn, bars["value"])

    def test_the_clusters_are_annotated_with_the_features(self):
        _call()
        labels = [text.get_text() for text in _panel().get_xticklabels()]
        assert labels == list(_simulated().args["feats"])

    def test_the_legend_lists_the_group_levels_in_draw_order(self):
        import matplotlib.pyplot as plt

        _call()
        legend = next(ax.get_legend() for ax in plt.gcf().axes if ax.get_legend() is not None)
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == list(_simulated().args["group_lv"])

    def test_the_y_axis_is_named_after_the_height_unless_told_otherwise(self):
        _call(mainbar="median")
        assert _panel().get_ylabel() == "median"
        _call(ylab="abundance")
        assert _panel().get_ylabel() == "abundance"

    def test_a_derived_range_includes_the_zero_a_bar_stands_on(self):
        _call()
        low, high = _panel().get_ylim()
        assert low <= 0 <= high

    def test_a_height_that_runs_both_ways_gets_a_baseline_to_be_measured_from(self):
        _call(mainbar="skewness")
        assert any(line.get_ydata()[0] == 0 for line in _panel().get_lines())

    def test_a_supplied_range_is_used_as_given(self):
        _call(ylim=(-5.0, 30.0))
        assert _panel().get_ylim() == (-5.0, 30.0)

    def test_a_grid_line_type_of_zero_draws_no_grid(self):
        _call(grid_lty=0)
        assert not any(line.get_visible() for line in _panel().get_ygridlines())

    def test_a_named_colour_is_recycled_over_the_levels(self):
        _call(col="black")
        colours = {patch.get_facecolor() for patch in _panel().patches}
        assert len(colours) == 1


class TestArgumentChecks:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mainbar": "nope"}, "`mainbar` must be one of"),
            ({"errorbar": "nope"}, "`errorbar` must be one of"),
            ({"conf_level": 1}, "`conf_level`"),
            ({"gap": -1}, "`gap`"),
            ({"lwd": 0}, "`lwd`"),
            ({"cex_legend": 0}, "`cex_legend`"),
            ({"ylim": (1.0,)}, "`ylim`"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        with pytest.raises(SaValueError, match=match):
            _call(**kwargs)

    def test_a_height_that_is_missing_everywhere_leaves_nothing_to_draw(self):
        """One observation per group has no spread, which is what ``sd`` is."""
        args = _simulated().args
        levels = list(args["group_lv"])[:2]
        with pytest.raises(SaValueError, match="is NA for every feature and group"):
            draw_grouped_barplot(args["data"].iloc[:2], args["feats"], levels, levels, mainbar="sd")
