"""What the boxes were, and where they were put.

The statistics come back rather than only the picture, so a box can be checked
against the observations behind it: the columns are the levels or the cells, in
the order they were drawn, and the fences and the notch are the definitions the
plot says they are.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pandas as pd
import pytest

from statassist import (
    compare_factorial_groups,
    draw_grouped_boxplot,
    simulate_factorial_groups,
    simulate_two_groups,
)
from statassist.core.errors import SaValueError
from statassist.plot.boxplot import BOX_ROWS, CONF_ROWS, NOTCH_WIDTH, WHISKER_REACH


@functools.lru_cache(maxsize=1)
def _one_factor():
    return simulate_two_groups(n_feats=3, n_case=9, n_control=9, n_up=1, n_down=1, seed=11)


@functools.lru_cache(maxsize=1)
def _crossed():
    return simulate_factorial_groups(n_feats=3, n_per_cell=6, seed=4)


def _call(**kwargs):
    args = _one_factor().args
    kwargs.setdefault("group", args["group"])
    kwargs.setdefault("group_lv", args["group_lv"])
    return draw_grouped_boxplot(args["data"], args["feats"], **kwargs)


def _call_crossed(**kwargs):
    args = _crossed().args
    kwargs.setdefault("factors", args["factors"])
    kwargs.setdefault("factor_lv", args["factor_lv"])
    return draw_grouped_boxplot(args["data"], args["feats"], **kwargs)


def _panels():
    """The axes holding boxes, which is every axis but the one holding the legend."""
    import matplotlib.pyplot as plt

    return [ax for ax in plt.gcf().axes if ax.get_legend() is None]


class TestSayingWhatTheBoxesAre:
    def test_group_and_factors_are_two_ways_of_saying_it_so_only_one_is_taken(self):
        args = _one_factor().args
        with pytest.raises(SaValueError, match="two ways of saying"):
            draw_grouped_boxplot(
                args["data"],
                args["feats"],
                group=args["group"],
                group_lv=args["group_lv"],
                factors={"a": args["group"], "b": args["group"]},
            )

    def test_neither_of_them_leaves_nothing_to_draw(self):
        args = _one_factor().args
        with pytest.raises(SaValueError, match="nothing says what the boxes are"):
            draw_grouped_boxplot(args["data"], args["feats"])

    def test_levels_without_the_factors_they_belong_to_are_refused(self):
        args = _one_factor().args
        with pytest.raises(SaValueError, match="`factor_lv` gives the levels"):
            draw_grouped_boxplot(args["data"], args["feats"], factor_lv={"sex": ["m", "f"]})

    def test_a_single_factor_states_its_reference_in_group_lv_not_twice(self):
        with pytest.raises(SaValueError, match="`control_label` names a reference level"):
            _call(control_label="control")

    def test_a_crossed_design_reads_control_label_and_draws_that_level_first(self):
        args = _crossed().args
        second = list(args["factor_lv"])[1]
        wanted = list(args["factor_lv"][second])[-1]
        stats = _call_crossed(factor_lv=None, control_label={second: wanted})
        columns = list(stats["box_summary_stats"][args["feats"][0]].columns)
        assert all(label.endswith(wanted) for label in columns[: len(args["factor_lv"][second])])


class TestReturnedStatistics:
    def test_the_columns_are_the_group_levels_in_the_order_they_were_named(self):
        args = _one_factor().args
        stats = _call()
        for feat in args["feats"]:
            assert list(stats["box_summary_stats"][feat].columns) == list(args["group_lv"])

    def test_the_columns_of_a_crossed_design_are_the_cell_labels_the_analysis_keys_on(self):
        args = _crossed().args
        levels = args["factor_lv"]
        expected = [".".join(cell) for cell in itertools.product(*levels.values())]
        drawn = list(_call_crossed()["box_summary_stats"][args["feats"][0]].columns)
        assert sorted(drawn) == sorted(expected)

    def test_the_boxes_are_the_cells_the_factorial_comparison_fits(self):
        """Both go through ``fact_layout``, so a box and a cell are one thing."""
        args = _crossed().args
        compared = compare_factorial_groups(
            args["data"], args["feats"], args["factors"], args["factor_lv"]
        )
        drawn = _call_crossed()["box_summary_stats"][args["feats"][0]]
        assert set(drawn.columns) == set(compared["cells"]["cell"])

    def test_the_box_count_is_the_cell_count_the_comparison_reports(self):
        args = _crossed().args
        compared = compare_factorial_groups(
            args["data"], args["feats"], args["factors"], args["factor_lv"]
        )
        feat = args["feats"][0]
        of_feature = compared["cells"][compared["cells"]["features"] == feat]
        counts = _call_crossed()["median_confidence_stats"][feat].loc["n"]
        assert list(counts[list(of_feature["cell"])]) == list(of_feature["n"].astype(float))

    def test_the_primary_factor_varies_fastest_so_it_is_what_a_cluster_holds(self):
        args = _crossed().args
        primary = list(args["factor_lv"])[0]
        levels = list(args["factor_lv"][primary])
        drawn = list(_call_crossed()["box_summary_stats"][args["feats"][0]].columns)
        assert [label.split(".")[0] for label in drawn[: len(levels)]] == levels

    def test_the_two_tables_hold_the_rows_they_say_they_do(self):
        stats = _call()
        feat = _one_factor().args["feats"][0]
        assert list(stats["box_summary_stats"][feat].index) == list(BOX_ROWS)
        assert list(stats["median_confidence_stats"][feat].index) == list(CONF_ROWS)

    def test_the_quartiles_and_fences_are_the_ones_the_observations_give(self):
        args = _one_factor().args
        feat, level = args["feats"][0], args["group_lv"][0]
        values = (
            args["data"]
            .loc[(pd.Series(args["group"]) == level).to_numpy(), feat]
            .to_numpy(dtype=float)
        )
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))

        column = _call()["box_summary_stats"][feat][level]
        assert column["Q1"] == pytest.approx(q1)
        assert column["median"] == pytest.approx(median)
        assert column["lower_bound"] == pytest.approx(q1 - WHISKER_REACH * (q3 - q1))
        assert column["upper_bound"] == pytest.approx(q3 + WHISKER_REACH * (q3 - q1))

    def test_the_notch_is_the_median_plus_and_minus_its_own_width(self):
        args = _one_factor().args
        feat, level = args["feats"][0], args["group_lv"][0]
        box = _call()
        summary = box["box_summary_stats"][feat][level]
        notch = box["median_confidence_stats"][feat][level]
        iqr = summary["Q3"] - summary["Q1"]
        half = NOTCH_WIDTH * iqr / np.sqrt(notch["n"])
        assert notch["lower_conf"] == pytest.approx(summary["median"] - half)
        assert notch["upper_conf"] == pytest.approx(summary["median"] + half)

    def test_out_statistics_false_draws_and_returns_nothing(self):
        assert _call(out_statistics=False) is None

    def test_a_box_with_nothing_in_it_comes_back_all_missing(self):
        args = _one_factor().args
        data = args["data"].copy()
        feat = args["feats"][0]
        data.loc[(pd.Series(args["group"]) == args["group_lv"][0]).to_numpy(), feat] = np.nan
        stats = draw_grouped_boxplot(
            data, args["feats"], group=args["group"], group_lv=args["group_lv"]
        )
        column = stats["box_summary_stats"][feat][args["group_lv"][0]]
        assert column.isna().all()

    def test_both_panel_arrangements_hold_the_same_boxes(self):
        by_feature = _call_crossed(panel_by="feature")["box_summary_stats"]
        by_factor = _call_crossed(panel_by="factor")["box_summary_stats"]
        for feat, table in by_feature.items():
            pd.testing.assert_frame_equal(table, by_factor[feat])


class TestWhatWasDrawn:
    def test_one_panel_per_feature_under_panel_by_feature(self):
        _call_crossed(panel_by="feature")
        assert len(_panels()) == len(_crossed().args["feats"])

    def test_one_panel_per_cell_of_the_remaining_factors_under_panel_by_factor(self):
        args = _crossed().args
        _call_crossed(panel_by="factor")
        second = list(args["factor_lv"])[1]
        assert len(_panels()) == len(args["factor_lv"][second])

    def test_panels_of_the_same_quantity_share_one_axis(self):
        _call_crossed(panel_by="factor")
        assert len({ax.get_ylim() for ax in _panels()}) == 1

    def test_panels_of_different_quantities_keep_their_own(self):
        _call_crossed(panel_by="feature")
        assert len({ax.get_ylim() for ax in _panels()}) > 1

    def test_a_supplied_range_is_shared_by_every_panel(self):
        _call_crossed(panel_by="feature", ylim=(0.0, 25.0))
        for ax in _panels():
            assert ax.get_ylim() == (0.0, 25.0)

    def test_the_clusters_are_annotated_with_what_they_hold(self):
        _call()
        labels = [text.get_text() for text in _panels()[0].get_xticklabels()]
        assert labels == list(_one_factor().args["feats"])

    def test_the_legend_lists_the_levels_inside_a_cluster(self):
        import matplotlib.pyplot as plt

        _call()
        legends = [ax.get_legend() for ax in plt.gcf().axes if ax.get_legend() is not None]
        labels = [text.get_text() for text in legends[0].get_texts()]
        assert labels == list(_one_factor().args["group_lv"])

    def test_the_legend_of_a_crossed_design_is_titled_with_the_primary_factor(self):
        import matplotlib.pyplot as plt

        _call_crossed()
        legends = [ax.get_legend() for ax in plt.gcf().axes if ax.get_legend() is not None]
        assert legends[0].get_title().get_text() == list(_crossed().args["factor_lv"])[0]

    def test_a_grid_line_type_of_zero_draws_no_grid(self):
        _call(grid_lty=0)
        assert not any(line.get_visible() for line in _panels()[0].get_ygridlines())


class TestArgumentChecks:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"panel_by": "nope"}, "`panel_by` must be one of"),
            ({"panel_nrow": 0}, "`panel_nrow`"),
            ({"gap": -1}, "`gap`"),
            ({"lwd": 0}, "`lwd`"),
            ({"cex_axis": 0}, "`cex_axis`"),
            ({"ylim": (1.0, 2.0, 3.0)}, "`ylim`"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        with pytest.raises(SaValueError, match=match):
            _call(**kwargs)

    def test_features_with_nothing_finite_in_them_cannot_share_an_axis(self):
        args = _crossed().args
        data = args["data"].copy()
        for feat in args["feats"]:
            data[feat] = np.nan
        with pytest.raises(SaValueError, match="no finite value in any cell"):
            draw_grouped_boxplot(
                data,
                args["feats"],
                factors=args["factors"],
                factor_lv=args["factor_lv"],
                panel_by="factor",
            )
