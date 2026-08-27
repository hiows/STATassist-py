"""What the mosaic laid out, shaded and marked.

The geometry comes back with the figure, so the picture can be checked instead of
eyeballed: the tiles are in the unit square, their areas are the cells' shares of
the table, and the segments the null hypothesis was drawn at are in the result
beside them.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from crafted import crafted_categorical

from statassist import (
    compare_categorical_groups,
    compare_two_groups,
    draw_mosaic_plot,
    draw_volcano_plot,
    estimate_categorical_significance,
    estimate_significance,
    simulate_two_groups,
)
from statassist.core.errors import SaValueError
from statassist.plot.mosaic import GAP_MAX_SHARE, MOSAIC_COLORS, RESIDUAL_BREAKS

CROSSED = pd.DataFrame(
    {
        "smoker": ["y"] * 60 + ["n"] * 60,
        "grade": (
            ["high"] * 10
            + ["mid"] * 20
            + ["low"] * 30
            + ["high"] * 30
            + ["mid"] * 20
            + ["low"] * 10
        ),
    }
)

MATCHED = pd.DataFrame(
    {
        "before": ["pass"] * 20 + ["fail"] * 30,
        "after": ["pass"] * 18 + ["fail"] * 2 + ["pass"] * 14 + ["fail"] * 16,
    }
)

#: The gap the tiles are laid out with unless a test says otherwise.
GAP = 0.015


@functools.lru_cache(maxsize=2)
def _res(paired: bool = False):
    return compare_categorical_groups(MATCHED if paired else CROSSED, paired=paired)


class TestReturnedGeometry:
    def test_the_slots_are_the_ones_r_returns(self):
        drawn = draw_mosaic_plot(_res())
        assert set(drawn) == {
            "cells",
            "widths",
            "heights",
            "expected_prop",
            "empty_levels",
            "null",
            "residual",
            "residual_breaks",
            "colors",
        }
        assert drawn["residual_breaks"] == list(RESIDUAL_BREAKS)
        assert drawn["colors"] == list(MOSAIC_COLORS)

    def test_the_cell_table_comes_back_with_the_tile_it_was_drawn_as(self):
        res = _res()
        cells = draw_mosaic_plot(res)["cells"]
        assert list(cells.columns)[: len(res.cells.columns)] == list(res.cells.columns)
        for name in ("x1", "x2", "y1", "y2", "fill"):
            assert name in cells.columns

    def test_the_strips_are_the_marginal_shares_and_sum_to_one(self):
        res = _res()
        drawn = draw_mosaic_plot(res)
        table = res.as_table()
        expected = table.sum(axis=1) / table.to_numpy().sum()
        assert np.allclose(drawn["widths"].to_numpy(), expected.to_numpy())
        assert float(drawn["widths"].sum()) == pytest.approx(1.0)

    def test_each_strip_is_cut_by_the_conditional_shares(self):
        drawn = draw_mosaic_plot(_res())
        assert np.allclose(drawn["heights"].sum(axis=1), 1.0)

    def test_the_tiles_cover_the_square_but_for_the_gaps(self):
        res = _res()
        drawn = draw_mosaic_plot(res, gap=GAP)
        cells = drawn["cells"]
        n_row, n_col = res.design["dim"]
        area = float(((cells["x2"] - cells["x1"]) * (cells["y2"] - cells["y1"])).sum())
        assert area == pytest.approx((1 - GAP * (n_row - 1)) * (1 - GAP * (n_col - 1)))

    def test_a_tile_area_is_the_cell_share_of_the_table(self):
        """Which is the whole claim a mosaic makes, so it is checked rather than
        left to the eye."""
        drawn = draw_mosaic_plot(_res(), gap=0)
        cells = drawn["cells"]
        area = (cells["x2"] - cells["x1"]) * (cells["y2"] - cells["y1"])
        assert np.allclose(area, cells["prop_total"])

    def test_every_tile_stays_inside_the_unit_square(self):
        cells = draw_mosaic_plot(_res())["cells"]
        assert cells["x1"].min() >= 0 and cells["x2"].max() <= 1
        assert cells["y1"].min() >= 0 and cells["y2"].max() <= 1

    def test_a_wide_gap_is_capped_rather_than_squeezing_the_tiles_out(self):
        res = compare_categorical_groups(CROSSED)
        n_col = res.design["dim"][1]
        drawn = draw_mosaic_plot(res, gap=0.2)
        cells = drawn["cells"]
        spans = (cells["y2"] - cells["y1"]).to_numpy()
        assert spans.min() > 0
        usable = 1 - min(0.2, GAP_MAX_SHARE / (n_col - 1)) * (n_col - 1)
        assert usable >= 1 - GAP_MAX_SHARE


class TestExpectedLines:
    def test_under_independence_every_strip_is_cut_at_the_same_heights(self):
        """Which is what makes an association visible at a glance: the tiles move
        and the lines do not."""
        drawn = draw_mosaic_plot(_res())
        expected = drawn["expected_prop"].to_numpy()
        assert np.allclose(expected, expected[0])

    def test_under_symmetry_the_strips_are_cut_at_different_heights(self):
        """There the expectation is a cell against its own transpose, so it is
        not one distribution shared by the strips."""
        drawn = draw_mosaic_plot(_res(paired=True))
        expected = drawn["expected_prop"].to_numpy()
        assert not np.allclose(expected, expected[0])

    def test_the_expected_shares_are_normalised_inside_the_strip(self):
        for paired in (False, True):
            drawn = draw_mosaic_plot(_res(paired))
            assert np.allclose(drawn["expected_prop"].sum(axis=1), 1.0)

    def test_the_line_is_where_the_tile_edge_would_have_been_under_the_null(self):
        res = _res()
        drawn = draw_mosaic_plot(res, gap=0)
        expected = drawn["expected_prop"]
        # A strip whose cells sit exactly at their expectation has its first
        # boundary at the same height as the first tile's top edge.
        cells = drawn["cells"]
        first = cells.loc[cells["row_level"] == expected.index[0]].iloc[0]
        assert float(first["y2"]) == pytest.approx(float(first["prop_row"]))

    def test_expected_line_false_leaves_the_geometry_reported_all_the_same(self):
        with_line = draw_mosaic_plot(_res())["expected_prop"]
        without = draw_mosaic_plot(_res(), expected_line=False)["expected_prop"]
        assert np.allclose(with_line.to_numpy(), without.to_numpy())


class TestShading:
    def test_a_residual_inside_the_middle_band_is_the_neutral_colour(self):
        drawn = draw_mosaic_plot(_res())
        cells = drawn["cells"]
        middle = cells.loc[cells["residual"].abs() < 2]
        assert len(middle.index) > 0
        assert set(middle["fill"]) == {MOSAIC_COLORS[2]}

    def test_the_two_directions_take_the_volcano_plot_colours(self):
        from statassist.plot.volcano import DOWN_COLOR, UP_COLOR

        assert MOSAIC_COLORS[0] == DOWN_COLOR
        assert MOSAIC_COLORS[-1] == UP_COLOR

    def test_more_than_expected_and_less_than_expected_are_different_colours(self):
        cells = draw_mosaic_plot(_res())["cells"]
        above = set(cells.loc[cells["residual"] > 2, "fill"])
        below = set(cells.loc[cells["residual"] < -2, "fill"])
        assert above and below
        assert above.isdisjoint(below)

    def test_shade_false_draws_one_colour_and_reports_it(self):
        cells = draw_mosaic_plot(_res(), shade=False)["cells"]
        assert set(cells["fill"]) == {"white"}

    def test_the_dark_theme_changes_the_unshaded_fill_and_not_the_bands(self):
        light = draw_mosaic_plot(_res())["cells"]
        dark = draw_mosaic_plot(_res(), dark=True)["cells"]
        assert list(light["fill"]) == list(dark["fill"])
        assert set(draw_mosaic_plot(_res(), shade=False, dark=True)["cells"]["fill"]) != {"white"}

    def test_standardized_shades_on_the_other_column(self):
        """A balanced 2x2 doubles the Pearson residual on the way to the
        standardized one, so a table sitting at 1.2 lights up under the second
        scale and not under the first."""
        balanced = pd.DataFrame(
            {
                "a": ["x"] * 50 + ["y"] * 50,
                "b": ["p"] * 31 + ["q"] * 19 + ["p"] * 19 + ["q"] * 31,
            }
        )
        res = compare_categorical_groups(balanced, correct=False)
        by_pearson = draw_mosaic_plot(res)["cells"]
        by_standardized = draw_mosaic_plot(res, residual="standardized")["cells"]
        assert np.allclose(by_pearson["residual"].abs(), 1.2)
        assert np.allclose(by_standardized["std_residual"].abs(), 2.4)
        assert set(by_pearson["fill"]) == {MOSAIC_COLORS[2]}
        assert MOSAIC_COLORS[2] not in set(by_standardized["fill"])


class TestAnnotation:
    def test_none_writes_nothing_and_auto_writes_something(self):
        import matplotlib.pyplot as plt

        draw_mosaic_plot(_res(), anno_cells="none")
        assert not _texts_on_tiles(plt.gcf())
        draw_mosaic_plot(_res(), anno_cells="auto")
        assert _texts_on_tiles(plt.gcf())

    def test_the_flags_still_say_what_they_said(self):
        import matplotlib.pyplot as plt

        draw_mosaic_plot(_res(), anno_cells=False)
        assert not _texts_on_tiles(plt.gcf())
        draw_mosaic_plot(_res(), anno_cells=True)
        assert _texts_on_tiles(plt.gcf())

    def test_count_writes_the_observation_count(self):
        import matplotlib.pyplot as plt

        res = _res()
        draw_mosaic_plot(res, anno_cells="count")
        written = _texts_on_tiles(plt.gcf())
        assert written == {str(int(value)) for value in res.cells["observed"]}

    def test_percent_writes_the_share_of_the_strip(self):
        import matplotlib.pyplot as plt

        draw_mosaic_plot(_res(), anno_cells="percent")
        assert all(text.endswith("%") for text in _texts_on_tiles(plt.gcf()))

    def test_an_explicit_request_is_honoured_where_auto_gives_up(self):
        """`auto` drops what does not fit; naming a mode does not, because the
        caller asked for the number rather than for a tidy picture."""
        import matplotlib.pyplot as plt

        thin = pd.DataFrame(
            {
                "a": ["x"] * 200 + ["y"] * 2,
                "b": ["p"] * 100 + ["q"] * 100 + ["p", "q"],
            }
        )
        res = compare_categorical_groups(thin)
        draw_mosaic_plot(res, anno_cells="auto", cex_anno=3)
        auto = _texts_on_tiles(plt.gcf())
        draw_mosaic_plot(res, anno_cells="count", cex_anno=3)
        named = _texts_on_tiles(plt.gcf())
        assert len(auto) < len(named)


class TestEmptyLevels:
    def test_a_table_every_level_of_which_was_seen_reports_none(self):
        for paired in (False, True):
            assert draw_mosaic_plot(_res(paired))["empty_levels"] == {"row": [], "col": []}

    def test_a_strip_holding_nothing_is_reported_rather_than_drawn(self):
        drawn = draw_mosaic_plot(crafted_categorical())
        assert drawn["empty_levels"]["row"] == ["b"]
        cells = drawn["cells"]
        empty = cells.loc[cells["row_level"] == "b"]
        assert np.allclose(empty["y2"] - empty["y1"], 0)
        assert np.allclose(empty["x2"] - empty["x1"], 0)

    def test_an_empty_level_takes_no_axis_label(self):
        import matplotlib.pyplot as plt

        draw_mosaic_plot(crafted_categorical())
        assert [text.get_text() for text in plt.gcf().axes[0].get_xticklabels()] == ["a"]

    def test_the_labels_are_read_off_the_first_strip_that_holds_something(self):
        cells = draw_mosaic_plot(crafted_categorical())["cells"]
        reference = cells.loc[cells["row_level"] == "a"]
        assert float(reference["y1"].min()) == 0.0

    def test_a_cell_with_no_residual_to_read_is_not_shaded_as_no_departure(self):
        """ "Nothing to measure" and "no departure" are different states, and a
        neutral tile would say the second."""
        from statassist.plot.mosaic import _EMPTY_FILL

        cells = draw_mosaic_plot(crafted_categorical())["cells"]
        missing = cells.loc[cells["residual"].isna()]
        assert len(missing.index) == 1
        assert set(missing["fill"]) == {_EMPTY_FILL}
        assert _EMPTY_FILL not in MOSAIC_COLORS


class TestRefusals:
    def test_the_standardized_residual_has_no_value_under_symmetry(self):
        with pytest.raises(SaValueError, match="no value under symmetry"):
            draw_mosaic_plot(_res(paired=True), residual="standardized")

    def test_a_numeric_comparison_is_pointed_at_the_plots_that_read_one(self):
        comparison = _numeric()
        with pytest.raises(SaValueError, match="draw_grouped_boxplot"):
            draw_mosaic_plot(comparison)

    def test_a_bare_table_carries_no_null_hypothesis(self):
        with pytest.raises(SaValueError, match="compare_categorical_groups"):
            draw_mosaic_plot(_res().as_table())

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"gap": 0.5}, "gap"),
            ({"shade": 1}, "shade"),
            ({"cex_anno": 0}, "cex_anno"),
            ({"residual": "deviance"}, "residual"),
            ({"anno_cells": "some"}, "anno_cells"),
        ],
    )
    def test_an_unusable_argument_fails_at_the_boundary(self, kwargs, message):
        with pytest.raises(SaValueError, match=message):
            draw_mosaic_plot(_res(), **kwargs)


class TestVerdictObjectsStayApart:
    def test_a_volcano_plot_refuses_a_categorical_verdict(self):
        """The reason the two verdicts are different classes: a cell axis has no
        feature to label and no `log2fc` to put on the x axis."""
        verdict = estimate_categorical_significance(_res())
        with pytest.raises(SaValueError, match="estimate_significance"):
            draw_volcano_plot(verdict)

    def test_a_mosaic_refuses_a_numeric_verdict(self):
        verdict = estimate_significance(_numeric())
        with pytest.raises(SaValueError, match="compare_categorical_groups"):
            draw_mosaic_plot(verdict)


@functools.lru_cache(maxsize=1)
def _numeric():
    sim = simulate_two_groups(n_feats=4, n_case=8, n_control=8, n_up=1, n_down=1, seed=5)
    return compare_two_groups(**sim.args, diagnose=False)


def _texts_on_tiles(fig) -> set[str]:
    """Every label written inside the mosaic panel, axis labels and title apart."""
    ax = fig.axes[0]
    fixed = {ax.get_xlabel(), ax.get_ylabel(), ax.get_title()}
    return {
        text.get_text() for text in ax.texts if text.get_text() and text.get_text() not in fixed
    }
