"""What the volcano plot colours, labels and caps.

The masks that colour the points are the masks that pick the labels, so the two
can never disagree; with the default arguments they reproduce the ``is_signif``
column of the verdict, which is what these tests read them against. The rest is
the axis: a p-value of zero and an infinite fold change have no coordinate, and
both are drawn at the edge rather than dropped.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from statassist import (
    compare_multiple_groups,
    compare_two_groups,
    draw_volcano_plot,
    estimate_significance,
    simulate_multiple_groups,
    simulate_two_groups,
)
from statassist.core.errors import SaValueError


@functools.lru_cache(maxsize=1)
def _verdict():
    sim = simulate_two_groups(n_feats=20, n_up=4, n_down=4, seed=13)
    res = compare_two_groups(**sim.args, diagnose=False)
    return sim, res, estimate_significance(res, log2fc_cutoff=0.5)


def _points(colour):
    """The points drawn in one colour, as (x, y) pairs."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    ax = plt.gcf().axes[0]
    for collection in ax.collections:
        drawn = collection.get_facecolor()
        if drawn.size and np.allclose(drawn[0], to_rgba(colour)):
            return collection.get_offsets()
    return np.empty((0, 2))


class TestMasks:
    def test_the_coloured_points_are_the_ones_the_verdict_called_significant(self):
        from statassist.plot.volcano import DOWN_COLOR, UP_COLOR

        _, _, sig = _verdict()
        table = sig["significance"]
        draw_volcano_plot(sig)
        called = int((table["is_signif"] == True).sum())  # noqa: E712 - NA must not count
        assert len(_points(UP_COLOR)) + len(_points(DOWN_COLOR)) == called

    def test_up_is_drawn_right_of_the_guide_and_down_left_of_it(self):
        from statassist.plot.volcano import DOWN_COLOR, UP_COLOR

        _, _, sig = _verdict()
        cutoff = sig["significance"].attrs["log2fc_cutoff"]
        draw_volcano_plot(sig)
        assert np.all(_points(UP_COLOR)[:, 0] >= cutoff)
        assert np.all(_points(DOWN_COLOR)[:, 0] <= -cutoff)

    def test_a_wider_cutoff_than_the_verdict_used_colours_fewer_points(self):
        from statassist.plot.volcano import DOWN_COLOR, UP_COLOR

        _, _, sig = _verdict()
        draw_volcano_plot(sig)
        loose = len(_points(UP_COLOR)) + len(_points(DOWN_COLOR))
        draw_volcano_plot(sig, log2fc_cutoff=3.0)
        strict = len(_points(UP_COLOR)) + len(_points(DOWN_COLOR))
        assert strict <= loose

    def test_labels_are_capped_at_anno_top_in_each_direction(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        draw_volcano_plot(sig, anno_top=1)
        labels = [text for text in plt.gcf().axes[0].texts if text.get_text()]
        assert len(labels) <= 2

    def test_anno_feats_false_draws_no_labels_at_all(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        draw_volcano_plot(sig, anno_feats=False)
        assert len(plt.gcf().axes[0].texts) == 0


class TestAxis:
    def test_a_p_value_of_zero_is_drawn_inside_the_panel_rather_than_dropped(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        table = sig["significance"].copy()
        table.attrs = dict(sig["significance"].attrs)
        table.loc[table.index[0], "adj_pvalue"] = 0.0
        draw_volcano_plot(table)
        ax = plt.gcf().axes[0]
        assert np.isfinite(ax.get_ylim()).all()
        drawn = np.concatenate([c.get_offsets() for c in ax.collections])
        assert np.all(drawn[:, 1] <= ax.get_ylim()[1])

    def test_an_infinite_log2fc_is_drawn_at_the_edge_it_points_to(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        table = sig["significance"].copy()
        table.attrs = dict(sig["significance"].attrs)
        table.loc[table.index[0], "log2fc"] = np.inf
        draw_volcano_plot(table)
        ax = plt.gcf().axes[0]
        drawn = np.concatenate([c.get_offsets() for c in ax.collections])
        assert np.all(drawn[:, 0] <= ax.get_xlim()[1])

    def test_a_supplied_range_is_used_as_given(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        draw_volcano_plot(sig, xlim=(-1.0, 1.0), ylim=(0.0, 2.0))
        ax = plt.gcf().axes[0]
        assert ax.get_xlim() == (-1.0, 1.0)
        assert ax.get_ylim() == (0.0, 2.0)

    def test_a_derived_range_is_symmetric_and_reaches_the_guides(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        attrs = sig["significance"].attrs
        draw_volcano_plot(sig)
        low, high = plt.gcf().axes[0].get_xlim()
        assert pytest.approx(-low) == high
        assert high >= attrs["log2fc_cutoff"]

    def test_a_multi_group_verdict_says_on_the_axis_what_its_log2fc_compares(self):
        import matplotlib.pyplot as plt

        sim = simulate_multiple_groups(n_feats=5, n_control=15, n_treat=(15, 15), n_up=2, seed=6)
        res = compare_multiple_groups(**sim.args, diagnose=False)
        sig = estimate_significance(res, log2fc_cutoff=0.5)
        draw_volcano_plot(sig)
        label = plt.gcf().axes[0].get_xlabel()
        assert "most extreme level" in label
        assert sim.args["group_lv"][0] in label


class TestArgumentChecks:
    def test_a_contrast_reading_is_sent_back_to_name_one_table(self):
        sim = simulate_multiple_groups(n_feats=4, n_control=15, n_treat=(15, 15), n_up=2, seed=6)
        res = compare_multiple_groups(**sim.args, diagnose=False)
        sig = estimate_significance(res, by="contrast", log2fc_cutoff=0.5)
        with pytest.raises(SaValueError, match="one verdict table per contrast"):
            draw_volcano_plot(sig)

    def test_naming_one_contrast_draws_it(self):
        sim = simulate_multiple_groups(n_feats=4, n_control=15, n_treat=(15, 15), n_up=2, seed=6)
        res = compare_multiple_groups(**sim.args, diagnose=False)
        sig = estimate_significance(res, by="contrast", log2fc_cutoff=0.5)
        first = next(iter(sig["significance"]))
        draw_volcano_plot(sig["significance"][first])

    def test_a_table_without_the_recorded_cutoffs_asks_for_them(self):
        _, _, sig = _verdict()
        stripped = pd.DataFrame(sig["significance"][["features", "log2fc", "adj_pvalue"]])
        with pytest.raises(SaValueError, match="does not carry the cutoffs"):
            draw_volcano_plot(stripped)
        draw_volcano_plot(stripped, log2fc_cutoff=0.5, pval_cutoff=0.05)

    def test_a_missing_column_names_itself(self):
        _, _, sig = _verdict()
        stripped = sig["significance"].drop(columns=["log2fc"])
        with pytest.raises(SaValueError, match="missing the column"):
            draw_volcano_plot(stripped)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"anno_top": -1}, "`anno_top`"),
            ({"cex_anno": 0}, "`cex_anno`"),
            ({"margin": (1, 2, 3)}, "`margin`"),
            ({"xlim": (1.0,)}, "`xlim`"),
            ({"pval_cutoff": 2}, "`pval_cutoff`"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        _, _, sig = _verdict()
        with pytest.raises(SaValueError, match=match):
            draw_volcano_plot(sig, **kwargs)

    def test_the_unadjusted_reading_says_so_on_the_axis(self):
        import matplotlib.pyplot as plt

        _, _, sig = _verdict()
        draw_volcano_plot(sig, use_adjusted=False)
        assert "adjusted" not in plt.gcf().axes[0].get_ylabel()


class TestTermPanels:
    def test_a_term_reading_draws_one_panel_per_default_term(self):
        import matplotlib.pyplot as plt

        from statassist import compare_factorial_groups, simulate_factorial_groups

        sim = simulate_factorial_groups(n_feats=6, n_per_cell=5, seed=21)
        res = compare_factorial_groups(**sim.args, diagnose=False)
        sig = estimate_significance(res, by="term", log2fc_cutoff=0.25)
        draw_volcano_plot(sig, main="by term")
        # Two mains + interaction for a two-factor design.
        assert len([ax for ax in plt.gcf().axes if ax.has_data() or ax.get_title()]) >= 3
        plt.close("all")
