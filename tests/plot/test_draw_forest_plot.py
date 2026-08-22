"""What the forest plot draws, and what it hands back.

A picture is graded by eye, so what is checked here is everything about the plot
that is not the picture: which view was chosen, which rows went into it in which
order, what the axis was scaled to and which arguments are refused. The returned
frame is the plot's own account of itself, so a test that reads it is reading the
same thing the drawing did.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from statassist import (
    compare_multiple_groups,
    compare_one_sample,
    compare_two_groups,
    draw_forest_plot,
    simulate_multiple_groups,
    simulate_two_groups,
)
from statassist.core.errors import SaValueError, SaWarning


@functools.lru_cache(maxsize=1)
def _two_group():
    sim = simulate_two_groups(n_feats=6, n_up=2, n_down=2, seed=11)
    return sim, compare_two_groups(**sim.args, diagnose=False)


@functools.lru_cache(maxsize=1)
def _multi_group():
    """A multi-group comparison whose omnibus test qualifies some features.

    The post-hoc view is the point of it, and the pairwise stage only runs for
    the features the omnibus test passed, so the effects have to be large enough
    for that to happen.
    """
    sim = simulate_multiple_groups(
        n_feats=5, n_control=20, n_treat=(20, 20), n_up=2, n_down=1, seed=4
    )
    return sim, compare_multiple_groups(**sim.args, diagnose=False)


class TestView:
    def test_a_two_group_table_has_intervals_so_auto_draws_the_estimates(self):
        _, res = _two_group()
        drawn = draw_forest_plot(res)
        assert drawn.attrs["view"] == "estimate"
        assert list(drawn["features"]) == list(res["effect"]["features"])

    def test_a_multi_group_omnibus_has_no_estimate_so_auto_falls_through(self):
        _, res = _multi_group()
        drawn = draw_forest_plot(res)
        assert drawn.attrs["view"] in ("posthoc", "pvalue")

    def test_the_posthoc_view_stays_on_one_feature_until_feats_names_more(self):
        _, res = _multi_group()
        qualified = res["posthoc"]["anova_test"]
        if len(qualified.index) == 0:  # pragma: no cover - the fixture qualifies some
            pytest.skip("no feature qualified for the post-hoc stage")
        drawn = draw_forest_plot(res, type="posthoc")
        assert drawn["features"].nunique() == 1
        assert drawn.attrs["view"] == "posthoc"

    def test_naming_two_features_draws_the_contrasts_of_both(self):
        _, res = _multi_group()
        qualified = list(dict.fromkeys(res["posthoc"]["welch_test"]["features"]))
        assert len(qualified) >= 2, "the fixture is meant to qualify several features"
        drawn = draw_forest_plot(res, test="welch_test", type="posthoc", feats=qualified[:2])
        assert set(drawn["features"]) == set(qualified[:2])

    def test_a_feature_without_contrasts_is_left_out_of_the_posthoc_view(self):
        """The pairwise stage runs only for the features the omnibus qualified."""
        _, res = _multi_group()
        table = res["tests"]["welch_test"]
        qualified = set(res["posthoc"]["welch_test"]["features"])
        left_out = [name for name in table["features"] if name not in qualified]
        wanted = [*sorted(qualified)[:1], left_out[0]]
        drawn = draw_forest_plot(res, test="welch_test", type="posthoc", feats=wanted)
        assert set(drawn["features"]) == {wanted[0]}

    def test_asking_for_an_estimate_an_omnibus_table_has_not_got_is_refused(self):
        _, res = _multi_group()
        with pytest.raises(SaValueError, match="holds no estimate to draw"):
            draw_forest_plot(res, type="estimate")

    def test_asking_for_contrasts_a_two_group_result_has_not_got_is_refused(self):
        _, res = _two_group()
        with pytest.raises(SaValueError, match="holds no contrasts to draw"):
            draw_forest_plot(res, type="posthoc")

    def test_the_pvalue_view_is_available_for_every_table(self):
        _, res = _multi_group()
        drawn = draw_forest_plot(res, type="pvalue")
        assert drawn.attrs["view"] == "pvalue"
        assert len(drawn.index) == len(res["tests"]["anova_test"].index)


class TestRows:
    def test_feats_selects_and_orders_the_rows_it_names(self):
        sim, res = _two_group()
        wanted = [sim.args["feats"][2], sim.args["feats"][0]]
        drawn = draw_forest_plot(res, feats=wanted)
        assert list(drawn["features"]) == wanted

    def test_an_unknown_feature_is_refused_with_the_ones_on_offer(self):
        _, res = _two_group()
        with pytest.raises(SaValueError, match="Not found: nope"):
            draw_forest_plot(res, feats=["nope"])

    def test_sorting_by_pvalue_draws_the_most_significant_row_first(self):
        _, res = _two_group()
        drawn = draw_forest_plot(res, sort_by="pvalue")
        p_values = drawn["pval_adj"].to_numpy(dtype=float)
        assert np.all(np.diff(p_values[~np.isnan(p_values)]) >= 0)

    def test_the_unadjusted_p_value_sorts_and_colours_on_its_own_column(self):
        _, res = _two_group()
        drawn = draw_forest_plot(res, sort_by="pvalue", use_adjusted=False)
        p_values = drawn["pval"].to_numpy(dtype=float)
        assert np.all(np.diff(p_values[~np.isnan(p_values)]) >= 0)


class TestAxis:
    def test_a_supplied_xlim_is_used_as_given(self):
        _, res = _two_group()
        draw_forest_plot(res, xlim=(-1.0, 1.0))
        import matplotlib.pyplot as plt

        assert plt.gcf().axes[0].get_xlim() == (-1.0, 1.0)

    def test_a_derived_range_covers_every_interval_that_was_drawn(self):
        _, res = _two_group()
        drawn = draw_forest_plot(res)
        import matplotlib.pyplot as plt

        low, high = plt.gcf().axes[0].get_xlim()
        assert low <= drawn["lower_conf"].min()
        assert high >= drawn["upper_conf"].max()

    def test_the_relative_effect_marks_its_own_null_rather_than_zero(self):
        """Brunner-Munzel is the one estimate in the package whose null is 0.5."""
        _, res = _two_group()
        draw_forest_plot(res, test="robust_test")
        import matplotlib.pyplot as plt

        guides = [
            line.get_xdata()[0]
            for line in plt.gcf().axes[0].get_lines()
            if line.get_linestyle() == "--"
        ]
        assert pytest.approx(0.5) == guides[0]

    def test_a_one_sample_result_is_drawn_from_the_same_three_views(self):
        sim, _ = _two_group()
        # Nothing here is binary, so the proportion test reports that it could
        # not run; the two tests that did are what is being drawn.
        with pytest.warns(SaWarning, match="proportion test"):
            res = compare_one_sample(sim.args["data"], sim.args["feats"], mu=0, diagnose=False)
        drawn = draw_forest_plot(res)
        assert drawn.attrs["view"] == "estimate"


class TestArgumentChecks:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"type": "nope"}, "`type` must be one of"),
            ({"sort_by": "nope"}, "`sort_by` must be"),
            ({"alpha": 0}, "`alpha`"),
            ({"dark": "yes"}, "`dark`"),
            ({"xlim": (1.0,)}, "`xlim`"),
            ({"test": "nope"}, "`test` must name one of"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, kwargs, match):
        _, res = _two_group()
        with pytest.raises(SaValueError, match=match):
            draw_forest_plot(res, **kwargs)

    def test_the_dark_theme_changes_the_background_and_nothing_else(self):
        _, res = _two_group()
        plain = draw_forest_plot(res)
        dark = draw_forest_plot(res, dark=True)
        assert list(plain["features"]) == list(dark["features"])
        import matplotlib.pyplot as plt

        assert plt.gcf().get_facecolor() != (1.0, 1.0, 1.0, 1.0)
