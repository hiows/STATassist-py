"""``draw_interaction_plot`` views and return contract."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from statassist import (  # noqa: E402
    compare_factorial_groups,
    draw_interaction_plot,
    simulate_factorial_groups,
)
from statassist.core.errors import SaValueError  # noqa: E402


@pytest.fixture
def factorial():
    sim = simulate_factorial_groups(n_feats=4, n_per_cell=6, seed=11)
    return compare_factorial_groups(**sim.args, diagnose=False)


class TestPairwise:
    def test_auto_pairwise_for_two_factors_returns_view_attr(self, factorial):
        drawn = draw_interaction_plot(factorial)
        assert drawn.attrs["view"] == "pairwise"
        assert set(drawn["features"]) == set(factorial.features)
        plt.close("all")

    def test_named_factors_flip_the_axes(self, factorial):
        drawn = draw_interaction_plot(factorial, x="treatment", trace="sex")
        assert drawn.attrs["view"] == "pairwise"
        assert set(drawn["x_factor"]) == {"treatment"}
        assert set(drawn["trace_factor"]) == {"sex"}
        plt.close("all")


class TestMatrix:
    def test_three_factor_auto_is_matrix(self):
        sim = simulate_factorial_groups(
            n_feats=2,
            n_per_cell=5,
            seed=12,
            factor_lv={
                "treatment": ["control", "treat_A"],
                "sex": ["male", "female"],
                "site": ["north", "south"],
            },
        )
        res = compare_factorial_groups(**sim.args, diagnose=False)
        drawn = draw_interaction_plot(res)
        assert drawn.attrs["view"] == "matrix"
        assert drawn["features"].nunique() == 1
        plt.close("all")


class TestFacet:
    def test_facet_keeps_the_third_factor_apart(self):
        sim = simulate_factorial_groups(
            n_feats=2,
            n_per_cell=5,
            seed=13,
            factor_lv={
                "treatment": ["control", "treat_A"],
                "sex": ["male", "female"],
                "site": ["north", "south"],
            },
        )
        res = compare_factorial_groups(**sim.args, diagnose=False)
        drawn = draw_interaction_plot(res, x="site", trace="treatment", facet="sex", type="facet")
        assert drawn.attrs["view"] == "facet"
        assert drawn["panel"].nunique() == 2
        plt.close("all")


class TestRefusal:
    def test_a_non_factorial_result_is_refused(self):
        from statassist import compare_two_groups, simulate_two_groups

        sim = simulate_two_groups(n_feats=3, n_up=1, n_down=1, seed=1)
        res = compare_two_groups(**sim.args, diagnose=False)
        with pytest.raises(SaValueError, match="factorial comparison"):
            draw_interaction_plot(res)
