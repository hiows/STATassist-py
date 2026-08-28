"""The two pictures of an evaluation.

Both are drawn from the tables the evaluation already computed rather than from
anything recomputed, which is what makes the picture and the numbers beside it
describe the same thing. That is what most of these tests check: the coordinates
on the figure are the ones in ``curves``, and the line in a panel is the
``calib_slope`` and ``calib_intercept`` of ``metrics``.

The two are also not interchangeable, and the object that carries one carries the
names of the other's slots, so the refusals are checked as carefully as the
drawing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from matplotlib.lines import Line2D

from statassist import (
    draw_prediction_plot,
    draw_roc_curve,
    evaluate_classification_models,
    evaluate_regression_models,
    fit_linear_regression,
    fit_logistic_regression,
    simulate_classification,
    simulate_regression,
)
from statassist.core import SaValueError
from statassist.plot import PREDICTION_VIEWS


def _figure():
    import matplotlib.pyplot as plt

    return plt.gcf()


def _curves(ax):
    """The lines an axes carries, longest first, so a guide is easy to skip."""
    return [artist for artist in ax.lines if isinstance(artist, Line2D)]


@pytest.fixture(scope="module")
def scored_reg():
    sim = simulate_regression(n_samples=160, n_pred=4, n_factor_pred=1, seed=41)
    frame = sim.args["data"]
    train, test = frame.iloc[:110], frame.iloc[110:]
    predictors = list(sim.args["predictors"])
    full = fit_linear_regression(
        train, outcome=sim.args["outcome"], predictors=predictors, cv=False
    )
    thin = fit_linear_regression(
        train, outcome=sim.args["outcome"], predictors=predictors[:1], cv=False
    )
    return evaluate_regression_models(full, {"one_predictor": thin}, newdata=test)


@pytest.fixture(scope="module")
def scored_clf():
    sim = simulate_classification(n_samples=200, n_pred=4, seed=42)
    frame = sim.args["data"]
    train, test = frame.iloc[:140], frame.iloc[140:]
    predictors = list(sim.args["predictors"])
    full = fit_logistic_regression(
        train,
        outcome=sim.args["outcome"],
        predictors=predictors,
        outcome_lv=sim.args["outcome_lv"],
        cv=False,
    )
    thin = fit_logistic_regression(
        train,
        outcome=sim.args["outcome"],
        predictors=predictors[:1],
        outcome_lv=sim.args["outcome_lv"],
        cv=False,
    )
    return evaluate_classification_models(full, {"one_predictor": thin}, newdata=test)


@pytest.fixture(scope="module")
def fitted_model():
    sim = simulate_regression(n_samples=60, n_pred=2, seed=43)
    return fit_linear_regression(**sim.args, cv=False)


class TestTheRocCurve:
    def test_it_returns_the_rows_of_metrics_that_were_drawn(self, scored_clf):
        drawn = draw_roc_curve(scored_clf)
        assert drawn["model"].tolist() == scored_clf["models"]

    def test_one_curve_is_drawn_per_model_beside_the_chance_diagonal(self, scored_clf):
        draw_roc_curve(scored_clf)
        ax = _figure().axes[0]
        assert len(_curves(ax)) == len(scored_clf["models"]) + 1

    def test_the_chance_diagonal_can_be_left_off(self, scored_clf):
        draw_roc_curve(scored_clf, chance=False)
        ax = _figure().axes[0]
        assert len(_curves(ax)) == len(scored_clf["models"])

    def test_the_points_on_the_figure_are_the_ones_in_the_curve_table(self, scored_clf):
        """Nothing is recomputed here, which is what stops the picture and the
        `auc` column from describing different curves."""
        draw_roc_curve(scored_clf, chance=False)
        ax = _figure().axes[0]
        curves = scored_clf["curves"]
        for line, name in zip(_curves(ax), scored_clf["models"], strict=True):
            mine = curves.loc[curves["model"] == name]
            x, y = line.get_xdata(), line.get_ydata()
            assert len(x) == len(mine.index)
            assert set(np.round(x, 12)) == set(np.round(1 - mine["specificity"], 12))
            assert set(np.round(y, 12)) == set(np.round(mine["sensitivity"], 12))

    def test_the_curve_is_walked_left_to_right_so_a_run_of_ties_is_crossed_once(self, scored_clf):
        draw_roc_curve(scored_clf, chance=False)
        ax = _figure().axes[0]
        for line in _curves(ax):
            assert np.all(np.diff(line.get_xdata()) >= 0)

    def test_the_area_under_the_drawn_curve_is_the_auc_in_the_legend(self, scored_clf):
        drawn = draw_roc_curve(scored_clf, chance=False)
        ax = _figure().axes[0]
        for line, area in zip(_curves(ax), drawn["auc"], strict=True):
            drawn_area = float(np.trapezoid(line.get_ydata(), line.get_xdata()))
            assert drawn_area == pytest.approx(area, abs=1e-8)

    def test_the_axes_are_the_unit_square(self, scored_clf):
        draw_roc_curve(scored_clf)
        ax = _figure().axes[0]
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)

    def test_the_title_names_the_class_being_ranked_against_the_other(self, scored_clf):
        draw_roc_curve(scored_clf)
        reference, event = scored_clf["design"]["outcome_lv"]
        assert _figure().axes[0].get_title() == f"ROC: {event} against {reference}"

    def test_the_legend_carries_the_model_names_and_the_auc_when_asked(self, scored_clf):
        draw_roc_curve(scored_clf)
        plain = [text.get_text() for text in _figure().axes[0].get_legend().get_texts()]
        assert plain == scored_clf["models"]

        draw_roc_curve(scored_clf, anno_auc=True)
        annotated = [text.get_text() for text in _figure().axes[0].get_legend().get_texts()]
        for entry, name in zip(annotated, scored_clf["models"], strict=True):
            assert entry.startswith(name + "  (")

    def test_the_legend_can_be_left_off(self, scored_clf):
        draw_roc_curve(scored_clf, legend_pos=None)
        assert _figure().axes[0].get_legend() is None

    def test_models_can_be_named_to_choose_which_curves_are_drawn_and_in_what_order(
        self, scored_clf
    ):
        reversed_order = scored_clf["models"][::-1]
        drawn = draw_roc_curve(scored_clf, models=reversed_order)
        assert drawn["model"].tolist() == reversed_order

        drawn = draw_roc_curve(scored_clf, models=scored_clf["models"][1])
        assert drawn["model"].tolist() == [scored_clf["models"][1]]
        assert len(_curves(_figure().axes[0])) == 2

    def test_each_model_is_drawn_in_a_colour_of_its_own(self, scored_clf):
        draw_roc_curve(scored_clf, chance=False)
        colours = {line.get_color() for line in _curves(_figure().axes[0])}
        assert len(colours) == len(scored_clf["models"])

    def test_one_line_type_applies_to_every_curve_and_a_set_applies_one_each(self, scored_clf):
        draw_roc_curve(scored_clf, chance=False, lty="dashed")
        styles = {line.get_linestyle() for line in _curves(_figure().axes[0])}
        assert styles == {"--"}

        draw_roc_curve(scored_clf, chance=False, lty=[1, 2])
        styles = [line.get_linestyle() for line in _curves(_figure().axes[0])]
        assert styles == ["-", "--"]

    def test_a_dark_background_darkens_the_figure_rather_than_the_curves(self, scored_clf):
        draw_roc_curve(scored_clf, dark=True)
        fig = _figure()
        assert fig.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
        assert fig.axes[0].get_title() != ""


class TestThePredictionPlot:
    def test_it_carries_the_view_it_resolved(self, scored_reg):
        """`type="auto"` is settled inside, so the caller would otherwise have no
        way to find out which of the two it got."""
        assert draw_prediction_plot(scored_reg).attrs["view"] == "panel"
        assert draw_prediction_plot(scored_reg, type="overlay").attrs["view"] == "overlay"
        single = draw_prediction_plot(scored_reg, models=scored_reg["models"][0])
        assert single.attrs["view"] == "overlay"

    def test_one_panel_is_drawn_per_model_and_one_axes_for_an_overlay(self, scored_reg):
        draw_prediction_plot(scored_reg, type="panel")
        assert len(_figure().axes) == len(scored_reg["models"])
        draw_prediction_plot(scored_reg, type="overlay")
        assert len(_figure().axes) == 1

    def test_the_panels_can_be_stacked(self, scored_reg):
        draw_prediction_plot(scored_reg, type="panel", panel_nrow=2)
        positions = {axes.get_position().y0 for axes in _figure().axes}
        assert len(positions) == 2

    def test_both_axes_span_the_same_range_in_every_panel(self, scored_reg):
        """So the panels are comparable and the identity line is the diagonal of
        the square rather than an arbitrary chord."""
        draw_prediction_plot(scored_reg, type="panel")
        spans = {(axes.get_xlim(), axes.get_ylim()) for axes in _figure().axes}
        assert len(spans) == 1
        (x_span, y_span) = next(iter(spans))
        assert x_span == y_span

    def test_the_range_covers_every_value_drawn(self, scored_reg):
        draw_prediction_plot(scored_reg, type="overlay")
        low, high = _figure().axes[0].get_xlim()
        held = scored_reg["predictions"]
        assert low <= float(held["observed"].min())
        assert low <= float(held["predicted"].min())
        assert high >= float(held["observed"].max())
        assert high >= float(held["predicted"].max())

    def test_the_range_can_be_fixed(self, scored_reg):
        draw_prediction_plot(scored_reg, type="overlay", lim=(-3, 3))
        assert _figure().axes[0].get_xlim() == (-3.0, 3.0)
        assert _figure().axes[0].get_ylim() == (-3.0, 3.0)

    def test_the_points_drawn_are_the_predictions_of_that_model(self, scored_reg):
        draw_prediction_plot(scored_reg, type="panel")
        held = scored_reg["predictions"]
        for axes, name in zip(_figure().axes, scored_reg["models"], strict=True):
            mine = held.loc[held["model"] == name]
            offsets = axes.collections[0].get_offsets()
            assert len(offsets) == len(mine.index)
            assert np.allclose(np.sort(offsets[:, 0]), np.sort(mine["observed"]))
            assert np.allclose(np.sort(offsets[:, 1]), np.sort(mine["predicted"]))

    def test_the_points_can_be_left_off_which_is_what_makes_an_overlay_readable(self, scored_reg):
        draw_prediction_plot(scored_reg, type="overlay", points=False)
        assert len(_figure().axes[0].collections) == 0

    def test_the_line_in_a_panel_is_the_calibration_line_the_metrics_report(self, scored_reg):
        """Taken from the two numbers the table holds rather than fitted again, so
        the picture and `metrics` cannot drift apart."""
        draw_prediction_plot(scored_reg, type="panel", points=False)
        for position, axes in enumerate(_figure().axes):
            row = scored_reg["metrics"].iloc[position]
            # The identity is drawn first, so the calibration line is the second.
            line = _curves(axes)[1]
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            assert slope == pytest.approx(float(row["calib_slope"]))
            assert y[0] - slope * x[0] == pytest.approx(float(row["calib_intercept"]))

    def test_the_identity_is_the_diagonal_of_the_square(self, scored_reg):
        draw_prediction_plot(scored_reg, type="overlay", points=False)
        axes = _figure().axes[0]
        identity = _curves(axes)[0]
        assert tuple(identity.get_xdata()) == axes.get_xlim()
        assert tuple(identity.get_ydata()) == axes.get_ylim()

    def test_the_annotations_are_written_only_when_asked(self, scored_reg):
        draw_prediction_plot(scored_reg, type="panel")
        assert all(len(axes.texts) == 0 for axes in _figure().axes)

        draw_prediction_plot(scored_reg, type="panel", anno_corr=True, anno_rsq=True)
        for position, axes in enumerate(_figure().axes):
            row = scored_reg["metrics"].iloc[position]
            written = " ".join(text.get_text() for text in axes.texts)
            assert "Corr =" in written
            assert "R-sq =" in written
            # Reported separately for the same reason the table carries both:
            # `r_squared` is measured against the outcome, not against a refitted
            # line, so the two are different numbers.
            assert float(row["r_squared"]) != pytest.approx(float(row["cor"]) ** 2)

    def test_the_calibration_line_is_annotated_as_an_equation(self, scored_reg):
        draw_prediction_plot(scored_reg, type="panel", anno_lm=True)
        for axes in _figure().axes:
            written = " ".join(text.get_text() for text in axes.texts)
            assert written.startswith("y = ")
            assert "x " in written

    def test_a_panel_is_titled_with_its_model_and_the_figure_with_main(self, scored_reg):
        draw_prediction_plot(scored_reg, type="panel", main="Held out")
        assert [axes.get_title() for axes in _figure().axes] == scored_reg["models"]
        assert _figure()._suptitle.get_text() == "Held out"

    def test_the_x_axis_names_the_outcome_that_was_predicted(self, scored_reg):
        draw_prediction_plot(scored_reg, type="overlay")
        axes = _figure().axes[0]
        assert axes.get_xlabel() == f"Observed {scored_reg['design']['outcome']}"
        assert axes.get_ylabel() == "Predicted"

    def test_a_model_with_no_calibration_line_still_earns_a_legend_entry(self, scored_reg):
        """A model the rows defeated has nothing to draw and is still one of the
        models, so leaving it out of the legend would renumber the colours."""
        broken = scored_reg["metrics"].copy()
        broken.loc[broken.index[1], "calib_slope"] = math.nan
        patched = type(scored_reg)({**scored_reg.to_dict(), "metrics": broken})
        draw_prediction_plot(patched, type="overlay", points=False)
        entries = [text.get_text() for text in _figure().axes[0].get_legend().get_texts()]
        assert entries == patched["models"]

    def test_a_model_with_no_calibration_line_says_so_in_its_annotation(self, scored_reg):
        broken = scored_reg["metrics"].copy()
        broken.loc[broken.index[1], "calib_slope"] = math.nan
        patched = type(scored_reg)({**scored_reg.to_dict(), "metrics": broken})
        draw_prediction_plot(patched, type="panel", anno_lm=True)
        written = " ".join(text.get_text() for text in _figure().axes[1].texts)
        assert written == "no calibration line"


class TestWhatIsRefused:
    def test_the_two_pictures_are_not_interchangeable(self, scored_reg, scored_clf):
        with pytest.raises(SaValueError, match="Use draw_roc_curve"):
            draw_prediction_plot(scored_clf)
        with pytest.raises(SaValueError, match="Use draw_prediction_plot"):
            draw_roc_curve(scored_reg)

    def test_a_fitted_model_is_told_what_it_is_missing(self, fitted_model):
        with pytest.raises(SaValueError, match="rather than an evaluation of one"):
            draw_roc_curve(fitted_model)
        with pytest.raises(SaValueError, match="rather than an evaluation of one"):
            draw_prediction_plot(fitted_model)

    def test_something_else_entirely_is_pointed_at_the_two_functions(self):
        with pytest.raises(SaValueError, match="must be an evaluation result"):
            draw_roc_curve({"metrics": None})

    def test_an_unknown_model_name_lists_the_ones_the_evaluation_holds(self, scored_clf):
        with pytest.raises(SaValueError, match="Available: baseline, one_predictor"):
            draw_roc_curve(scored_clf, models=["nothing"])

    def test_a_duplicated_model_name_is_refused(self, scored_clf):
        with pytest.raises(SaValueError, match="duplicated names: baseline"):
            draw_roc_curve(scored_clf, models=["baseline", "baseline"])

    def test_an_empty_set_of_models_is_refused_rather_than_read_as_all_of_them(self, scored_clf):
        with pytest.raises(SaValueError, match="non-empty sequence of model names"):
            draw_roc_curve(scored_clf, models=[])

    def test_a_colour_set_that_is_neither_one_nor_one_each_is_refused(self, scored_clf):
        """Recycling a shorter set would give two models the same colour, which is
        the one thing an overlaid plot cannot survive."""
        with pytest.raises(SaValueError, match=r"one per drawn model \(2\)"):
            draw_roc_curve(scored_clf, col=["red", "blue", "green"])

    def test_a_line_type_set_that_is_neither_one_nor_one_each_is_refused(self, scored_clf):
        with pytest.raises(SaValueError, match=r"one per drawn model \(2\)"):
            draw_roc_curve(scored_clf, lty=[1, 2, 3])

    def test_an_unknown_view_lists_the_ones_there_are(self, scored_reg):
        with pytest.raises(SaValueError, match="`type` must be one of"):
            draw_prediction_plot(scored_reg, type="grid")
        assert PREDICTION_VIEWS[0] == "auto"
