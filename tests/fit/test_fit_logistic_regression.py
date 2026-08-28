"""What a fitted logistic regression reports, and which way it points.

The direction rule is most of this file. A classification has a class of
interest, and every coefficient, odds ratio and predicted probability in the
result is a statement about that class rather than the other one, so a test that
does not pin the direction down has not tested the model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statassist import fit_logistic_regression, simulate_classification
from statassist.core import SaValueError, SaWarning


@pytest.fixture(scope="module")
def sim():
    """A classification with planted coefficients and a known event rate."""
    return simulate_classification(n_samples=300, n_pred=6, n_pos=2, n_neg=1, seed=13)


@pytest.fixture(scope="module")
def fitted(sim):
    return fit_logistic_regression(**sim.args, cv=False, seed=1)


def _args(sim, **override):
    """The simulator's own call, with an argument said differently.

    ``sim.args`` already names ``outcome_lv``, since the simulator knows which
    class it planted the coefficients for. Overriding it takes a copy rather than
    a second keyword.
    """
    return {**sim.args, **override}


class TestWhatItFitted:
    def test_the_analysis_names_the_model(self, fitted):
        assert fitted["analysis"] == "logistic_regression"

    def test_the_coefficient_table_is_in_the_order_terms_gives(self, fitted):
        assert fitted["coefficients"]["terms"].tolist() == fitted["terms"]

    def test_the_design_records_the_two_classes_and_how_many_events(self, sim, fitted):
        design = fitted["design"]
        assert design["outcome_type"] == "two classes"
        assert design["outcome_lv"] == ["control", "case"]
        events = int((sim.args["data"][sim.args["outcome"]] == "case").sum())
        assert design["n_events"] == events
        assert design["event_rate"] == pytest.approx(events / design["n_used"])

    def test_the_statistic_is_a_wald_z_and_is_referred_to_no_degrees_of_freedom(self, fitted):
        table = fitted["coefficients"]
        assert bool(table["df"].isna().all())
        assert float(table["statistic"].iloc[1]) == pytest.approx(
            float(table["estimate"].iloc[1]) / float(table["stderr"].iloc[1])
        )

    def test_the_odds_ratio_columns_are_the_exponentiated_ones(self, fitted):
        table = fitted["coefficients"]
        assert np.allclose(table["odds_ratio"], np.exp(table["estimate"]))
        assert np.allclose(table["or_lower_conf"], np.exp(table["lower_conf"]))
        assert np.allclose(table["or_upper_conf"], np.exp(table["upper_conf"]))


class TestDirection:
    def test_the_second_level_is_the_class_the_coefficients_describe(self, sim, fitted):
        """A planted positive coefficient raises the chance of ``case``, so with
        ``case`` second its odds ratio is above 1."""
        truth = sim.truth_term.set_index("terms")["beta"]
        table = fitted["coefficients"].set_index("terms")
        for term in truth[truth > 0].index:
            assert float(table.loc[term, "odds_ratio"]) > 1
        for term in truth[truth < 0].index:
            assert float(table.loc[term, "odds_ratio"]) < 1

    def test_swapping_the_levels_turns_every_coefficient_around(self, sim, fitted):
        flipped = fit_logistic_regression(**_args(sim, outcome_lv=["case", "control"]), cv=False)
        assert np.allclose(flipped["coefficients"]["estimate"], -fitted["coefficients"]["estimate"])

    def test_swapping_the_levels_leaves_the_uncertainty_alone(self, sim, fitted):
        flipped = fit_logistic_regression(**_args(sim, outcome_lv=["case", "control"]), cv=False)
        assert np.allclose(flipped["coefficients"]["stderr"], fitted["coefficients"]["stderr"])
        assert np.allclose(flipped["coefficients"]["pval"], fitted["coefficients"]["pval"])

    def test_naming_the_reference_alone_says_the_same_thing(self, sim, fitted):
        named = fit_logistic_regression(
            **_args(sim, outcome_lv=None), control_label="control", cv=False
        )
        pd.testing.assert_frame_equal(named["coefficients"], fitted["coefficients"])

    def test_swapping_the_levels_swaps_which_rows_are_the_events(self, sim, fitted):
        flipped = fit_logistic_regression(**_args(sim, outcome_lv=["case", "control"]), cv=False)
        assert (
            flipped["design"]["n_events"] + fitted["design"]["n_events"]
            == fitted["design"]["n_used"]
        )

    def test_the_default_order_is_the_sorted_one(self):
        frame = pd.DataFrame({"y": ["treated", "control"] * 20, "x": list(range(40))})
        fit = fit_logistic_regression(frame, outcome="y", cv=False)
        assert fit["design"]["outcome_lv"] == ["control", "treated"]


class TestTheEstimates:
    def test_the_fit_solves_the_likelihood_equations(self, fitted):
        """Which is what makes it a maximum likelihood fit: at the optimum the
        design is orthogonal to the difference between the outcome and the fitted
        probabilities."""
        engine = fitted.fit
        probability = np.asarray(engine.estimator.predict_proba(engine.x))[:, 1]
        matrix = np.column_stack([np.ones(len(engine.x.index)), np.asarray(engine.x, dtype=float)])
        assert np.allclose(matrix.T @ (engine.y - probability), 0, atol=1e-4)

    def test_a_planted_coefficient_is_recovered_within_its_own_uncertainty(self, sim, fitted):
        truth = sim.truth_term.set_index("terms")["beta"]
        table = fitted["coefficients"].set_index("terms")
        for term in truth[truth != 0].index:
            row = table.loc[term]
            assert abs(float(row["estimate"]) - float(truth[term])) < 3 * float(row["stderr"])

    def test_the_intercept_alone_would_predict_the_event_rate(self):
        """The one coefficient a logistic regression has a closed form for."""
        rng = np.random.default_rng(5)
        y = np.where(rng.random(200) < 0.3, "case", "control")
        frame = pd.DataFrame({"y": y, "x": rng.normal(size=200)})
        fit = fit_logistic_regression(
            frame, outcome="y", outcome_lv=["control", "case"], predictors=["x"], cv=False
        )
        rate = fit["design"]["event_rate"]
        # `x` is noise, so the intercept sits near the log odds of the rate.
        assert float(fit["coefficients"]["estimate"].iloc[0]) == pytest.approx(
            math.log(rate / (1 - rate)), abs=0.2
        )

    def test_a_predictor_that_carries_nothing_is_reported_as_carrying_nothing(self, sim, fitted):
        truth = sim.truth_term.set_index("terms")["beta"]
        table = fitted["coefficients"].set_index("terms")
        null = truth[truth == 0].index
        assert len(null) > 0
        assert bool((table.loc[null, "pval"] > 0.01).all())


class TestFitStats:
    def test_the_residual_deviance_is_below_the_null_one(self, fitted):
        stats = fitted["fit_stats"]
        assert stats["residual_deviance"] < stats["null_deviance"]

    def test_mcfadden_is_the_share_of_the_null_deviance_explained(self, fitted):
        stats = fitted["fit_stats"]
        assert stats["mcfadden_r2"] == pytest.approx(
            1 - stats["residual_deviance"] / stats["null_deviance"]
        )
        assert 0 < stats["mcfadden_r2"] < 1

    def test_the_likelihood_ratio_test_is_the_difference_of_the_two_deviances(self, fitted):
        stats = fitted["fit_stats"]
        assert stats["lr_stat"] == pytest.approx(
            stats["null_deviance"] - stats["residual_deviance"]
        )
        assert stats["lr_df"] == len(fitted["terms"]) - 1
        assert stats["lr_pval"] < 1e-6

    def test_the_degrees_of_freedom_are_the_ones_a_glm_reports(self, fitted):
        stats = fitted["fit_stats"]
        n_used = fitted["design"]["n_used"]
        assert stats["df_null"] == n_used - 1
        assert stats["df_residual"] == n_used - len(fitted["terms"])

    def test_the_criteria_penalise_the_deviance_by_the_parameter_count(self, fitted):
        stats = fitted["fit_stats"]
        rank = len(fitted["terms"])
        assert stats["aic"] == pytest.approx(stats["residual_deviance"] + 2 * rank)
        assert stats["bic"] > stats["aic"]


class TestResampling:
    def test_the_folds_are_stratified_so_a_fold_cannot_hold_one_class(self):
        """Unstratified folds of a rare event give a fold with no event in it,
        and a fold with one class has no accuracy worth reporting."""
        rng = np.random.default_rng(8)
        n = 80
        y = np.array(["control"] * n)
        y[rng.permutation(n)[:12]] = "case"
        frame = pd.DataFrame({"y": y, "x": rng.normal(size=n)})
        fit = fit_logistic_regression(
            frame, outcome="y", cv=True, cv_method="kfold", n_fold=5, seed=1
        )
        assert bool(fit["resampling"]["Accuracy"].notna().all())

    def test_the_metrics_are_the_classification_ones(self, sim):
        fit = fit_logistic_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        assert fit["engine"]["metrics"] == ["Accuracy", "Kappa"]
        assert 0 <= float(fit["performance"]["Accuracy"].iloc[0]) <= 1

    def test_a_model_with_planted_signal_beats_guessing_the_common_class(self, sim):
        fit = fit_logistic_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        rate = fit["design"]["event_rate"]
        assert float(fit["performance"]["Accuracy"].iloc[0]) > max(rate, 1 - rate)

    def test_the_coefficients_do_not_depend_on_whether_it_was_scored(self, sim, fitted):
        scored = fit_logistic_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        pd.testing.assert_frame_equal(scored["coefficients"], fitted["coefficients"])


class TestPredict:
    def test_the_raw_prediction_is_a_class_label_rather_than_a_code(self, sim, fitted):
        predicted = fitted.predict(sim.args["data"])
        assert set(np.unique(predicted)) <= set(fitted["design"]["outcome_lv"])

    def test_the_response_is_the_probability_of_the_class_being_modelled(self, sim, fitted):
        probability = fitted.predict(sim.args["data"], type="response")
        assert bool(((probability >= 0) & (probability <= 1)).all())
        # Which class that is has to be the second level, not whichever the
        # engine happened to label 1.
        event = fitted["design"]["outcome_lv"][1]
        table = fitted.predict(sim.args["data"], type="prob")
        assert np.allclose(probability, table[event])

    def test_the_raw_prediction_agrees_with_the_probability_it_came_from(self, sim, fitted):
        levels = fitted["design"]["outcome_lv"]
        probability = fitted.predict(sim.args["data"], type="response")
        predicted = fitted.predict(sim.args["data"])
        expected = np.where(probability > 0.5, levels[1], levels[0])
        assert (predicted == expected).all()

    def test_the_probability_table_has_one_column_per_class_and_they_sum_to_one(self, sim, fitted):
        table = fitted.predict(sim.args["data"], type="prob")
        assert list(table.columns) == fitted["design"]["outcome_lv"]
        assert np.allclose(table.sum(axis=1), 1)

    def test_a_row_missing_a_predictor_gets_a_missing_prediction_of_every_kind(self, sim, fitted):
        rows = sim.args["data"].copy()
        rows.loc[rows.index[0], sim.args["predictors"][0]] = np.nan
        assert fitted.predict(rows)[0] is None
        assert math.isnan(float(fitted.predict(rows, type="response")[0]))
        assert bool(fitted.predict(rows, type="prob").iloc[0].isna().all())

    def test_swapping_the_levels_swaps_the_predicted_probability(self, sim, fitted):
        flipped = fit_logistic_regression(**_args(sim, outcome_lv=["case", "control"]), cv=False)
        rows = sim.args["data"]
        assert np.allclose(
            flipped.predict(rows, type="response"),
            1 - fitted.predict(rows, type="response"),
        )


class TestRefusals:
    def test_a_third_class_is_an_error_rather_than_dropped_rows(self):
        frame = pd.DataFrame({"y": ["a", "b", "c"] * 10, "x": list(range(30))})
        with pytest.raises(SaValueError, match="holds 3 classes"):
            fit_logistic_regression(frame, outcome="y", cv=False)

    def test_naming_two_of_three_would_leave_rows_out(self):
        frame = pd.DataFrame({"y": ["a", "b", "c"] * 10, "x": list(range(30))})
        with pytest.raises(SaValueError, match="silently left out"):
            fit_logistic_regression(frame, outcome="y", outcome_lv=["a", "b"], cv=False)

    def test_the_two_ways_of_naming_the_reference_must_agree(self, sim):
        with pytest.raises(SaValueError, match="disagree about which class"):
            fit_logistic_regression(**sim.args, control_label="case", cv=False)

    def test_a_single_class_has_nothing_to_classify(self):
        frame = pd.DataFrame({"y": ["a"] * 20, "x": list(range(20))})
        with pytest.raises(SaValueError, match="nothing to classify"):
            fit_logistic_regression(frame, outcome="y", cv=False)

    def test_a_numeric_outcome_is_read_as_class_labels(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"y": rng.integers(0, 2, 60), "x": rng.normal(size=60)})
        fit = fit_logistic_regression(frame, outcome="y", cv=False)
        assert fit["design"]["outcome_lv"] == ["0", "1"]


class TestAliasedTerms:
    def test_a_term_the_data_cannot_support_is_named_and_left_missing(self):
        rng = np.random.default_rng(10)
        a = rng.normal(size=80)
        frame = pd.DataFrame(
            {
                "y": np.where(a + rng.normal(0, 0.5, 80) > 0, "case", "control"),
                "a": a,
                "b": 3 * a,
            }
        )
        with pytest.warns(SaWarning, match="already span them"):
            fit = fit_logistic_regression(frame, outcome="y", cv=False)
        table = fit["coefficients"].set_index("terms")
        assert math.isnan(float(table.loc["b", "estimate"]))
        assert math.isfinite(float(table.loc["a", "estimate"]))
