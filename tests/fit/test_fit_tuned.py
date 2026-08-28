"""The three models whose settings are chosen by resampling rather than fitted.

They have almost nothing in common statistically - a penalty, a forest and a
kernel - which is the point of testing them together: what has to hold is the
contract, that a caller who can read one result can read the other two.

What is specific to each is checked separately: that a penalty shrinks, that a
forest scores itself out of bag, and that a machine reports how many rows it
needed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statassist import (
    fit_elastic_net,
    fit_rf,
    fit_svm,
    simulate_classification,
    simulate_regression,
)
from statassist.core import SaValueError


@pytest.fixture(scope="module")
def reg():
    """A regression with a factor predictor, so the coding is exercised."""
    return simulate_regression(n_samples=140, n_pred=5, n_factor_pred=1, seed=21)


@pytest.fixture(scope="module")
def clf():
    return simulate_classification(n_samples=180, n_pred=4, n_factor_pred=1, seed=22)


def _args(sim, **override):
    """The simulator's own call, with an argument said differently."""
    return {**sim.args, **override}


@pytest.fixture(scope="module")
def fitted(reg):
    """One fit of each, on the same data, so the contract can be compared."""
    return {
        "elastic_net": fit_elastic_net(**reg.args, penalty="lasso", lambda_=0.2, cv=False),
        "random_forest": fit_rf(**reg.args, ntree=60, cv=False, seed=1),
        "svm": fit_svm(**reg.args, C=1, cv=False, seed=1),
    }


ANALYSES = ["elastic_net", "random_forest", "svm"]


class TestTheSharedContract:
    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_the_analysis_names_the_model(self, fitted, analysis):
        assert fitted[analysis]["analysis"] == analysis

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_every_slot_of_the_contract_is_there(self, fitted, analysis):
        assert list(fitted[analysis].keys()) == [
            "analysis",
            "terms",
            "design",
            "parameters",
            "coefficients",
            "fit_stats",
            "performance",
            "resampling",
            "engine",
            "metadata",
        ]

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_the_table_is_in_the_order_terms_gives(self, fitted, analysis):
        model = fitted[analysis]
        assert model["coefficients"]["terms"].tolist() == model["terms"]

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_no_inference_is_reported_for_a_model_that_has_none(self, fitted, analysis):
        """None of the three has a sampling distribution its estimate came from,
        so none of them carries a p-value."""
        assert "pval" not in fitted[analysis]["coefficients"]
        assert "stderr" not in fitted[analysis]["coefficients"]

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_the_design_says_the_same_thing_whichever_model_read_it(self, fitted, analysis, reg):
        design = fitted[analysis]["design"]
        assert design["outcome"] == reg.args["outcome"]
        assert design["outcome_type"] == "continuous"
        assert design["predictors"] == list(reg.args["predictors"])
        assert design["n_used"] + design["n_dropped"] == design["n_obs"]

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_the_engine_names_itself_and_the_columns_it_was_handed(self, fitted, analysis):
        engine = fitted[analysis]["engine"]
        assert engine["package"] == "scikit-learn"
        assert engine["metrics"] == ["RMSE", "Rsquared", "MAE"]
        # A factor predictor is coded, so the columns the engine saw are more
        # numerous than the predictors whichever model it was.
        assert len(engine["x_names"]) > len(fitted[analysis]["design"]["predictors"])

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_nothing_resampled_reports_neither_table_and_one_candidate(self, fitted, analysis):
        model = fitted[analysis]
        assert model["performance"] is None
        assert model["resampling"] is None
        assert model["parameters"]["n_candidates"] == 1
        assert model["parameters"]["cv"] is False

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_the_engine_object_is_not_among_the_slots(self, fitted, analysis):
        assert "fit" not in fitted[analysis]
        assert fitted[analysis].fit is not None

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_there_is_one_prediction_per_row_of_newdata(self, fitted, analysis, reg):
        rows = reg.args["data"]
        assert len(fitted[analysis].predict(rows)) == len(rows.index)

    @pytest.mark.parametrize("analysis", ANALYSES)
    def test_a_row_missing_a_predictor_gets_a_missing_prediction(self, fitted, analysis, reg):
        rows = reg.args["data"].copy()
        rows.loc[rows.index[0], reg.args["predictors"][0]] = np.nan
        predicted = fitted[analysis].predict(rows)
        assert math.isnan(float(predicted[0]))


class TestResamplingIsShared:
    """The three are tuned differently and scored identically, which is what
    makes their ``performance`` tables comparable."""

    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": [0.05, 0.5]}),
            (fit_rf, {"ntree": 40, "mtry": [2, 4]}),
            (fit_svm, {"C": [1, 10]}),
        ],
    )
    def test_a_search_reports_a_row_per_candidate_and_a_row_per_fold(self, reg, fit, tuning):
        model = fit(**reg.args, **tuning, cv_method="kfold", n_fold=4, seed=1)
        assert len(model["performance"].index) == 2
        assert model["parameters"]["n_candidates"] == 2
        assert len(model["resampling"].index) == 4
        assert list(model["resampling"]["Resample"]) == [f"Fold{at}" for at in range(1, 5)]

    @pytest.mark.parametrize(
        ("fit", "tuning", "chosen"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": [0.05, 0.5]}, "lambda"),
            (fit_rf, {"ntree": 40, "mtry": [2, 4]}, "mtry"),
            (fit_svm, {"C": [1, 10]}, "C"),
        ],
    )
    def test_the_recorded_setting_is_one_the_search_actually_scored(self, reg, fit, tuning, chosen):
        """``parameters`` holds what ran, not the grid that was asked for."""
        model = fit(**reg.args, **tuning, cv_method="kfold", n_fold=4, seed=1)
        column = "lambda_" if chosen == "lambda" else chosen
        assert model["parameters"][chosen] in set(model["performance"][column])

    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": 0.2}),
            (fit_rf, {"ntree": 40}),
            (fit_svm, {"C": 1}),
        ],
    )
    def test_the_metrics_are_the_same_three_with_their_spread(self, reg, fit, tuning):
        model = fit(**reg.args, **tuning, cv_method="kfold", n_fold=4, seed=1)
        for metric in ("RMSE", "Rsquared", "MAE"):
            assert math.isfinite(float(model["performance"][metric].iloc[0]))
            assert math.isfinite(float(model["performance"][f"{metric}SD"].iloc[0]))

    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "alpha": 1, "lambda_": [0.05, 0.5]}),
            (fit_rf, {"mtry": [2, 4]}),
            (fit_svm, {"C": [1, 10]}),
        ],
    )
    def test_more_than_one_candidate_without_resampling_has_nothing_to_choose_with(
        self, reg, fit, tuning
    ):
        with pytest.raises(SaValueError, match="must hold one candidate"):
            fit(**reg.args, **tuning, cv=False)


class TestClassificationIsShared:
    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": 0.05}),
            (fit_rf, {"ntree": 60}),
            (fit_svm, {"C": 1}),
        ],
    )
    def test_a_class_outcome_is_recorded_the_same_way_by_all_three(self, clf, fit, tuning):
        model = fit(**clf.args, **tuning, cv=False, seed=1)
        design = model["design"]
        assert design["outcome_type"] == "two classes"
        assert design["outcome_lv"] == ["control", "case"]
        events = int((clf.args["data"][clf.args["outcome"]] == "case").sum())
        assert design["n_events"] == events
        assert design["event_rate"] == pytest.approx(events / design["n_used"])
        assert model["engine"]["metrics"] == ["Accuracy", "Kappa"]

    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": 0.05}),
            (fit_rf, {"ntree": 60}),
            (fit_svm, {"C": 1}),
        ],
    )
    def test_the_predictions_point_at_the_second_level(self, clf, fit, tuning):
        model = fit(**clf.args, **tuning, cv=False, seed=1)
        rows = clf.args["data"]
        levels = model["design"]["outcome_lv"]
        table = model.predict(rows, type="prob")
        assert list(table.columns) == levels
        assert np.allclose(model.predict(rows, type="response"), table[levels[1]])
        assert set(np.unique(model.predict(rows))) <= set(levels)

    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": 0.05}),
            (fit_rf, {"ntree": 60}),
            (fit_svm, {"C": 1}),
        ],
    )
    def test_a_third_class_is_refused_rather_than_dropped(self, fit, tuning):
        frame = pd.DataFrame(
            {"y": ["a", "b", "c"] * 12, "x": list(range(36)), "z": list(range(36, 72))}
        )
        with pytest.raises(SaValueError, match="holds 3 classes"):
            fit(frame, outcome="y", outcome_lv=["a", "b"], **tuning, cv=False)

    @pytest.mark.parametrize(
        ("fit", "tuning"),
        [
            (fit_elastic_net, {"penalty": "lasso", "lambda_": 0.05}),
            (fit_rf, {"ntree": 60}),
            (fit_svm, {"C": 1}),
        ],
    )
    def test_a_numeric_two_valued_outcome_is_a_regression_unless_asked_otherwise(self, fit, tuning):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {
                "y": rng.integers(0, 2, 60).astype(float),
                "a": rng.normal(size=60),
                "b": rng.normal(size=60),
            }
        )
        assert fit(frame, outcome="y", **tuning, cv=False)["design"]["outcome_type"] == "continuous"
        asked = fit(frame, outcome="y", control_label="0.0", **tuning, cv=False)
        assert asked["design"]["outcome_type"] == "two classes"


class TestElasticNet:
    def test_a_heavier_penalty_keeps_fewer_terms(self, reg):
        light = fit_elastic_net(**reg.args, penalty="lasso", lambda_=0.01, cv=False)
        heavy = fit_elastic_net(**reg.args, penalty="lasso", lambda_=3.0, cv=False)
        assert heavy["fit_stats"]["n_selected"] < light["fit_stats"]["n_selected"]

    def test_a_ridge_shrinks_every_term_and_drops_none(self, reg):
        ridge = fit_elastic_net(**reg.args, penalty="ridge", lambda_=3.0, cv=False)
        assert ridge["fit_stats"]["n_zero"] == 0
        assert bool(ridge["coefficients"]["selected"].all())

    def test_shrinking_moves_the_estimates_towards_zero(self, reg):
        """Which is what the penalty is: not a different fit, a pulled one."""
        from statassist import fit_linear_regression

        plain = fit_linear_regression(**reg.args, cv=False)
        shrunk = fit_elastic_net(**reg.args, penalty="ridge", lambda_=3.0, cv=False)
        merged = plain["coefficients"].merge(
            shrunk["coefficients"], on="terms", suffixes=("_plain", "_shrunk")
        )
        slopes = merged.loc[merged["terms"] != "(Intercept)"]
        assert float(slopes["estimate_shrunk"].abs().sum()) < float(
            slopes["estimate_plain"].abs().sum()
        )

    def test_the_intercept_is_kept_whatever_the_penalty_does(self, reg):
        heavy = fit_elastic_net(**reg.args, penalty="lasso", lambda_=1e3, cv=False)
        row = heavy["coefficients"].iloc[0]
        assert row["terms"] == "(Intercept)"
        assert bool(row["selected"])
        assert heavy["fit_stats"]["n_selected"] == 0

    def test_the_estimates_are_on_the_scale_the_predictors_came_in_on(self):
        """The columns are standardized before the penalty is applied, so an
        estimate that had not been put back would be a thousand times off for a
        column measured a thousand times smaller."""
        rng = np.random.default_rng(4)
        a = rng.normal(size=200)
        b = rng.normal(size=200)
        frame = pd.DataFrame({"y": 2 * a + b + rng.normal(0, 0.2, 200), "a": a, "b": b})
        rescaled = frame.assign(a=frame["a"] * 1000)

        plain = fit_elastic_net(frame, outcome="y", penalty="ridge", lambda_=0.01, cv=False)
        large = fit_elastic_net(rescaled, outcome="y", penalty="ridge", lambda_=0.01, cv=False)
        at = plain["terms"].index("a")
        assert float(large["coefficients"]["estimate"].iloc[at]) == pytest.approx(
            float(plain["coefficients"]["estimate"].iloc[at]) / 1000, rel=1e-3
        )

    def test_a_penalty_of_zero_is_the_unpenalized_fit(self, reg):
        from statassist import fit_linear_regression

        plain = fit_linear_regression(**reg.args, cv=False)
        unpenalized = fit_elastic_net(**reg.args, penalty="lasso", lambda_=0, cv=False)
        assert np.allclose(
            unpenalized["coefficients"]["estimate"], plain["coefficients"]["estimate"]
        )

    def test_a_classification_reports_the_odds_ratio_of_each_estimate(self, clf):
        model = fit_elastic_net(**clf.args, penalty="lasso", lambda_=0.05, cv=False)
        table = model["coefficients"]
        assert list(table.columns) == ["terms", "estimate", "selected", "odds_ratio"]
        assert np.allclose(table["odds_ratio"], np.exp(table["estimate"]))

    def test_a_classification_reports_the_deviances_it_was_fitted_by(self, clf):
        stats = fit_elastic_net(**clf.args, penalty="lasso", lambda_=0.05, cv=False)["fit_stats"]
        assert stats["residual_deviance"] < stats["null_deviance"]
        assert 0 < stats["mcfadden_r2"] < 1

    def test_a_one_term_model_has_no_budget_to_divide(self):
        frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "a": [1.0, 2.0, 3.0, 5.0]})
        with pytest.raises(SaValueError, match="nothing to divide"):
            fit_elastic_net(frame, outcome="y", penalty="lasso", lambda_=0.1, cv=False)

    def test_an_unknown_penalty_lists_the_ones_there_are(self, reg):
        with pytest.raises(SaValueError, match="`penalty` must be one of"):
            fit_elastic_net(**reg.args, penalty="l1")


class TestRandomForest:
    def test_the_terms_are_the_predictors_rather_than_the_coded_columns(self, fitted, reg):
        """A factor reaches the engine as several columns and comes back as one
        predictor, which is what makes this table readable beside R's."""
        model = fitted["random_forest"]
        assert set(model["terms"]) == set(reg.args["predictors"])
        assert len(model["engine"]["x_names"]) > len(model["terms"])

    def test_the_importance_of_a_factor_is_the_sum_over_its_own_columns(self, reg):
        """Summed rather than averaged: what a predictor was worth does not
        depend on how many columns it was spread over."""
        model = fit_rf(**reg.args, ntree=60, cv=False, seed=1)
        engine = model.fit
        source = model["engine"]["x_names"]
        factor_columns = [name for name in source if name.startswith("x_cat_1")]
        assert len(factor_columns) > 1
        by_column = dict(
            zip(source, np.asarray(engine.estimator.feature_importances_), strict=True)
        )
        rolled = float(sum(by_column[name] for name in factor_columns))
        row = model["coefficients"].set_index("terms").loc["x_cat_1"]
        assert float(row["impurity"]) == pytest.approx(rolled)

    def test_the_table_is_sorted_by_importance_rather_than_by_the_term_order(self, fitted):
        estimate = fitted["random_forest"]["coefficients"]["estimate"].to_numpy()
        assert np.all(np.diff(estimate) <= 0)

    def test_both_kinds_of_importance_are_reported(self, fitted):
        assert list(fitted["random_forest"]["coefficients"].columns) == [
            "terms",
            "estimate",
            "impurity",
        ]

    def test_the_scores_are_out_of_bag_and_say_how_many_rows_had_one(self, fitted):
        stats = fitted["random_forest"]["fit_stats"]
        assert set(stats) == {"oob_r_squared", "oob_rmse", "oob_mae", "n_oob"}
        assert stats["n_oob"] == fitted["random_forest"]["design"]["n_used"]

    def test_the_out_of_bag_score_is_worse_than_the_score_on_the_fitted_rows(self, fitted):
        """Which is the whole reason it is the one reported: a forest predicts its
        own training rows nearly perfectly."""
        model = fitted["random_forest"]
        engine = model.fit
        fitted_values = np.asarray(
            engine.estimator.predict(np.asarray(engine.x, dtype=float)), dtype=float
        )
        total = float(np.sum((engine.y - engine.y.mean()) ** 2))
        in_sample = 1 - float(np.sum((engine.y - fitted_values) ** 2)) / total
        assert model["fit_stats"]["oob_r_squared"] < in_sample

    def test_a_classification_scores_both_sides_of_the_confusion(self, clf):
        stats = fit_rf(**clf.args, ntree=60, cv=False, seed=1)["fit_stats"]
        assert set(stats) == {
            "oob_accuracy",
            "oob_error",
            "oob_kappa",
            "oob_sensitivity",
            "oob_specificity",
            "n_oob",
        }
        assert stats["oob_accuracy"] + stats["oob_error"] == pytest.approx(1)

    def test_mtry_defaults_to_the_rule_of_thumb_on_the_terms_the_engine_sees(self, fitted):
        model = fitted["random_forest"]
        n_terms = len(model["engine"]["x_names"])
        assert model["parameters"]["mtry"] == max(1, n_terms // 3)

    def test_the_leaf_size_default_differs_by_outcome_type(self, reg, clf):
        assert fit_rf(**reg.args, ntree=20, cv=False)["parameters"]["nodesize"] == 5
        assert fit_rf(**clf.args, ntree=20, cv=False)["parameters"]["nodesize"] == 1

    def test_what_the_port_changed_about_the_engine_is_declared(self, fitted):
        overridden = fitted["random_forest"]["engine"]["overridden"]
        assert "oob_score = True" in overridden
        assert any("summed back" in note for note in overridden)

    def test_the_seed_makes_the_forest_reproducible(self, reg):
        one = fit_rf(**reg.args, ntree=40, cv=False, seed=7)
        two = fit_rf(**reg.args, ntree=40, cv=False, seed=7)
        other = fit_rf(**reg.args, ntree=40, cv=False, seed=8)
        pd.testing.assert_frame_equal(one["coefficients"], two["coefficients"])
        assert not one["coefficients"].equals(other["coefficients"])

    def test_mtry_above_the_term_count_is_refused(self, reg):
        n_terms = len(fit_rf(**reg.args, ntree=20, cv=False)["engine"]["x_names"])
        with pytest.raises(SaValueError, match="cannot exceed"):
            fit_rf(**reg.args, ntree=20, mtry=n_terms + 1, cv=False)


class TestSupportVectorMachine:
    def test_the_terms_are_the_coded_columns_rather_than_the_predictors(self, fitted):
        """Unlike the forest: a machine has no notion of a predictor at all, only
        of the distances the kernel reads over the columns."""
        model = fitted["svm"]
        assert set(model["terms"]) == set(model["engine"]["x_names"])

    def test_only_one_kind_of_importance_is_reported(self, fitted):
        assert list(fitted["svm"]["coefficients"].columns) == ["terms", "estimate"]

    def test_the_table_is_sorted_by_importance(self, fitted):
        estimate = fitted["svm"]["coefficients"]["estimate"].to_numpy()
        assert np.all(np.diff(estimate) <= 0)

    def test_how_many_rows_the_boundary_needed_is_reported(self, fitted):
        stats = fitted["svm"]["fit_stats"]
        n_used = fitted["svm"]["design"]["n_used"]
        assert stats["support_vector_rate"] == pytest.approx(stats["n_support_vector"] / n_used)
        assert 0 < stats["support_vector_rate"] <= 1

    def test_the_kernel_is_the_only_one_offered_and_is_recorded_as_such(self, fitted):
        model = fitted["svm"]
        assert model["parameters"]["kernel"] == "radial"
        assert model["engine"]["kernel"] == "radial"

    def test_the_width_defaults_to_one_over_the_term_count(self, fitted):
        model = fitted["svm"]
        assert model["parameters"]["sigma"] == pytest.approx(1 / len(model["engine"]["x_names"]))
        assert any("heuristic" in note for note in model["engine"]["overridden"])

    def test_naming_the_width_leaves_the_heuristic_undeclared(self, reg):
        model = fit_svm(**reg.args, C=1, sigma=0.5, cv=False, seed=1)
        assert model["parameters"]["sigma"] == 0.5
        assert not any("heuristic" in note for note in model["engine"]["overridden"])

    def test_a_classification_fits_the_calibration_its_probabilities_need(self, clf):
        model = fit_svm(**clf.args, C=1, cv=False, seed=1)
        assert "probability = True" in model["engine"]["overridden"]
        probability = model.predict(clf.args["data"], type="response")
        assert bool(((probability >= 0) & (probability <= 1)).all())

    def test_a_classification_scores_both_sides_of_the_confusion(self, clf):
        stats = fit_svm(**clf.args, C=1, cv=False, seed=1)["fit_stats"]
        assert set(stats) == {
            "accuracy",
            "error",
            "kappa",
            "sensitivity",
            "specificity",
            "n_support_vector",
            "support_vector_rate",
        }

    def test_a_zero_cost_is_refused_by_name(self, reg):
        with pytest.raises(SaValueError, match="`C` must be above 0"):
            fit_svm(**reg.args, C=0, cv=False)

    def test_a_zero_width_is_refused_by_name(self, reg):
        with pytest.raises(SaValueError, match="`sigma` must be above 0"):
            fit_svm(**reg.args, C=1, sigma=0, cv=False)

    def test_the_columns_are_scaled_so_the_kernel_is_not_one_predictor(self):
        """A kernel reads one distance over every term at once, so a term on a
        larger scale would be the only thing the machine could see."""
        rng = np.random.default_rng(6)
        a = rng.normal(size=150)
        b = rng.normal(size=150)
        frame = pd.DataFrame({"y": b + rng.normal(0, 0.2, 150), "a": a * 1000, "b": b})
        model = fit_svm(frame, outcome="y", C=10, cv=False, seed=1)
        top = model["coefficients"]["terms"].iloc[0]
        assert top == "b"
        assert "columns centred and scaled" in model["engine"]["overridden"]
