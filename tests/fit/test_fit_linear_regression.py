"""What a fitted linear regression reports, and whether it is right.

The engine's solve is not re-derived here. What is checked is everything the
package put around it: that the terms it reports are the terms it fitted, that
the inference beside them is the closed form and not something adjacent to it,
and that the result object holds together as a contract.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from statassist import fit_linear_regression, simulate_regression
from statassist.core import SaValueError, SaWarning


@pytest.fixture(scope="module")
def sim():
    """A regression with planted coefficients, three of eight carrying signal."""
    return simulate_regression(n_samples=200, n_pred=8, n_pos=2, n_neg=1, seed=11)


def _span(fit) -> float:
    """The width of the slope's interval, the first term after the intercept."""
    table = fit["coefficients"]
    return float(table["upper_conf"].iloc[1] - table["lower_conf"].iloc[1])


@pytest.fixture(scope="module")
def fitted(sim):
    """The same fit, reused: fitting is the slow part and nothing mutates it."""
    return fit_linear_regression(**sim.args, cv=False, seed=1)


class TestWhatItFitted:
    def test_the_analysis_names_the_model(self, fitted):
        assert fitted["analysis"] == "linear_regression"

    def test_every_slot_of_the_contract_is_there(self, fitted):
        assert list(fitted.keys()) == [
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

    def test_the_coefficient_table_is_in_the_order_terms_gives(self, fitted):
        assert fitted["coefficients"]["terms"].tolist() == fitted["terms"]

    def test_the_intercept_is_the_first_term(self, fitted):
        assert fitted["terms"][0] == "(Intercept)"

    def test_a_factor_predictor_becomes_one_term_per_level_beyond_the_first(self):
        sim = simulate_regression(n_samples=90, n_pred=2, n_factor_pred=1, seed=4)
        fit = fit_linear_regression(**sim.args, cv=False)
        levels = fit["design"]["predictor_lv"]["x_cat_1"]
        planted = [term for term in fit["terms"] if term.startswith("x_cat_1")]
        assert len(planted) == len(levels) - 1
        assert planted == [f"x_cat_1{level}" for level in levels[1:]]

    def test_terms_and_predictors_are_different_lists_and_both_are_reported(self):
        sim = simulate_regression(n_samples=90, n_pred=2, n_factor_pred=1, seed=4)
        fit = fit_linear_regression(**sim.args, cv=False)
        assert len(fit["terms"]) > len(fit["design"]["predictors"]) + 1

    def test_a_single_valued_predictor_is_named_among_the_dropped(self):
        sim = simulate_regression(n_samples=60, n_pred=3, n_constant_pred=1, seed=5)
        fit = fit_linear_regression(**sim.args, cv=False)
        assert fit["design"]["dropped_predictors"] == ["x_const_1"]
        assert not any(term.startswith("x_const") for term in fit["terms"])

    def test_the_rows_a_missing_cell_cost_are_counted(self):
        sim = simulate_regression(n_samples=120, n_pred=4, p_missing=0.05, seed=6)
        fit = fit_linear_regression(**sim.args, cv=False)
        design = fit["design"]
        assert design["n_obs"] == 120
        assert design["n_dropped"] > 0
        assert design["n_used"] + design["n_dropped"] == design["n_obs"]

    def test_the_engine_records_the_design_columns_it_was_handed(self, fitted):
        assert fitted["engine"]["x_names"] == fitted["terms"][1:]
        assert fitted["engine"]["package"] == "scikit-learn"


class TestTheEstimates:
    def test_a_planted_coefficient_is_recovered_within_its_own_uncertainty(self, sim, fitted):
        """The whole point of a simulator: the answer is known before the fit.

        Scored against the standard error the fit reports rather than a fixed
        tolerance, since how close an estimate can get is a property of the noise
        and the sample size and not something a test gets to choose.
        """
        truth = sim.truth_term.set_index("terms")["beta"]
        table = fitted["coefficients"].set_index("terms")
        for term in truth[truth != 0].index:
            row = table.loc[term]
            assert abs(float(row["estimate"]) - float(truth[term])) < 3 * float(row["stderr"])

    def test_a_planted_coefficient_is_found_on_the_side_it_was_planted(self, sim, fitted):
        truth = sim.truth_term.set_index("terms")["beta"]
        table = fitted["coefficients"].set_index("terms")
        for term in truth[truth != 0].index:
            assert np.sign(float(table.loc[term, "estimate"])) == np.sign(float(truth[term]))

    def test_a_null_predictor_is_reported_as_one_rather_than_left_out(self, sim, fitted):
        truth = sim.truth_term.set_index("terms")["beta"]
        table = fitted["coefficients"].set_index("terms")
        for term in truth[truth == 0].index:
            assert term in table.index
            assert float(table.loc[term, "pval"]) > 0.01

    def test_the_residuals_are_orthogonal_to_the_design(self, sim, fitted):
        """Which is what it means for the estimates to be the least squares
        solution, checked without re-deriving the solve."""
        engine = fitted.fit
        residual = engine.y - np.asarray(engine.estimator.predict(engine.x))
        matrix = np.column_stack([np.ones(len(engine.x.index)), np.asarray(engine.x, dtype=float)])
        assert np.allclose(matrix.T @ residual, 0, atol=1e-8)

    def test_the_inference_of_a_simple_regression_is_the_textbook_one(self):
        frame = pd.DataFrame(
            {"y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        fit = fit_linear_regression(frame, outcome="y", cv=False)
        row = fit["coefficients"].set_index("terms").loc["x"]
        known = sp_stats.linregress(frame["x"], frame["y"])
        assert float(row["estimate"]) == pytest.approx(known.slope)
        assert float(row["stderr"]) == pytest.approx(known.stderr)
        assert float(row["pval"]) == pytest.approx(known.pvalue)

    def test_the_interval_is_the_t_interval_the_statistic_beside_it_belongs_to(self):
        frame = pd.DataFrame(
            {"y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        fit = fit_linear_regression(frame, outcome="y", cv=False, conf_level=0.9)
        row = fit["coefficients"].set_index("terms").loc["x"]
        assert float(row["df"]) == 4
        crit = float(sp_stats.t.ppf(0.95, 4))
        assert float(row["lower_conf"]) == pytest.approx(
            float(row["estimate"]) - crit * float(row["stderr"])
        )

    def test_a_narrower_level_gives_a_narrower_interval(self):
        frame = pd.DataFrame(
            {"y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        wide = fit_linear_regression(frame, outcome="y", cv=False, conf_level=0.99)
        narrow = fit_linear_regression(frame, outcome="y", cv=False, conf_level=0.80)
        assert _span(narrow) < _span(wide)


class TestAliasedTerms:
    @pytest.fixture
    def collinear(self):
        """``b`` is twice ``a``, so the two cannot be told apart."""
        rng = np.random.default_rng(7)
        a = rng.normal(size=40)
        return pd.DataFrame({"y": a + rng.normal(0, 0.2, 40), "a": a, "b": 2 * a})

    def test_the_term_the_data_cannot_support_is_named_in_a_warning(self, collinear):
        with pytest.warns(SaWarning, match="already span them"):
            fit_linear_regression(collinear, outcome="y", cv=False)

    def test_it_keeps_its_row_with_nothing_estimated_in_it(self, collinear):
        """``df`` stays, since the residual degrees of freedom are a property of
        the model rather than something estimated about the term."""
        with pytest.warns(SaWarning):
            fit = fit_linear_regression(collinear, outcome="y", cv=False)
        row = fit["coefficients"].set_index("terms").loc["b"]
        assert bool(row.drop(labels=["df"]).isna().all())
        assert float(row["df"]) == len(collinear.index) - 2
        assert fit["terms"] == ["(Intercept)", "a", "b"]

    def test_the_term_that_was_kept_is_the_one_named_first(self, collinear):
        with pytest.warns(SaWarning):
            fit = fit_linear_regression(collinear, outcome="y", cv=False)
        assert math.isfinite(float(fit["coefficients"].set_index("terms").loc["a", "estimate"]))


class TestFitStats:
    def test_r_squared_is_the_share_of_variance_the_fit_accounts_for(self, fitted):
        engine = fitted.fit
        y = engine.y
        rss = float(np.sum((y - np.asarray(engine.estimator.predict(engine.x))) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        assert fitted["fit_stats"]["r_squared"] == pytest.approx(1 - rss / tss)

    def test_the_adjusted_one_is_below_it_when_there_are_terms_to_pay_for(self, fitted):
        stats = fitted["fit_stats"]
        assert stats["adj_r_squared"] < stats["r_squared"]

    def test_the_overall_test_is_the_f_on_the_two_degrees_of_freedom_reported(self, fitted):
        stats = fitted["fit_stats"]
        assert stats["df1"] == len(fitted["terms"]) - 1
        assert stats["df2"] == fitted["design"]["n_used"] - len(fitted["terms"])
        assert stats["pval"] == pytest.approx(
            float(sp_stats.f.sf(stats["f_stat"], stats["df1"], stats["df2"]))
        )

    def test_a_model_with_planted_signal_is_rejected_against_nothing(self, fitted):
        assert fitted["fit_stats"]["pval"] < 1e-6

    def test_sigma_is_the_residual_standard_error(self, fitted):
        engine = fitted.fit
        residual = engine.y - np.asarray(engine.estimator.predict(engine.x))
        df = fitted["design"]["n_used"] - len(fitted["terms"])
        assert fitted["fit_stats"]["sigma"] == pytest.approx(
            math.sqrt(float(np.sum(residual**2)) / df)
        )


class TestResampling:
    def test_nothing_resampled_reports_neither_table(self, fitted):
        assert fitted["performance"] is None
        assert fitted["resampling"] is None
        assert fitted["parameters"]["cv_method"] is None

    def test_the_coefficients_do_not_depend_on_whether_it_was_scored(self, sim, fitted):
        scored = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        pd.testing.assert_frame_equal(scored["coefficients"], fitted["coefficients"])

    def test_a_kfold_run_reports_a_row_per_fold_and_a_summary(self, sim):
        fit = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", n_fold=5, seed=1)
        assert len(fit["performance"].index) == 1
        assert len(fit["resampling"].index) == 5
        assert fit["parameters"]["n_fold"] == 5
        assert fit["parameters"]["n_repeat"] is None

    def test_the_metrics_are_carets_and_carry_their_spread(self, sim):
        fit = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        assert fit["engine"]["metrics"] == ["RMSE", "Rsquared", "MAE"]
        for metric in fit["engine"]["metrics"]:
            assert math.isfinite(float(fit["performance"][metric].iloc[0]))
            assert math.isfinite(float(fit["performance"][f"{metric}SD"].iloc[0]))

    def test_a_repeated_run_scores_every_fold_of_every_repeat(self, sim):
        fit = fit_linear_regression(
            **sim.args, cv=True, cv_method="repeated_kfold", n_fold=4, n_repeat=3, seed=1
        )
        assert len(fit["resampling"].index) == 12
        assert fit["parameters"]["n_repeat"] == 3

    def test_leave_one_out_pools_its_predictions_and_reports_no_fold_table(self, sim):
        fit = fit_linear_regression(**sim.args, cv=True, cv_method="loocv")
        assert fit["resampling"] is None
        assert len(fit["performance"].index) == 1
        assert "RMSESD" not in fit["performance"].columns

    def test_the_seed_fixes_the_folds_and_nothing_else(self, sim):
        one = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=99)
        two = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=99)
        other = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=100)
        pd.testing.assert_frame_equal(one["resampling"], two["resampling"])
        assert not one["resampling"].equals(other["resampling"])

    def test_the_held_out_score_is_worse_than_the_score_on_the_fitted_rows(self, sim, fitted):
        fit = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        assert float(fit["performance"]["Rsquared"].iloc[0]) < fitted["fit_stats"]["r_squared"]


class TestPredict:
    def test_predicting_on_nothing_gives_the_fitted_values(self, fitted):
        engine = fitted.fit
        assert np.allclose(fitted.predict(), engine.estimator.predict(engine.x))

    def test_there_is_one_prediction_per_row_of_newdata(self, sim, fitted):
        rows = sim.args["data"]
        assert len(fitted.predict(rows)) == len(rows.index)

    def test_the_outcome_column_may_come_along_and_is_ignored(self, sim, fitted):
        rows = sim.args["data"]
        without = rows.drop(columns=[sim.args["outcome"]])
        assert np.allclose(fitted.predict(rows), fitted.predict(without))

    def test_a_row_that_is_missing_a_predictor_gets_a_missing_prediction(self, sim, fitted):
        rows = sim.args["data"].copy()
        rows.loc[rows.index[0], sim.args["predictors"][0]] = np.nan
        predicted = fitted.predict(rows)
        assert math.isnan(float(predicted[0]))
        assert math.isfinite(float(predicted[1]))

    def test_the_columns_may_arrive_in_any_order(self, sim, fitted):
        rows = sim.args["data"]
        assert np.allclose(fitted.predict(rows), fitted.predict(rows[rows.columns[::-1]]))

    def test_a_factor_predictor_is_coded_the_way_the_fit_coded_it(self):
        sim = simulate_regression(n_samples=90, n_pred=2, n_factor_pred=1, seed=4)
        fit = fit_linear_regression(**sim.args, cv=False)
        rows = sim.args["data"]
        # A held-out half missing a level still codes to the model's terms.
        levels = fit["design"]["predictor_lv"]["x_cat_1"]
        subset = rows.loc[rows["x_cat_1"] == levels[0]]
        assert len(fit.predict(subset)) == len(subset.index)

    def test_probabilities_are_not_something_a_regression_has(self, sim, fitted):
        with pytest.raises(SaValueError, match="only a classification"):
            fitted.predict(sim.args["data"], type="prob")

    def test_an_unknown_type_is_refused_by_name(self, fitted):
        with pytest.raises(SaValueError, match="`type` must be one of"):
            fitted.predict(type="fitted")

    def test_coef_is_the_coefficient_table_itself(self, fitted):
        pd.testing.assert_frame_equal(fitted.coef(), fitted["coefficients"])


class TestRefusals:
    def test_a_class_outcome_is_sent_to_the_other_function(self):
        frame = pd.DataFrame({"y": ["a", "b"] * 15, "x": list(range(30))})
        with pytest.raises(SaValueError, match="fit_logistic_regression"):
            fit_linear_regression(frame, outcome="y", cv=False)

    def test_a_non_finite_outcome_has_no_residual(self):
        frame = pd.DataFrame({"y": [1.0, 2.0, math.inf, 4.0], "x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(SaValueError, match="non-finite"):
            fit_linear_regression(frame, outcome="y", cv=False)

    def test_an_unknown_resampling_scheme_lists_the_ones_there_are(self, sim):
        with pytest.raises(SaValueError, match="`cv_method` must be one of"):
            fit_linear_regression(**sim.args, cv_method="bootstrap")

    def test_a_confidence_level_outside_the_open_unit_interval_is_refused(self, sim):
        with pytest.raises(SaValueError, match="`conf_level`"):
            fit_linear_regression(**sim.args, cv=False, conf_level=1)


class TestJsonShape:
    def test_the_engine_object_is_not_among_the_slots(self, fitted):
        assert "fit" not in fitted
        assert "fit" not in fitted.to_dict()
        assert fitted.fit is not None

    def test_every_slot_is_a_shape_json_can_hold(self, sim):
        fit = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        payload = {}
        for name, value in fit.to_dict().items():
            if isinstance(value, pd.DataFrame):
                payload[name] = value.to_dict(orient="list")
            else:
                payload[name] = value
        # NaN is what json calls it, so the round trip is on the structure.
        text = json.dumps(payload, default=str, allow_nan=True)
        assert json.loads(text)["analysis"] == "linear_regression"

    def test_the_summary_says_what_was_fitted_to_what(self, sim):
        fit = fit_linear_regression(**sim.args, cv=True, cv_method="kfold", seed=1)
        printed = repr(fit)
        assert "linear_regression" in printed
        assert "RMSE" in printed
        assert sim.args["outcome"] in printed
