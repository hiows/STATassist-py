"""Scoring fitted models on rows they were not fitted on.

The two functions share everything up to the point where the scoring begins, so
what is checked here in one place is the shared part: the contract of the result,
the intersection of the rows, and the refusals that stop a table of numbers from
being assembled out of models that were not answering the same question.

What is specific to each is checked separately. A regression reports differences
without tests, so what has to hold is arithmetic: that the deltas are
``new - baseline`` and that the calibration line is the line the metrics describe.
A classification reports three paired tests, so what has to hold is that they
agree with each other on which model was better, and that the curve the result
carries is the curve its own AUC was measured on.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from statassist import (
    evaluate_classification_models,
    evaluate_regression_models,
    fit_linear_regression,
    fit_logistic_regression,
    fit_rf,
    simulate_classification,
    simulate_regression,
)
from statassist.core import SaValueError, SaWarning
from statassist.core.contracts import (
    classification_comparison_columns,
    classification_metric_columns,
    curve_columns,
    prediction_columns,
    regression_comparison_columns,
    regression_metric_columns,
)


@pytest.fixture(scope="module")
def reg():
    """A regression with a factor predictor, split into a fitted and a held-out half."""
    sim = simulate_regression(n_samples=160, n_pred=4, n_factor_pred=1, seed=31)
    frame = sim.args["data"]
    return {
        "sim": sim,
        "outcome": sim.args["outcome"],
        "predictors": list(sim.args["predictors"]),
        "train": frame.iloc[:110].reset_index(drop=True),
        "test": frame.iloc[110:].reset_index(drop=True),
    }


@pytest.fixture(scope="module")
def clf():
    sim = simulate_classification(n_samples=200, n_pred=4, n_factor_pred=1, seed=32)
    frame = sim.args["data"]
    return {
        "sim": sim,
        "outcome": sim.args["outcome"],
        "outcome_lv": list(sim.args["outcome_lv"]),
        "predictors": list(sim.args["predictors"]),
        "train": frame.iloc[:140].reset_index(drop=True),
        "test": frame.iloc[140:].reset_index(drop=True),
    }


@pytest.fixture(scope="module")
def reg_models(reg):
    """A full fit and a one-predictor fit of the same outcome."""
    full = fit_linear_regression(
        reg["train"], outcome=reg["outcome"], predictors=reg["predictors"], cv=False
    )
    thin = fit_linear_regression(
        reg["train"], outcome=reg["outcome"], predictors=reg["predictors"][:1], cv=False
    )
    return full, thin


@pytest.fixture(scope="module")
def clf_models(clf):
    full = fit_logistic_regression(
        clf["train"],
        outcome=clf["outcome"],
        predictors=clf["predictors"],
        outcome_lv=clf["outcome_lv"],
        cv=False,
    )
    thin = fit_logistic_regression(
        clf["train"],
        outcome=clf["outcome"],
        predictors=clf["predictors"][:1],
        outcome_lv=clf["outcome_lv"],
        cv=False,
    )
    return full, thin


@pytest.fixture(scope="module")
def scored_reg(reg, reg_models):
    full, thin = reg_models
    return evaluate_regression_models(full, {"one_predictor": thin}, newdata=reg["test"])


@pytest.fixture(scope="module")
def scored_clf(clf, clf_models):
    full, thin = clf_models
    return evaluate_classification_models(full, {"one_predictor": thin}, newdata=clf["test"])


class TestTheSharedContract:
    def test_a_regression_evaluation_holds_the_eight_slots_and_no_curve(self, scored_reg):
        assert list(scored_reg.keys()) == [
            "analysis",
            "models",
            "design",
            "parameters",
            "predictions",
            "metrics",
            "comparisons",
            "metadata",
        ]

    def test_a_classification_evaluation_holds_the_curve_as_well(self, scored_clf):
        assert list(scored_clf.keys()) == [
            "analysis",
            "models",
            "design",
            "parameters",
            "predictions",
            "metrics",
            "comparisons",
            "curves",
            "metadata",
        ]

    @pytest.mark.parametrize("kind", ["reg", "clf"])
    def test_the_baseline_is_the_first_model_and_the_tables_follow_that_order(
        self, scored_reg, scored_clf, kind
    ):
        res = scored_reg if kind == "reg" else scored_clf
        assert res["models"] == ["baseline", "one_predictor"]
        assert res["design"]["baseline"] == "baseline"
        assert res["metrics"]["model"].tolist() == res["models"]
        assert res["comparisons"]["model"].tolist() == res["models"][1:]

    def test_the_metric_table_carries_its_contract_columns(self, scored_reg, scored_clf):
        for name in regression_metric_columns():
            assert name in scored_reg["metrics"].columns
        for name in classification_metric_columns():
            assert name in scored_clf["metrics"].columns

    def test_the_comparison_table_carries_its_contract_columns(self, scored_reg, scored_clf):
        assert list(scored_reg["comparisons"].columns) == regression_comparison_columns()
        assert list(scored_clf["comparisons"].columns) == classification_comparison_columns()

    def test_the_prediction_table_is_long_and_holds_each_model_once_in_order(
        self, scored_reg, scored_clf
    ):
        for res in (scored_reg, scored_clf):
            table = res["predictions"]
            for name in prediction_columns():
                assert name in table.columns
            assert list(dict.fromkeys(table["model"])) == res["models"]
            assert len(table.index) == len(res["models"]) * res["design"]["n_used"]

    def test_the_curve_table_carries_its_contract_columns(self, scored_clf):
        assert list(scored_clf["curves"].columns) == curve_columns()

    @pytest.mark.parametrize("kind", ["reg", "clf"])
    def test_every_slot_survives_being_written_out_as_json(self, scored_reg, scored_clf, kind):
        """The reason there is no engine handle here: an evaluation has nothing to
        predict with, so unlike a model it needs no exception to the rule."""
        import json

        res = scored_reg if kind == "reg" else scored_clf
        rebuilt = {}
        for name, slot in res.to_dict().items():
            rebuilt[name] = (
                json.loads(slot.to_json(orient="records"))
                if isinstance(slot, pd.DataFrame)
                else json.loads(json.dumps(slot))
            )
        assert rebuilt["models"] == res["models"]
        assert len(rebuilt["metrics"]) == len(res["metrics"].index)

    def test_the_infinite_first_threshold_of_a_curve_becomes_null_in_json(self, scored_clf):
        """The one cell of the whole object JSON has no number for. The
        coordinates it labels survive, which is what the curve is drawn from."""
        import json

        opening = json.loads(scored_clf["curves"].head(1).to_json(orient="records"))[0]
        assert opening["threshold"] is None
        assert opening["sensitivity"] == 0.0
        assert opening["specificity"] == 1.0

    @pytest.mark.parametrize("kind", ["reg", "clf"])
    def test_a_single_model_is_compared_to_nothing_and_the_slot_goes(
        self, reg, clf, reg_models, clf_models, kind
    ):
        if kind == "reg":
            res = evaluate_regression_models(reg_models[0], newdata=reg["test"])
        else:
            res = evaluate_classification_models(clf_models[0], newdata=clf["test"])
        assert "comparisons" not in res
        assert res["models"] == ["baseline"]

    @pytest.mark.parametrize("kind", ["reg", "clf"])
    def test_the_baseline_can_be_called_something_else(
        self, reg, clf, reg_models, clf_models, kind
    ):
        if kind == "reg":
            res = evaluate_regression_models(
                reg_models[0], newdata=reg["test"], baseline_label="full"
            )
        else:
            res = evaluate_classification_models(
                clf_models[0], newdata=clf["test"], baseline_label="full"
            )
        assert res["models"] == ["full"]
        assert res["design"]["baseline"] == "full"

    @pytest.mark.parametrize("kind", ["reg", "clf"])
    def test_repr_summarises_without_printing_the_tables(self, scored_reg, scored_clf, kind):
        res = scored_reg if kind == "reg" else scored_clf
        text = repr(res)
        assert res["analysis"] in text
        assert "metrics" in text and "comparisons" in text
        for name in res["models"]:
            assert name in text
        assert len(text.splitlines()) < len(res["predictions"].index)


class TestWhatIsRefused:
    def test_a_model_of_the_other_family_is_named_and_pointed_elsewhere(
        self, reg, clf, reg_models, clf_models
    ):
        with pytest.raises(SaValueError, match="evaluate_classification_models"):
            evaluate_regression_models(clf_models[0], newdata=clf["test"])
        with pytest.raises(SaValueError, match="evaluate_regression_models"):
            evaluate_classification_models(reg_models[0], newdata=reg["test"])

    def test_something_that_is_not_a_fitted_model_is_refused(self, reg):
        with pytest.raises(SaValueError, match="must be a fitted model"):
            evaluate_regression_models(reg["train"], newdata=reg["test"])

    def test_an_unnamed_collection_of_new_models_is_refused_rather_than_numbered(
        self, reg, reg_models
    ):
        full, thin = reg_models
        with pytest.raises(SaValueError, match="mapping of name to fitted model"):
            evaluate_regression_models(full, [thin], newdata=reg["test"])

    def test_a_new_model_that_is_not_a_model_is_named(self, reg, reg_models):
        with pytest.raises(SaValueError, match="Not a model: junk"):
            evaluate_regression_models(reg_models[0], {"junk": reg["train"]}, newdata=reg["test"])

    def test_a_new_model_may_not_be_called_what_the_baseline_is_called(self, reg, reg_models):
        full, thin = reg_models
        with pytest.raises(SaValueError, match="which is what the baseline is called"):
            evaluate_regression_models(full, {"baseline": thin}, newdata=reg["test"])

    def test_models_of_different_outcomes_are_refused(self, reg, reg_models):
        """Two such models can both be scored, and the scores can be put in one
        table, and the table means nothing."""
        full, _ = reg_models
        other = fit_linear_regression(
            reg["train"],
            outcome=reg["predictors"][0],
            predictors=[reg["outcome"]],
            cv=False,
        )
        with pytest.raises(SaValueError, match="fitted to the same outcome"):
            evaluate_regression_models(full, {"other": other}, newdata=reg["test"])

    def test_classifications_pointed_at_different_classes_are_refused(self, clf, clf_models):
        """Both answer a probability from `type="response"` and one of them is the
        probability of the other class, so every comparison between them is
        reversed."""
        full, _ = clf_models
        reversed_fit = fit_logistic_regression(
            clf["train"],
            outcome=clf["outcome"],
            predictors=clf["predictors"],
            outcome_lv=clf["outcome_lv"][::-1],
            cv=False,
        )
        with pytest.raises(SaValueError, match="same `outcome_lv`"):
            evaluate_classification_models(full, {"flipped": reversed_fit}, newdata=clf["test"])

    def test_newdata_with_no_rows_is_refused(self, reg, reg_models):
        with pytest.raises(SaValueError, match="zero rows"):
            evaluate_regression_models(reg_models[0], newdata=reg["test"].iloc[:0])

    def test_fewer_than_two_scorable_rows_is_refused(self, reg, reg_models):
        with pytest.raises(SaValueError, match="at least 2 are needed"):
            evaluate_regression_models(reg_models[0], newdata=reg["test"].iloc[:1])

    def test_an_answer_that_is_not_numeric_is_refused_for_a_regression(self, reg, reg_models):
        labels = ["a"] * len(reg["test"].index)
        with pytest.raises(SaValueError, match="must be numeric"):
            evaluate_regression_models(reg_models[0], newdata=reg["test"], answer=labels)

    def test_a_missing_outcome_column_asks_for_the_answer_by_name(self, reg, reg_models):
        without = reg["test"].drop(columns=[reg["outcome"]])
        with pytest.raises(SaValueError, match="Name the observed values with `answer`"):
            evaluate_regression_models(reg_models[0], newdata=without)

    def test_a_third_class_among_the_answers_is_refused(self, clf, clf_models):
        answers = clf["test"][clf["outcome"]].astype(object).copy()
        answers.iloc[0] = "other"
        with pytest.raises(SaValueError, match="class\\(es\\) the models were not fitted on"):
            evaluate_classification_models(
                clf_models[0], newdata=clf["test"], answer=answers.tolist()
            )

    def test_one_class_among_the_scored_rows_leaves_nothing_to_discriminate(self, clf, clf_models):
        one_class = clf["test"].loc[clf["test"][clf["outcome"]] == clf["outcome_lv"][0]]
        with pytest.raises(SaValueError, match="single class"):
            evaluate_classification_models(clf_models[0], newdata=one_class)

    def test_naming_the_other_class_is_a_disagreement_rather_than_a_reversal(self, clf, clf_models):
        """A fitted classification predicts the probability of one particular
        class and cannot be re-pointed after the fact."""
        with pytest.raises(SaValueError, match="cannot be re-pointed"):
            evaluate_classification_models(
                clf_models[0], newdata=clf["test"], outcome_lv=clf["outcome_lv"][::-1]
            )
        with pytest.raises(SaValueError, match="cannot be re-pointed"):
            evaluate_classification_models(
                clf_models[0], newdata=clf["test"], control_label=clf["outcome_lv"][1]
            )

    def test_naming_the_classes_the_way_they_were_fitted_is_accepted(self, clf, clf_models):
        res = evaluate_classification_models(
            clf_models[0],
            newdata=clf["test"],
            outcome_lv=clf["outcome_lv"],
            control_label=clf["outcome_lv"][0],
        )
        assert res["design"]["outcome_lv"] == clf["outcome_lv"]


class TestTheIntersectionOfTheRows:
    def test_a_row_one_model_cannot_predict_is_left_out_of_all_of_them(self, reg, reg_models):
        """Which is what stops two numbers from two samples being put in one table
        and their difference being called an improvement."""
        full, thin = reg_models
        holed = reg["test"].copy()
        # A hole in a column only the full model was fitted on: the thin model
        # could answer this row, and does not get to.
        holed.loc[holed.index[0], reg["predictors"][-1]] = np.nan
        res = evaluate_regression_models(full, {"one_predictor": thin}, newdata=holed)
        assert res["design"]["n_obs"] == len(holed.index)
        assert res["design"]["n_dropped"] == 1
        assert res["design"]["n_used"] == len(holed.index) - 1
        assert set(res["predictions"]["row"]) == set(range(1, len(holed.index)))
        for name in res["models"]:
            scored = res["predictions"].loc[res["predictions"]["model"] == name, "row"]
            assert scored.tolist() == sorted(set(res["predictions"]["row"]))

    def test_a_row_with_no_observed_outcome_cannot_be_scored_either(self, reg, reg_models):
        holed = reg["test"].copy()
        holed.loc[holed.index[0], reg["outcome"]] = np.nan
        res = evaluate_regression_models(reg_models[0], newdata=holed)
        assert res["design"]["n_dropped"] == 1
        assert 0 not in set(res["predictions"]["row"])

    def test_the_rows_that_went_are_reported_once_with_the_reason(self, reg, reg_models, caplog):
        full, thin = reg_models
        holed = reg["test"].copy()
        holed.loc[holed.index[0], reg["predictors"][-1]] = np.nan
        holed.loc[holed.index[1], reg["outcome"]] = np.nan
        with caplog.at_level(logging.INFO, logger="statassist"):
            evaluate_regression_models(full, {"one_predictor": thin}, newdata=holed)
        notes = [record.message for record in caplog.records if "left out" in record.message]
        assert len(notes) == 1
        assert "with no observed outcome" in notes[0]
        assert "incomplete across the predictors of baseline" in notes[0]

    def test_the_row_column_is_the_position_in_newdata_rather_than_a_running_count(
        self, reg, reg_models
    ):
        holed = reg["test"].copy()
        holed.loc[holed.index[0], reg["outcome"]] = np.nan
        res = evaluate_regression_models(reg_models[0], newdata=holed)
        rows = res["predictions"]["row"].to_numpy()
        assert rows.min() == 1
        assert rows.max() == len(holed.index) - 1


class TestARegressionEvaluation:
    def test_the_predictions_are_the_ones_the_model_itself_answers(
        self, reg, reg_models, scored_reg
    ):
        full, _ = reg_models
        direct = np.asarray(full.predict(reg["test"], type="response"), dtype=float)
        held = scored_reg["predictions"]
        mine = held.loc[held["model"] == "baseline"]
        assert np.allclose(mine["predicted"].to_numpy(dtype=float), direct)

    def test_every_delta_is_new_minus_baseline(self, scored_reg):
        metrics = scored_reg["metrics"]
        comparisons = scored_reg["comparisons"]
        for column in ("cor", "r_squared", "rmse", "mae"):
            assert comparisons[f"delta_{column}"].iloc[0] == pytest.approx(
                metrics[column].iloc[1] - metrics[column].iloc[0]
            )

    def test_the_full_model_beats_the_one_predictor_model_on_the_held_out_rows(self):
        """A property rather than a number, and it needs a signal strong enough to
        be the property: the planted effect is spread over every predictor, so
        dropping all but one of them has to cost something. On the noisier fixture
        the two models are within each other's sampling error and either can come
        out ahead, which is a fact about fifty rows rather than about the scoring.
        """
        sim = simulate_regression(n_samples=200, n_pred=4, noise_sd=0.5, seed=7)
        frame = sim.args["data"]
        train, test = frame.iloc[:150], frame.iloc[150:]
        predictors = list(sim.args["predictors"])
        full = fit_linear_regression(
            train, outcome=sim.args["outcome"], predictors=predictors, cv=False
        )
        thin = fit_linear_regression(
            train, outcome=sim.args["outcome"], predictors=predictors[:1], cv=False
        )
        row = evaluate_regression_models(full, {"one_predictor": thin}, newdata=test)[
            "comparisons"
        ].iloc[0]
        assert row["delta_cor"] < 0
        assert row["delta_r_squared"] < 0
        assert row["delta_rmse"] > 0
        assert row["delta_mae"] > 0

    def test_rmse_and_mae_are_what_the_residuals_say(self, scored_reg):
        held = scored_reg["predictions"]
        for position, name in enumerate(scored_reg["models"]):
            mine = held.loc[held["model"] == name]
            residual = mine["predicted"].to_numpy(dtype=float) - mine["observed"].to_numpy(
                dtype=float
            )
            row = scored_reg["metrics"].iloc[position]
            assert row["rmse"] == pytest.approx(math.sqrt(float(np.mean(residual**2))))
            assert row["mae"] == pytest.approx(float(np.mean(np.abs(residual))))
            assert row["bias"] == pytest.approx(float(np.mean(residual)))

    def test_r_squared_is_measured_against_the_outcome_rather_than_against_a_refitted_line(
        self, scored_reg
    ):
        """`cor**2` is what `r_squared` would be if the predictions were first
        rescaled by a line fitted to these same rows, so the two agree only for a
        model that needs no calibration."""
        row = scored_reg["metrics"].iloc[0]
        assert row["r_squared"] < row["cor"] ** 2
        assert row["calib_slope"] != pytest.approx(1.0, abs=0.05)

    def test_the_calibration_line_is_predicted_on_observed(self, scored_reg):
        held = scored_reg["predictions"]
        for position, name in enumerate(scored_reg["models"]):
            mine = held.loc[held["model"] == name]
            observed = mine["observed"].to_numpy(dtype=float)
            predicted = mine["predicted"].to_numpy(dtype=float)
            slope = float(np.cov(observed, predicted, ddof=1)[0, 1] / np.var(observed, ddof=1))
            row = scored_reg["metrics"].iloc[position]
            assert row["calib_slope"] == pytest.approx(slope)
            assert row["calib_intercept"] == pytest.approx(
                float(predicted.mean()) - slope * float(observed.mean())
            )

    def test_a_prediction_that_does_not_vary_ranks_nothing_and_says_so(self, reg, reg_models):
        """A forest with one leaf per tree answers the same value for every row,
        which leaves it a calibration line and no correlation."""
        full, _ = reg_models
        flat = fit_rf(
            reg["train"],
            outcome=reg["outcome"],
            predictors=reg["predictors"],
            mtry=1,
            ntree=5,
            nodesize=len(reg["train"].index),
            cv=False,
            seed=1,
        )
        with pytest.warns(SaWarning, match="`cor` is missing for 1 model"):
            res = evaluate_regression_models(full, {"flat": flat}, newdata=reg["test"])
        assert math.isnan(res["metrics"]["cor"].iloc[1])
        assert res["metrics"]["calib_slope"].iloc[1] == pytest.approx(0.0)

    def test_an_outcome_that_does_not_vary_leaves_no_variance_to_explain(self, reg, reg_models):
        constant = reg["test"].copy()
        constant[reg["outcome"]] = 1.0
        with pytest.warns(SaWarning, match="single value over the"):
            res = evaluate_regression_models(reg_models[0], newdata=constant)
        row = res["metrics"].iloc[0]
        assert math.isnan(row["cor"])
        assert math.isnan(row["r_squared"])
        assert math.isnan(row["calib_slope"])
        # Measured against the observed value rather than against its spread, so
        # these three survive the rows being what they are.
        assert math.isfinite(row["rmse"])
        assert math.isfinite(row["mae"])
        assert math.isfinite(row["bias"])

    def test_scoring_a_regression_takes_no_choices(self, scored_reg):
        assert scored_reg["parameters"] == {}


class TestAClassificationEvaluation:
    def test_the_predictions_are_the_probability_of_the_second_level(
        self, clf, clf_models, scored_clf
    ):
        full, _ = clf_models
        direct = np.asarray(full.predict(clf["test"], type="response"), dtype=float)
        held = scored_clf["predictions"]
        mine = held.loc[held["model"] == "baseline"]
        assert np.allclose(mine["predicted"].to_numpy(dtype=float), direct)
        assert mine["predicted"].between(0, 1).all()

    def test_the_observed_column_is_one_for_the_second_level(self, clf, scored_clf):
        held = scored_clf["predictions"]
        mine = held.loc[held["model"] == "baseline"]
        answers = clf["test"][clf["outcome"]].astype(str).to_numpy()
        expected = (answers[mine["row"].to_numpy()] == clf["outcome_lv"][1]).astype(float)
        assert np.array_equal(mine["observed"].to_numpy(dtype=float), expected)
        assert scored_clf["design"]["n_events"] == int(expected.sum())

    def test_the_row_counts_are_the_same_for_every_model(self, scored_clf):
        metrics = scored_clf["metrics"]
        assert metrics["n_used"].nunique() == 1
        assert metrics["n_events"].nunique() == 1
        assert int(metrics["n_used"].iloc[0]) == scored_clf["design"]["n_used"]

    def test_the_full_model_discriminates_better_and_all_three_tests_agree(self, scored_clf):
        row = scored_clf["comparisons"].iloc[0]
        assert row["delta_auc"] < 0
        assert row["idi"] < 0
        assert row["nri"] < 0

    def test_the_nri_total_is_its_two_class_wise_halves(self, scored_clf):
        row = scored_clf["comparisons"].iloc[0]
        assert row["nri"] == pytest.approx(row["nri_event"] + row["nri_nonevent"])

    def test_every_interval_brackets_its_estimate(self, scored_clf):
        row = scored_clf["comparisons"].iloc[0]
        for stem in ("delta_auc", "idi", "nri"):
            assert row[f"{stem}_lower_conf"] < row[stem] < row[f"{stem}_upper_conf"]

    def test_a_narrower_confidence_level_gives_a_narrower_interval(self, clf, clf_models):
        full, thin = clf_models
        wide = evaluate_classification_models(
            full, {"thin": thin}, newdata=clf["test"], conf_level=0.99
        )
        narrow = evaluate_classification_models(
            full, {"thin": thin}, newdata=clf["test"], conf_level=0.80
        )

        def width(res, stem):
            row = res["comparisons"].iloc[0]
            return row[f"{stem}_upper_conf"] - row[f"{stem}_lower_conf"]

        for stem in ("delta_auc", "idi", "nri"):
            assert width(narrow, stem) < width(wide, stem)
        # The p-value is a property of the data rather than of the interval asked
        # for, so it does not move with the level.
        assert wide["comparisons"]["delta_auc_pval"].iloc[0] == pytest.approx(
            narrow["comparisons"]["delta_auc_pval"].iloc[0]
        )

    def test_the_threshold_free_scores_do_not_move_with_the_threshold(self, clf, clf_models):
        full, _ = clf_models
        half = evaluate_classification_models(full, newdata=clf["test"], threshold=0.5)
        strict = evaluate_classification_models(full, newdata=clf["test"], threshold=0.9)
        for column in ("auc", "brier"):
            assert half["metrics"][column].iloc[0] == pytest.approx(
                strict["metrics"][column].iloc[0]
            )
        # A stricter cut calls fewer rows an event, so it can only lose
        # sensitivity and gain specificity.
        assert strict["metrics"]["sensitivity"].iloc[0] <= half["metrics"]["sensitivity"].iloc[0]
        assert strict["metrics"]["specificity"].iloc[0] >= half["metrics"]["specificity"].iloc[0]
        assert strict["parameters"]["threshold"] == 0.9

    def test_the_threshold_scores_are_the_calls_the_threshold_makes(self, scored_clf):
        threshold = scored_clf["parameters"]["threshold"]
        held = scored_clf["predictions"]
        for position, name in enumerate(scored_clf["models"]):
            mine = held.loc[held["model"] == name]
            observed = mine["observed"].to_numpy(dtype=float) == 1
            called = mine["predicted"].to_numpy(dtype=float) >= threshold
            row = scored_clf["metrics"].iloc[position]
            assert row["accuracy"] == pytest.approx(float(np.mean(called == observed)))
            assert row["sensitivity"] == pytest.approx(float(np.mean(called[observed])))
            assert row["specificity"] == pytest.approx(float(np.mean(~called[~observed])))

    def test_the_curve_holds_every_model_and_opens_at_the_corner(self, scored_clf):
        curves = scored_clf["curves"]
        assert list(dict.fromkeys(curves["model"])) == scored_clf["models"]
        for name in scored_clf["models"]:
            mine = curves.loc[curves["model"] == name]
            first = mine.iloc[0]
            assert math.isinf(float(first["threshold"]))
            assert first["sensitivity"] == 0.0
            assert first["specificity"] == 1.0
            assert mine["sensitivity"].iloc[-1] == pytest.approx(1.0)
            assert mine["specificity"].iloc[-1] == pytest.approx(0.0)

    def test_the_area_under_the_drawn_curve_is_the_auc_beside_it(self, scored_clf):
        """Which is what makes the picture and the table describe the same curve:
        consecutive points joined by straight lines, so a run of ties is crossed
        diagonally."""
        curves = scored_clf["curves"]
        for position, name in enumerate(scored_clf["models"]):
            mine = curves.loc[curves["model"] == name]
            x = 1 - mine["specificity"].to_numpy(dtype=float)
            y = mine["sensitivity"].to_numpy(dtype=float)
            order = np.lexsort((y, x))
            area = float(np.trapezoid(y[order], x[order]))
            assert area == pytest.approx(scored_clf["metrics"]["auc"].iloc[position], abs=1e-8)

    def test_the_brier_score_is_the_mean_squared_distance_to_the_outcome(self, scored_clf):
        """Which an AUC is blind to: a model that ranks perfectly and predicts
        every event at 0.6 has an AUC of 1."""
        held = scored_clf["predictions"]
        for position, name in enumerate(scored_clf["models"]):
            mine = held.loc[held["model"] == name]
            gap = mine["predicted"].to_numpy(dtype=float) - mine["observed"].to_numpy(dtype=float)
            assert scored_clf["metrics"]["brier"].iloc[position] == pytest.approx(
                float(np.mean(gap**2))
            )

    def test_the_auc_interval_is_a_wald_one_and_is_not_clamped(self, scored_clf):
        """An interval on a bounded quantity from an unbounded approximation. It
        is left as it is rather than clipped, so that a reader can see the
        approximation is what it rests on."""
        row = scored_clf["metrics"].iloc[0]
        assert row["auc_lower_conf"] < row["auc"] < row["auc_upper_conf"]

    def test_a_model_of_a_different_engine_is_scored_by_the_same_arithmetic(self, clf, clf_models):
        """The whole point of reading every model through `predict`: a forest and
        a logistic regression are interchangeable here."""
        full, _ = clf_models
        forest = fit_rf(
            clf["train"],
            outcome=clf["outcome"],
            predictors=clf["predictors"],
            outcome_lv=clf["outcome_lv"],
            ntree=60,
            cv=False,
            seed=1,
        )
        res = evaluate_classification_models(full, {"forest": forest}, newdata=clf["test"])
        assert res["models"] == ["baseline", "forest"]
        assert res["metrics"]["auc"].between(0, 1).all()
        assert math.isfinite(res["comparisons"]["delta_auc_pval"].iloc[0])
