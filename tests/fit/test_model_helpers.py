"""The shared model helpers: input resolution, coding, direction, grids.

The port of what ``utils_model.R`` is checked on. These are the questions every
``fit_*`` function asks before any engine runs, so a mistake here is a mistake in
five models at once.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statassist.core import SaInternalError, SaValueError, new_model
from statassist.fit._shared import (
    CV_METHODS,
    design_lv,
    design_matrix,
    encode_outcome,
    enet_grid,
    inference_table,
    model_frame,
    no_grid,
    outcome_levels,
    predict_frame,
    resample_grid,
    resolve_model_input,
    rf_grid,
    svm_grid,
    train_control,
)


@pytest.fixture
def frame():
    """A small model frame with one of each kind of predictor."""
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "cat": ["lo", "mid", "hi", "lo", "mid", "hi"],
            "flag": [True, False, True, False, True, False],
            "const": [7.0] * 6,
        }
    )


class TestResolveModelInput:
    def test_the_outcome_column_is_not_a_candidate_predictor(self, frame):
        resolved = resolve_model_input(frame, "y")
        assert "y" not in resolved.predictors
        assert resolved.outcome == "y"

    def test_a_single_valued_predictor_is_left_out_and_named(self, frame):
        resolved = resolve_model_input(frame, "y")
        assert resolved.dropped_predictors == ["const"]
        assert "const" not in resolved.x.columns

    def test_the_outcome_among_the_predictors_is_refused(self, frame):
        with pytest.raises(SaValueError, match="predict from the answer"):
            resolve_model_input(frame, "y", ["num", "y"])

    def test_an_unknown_predictor_is_named(self, frame):
        with pytest.raises(SaValueError, match="not found in `data`: nope"):
            resolve_model_input(frame, "y", ["num", "nope"])

    def test_a_duplicated_predictor_is_named(self, frame):
        with pytest.raises(SaValueError, match="duplicated names: num"):
            resolve_model_input(frame, "y", ["num", "num"])

    def test_rows_missing_anything_the_model_needs_are_dropped_and_counted(self, frame):
        holed = frame.copy()
        holed.loc[0, "num"] = np.nan
        holed.loc[1, "y"] = np.nan
        resolved = resolve_model_input(holed, "y", ["num", "cat"])
        assert (resolved.n_obs, resolved.n_used, resolved.n_dropped) == (6, 4, 2)

    def test_a_string_predictor_becomes_a_categorical_with_its_levels_recorded(self, frame):
        resolved = resolve_model_input(frame, "y", ["cat"])
        assert sorted(resolved.predictor_lv["cat"]) == ["hi", "lo", "mid"]

    def test_levels_left_over_from_the_row_filtering_go(self, frame):
        holed = frame.copy()
        holed.loc[holed["cat"] == "hi", "num"] = np.nan
        resolved = resolve_model_input(holed, "y", ["num", "cat"])
        assert "hi" not in resolved.predictor_lv["cat"]

    def test_a_frame_whose_predictors_are_all_constant_has_nothing_to_fit(self):
        flat = pd.DataFrame({"y": [1.0, 2.0, 3.0], "a": [1.0] * 3, "b": [2.0] * 3})
        with pytest.raises(SaValueError, match="nothing to fit"):
            resolve_model_input(flat, "y")

    def test_an_outcome_given_as_a_vector_leaves_every_column_a_candidate(self, frame):
        resolved = resolve_model_input(frame[["num", "cat"]], frame["y"].to_numpy())
        assert resolved.predictors == ["num", "cat"]


class TestDesignMatrix:
    def test_a_k_level_factor_becomes_k_minus_one_terms_named_after_the_level(self, frame):
        resolved = resolve_model_input(frame, "y", ["num", "cat"])
        matrix = design_matrix(resolved.x)
        assert list(matrix.columns) == ["num", "catlo", "catmid"]

    def test_a_logical_predictor_is_coded_the_way_model_matrix_codes_it(self, frame):
        resolved = resolve_model_input(frame, "y", ["flag"])
        matrix = design_matrix(resolved.x)
        assert list(matrix.columns) == ["flagTRUE"]
        assert matrix["flagTRUE"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def test_the_intercept_is_not_among_the_terms(self, frame):
        resolved = resolve_model_input(frame, "y", ["num"])
        assert "(Intercept)" not in design_matrix(resolved.x).columns

    def test_a_level_no_row_takes_still_gets_its_column_of_zeroes(self):
        """Which is the whole point of coding `newdata` against the fit's levels:
        a held-out half missing a level would otherwise code to a narrower
        matrix than the model has coefficients for."""
        held_out = pd.DataFrame({"cat": pd.Categorical(["lo", "lo"], categories=["hi", "lo"])})
        matrix = design_matrix(held_out, xlev={"cat": ["hi", "lo", "mid"]})
        assert list(matrix.columns) == ["catlo", "catmid"]
        assert matrix["catmid"].tolist() == [0.0, 0.0]

    def test_a_missing_cell_reaches_every_dummy_of_the_factor_it_came_from(self):
        frame = pd.DataFrame({"cat": pd.Categorical([None, "lo"], categories=["hi", "lo"])})
        matrix = design_matrix(frame)
        assert bool(matrix.iloc[0].isna().all())

    def test_want_fixes_the_order_by_name(self, frame):
        resolved = resolve_model_input(frame, "y", ["num", "cat"])
        matrix = design_matrix(resolved.x, want=["catmid", "num", "catlo"])
        assert list(matrix.columns) == ["catmid", "num", "catlo"]

    def test_a_coding_missing_a_term_the_model_has_is_an_internal_error(self, frame):
        resolved = resolve_model_input(frame, "y", ["num"])
        with pytest.raises(SaInternalError, match="missing term"):
            design_matrix(resolved.x, want=["num", "catmid"])


class TestPredictFrame:
    @pytest.fixture
    def design(self, frame):
        resolved = resolve_model_input(frame, "y", ["num", "cat"])
        return {"predictors": resolved.predictors, "predictor_lv": resolved.predictor_lv}

    def test_columns_the_model_never_saw_are_ignored_rather_than_refused(self, frame, design):
        kept = predict_frame(frame, design)
        assert list(kept.columns) == ["num", "cat"]

    def test_a_missing_predictor_is_an_error_naming_it(self, frame, design):
        with pytest.raises(SaValueError, match="missing predictor column"):
            predict_frame(frame.drop(columns=["num"]), design)

    def test_a_level_the_fit_never_saw_has_no_coefficient_to_apply(self, frame, design):
        unseen = frame.copy()
        unseen.loc[0, "cat"] = "enormous"
        with pytest.raises(SaValueError, match="was not fitted on"):
            predict_frame(unseen, design)

    def test_the_levels_are_put_back_in_the_order_the_fit_used(self, frame, design):
        kept = predict_frame(frame.iloc[[0, 1]], design)
        assert list(kept["cat"].cat.categories) == design["predictor_lv"]["cat"]

    def test_a_numeric_predictor_that_arrived_as_text_is_refused(self, frame, design):
        retyped = frame.copy()
        retyped["num"] = retyped["num"].astype(str)
        with pytest.raises(SaValueError, match="cannot change between fitting"):
            predict_frame(retyped, design)


class TestOutcomeLevels:
    def test_sorting_is_what_puts_control_before_treated(self):
        y = pd.Series(["treated", "control", "control"])
        assert outcome_levels(y) == ["control", "treated"]

    def test_a_named_pair_is_taken_reference_first(self):
        y = pd.Series(["a", "b", "b"])
        assert outcome_levels(y, outcome_lv=["b", "a"]) == ["b", "a"]

    def test_control_label_states_the_same_direction_with_one_name(self):
        y = pd.Series(["a", "b", "b"])
        assert outcome_levels(y, control_label="b") == ["b", "a"]

    def test_naming_the_reference_twice_and_disagreeing_is_an_error(self):
        y = pd.Series(["a", "b"])
        with pytest.raises(SaValueError, match="disagree about which class"):
            outcome_levels(y, outcome_lv=["a", "b"], control_label="b")

    def test_a_third_class_is_an_error_rather_than_dropped_rows(self):
        y = pd.Series(["a", "b", "c"])
        with pytest.raises(SaValueError, match="holds 3 classes"):
            outcome_levels(y)

    def test_naming_two_of_three_would_leave_rows_out_silently(self):
        y = pd.Series(["a", "b", "c"])
        with pytest.raises(SaValueError, match="silently left out: c"):
            outcome_levels(y, outcome_lv=["a", "b"])

    def test_a_single_class_has_nothing_to_classify(self):
        with pytest.raises(SaValueError, match="nothing to classify"):
            outcome_levels(pd.Series(["a", "a"]))

    def test_the_second_level_is_the_one_that_becomes_the_event(self):
        y = pd.Series(["control", "case", "case"])
        assert encode_outcome(y, ["control", "case"]).tolist() == [0, 1, 1]
        assert encode_outcome(y, ["case", "control"]).tolist() == [1, 0, 0]


class TestGrids:
    def test_a_lasso_is_alpha_one_and_a_ridge_alpha_zero(self):
        assert enet_grid("lasso", alpha=0.3, lambda_=0.5, cv=False)["alpha"].tolist() == [1.0]
        assert enet_grid("ridge", alpha=0.3, lambda_=0.5, cv=False)["alpha"].tolist() == [0.0]

    def test_an_elastic_net_crosses_the_two_arguments(self):
        grid = enet_grid("elastic_net", alpha=[0.2, 0.8], lambda_=[0.1, 1.0], cv=True)
        assert len(grid.index) == 4
        assert set(grid.columns) == {"alpha", "lambda_"}

    def test_no_resampling_leaves_nothing_to_choose_between_candidates(self):
        with pytest.raises(SaValueError, match="must hold one candidate"):
            enet_grid("lasso", alpha=1, lambda_=[0.1, 1.0], cv=False)

    def test_alpha_is_validated_even_when_the_penalty_fixes_it(self):
        with pytest.raises(SaValueError, match="`alpha`"):
            enet_grid("lasso", alpha=2, lambda_=0.5, cv=False)

    def test_mtry_defaults_to_the_rule_of_thumb_for_the_outcome_type(self):
        assert rf_grid(None, p=9, classify=True, cv=True)["mtry"].tolist() == [3]
        assert rf_grid(None, p=9, classify=False, cv=True)["mtry"].tolist() == [3]
        assert rf_grid(None, p=4, classify=False, cv=True)["mtry"].tolist() == [1]

    def test_mtry_above_the_predictor_count_is_refused(self):
        with pytest.raises(SaValueError, match="cannot exceed the 3 predictor"):
            rf_grid([2, 5], p=3, classify=False, cv=True)

    def test_mtry_counts_predictors_so_it_must_be_whole(self):
        with pytest.raises(SaValueError, match="whole numbers"):
            rf_grid(2.5, p=5, classify=False, cv=True)

    def test_zero_is_refused_by_name_for_both_machine_arguments(self):
        with pytest.raises(SaValueError, match="`C` must be above 0"):
            svm_grid(C=0, sigma=1, cv=False)
        with pytest.raises(SaValueError, match="`sigma` must be above 0"):
            svm_grid(C=1, sigma=0, cv=False)

    def test_the_machine_grid_crosses_cost_and_width(self):
        grid = svm_grid(C=[1, 10], sigma=[0.1, 1], cv=True)
        assert len(grid.index) == 4
        assert list(grid.columns) == ["sigma", "C"]


class TestResampleControl:
    def test_no_resampling_records_none_of_the_three_arguments(self):
        control = train_control(False, "kfold", 5, 5, n_obs=50)
        assert (control.splitter, control.cv_method) == (None, None)
        assert (control.n_fold, control.n_repeat) == (None, None)

    def test_a_scheme_records_only_what_it_used(self):
        repeated = train_control(True, "repeated_kfold", 4, 3, n_obs=50)
        assert (repeated.n_fold, repeated.n_repeat) == (4, 3)
        single = train_control(True, "kfold", 4, 3, n_obs=50)
        assert (single.n_fold, single.n_repeat) == (4, None)
        loo = train_control(True, "loocv", 4, 3, n_obs=50)
        assert (loo.n_fold, loo.n_repeat) == (None, None)

    def test_every_argument_is_validated_whatever_the_scheme_reads(self):
        with pytest.raises(SaValueError, match="`n_fold`"):
            train_control(True, "loocv", 1, 5, n_obs=50)

    def test_more_folds_than_observations_would_leave_a_fold_empty(self):
        with pytest.raises(SaValueError, match="exceeds the 4 usable"):
            train_control(True, "kfold", 5, 5, n_obs=4)

    def test_an_unknown_scheme_is_refused_by_the_public_check(self):
        from statassist.fit._shared import check_cv_method

        with pytest.raises(SaValueError, match="must be one of"):
            check_cv_method("bootstrap")
        assert check_cv_method(CV_METHODS[0]) == "repeated_kfold"


class TestResampleGrid:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(3)
        x = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
        y = 2 * x["a"].to_numpy() - x["b"].to_numpy() + rng.normal(0, 0.3, 60)
        return x, y

    def _build(self, params):
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    def test_nothing_resampled_reports_no_tables_and_still_names_the_metrics(self, data):
        x, y = data
        control = train_control(False, "kfold", 5, 5, n_obs=len(y))
        scored = resample_grid(self._build, x, y, no_grid(), control, classify=False)
        assert (scored.results, scored.resampling) == (None, None)
        assert scored.metrics == ["RMSE", "Rsquared", "MAE"]

    def test_a_kfold_run_reports_one_row_per_candidate_and_one_per_resample(self, data):
        x, y = data
        control = train_control(True, "kfold", 5, 5, n_obs=len(y), seed=1)
        scored = resample_grid(self._build, x, y, no_grid(), control, classify=False)
        assert len(scored.results.index) == 1
        assert len(scored.resampling.index) == 5
        assert list(scored.resampling["Resample"]) == [f"Fold{at}" for at in range(1, 6)]
        for metric in scored.metrics:
            assert f"{metric}SD" in scored.results.columns

    def test_a_repeated_run_labels_the_fold_and_the_repeat(self, data):
        x, y = data
        control = train_control(True, "repeated_kfold", 3, 2, n_obs=len(y), seed=1)
        scored = resample_grid(self._build, x, y, no_grid(), control, classify=False)
        assert list(scored.resampling["Resample"]) == [
            "Fold1.Rep1",
            "Fold2.Rep1",
            "Fold3.Rep1",
            "Fold1.Rep2",
            "Fold2.Rep2",
            "Fold3.Rep2",
        ]

    def test_leave_one_out_is_scored_on_the_pooled_predictions(self, data):
        """A fold of one row has no correlation and no spread, so a per-fold
        table would be missing throughout and its standard deviation would
        describe the folds rather than the model."""
        x, y = data
        control = train_control(True, "loocv", 5, 5, n_obs=len(y))
        scored = resample_grid(self._build, x[:20], y[:20], no_grid(), control, classify=False)
        assert scored.resampling is None
        assert len(scored.results.index) == 1
        assert math.isfinite(float(scored.results["Rsquared"].iloc[0]))
        assert "RsquaredSD" not in scored.results.columns

    def test_the_candidate_that_placed_first_is_the_one_reported(self, data):
        """Chosen on the first metric, and its direction is what the metric
        means rather than a setting: a smaller RMSE is better."""
        from sklearn.linear_model import Ridge

        x, y = data
        grid = pd.DataFrame({"alpha": [1e-6, 1e6]})
        control = train_control(True, "kfold", 5, 5, n_obs=len(y), seed=1)
        scored = resample_grid(
            lambda params: Ridge(alpha=params["alpha"]), x, y, grid, control, classify=False
        )
        assert scored.best["alpha"] == pytest.approx(1e-6)


class TestSmallHelpers:
    def test_a_table_with_its_columns_and_no_rows_is_dropped(self):
        assert model_frame(pd.DataFrame({"a": []})) is None
        assert model_frame(None) is None
        assert model_frame(pd.DataFrame({"a": [1]})) is not None

    def test_the_levels_entry_is_left_out_when_nothing_has_levels(self):
        assert design_lv({}) == {}
        assert design_lv({"g": ["a", "b"]}) == {"predictor_lv": {"g": ["a", "b"]}}

    def test_the_inference_table_carries_the_contract_columns_in_order(self):
        table = inference_table(
            ["(Intercept)", "a"],
            np.array([1.0, 2.0]),
            np.array([0.5, 0.5]),
            conf_level=0.95,
            df=10,
        )
        assert list(table.columns) == [
            "terms",
            "estimate",
            "stderr",
            "statistic",
            "df",
            "pval",
            "lower_conf",
            "upper_conf",
        ]
        assert table["df"].tolist() == [10.0, 10.0]

    def test_a_wald_z_is_referred_to_no_degrees_of_freedom(self):
        table = inference_table(["a"], np.array([2.0]), np.array([1.0]), conf_level=0.95, df=None)
        assert bool(table["df"].isna().all())
        # The interval is the estimate plus and minus 1.96 standard errors.
        assert float(table["upper_conf"].iloc[0]) == pytest.approx(2 + 1.959963985, abs=1e-6)


class TestModelContract:
    def _table(self, terms):
        return pd.DataFrame({"terms": terms, "estimate": [1.0] * len(terms)})

    def _engine(self):
        return {"package": "p", "method": "m", "label": "l", "metrics": ["RMSE"]}

    def test_a_coefficient_table_out_of_order_is_an_internal_error(self):
        with pytest.raises(SaInternalError, match="not aligned with `terms`"):
            new_model(
                "m",
                ["a", "b"],
                {},
                {},
                self._table(["b", "a"]),
                {},
                self._engine(),
                fit=None,
            )

    def test_the_inference_columns_come_as_a_group(self):
        table = self._table(["a"])
        table["pval"] = [0.5]
        with pytest.raises(SaInternalError, match="some inference column"):
            new_model("m", ["a"], {}, {}, table, {}, self._engine(), fit=None)

    def test_an_engine_that_does_not_name_itself_is_an_internal_error(self):
        engine = self._engine()
        del engine["metrics"]
        with pytest.raises(SaInternalError, match="missing `metrics`"):
            new_model("m", ["a"], {}, {}, self._table(["a"]), {}, engine, fit=None)

    def test_the_engine_object_is_reached_as_an_attribute_and_is_not_a_slot(self):
        """Which is what keeps every slot JSON-shaped without anything having to
        be dropped first."""
        model = new_model("m", ["a"], {}, {}, self._table(["a"]), {}, self._engine(), fit="engine")
        assert model.fit == "engine"
        assert "fit" not in model
        assert "fit" not in model.to_dict()
