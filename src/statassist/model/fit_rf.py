"""Random forest regression and classification."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from statassist.contracts.model import sa_new_model
from statassist.utils.model_utils import (
    sa_design_lv,
    sa_outcome_levels,
    sa_resolve_model_input,
    sa_rf_grid,
    sa_train_control,
)
from statassist.utils.validate import sa_check_count, sa_preserve_seed


def _rf_importance(model, classify: bool) -> pd.DataFrame:
    imp = model.feature_importances_
    if classify:
        perm_col, purity_col = "MeanDecreaseAccuracy", "MeanDecreaseGini"
    else:
        perm_col, purity_col = "%IncMSE", "IncNodePurity"
    out = pd.DataFrame(
        {
            "terms": model.feature_names_in_,
            "estimate": imp,
            "impurity": getattr(model, "feature_importances_", imp),
        }
    )
    out = out.sort_values("estimate", ascending=False).reset_index(drop=True)
    return out


def _rf_reg_stats(oob_prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    at = ~np.isnan(oob_prediction)
    fitted = oob_prediction[at]
    observed = observed[at]
    residual = observed - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "oob_r_squared": 1 - sse / sst if sst > 0 else np.nan,
        "oob_rmse": float(np.sqrt(np.mean(residual**2))),
        "oob_mae": float(np.mean(np.abs(residual))),
        "n_oob": int(at.sum()),
    }


def _rf_class_stats(oob_prediction, observed, outcome_lv: list[str]) -> dict[str, float]:
    at = ~pd.isna(oob_prediction)
    called = (np.asarray(oob_prediction)[at].astype(str) == outcome_lv[1])
    positive = (np.asarray(observed)[at].astype(str) == outcome_lv[1])
    accuracy = float(np.mean(called == positive))
    expected = called.mean() * positive.mean() + (~called).mean() * (~positive).mean()
    return {
        "oob_accuracy": accuracy,
        "oob_error": 1 - accuracy,
        "oob_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else np.nan,
        "oob_sensitivity": float(called[positive].mean()) if positive.any() else np.nan,
        "oob_specificity": float((~called[~positive]).mean()) if (~positive).any() else np.nan,
        "n_oob": int(at.sum()),
    }


def fit_rf(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    mtry: int | list[int] | None = None,
    ntree: int = 500,
    nodesize: int | None = None,
    cv: bool = True,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    ntree = sa_check_count(ntree, "ntree", 1)
    input_ = sa_resolve_model_input(data, outcome, predictors)

    classify = (
        outcome_lv is not None
        or control_label is not None
        or not np.issubdtype(input_["y"].dtype, np.number)
    )
    if not classify and len(np.unique(input_["y"])) == 2:
        warnings.warn(
            "`outcome` is numeric and takes two values, so it was fitted as a regression.",
            stacklevel=2,
        )

    if classify:
        y = sa_outcome_levels(input_["y"], outcome_lv, control_label, model="a random forest")
        outcome_lv = list(y.categories)
        y_fit = y.astype(str).to_numpy()
    else:
        if not np.all(np.isfinite(input_["y"])):
            raise ValueError("`outcome` holds non-finite value(s).")
        y_fit = input_["y"].astype(float)

    if nodesize is None:
        nodesize = 1 if classify else 5
    nodesize = sa_check_count(nodesize, "nodesize", 1)
    grid = sa_rf_grid(
        None if mtry is None else np.asarray(mtry),
        len(input_["predictors"]),
        classify,
        cv,
    )
    ctrl = sa_train_control(cv, cv_method, n_fold, n_repeat, input_["n_used"])
    x_df = input_["x"]

    with sa_preserve_seed(seed):
        if classify:
            base = RandomForestClassifier(
                n_estimators=ntree,
                min_samples_leaf=nodesize,
                oob_score=True,
                random_state=seed,
            )
            scoring = "accuracy"
        else:
            base = RandomForestRegressor(
                n_estimators=ntree,
                min_samples_leaf=nodesize,
                oob_score=True,
                random_state=seed,
            )
            scoring = "neg_root_mean_squared_error"

        param_grid = {"max_features": grid["mtry"].tolist()}
        if cv and ctrl["cv"] is not None and len(grid) > 1:
            search = GridSearchCV(base, param_grid=param_grid, cv=ctrl["cv"], scoring=scoring)
            search.fit(x_df, y_fit)
            model = search.best_estimator_
            perf = pd.DataFrame(search.cv_results_)
            best_mtry = search.best_params_["max_features"]
        else:
            model = base.set_params(max_features=int(grid["mtry"].iloc[0])).fit(x_df, y_fit)
            perf = None
            best_mtry = int(grid["mtry"].iloc[0])

    pi = permutation_importance(model, x_df, y_fit, n_repeats=5, random_state=seed)
    coefs = pd.DataFrame(
        {
            "terms": input_["predictors"],
            "estimate": pi.importances_mean,
            "impurity": model.feature_importances_,
        }
    ).sort_values("estimate", ascending=False).reset_index(drop=True)

    if classify:
        oob_pred = model.predict(x_df)
        fit_stats = _rf_class_stats(oob_pred, y_fit, outcome_lv)
    else:
        oob_pred = getattr(model, "oob_prediction_", np.full(len(y_fit), np.nan))
        fit_stats = _rf_reg_stats(oob_pred, y_fit)

    design: dict[str, Any] = {
        "outcome": input_["outcome"],
        "outcome_type": "two classes" if classify else "continuous",
        "n_obs": input_["n_obs"],
        "n_used": input_["n_used"],
        "n_dropped": input_["n_dropped"],
        "predictors": input_["predictors"],
        "dropped_predictors": input_["dropped_predictors"],
        **sa_design_lv(input_["predictor_lv"]),
    }
    if classify:
        n_events = int((y == outcome_lv[1]).sum())
        design.update(
            {"outcome_lv": outcome_lv, "n_events": n_events, "event_rate": n_events / input_["n_used"]}
        )

    return sa_new_model(
        analysis="random_forest",
        terms=coefs["terms"].tolist(),
        design=design,
        parameters={
            "mtry": best_mtry,
            "ntree": ntree,
            "nodesize": nodesize,
            "n_candidates": len(grid),
            "cv": cv,
            "cv_method": ctrl["cv_method"],
            "n_fold": ctrl["n_fold"],
            "n_repeat": ctrl["n_repeat"],
            "seed": seed,
        },
        coefficients=coefs,
        fit_stats=fit_stats,
        performance=perf,
        resampling=None,
        engine={
            "package": "sklearn",
            "method": "rf",
            "label": f"Random forest {'classification' if classify else 'regression'}",
            "metrics": ["RMSE", "Accuracy"] if classify else ["RMSE", "Rsquared", "MAE"],
        },
        fit=model,
    )
