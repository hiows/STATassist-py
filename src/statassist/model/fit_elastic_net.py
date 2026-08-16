"""Elastic net / lasso / ridge regression."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from statassist.contracts.model import sa_new_model
from statassist.utils.model_utils import (
    sa_design_lv,
    sa_design_matrix,
    sa_enet_grid,
    sa_outcome_levels,
    sa_resolve_model_input,
    sa_train_control,
)
from statassist.utils.validate import sa_preserve_seed


def _enet_gaussian_stats(fitted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = observed - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "r_squared": 1 - sse / sst if sst > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def _enet_binomial_stats(fitted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    eps = np.finfo(float).eps
    p = np.clip(fitted, eps, 1 - eps)
    event = observed.astype(float)
    deviance = -2 * np.sum(event * np.log(p) + (1 - event) * np.log(1 - p))
    rate = event.mean()
    null_deviance = (
        -2 * len(event) * (rate * np.log(rate) + (1 - rate) * np.log(1 - rate))
        if 0 < rate < 1
        else np.nan
    )
    return {
        "null_deviance": null_deviance,
        "residual_deviance": deviance,
        "mcfadden_r2": 1 - deviance / null_deviance
        if null_deviance and not np.isnan(null_deviance)
        else np.nan,
    }


def fit_elastic_net(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    penalty: str = "elastic_net",
    alpha: np.ndarray | list[float] | None = None,
    lambda_: np.ndarray | list[float] | None = None,
    cv: bool = True,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    if alpha is None:
        alpha = np.linspace(0, 1, 11)
    if lambda_ is None:
        lambda_ = 10 ** np.linspace(-4, 1, 50)

    input_ = sa_resolve_model_input(data, outcome, predictors)
    grid = sa_enet_grid(penalty, np.asarray(alpha), np.asarray(lambda_), cv)
    ctrl = sa_train_control(cv, cv_method, n_fold, n_repeat, input_["n_used"])
    x = sa_design_matrix(input_["x"], xlev=input_["predictor_lv"])
    if x.shape[1] < 2:
        raise ValueError(
            f"a penalty divides its budget between terms, but the model has {x.shape[1]} term(s)."
        )

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
        y = sa_outcome_levels(input_["y"], outcome_lv, control_label, model="an elastic net")
        outcome_lv = list(y.categories)
        y_fit = np.asarray((y == outcome_lv[1]).astype(int))
        family = "binomial"
    else:
        if not np.all(np.isfinite(input_["y"])):
            raise ValueError("`outcome` holds non-finite value(s).")
        y_fit = input_["y"].astype(float)
        family = "gaussian"

    param_grid = [{"alpha": row.alpha, "l1_ratio": row.alpha if penalty == "elastic_net" else (1 if penalty == "lasso" else 0)} for _, row in grid.iterrows()]

    with sa_preserve_seed(seed):
        if family == "gaussian":
            base = ElasticNet(max_iter=10000)
            scoring = "neg_root_mean_squared_error"
        else:
            base = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000)
            scoring = "accuracy"

        if cv and ctrl["cv"] is not None:
            search = GridSearchCV(base, param_grid=param_grid, cv=ctrl["cv"], scoring=scoring)
            search.fit(x.to_numpy(), y_fit)
            fit_obj = search.best_estimator_
            perf = pd.DataFrame(search.cv_results_)
            best = search.best_params_
        else:
            row = grid.iloc[0]
            if family == "gaussian":
                fit_obj = ElasticNet(alpha=row["lambda"], l1_ratio=row["alpha"], max_iter=10000).fit(
                    x, y_fit
                )
            else:
                fit_obj = LogisticRegression(
                    penalty="elasticnet",
                    C=1 / row["lambda"] if row["lambda"] > 0 else 1e6,
                    l1_ratio=row["alpha"],
                    solver="saga",
                    max_iter=10000,
                ).fit(x, y_fit)
            perf = None
            best = {"alpha": row["alpha"], "lambda": row["lambda"]}

    if hasattr(fit_obj, "coef_"):
        coef = np.concatenate([np.ravel(fit_obj.intercept_), np.ravel(fit_obj.coef_)])
        terms = ["(Intercept)"] + list(x.columns)
    else:
        coef = fit_obj.coef_.ravel()
        terms = list(x.columns)

    coefs = pd.DataFrame(
        {
            "terms": terms,
            "estimate": coef,
            "selected": coef != 0,
        }
    )
    coefs.loc[coefs["terms"] == "(Intercept)", "selected"] = True
    if classify:
        coefs["odds_ratio"] = np.exp(coefs["estimate"])

    if family == "gaussian":
        fitted_value = fit_obj.predict(x)
        fit_stats = _enet_gaussian_stats(fitted_value, y_fit)
    else:
        fitted_value = fit_obj.predict_proba(x)[:, 1]
        fit_stats = _enet_binomial_stats(fitted_value, y_fit)

    penalised = coefs[coefs["terms"] != "(Intercept)"]
    n_selected = int(penalised["selected"].sum())
    fit_stats = {**fit_stats, "n_selected": n_selected, "n_zero": len(penalised) - n_selected}

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
            {
                "outcome_lv": outcome_lv,
                "n_events": n_events,
                "event_rate": n_events / input_["n_used"],
            }
        )

    penalty_label = {"lasso": "Lasso (L1 penalty)", "ridge": "Ridge (L2 penalty)", "elastic_net": "Elastic net (L1 and L2 penalties)"}[penalty]

    return sa_new_model(
        analysis="elastic_net",
        terms=coefs["terms"].tolist(),
        design=design,
        parameters={
            "penalty": penalty,
            "alpha": best.get("alpha", best.get("l1_ratio")),
            "lambda": best.get("lambda", np.nan),
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
            "method": "glmnet",
            "family": family,
            "label": f"{penalty_label} {'binomial classification' if classify else 'linear regression'}",
            "metrics": ["RMSE", "Accuracy"] if classify else ["RMSE", "Rsquared", "MAE"],
            "x_names": list(x.columns),
        },
        fit=fit_obj,
    )
