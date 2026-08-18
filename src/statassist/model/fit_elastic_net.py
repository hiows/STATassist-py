"""Elastic net / lasso / ridge regression."""

from __future__ import annotations

import warnings

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from statassist.contracts.model import sa_new_model
from statassist.utils.glmnet_r import (
    sa_glmnet_path,
    sa_post_resample,
    sa_post_resample_class,
)
from statassist.utils.model_utils import (
    sa_design_lv,
    sa_design_matrix,
    sa_enet_grid,
    sa_outcome_levels,
    sa_resample_sets,
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


def _enet_final_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lam: float,
    family: str,
) -> tuple[float, np.ndarray, Any]:
    """Refit the chosen candidate on every row, as caret's `finalModel` is."""
    if family == "gaussian":
        intercept, beta = sa_glmnet_path(x, y, alpha, np.array([lam]))
        b0 = float(intercept[0])
        b = beta[:, 0]
        return b0, b, lambda new: new @ b + b0

    fit = _logistic_glmnet(x, y, alpha, lam)
    b0, b = float(fit.intercept_[0]), fit.coef_.ravel()
    return b0, b, lambda new: fit.predict_proba(new)[:, 1]


def _logistic_glmnet(
    x: np.ndarray, y: np.ndarray, alpha: float, lam: float
) -> LogisticRegression:
    """glmnet's binomial penalty written in scikit-learn's parameters.

    glmnet minimises ``-loglik/n + lambda * penalty`` and scikit-learn minimises
    ``-loglik + penalty/C``, so ``C = 1 / (n * lambda)``.
    """
    n = len(y)
    c = 1.0 / (n * lam) if lam > 0 else 1e12
    if alpha <= 0:
        model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=10000)
    elif alpha >= 1:
        model = LogisticRegression(penalty="l1", C=c, solver="saga", max_iter=20000)
    else:
        model = LogisticRegression(
            penalty="elasticnet", C=c, l1_ratio=alpha, solver="saga", max_iter=20000
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(x, y)


def _enet_resample(
    x: np.ndarray,
    y: np.ndarray,
    grid: pd.DataFrame,
    ctrl: dict[str, Any],
    family: str,
    strata: np.ndarray,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Score every grid candidate on caret's own folds.

    One path fit per fold and per mixing parameter rather than one per grid row:
    a penalised path is solved for every lambda at once, which is how glmnet is
    handed the whole grid too.
    """
    train_sets, names = sa_resample_sets(strata, ctrl, seed)

    metric_names = (
        ["RMSE", "Rsquared", "MAE"] if family == "gaussian" else ["Accuracy", "Kappa"]
    )
    alphas = list(dict.fromkeys(grid["alpha"].tolist()))
    lambdas = np.asarray(grid["lambda"], dtype=float)
    # (candidate, fold) for every metric.
    scores = {m: np.full((len(grid), len(train_sets)), np.nan) for m in metric_names}

    all_rows = np.arange(len(y))
    for f, train in enumerate(train_sets):
        test = np.setdiff1d(all_rows, train)
        for a in alphas:
            at = np.flatnonzero(grid["alpha"].to_numpy() == a)
            path_lambdas = lambdas[at]
            if family == "gaussian":
                intercept, beta = sa_glmnet_path(x[train], y[train], a, path_lambdas)
                preds = x[test] @ beta + intercept
                for col, k in enumerate(at):
                    for m, v in sa_post_resample(preds[:, col], y[test]).items():
                        scores[m][k, f] = v
            else:
                for col, k in enumerate(at):
                    fit = _logistic_glmnet(x[train], y[train], a, path_lambdas[col])
                    pred = fit.predict(x[test])
                    for m, v in sa_post_resample_class(pred, y[test]).items():
                        scores[m][k, f] = v

    perf = grid.copy().reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for m in metric_names:
            perf[m] = np.nanmean(scores[m], axis=1)
        for m in metric_names:
            perf[f"{m}SD"] = np.nanstd(scores[m], axis=1, ddof=1)

    # caret takes the smallest RMSE for a regression and the largest accuracy
    # for a classification, and the first of them when they tie.
    if family == "gaussian":
        best_index = int(np.nanargmin(perf["RMSE"].to_numpy()))
    else:
        best_index = int(np.nanargmax(perf["Accuracy"].to_numpy()))

    resampling = pd.DataFrame(
        {m: scores[m][best_index, :] for m in metric_names} | {"Resample": names}
    )
    return perf, resampling, best_index


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

    x_mat = x.to_numpy(dtype=float)
    perf: pd.DataFrame | None = None
    resampling: pd.DataFrame | None = None

    with sa_preserve_seed(seed):
        if cv and ctrl["cv"] is not None:
            strata = np.asarray(y).astype(str) if classify else y_fit
            perf, resampling, best_index = _enet_resample(
                x_mat, y_fit, grid, ctrl, family, strata, seed
            )
        else:
            best_index = 0

    row = grid.iloc[best_index]
    best_alpha = float(row["alpha"])
    best_lambda = float(row["lambda"])
    intercept, beta, predict = _enet_final_fit(
        x_mat, y_fit, best_alpha, best_lambda, family
    )

    coef = np.concatenate([[intercept], beta])
    terms = ["(Intercept)"] + list(x.columns)
    coefs = pd.DataFrame(
        {
            "terms": terms,
            "estimate": coef,
            # The intercept is not penalized, so it is not the penalty that
            # would have set it to zero and it counts as kept whatever its
            # value.
            "selected": (coef != 0) | (np.arange(len(coef)) == 0),
        }
    )
    if classify:
        coefs["odds_ratio"] = np.exp(coefs["estimate"])

    fitted_value = predict(x_mat)
    if family == "gaussian":
        fit_stats = _enet_gaussian_stats(fitted_value, y_fit)
    else:
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
            "alpha": best_alpha,
            "lambda": best_lambda,
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
        resampling=resampling,
        engine={
            "package": "sklearn",
            "method": "glmnet",
            "family": family,
            "label": f"{penalty_label} {'binomial classification' if classify else 'linear regression'}",
            "metrics": ["Accuracy", "Kappa"] if classify else ["RMSE", "Rsquared", "MAE"],
            "x_names": list(x.columns),
        },
        fit=None,
    )
