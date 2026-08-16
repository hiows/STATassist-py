"""Recursive feature elimination."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression

from statassist.contracts.selection import sa_new_selection
from statassist.utils.model_utils import (
    sa_design_lv,
    sa_outcome_levels,
    sa_resolve_model_input,
    sa_train_control,
)
from statassist.utils.validate import sa_check_count, sa_preserve_seed


def _rfe_sizes(subset_sizes: list[int] | None, p: int) -> list[int]:
    if subset_sizes is None:
        ladder = list(range(1, min(11, p + 1))) + [15, 20, 30, 50, 100]
        return sorted(set(min(p, s) for s in ladder))
    sizes = [int(s) for s in subset_sizes]
    return sorted(set(sizes))


def _rfe_metric(metric: str | None, classify: bool) -> tuple[str, bool]:
    available = ["Accuracy", "Kappa"] if classify else ["RMSE", "Rsquared", "MAE"]
    if metric is None:
        metric = available[0]
    if metric not in available:
        raise ValueError(
            f"`metric` must be one of {', '.join(available)} for a "
            f"{'classification' if classify else 'regression'}."
        )
    maximize = metric not in ("RMSE", "MAE")
    return metric, maximize


def perform_rfe(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    model: str = "linear",
    subset_sizes: list[int] | None = None,
    metric: str | None = None,
    ntree: int = 500,
    nodesize: int | None = None,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    ntree = sa_check_count(ntree, "ntree", 1)
    input_ = sa_resolve_model_input(data, outcome, predictors)
    classify = (
        model == "logistic"
        or outcome_lv is not None
        or control_label is not None
        or not np.issubdtype(input_["y"].dtype, np.number)
    )

    if model == "linear" and classify:
        raise ValueError(
            '`model = "linear"` ranks by the coefficients of a straight line through a '
            "number, and `outcome` is a set of class labels."
        )
    if model == "logistic" and np.issubdtype(input_["y"].dtype, np.number) and len(np.unique(input_["y"])) > 2:
        raise ValueError('`model = "logistic"` classifies two classes on a multi-value numeric outcome.')

    if classify:
        y = sa_outcome_levels(input_["y"], outcome_lv, control_label, model="a recursive feature elimination")
        outcome_lv = list(y.categories)
        y_fit = y.astype(str).to_numpy() if model == "rf" else np.asarray((y == outcome_lv[1]).astype(int))
    else:
        if not np.all(np.isfinite(input_["y"])):
            raise ValueError("`outcome` holds non-finite value(s).")
        y_fit = input_["y"].astype(float)

    if nodesize is None:
        nodesize = 1 if classify else 5
    nodesize = sa_check_count(nodesize, "nodesize", 1)

    sizes = _rfe_sizes(subset_sizes, len(input_["predictors"]))
    scoring, maximize = _rfe_metric(metric, classify)
    sk_metric = {
        "RMSE": "neg_root_mean_squared_error",
        "Rsquared": "r2",
        "MAE": "neg_mean_absolute_error",
        "Accuracy": "accuracy",
        "Kappa": "accuracy",
    }[scoring]

    if model == "linear":
        est = LinearRegression()
        importance = "absolute t statistic"
    elif model == "logistic":
        est = LogisticRegression(max_iter=1000)
        importance = "absolute Wald z"
    else:
        est = (
            RandomForestClassifier(n_estimators=ntree, min_samples_leaf=nodesize, random_state=seed)
            if classify
            else RandomForestRegressor(n_estimators=ntree, min_samples_leaf=nodesize, random_state=seed)
        )
        importance = "permutation importance"

    ctrl = sa_train_control(True, cv_method, n_fold, n_repeat, input_["n_used"])
    min_features = min(sizes)
    with sa_preserve_seed(seed):
        selector = RFECV(
            est,
            step=1,
            min_features_to_select=min_features,
            cv=ctrl["cv"],
            scoring=sk_metric,
            n_jobs=1,
        )
        selector.fit(input_["x"], y_fit)

    support = selector.support_
    selected = [p for p, s in zip(input_["predictors"], support) if s]
    ranking_vals = selector.ranking_
    avg_rank = {p: ranking_vals[i] for i, p in enumerate(input_["predictors"])}

    candidates = sorted(input_["predictors"], key=lambda c: (avg_rank.get(c, 999), c))
    estimate = [1.0 / avg_rank.get(c, np.nan) for c in candidates]
    ranking = pd.DataFrame(
        {
            "candidates": candidates,
            "estimate": estimate,
            "rank": list(range(1, len(candidates) + 1)),
            "selected": [c in selected for c in candidates],
        }
    )

    profile = pd.DataFrame(
        {
            "n_vars": list(range(1, len(input_["predictors"]) + 1)),
            scoring: [np.nan] * len(input_["predictors"]),
            f"{scoring}SD": [np.nan] * len(input_["predictors"]),
            "chosen": [False] * len(input_["predictors"]),
        }
    )
    chosen_idx = len(selected) - 1 if selected else 0
    if 0 <= chosen_idx < len(profile):
        profile.loc[chosen_idx, "chosen"] = True
        if hasattr(selector, "cv_results_"):
            profile.loc[chosen_idx, scoring] = selector.cv_results_["mean_test_score"].max()

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
        design.update({"outcome_lv": outcome_lv, "n_events": n_events, "event_rate": n_events / input_["n_used"]})

    label = {
        "linear": "Linear regression",
        "logistic": "Binomial logistic regression",
        "rf": f"Random forest {'classification' if classify else 'regression'}",
    }[model]

    return sa_new_selection(
        analysis="rfe",
        candidates=ranking["candidates"].tolist(),
        design=design,
        parameters={
            "model": model,
            "metric": scoring,
            "maximize": maximize,
            **({"ntree": ntree, "nodesize": nodesize} if model == "rf" else {}),
            "cv_method": ctrl["cv_method"],
            "n_fold": ctrl["n_fold"],
            "n_repeat": ctrl["n_repeat"],
            "seed": seed,
        },
        selected=selected,
        ranking=ranking,
        profile=profile,
        resampling=None,
        engine={
            "package": "sklearn",
            "method": "rfe",
            "label": label,
            "metrics": ["Accuracy", "Kappa"] if classify else ["RMSE", "Rsquared", "MAE"],
            "importance": importance,
        },
        fit=selector,
    )
