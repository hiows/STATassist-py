"""Support vector machine with radial kernel."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from statassist.contracts.model import sa_new_model
from statassist.utils.model_utils import (
    sa_bind_folds,
    sa_design_lv,
    sa_design_matrix,
    sa_outcome_levels,
    sa_resolve_model_input,
    sa_svm_grid,
    sa_svm_sigma,
    sa_train_control,
)
from statassist.utils.validate import sa_check_count, sa_preserve_seed


def _svm_reg_stats(fitted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = observed - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "r_squared": 1 - sse / sst if sst > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def _svm_class_stats(fitted, observed, outcome_lv: list[str]) -> dict[str, float]:
    called = np.asarray(fitted).astype(str) == outcome_lv[1]
    positive = np.asarray(observed).astype(str) == outcome_lv[1]
    accuracy = float(np.mean(called == positive))
    expected = called.mean() * positive.mean() + (~called).mean() * (~positive).mean()
    return {
        "accuracy": accuracy,
        "error": 1 - accuracy,
        "kappa": (accuracy - expected) / (1 - expected) if expected < 1 else np.nan,
        "sensitivity": float(called[positive].mean()) if positive.any() else np.nan,
        "specificity": float((~called[~positive]).mean()) if (~positive).any() else np.nan,
    }


def fit_svm(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    C: np.ndarray | list[float] | None = None,
    sigma: np.ndarray | list[float] | None = None,
    n_permute: int = 10,
    cv: bool = True,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    n_permute = sa_check_count(n_permute, "n_permute", 1)
    if C is None:
        C = 2 ** np.arange(-5, 11, 2)
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
        y = sa_outcome_levels(input_["y"], outcome_lv, control_label, model="a support vector machine")
        outcome_lv = list(y.categories)
        y_fit = y.astype(str).to_numpy()
    else:
        if not np.all(np.isfinite(input_["y"])):
            raise ValueError("`outcome` holds non-finite value(s).")
        y_fit = input_["y"].astype(float)

    x = sa_design_matrix(input_["x"], xlev=input_["predictor_lv"])
    ctrl = sa_train_control(cv, cv_method, n_fold, n_repeat, input_["n_used"])
    sigma_arr = sa_svm_sigma(None if sigma is None else np.asarray(sigma), x.to_numpy())
    grid = sa_svm_grid(np.asarray(C), sigma_arr, cv)

    with sa_preserve_seed(seed):
        if classify:
            base = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "svm",
                        SVC(kernel="rbf", probability=True, random_state=seed),
                    ),
                ]
            )
            scoring = "accuracy"
        else:
            base = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("svm", SVR(kernel="rbf")),
                ]
            )
            scoring = "neg_root_mean_squared_error"

        param_grid = [
            {"svm__C": row.C, "svm__gamma": 1 / (2 * row.sigma**2)}
            for _, row in grid.iterrows()
        ]
        folded = sa_bind_folds(
            ctrl, np.asarray(y).astype(str) if classify else y_fit, seed
        )
        if cv and folded["cv"] is not None and len(grid) > 1:
            search = GridSearchCV(base, param_grid=param_grid, cv=folded["cv"], scoring=scoring)
            search.fit(x, y_fit)
            model = search.best_estimator_
            perf = pd.DataFrame(search.cv_results_)
            best = search.best_params_
        else:
            row = grid.iloc[0]
            model = base.set_params(
                svm__C=row["C"], svm__gamma=1 / (2 * row["sigma"] ** 2)
            ).fit(x, y_fit)
            perf = None
            best = {"svm__C": row["C"], "svm__gamma": 1 / (2 * row["sigma"] ** 2)}

    pi = permutation_importance(model, x, y_fit, n_repeats=n_permute, random_state=seed)
    coefs = pd.DataFrame(
        {"terms": list(x.columns), "estimate": pi.importances_mean}
    ).sort_values("estimate", ascending=False).reset_index(drop=True)

    fitted_value = model.predict(x)
    if classify:
        fit_stats = _svm_class_stats(fitted_value, y_fit, outcome_lv)
    else:
        fit_stats = _svm_reg_stats(fitted_value, y_fit)

    n_sv = int(getattr(model.named_steps["svm"], "n_support_", [0]).sum())
    fit_stats = {
        **fit_stats,
        "n_support_vector": n_sv,
        "support_vector_rate": n_sv / input_["n_used"],
    }

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

    sigma_best = np.sqrt(1 / (2 * best["svm__gamma"])) if best["svm__gamma"] > 0 else np.nan

    return sa_new_model(
        analysis="svm",
        terms=coefs["terms"].tolist(),
        design=design,
        parameters={
            "kernel": "radial",
            "C": best["svm__C"],
            "sigma": sigma_best,
            "n_candidates": len(grid),
            "n_permute": n_permute,
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
            "method": "svmRadial",
            "kernel": "radial",
            "label": f"Support vector machine {'classification' if classify else 'regression'} (radial basis kernel)",
            "metrics": ["RMSE", "Accuracy"] if classify else ["RMSE", "Rsquared", "MAE"],
            "x_names": list(x.columns),
        },
        fit=model,
    )
