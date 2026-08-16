"""Binomial logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from statassist.contracts.model import sa_new_model
from statassist.utils.model_utils import (
    sa_coef_table,
    sa_design_lv,
    sa_outcome_levels,
    sa_resolve_model_input,
    sa_train_control,
    sa_wald_interval,
)
from statassist.utils.validate import sa_check_scalar_num, sa_preserve_seed


class _GLMWrapper(BaseEstimator, ClassifierMixin):
    def fit(self, x, y):
        x = sm.add_constant(x, has_constant="add")
        y = np.asarray(y, dtype=float)
        self.model_ = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, x):
        x = sm.add_constant(x, has_constant="add")
        proba = self.model_.predict(x)
        return np.where(proba >= 0.5, 1, 0)

    def predict_proba(self, x):
        x = sm.add_constant(x, has_constant="add")
        p = self.model_.predict(x)
        return np.column_stack([1 - p, p])


def fit_logistic_regression(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    cv: bool = True,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    input_ = sa_resolve_model_input(data, outcome, predictors)
    y = sa_outcome_levels(input_["y"], outcome_lv, control_label)
    outcome_lv = list(y.categories)
    y_bin = np.asarray((y == outcome_lv[1]).astype(int))

    ctrl = sa_train_control(cv, cv_method, n_fold, n_repeat, input_["n_used"])
    x_df = input_["x"]

    with sa_preserve_seed(seed):
        wrapper = _GLMWrapper()
        if ctrl["cv"] is not None:
            search = GridSearchCV(
                wrapper, param_grid={}, cv=ctrl["cv"], scoring="accuracy"
            )
            search.fit(x_df, y_bin)
            fit_obj = search.best_estimator_
            perf = pd.DataFrame({"Accuracy": [search.best_score_]})
            resampling = None
        else:
            fit_obj = wrapper.fit(x_df, y_bin)
            perf = None
            resampling = None

    model = fit_obj.model_
    interval = sa_wald_interval(
        np.column_stack([model.params.to_numpy(), model.bse.to_numpy()]),
        conf_level,
        df=None,
    )
    coefs = sa_coef_table(model, interval, df=np.nan)
    coefs["odds_ratio"] = np.exp(coefs["estimate"])
    coefs["or_lower_conf"] = np.exp(coefs["lower_conf"])
    coefs["or_upper_conf"] = np.exp(coefs["upper_conf"])

    lr_stat = model.null_deviance - model.deviance
    df_null = float(getattr(model, "df_null", model.nobs - 1))
    lr_df = df_null - model.df_resid
    fit_stats = {
        "null_deviance": float(model.null_deviance),
        "residual_deviance": float(model.deviance),
        "df_null": df_null,
        "df_residual": float(model.df_resid),
        "mcfadden_r2": float(1 - model.deviance / model.null_deviance)
        if model.null_deviance > 0
        else np.nan,
        "lr_stat": float(lr_stat),
        "lr_df": float(lr_df),
        "lr_pval": float(stats.chi2.sf(lr_stat, lr_df)) if lr_df > 0 else np.nan,
        "aic": float(model.aic),
        "bic": float(model.bic),
    }

    n_events = int((y == outcome_lv[1]).sum())
    design = {
        "outcome": input_["outcome"],
        "outcome_type": "two classes",
        "outcome_lv": outcome_lv,
        "n_events": n_events,
        "event_rate": n_events / input_["n_used"],
        "n_obs": input_["n_obs"],
        "n_used": input_["n_used"],
        "n_dropped": input_["n_dropped"],
        "predictors": input_["predictors"],
        "dropped_predictors": input_["dropped_predictors"],
        **sa_design_lv(input_["predictor_lv"]),
    }

    return sa_new_model(
        analysis="logistic_regression",
        terms=coefs["terms"].tolist(),
        design=design,
        parameters={
            "cv": cv,
            "cv_method": ctrl["cv_method"],
            "n_fold": ctrl["n_fold"],
            "n_repeat": ctrl["n_repeat"],
            "conf_level": conf_level,
            "seed": seed,
        },
        coefficients=coefs,
        fit_stats=fit_stats,
        performance=perf,
        resampling=resampling,
        engine={
            "package": "sklearn",
            "method": "glm",
            "family": "binomial",
            "label": "Binomial logistic regression",
            "metrics": ["Accuracy", "Kappa"],
        },
        fit=fit_obj,
    )
