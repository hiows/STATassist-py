"""Ordinary least squares linear regression."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from statassist.contracts.model import sa_new_model
from statassist.utils.model_utils import (
    sa_coef_table,
    sa_design_lv,
    sa_resolve_model_input,
    sa_train_control,
    sa_wald_interval,
)
from statassist.utils.validate import sa_check_scalar_num, sa_preserve_seed


class _LMWrapper(BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        x = sm.add_constant(x, has_constant="add")
        self.model_ = sm.OLS(y, x).fit()
        return self

    def predict(self, x):
        x = sm.add_constant(x, has_constant="add")
        return self.model_.predict(x)


def fit_linear_regression(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    cv: bool = True,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    input_ = sa_resolve_model_input(data, outcome, predictors)
    y = input_["y"]
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            f"`outcome` must be a numeric column for a linear regression, but is {y.dtype}. "
            "Use fit_logistic_regression() for an outcome with two classes."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError(
            "`outcome` holds non-finite value(s), which least squares has no residual for."
        )

    ctrl = sa_train_control(cv, cv_method, n_fold, n_repeat, input_["n_used"])
    x_df = input_["x"]

    with sa_preserve_seed(seed):
        wrapper = _LMWrapper()
        if ctrl["cv"] is not None:
            search = GridSearchCV(wrapper, param_grid={}, cv=ctrl["cv"], scoring="neg_root_mean_squared_error")
            search.fit(x_df, y)
            fit_obj = search.best_estimator_
            perf = pd.DataFrame(
                {
                    "RMSE": [-search.best_score_],
                    "RMSESD": [np.std(-search.cv_results_["mean_test_score"], ddof=1)],
                    "Rsquared": [np.nan],
                    "MAE": [np.nan],
                }
            )
            resampling = None
        else:
            fit_obj = wrapper.fit(x_df, y)
            perf = None
            resampling = None

    model = fit_obj.model_
    summ = model.summary2()
    interval = sa_wald_interval(
        np.column_stack([model.params.to_numpy(), model.bse.to_numpy()]),
        conf_level,
        df=model.df_resid,
    )
    coefs = sa_coef_table(model, interval, df=model.df_resid)

    f_stat = model.fvalue
    fit_stats = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "sigma": float(np.sqrt(model.mse_resid)),
        "f_stat": float(f_stat) if f_stat is not None else np.nan,
        "df1": float(model.df_model),
        "df2": float(model.df_resid),
        "pval": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "aic": float(model.aic),
        "bic": float(model.bic),
    }

    design = {
        "outcome": input_["outcome"],
        "outcome_type": "continuous",
        "n_obs": input_["n_obs"],
        "n_used": input_["n_used"],
        "n_dropped": input_["n_dropped"],
        "predictors": input_["predictors"],
        "dropped_predictors": input_["dropped_predictors"],
        **sa_design_lv(input_["predictor_lv"]),
    }

    return sa_new_model(
        analysis="linear_regression",
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
            "method": "lm",
            "label": "Ordinary least squares linear regression",
            "metrics": ["RMSE", "Rsquared", "MAE"],
        },
        fit=fit_obj,
    )
