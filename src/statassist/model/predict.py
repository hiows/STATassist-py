"""Predict and coef for sa_model results."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from statassist.utils.model_utils import sa_design_matrix, sa_predict_frame


def coef(object: dict[str, Any]) -> pd.DataFrame:
    return object["coefficients"]


def predict(
    object: dict[str, Any],
    newdata: pd.DataFrame | None = None,
    type: Literal["raw", "response", "prob"] = "raw",
) -> np.ndarray | pd.Series | pd.DataFrame:
    fit = object["fit"]
    design = object["design"]
    classify = design.get("outcome_type") == "two classes"
    outcome_lv = design.get("outcome_lv")

    if newdata is None:
        if hasattr(fit, "predict"):
            out = fit.predict()
        else:
            out = fit["predict"]()
        return _shape_prediction(out, type, classify, outcome_lv)

    x = sa_predict_frame(newdata, design)
    usable = x.notna().all(axis=1).to_numpy()
    if not np.any(usable):
        raise ValueError(
            "no row of `newdata` is complete across the predictor(s) the model was "
            "fitted on, so there is nothing to predict from."
        )

    ready = x.loc[usable].copy()
    x_names = object.get("engine", {}).get("x_names")
    if x_names is not None:
        ready = sa_design_matrix(ready, xlev=design.get("predictor_lv"), want=x_names)

    if hasattr(fit, "predict_proba") and type in ("response", "prob"):
        if type == "prob" and hasattr(fit, "predict_proba"):
            proba = fit.predict_proba(ready)
            if classify and outcome_lv is not None:
                cols = list(getattr(fit, "classes_", outcome_lv))
                out_df = pd.DataFrame(proba, columns=cols)
                return _scatter(out_df, usable)
            return _scatter(pd.DataFrame(proba), usable)
        if type == "response" and classify:
            if hasattr(fit, "predict_proba"):
                proba = fit.predict_proba(ready)
                classes = list(getattr(fit, "classes_", outcome_lv or []))
                if outcome_lv and len(classes) >= 2:
                    idx = classes.index(outcome_lv[1]) if outcome_lv[1] in classes else 1
                else:
                    idx = 1
                out = proba[:, idx]
            else:
                out = fit.predict(ready)
            return _scatter(np.asarray(out, dtype=float), usable)

    if hasattr(fit, "predict"):
        out = fit.predict(ready)
    else:
        out = fit["predict"](ready, type=type)

    return _scatter(_shape_prediction(out, type, classify, outcome_lv), usable)


def _shape_prediction(
    value: Any,
    type: str,
    classify: bool,
    outcome_lv: list[str] | None,
) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    if isinstance(value, pd.Series):
        return value.reset_index(drop=True).to_numpy()
    return np.asarray(value)


def _scatter(value: Any, usable: np.ndarray) -> Any:
    if usable.all():
        return value
    n = len(usable)
    if isinstance(value, pd.DataFrame):
        full = pd.DataFrame(index=range(n), columns=value.columns, dtype=float)
        full.loc[usable] = value.to_numpy()
        return full.reset_index(drop=True)
    full = np.full(n, np.nan, dtype=float)
    arr = np.asarray(value, dtype=float)
    full[usable] = arr
    return full
