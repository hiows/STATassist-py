"""Evaluation helpers shared by evaluate_* functions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.model.predict import predict
from statassist.utils.validate import sa_resolve_row_vector


def sa_evaluate_newdata(newdata: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(newdata, np.ndarray):
        newdata = pd.DataFrame(newdata)
    if not isinstance(newdata, pd.DataFrame):
        raise ValueError("`newdata` must be a data.frame or a matrix.")
    if newdata.empty:
        raise ValueError("`newdata` has zero rows, so there is nothing to score.")
    return newdata


def sa_resolve_models(
    baseline_model: dict[str, Any],
    new_models: dict[str, Any] | None,
    baseline_label: str,
) -> dict[str, Any]:
    if baseline_model.get("__class__", ("",))[0] != "sa_model":
        raise ValueError("`baseline_model` must be a fitted sa_model result.")
    if not baseline_label:
        raise ValueError("`baseline_label` must be a single non-empty name.")
    if not new_models:
        return {baseline_label: baseline_model}
    if not isinstance(new_models, dict):
        raise ValueError("`new_models` must be a named dict of fitted models, or None.")
    labels = list(new_models.keys())
    if not labels or any(not k for k in labels):
        raise ValueError("every element of `new_models` must be named.")
    dup = sorted({k for k in labels if labels.count(k) > 1})
    if dup:
        raise ValueError(f"`new_models` contains duplicated names: {', '.join(dup)}.")
    if baseline_label in labels:
        raise ValueError(f"`new_models` holds a model called `{baseline_label}`.")
    for k, v in new_models.items():
        if v.get("__class__", ("",))[0] != "sa_model":
            raise ValueError(f"every element of `new_models` must be a fitted model. Not a model: {k}.")
    return {baseline_label: baseline_model, **new_models}


def sa_check_model_family(models: dict[str, Any], want: str, other: str) -> None:
    types = {k: m["design"]["outcome_type"] for k, m in models.items()}
    wrong = [k for k, t in types.items() if t != want]
    if wrong:
        raise ValueError(
            f"every model must have been fitted to {want} outcome. Not {want}: "
            + ", ".join(f"{k} ({types[k]})" for k in wrong)
            + f". Use {other} for those."
        )


def sa_check_model_agreement(models: dict[str, Any]) -> None:
    outcomes = {k: m["design"]["outcome"] for k, m in models.items()}
    if len(set(outcomes.values())) > 1:
        raise ValueError(
            "every model must have been fitted to the same outcome. Got "
            + ", ".join(f"{k} = {v}" for k, v in outcomes.items())
        )
    levels = {k: m["design"].get("outcome_lv") for k, m in models.items()}
    named = {k: v for k, v in levels.items() if v is not None}
    if named:
        first = next(iter(named.values()))
        disagree = [k for k, v in named.items() if v != first]
        if disagree:
            raise ValueError(
                "every model must hold the same `outcome_lv`, in the same order."
            )


def sa_resolve_answer(answer: Any, newdata: pd.DataFrame, baseline_model: dict[str, Any]) -> dict[str, Any]:
    if answer is not None:
        return sa_resolve_row_vector(answer, "answer", newdata, allow_na=True)
    label = baseline_model["design"]["outcome"]
    if label in (np.nan, "<vector>") or label not in newdata.columns:
        raise ValueError(
            "`answer` is required when the baseline model was fitted from a vector outcome."
        )
    return {"value": newdata[label].to_numpy(), "label": label}


def sa_collect_predictions(
    models: dict[str, Any],
    newdata: pd.DataFrame,
    observed: np.ndarray,
) -> dict[str, Any]:
    n_obs = len(newdata)
    columns = []
    for nm, m in models.items():
        out = predict(m, newdata=newdata, type="response")
        columns.append(np.asarray(out, dtype=float))
    predicted = np.column_stack(columns)
    model_names = list(models.keys())
    answered = ~pd.isna(observed)
    predictable = ~np.any(np.isnan(predicted), axis=1)
    keep = np.where(answered & predictable)[0]
    n_dropped = n_obs - len(keep)
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} of {n_obs} row(s) of `newdata` were left out because not every "
            "model could predict them. Every model is scored on the same rows.",
            stacklevel=2,
        )
    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} row(s) have an observed outcome and a prediction from every model."
        )
    return {
        "predicted": predicted[keep, :],
        "keep": keep + 1,
        "n_obs": n_obs,
        "n_dropped": n_dropped,
        "model_names": model_names,
    }


def sa_prediction_table(
    predicted: np.ndarray,
    keep: np.ndarray,
    observed: np.ndarray,
    model_names: list[str],
) -> pd.DataFrame:
    rows = []
    for j, model in enumerate(model_names):
        for i, row in enumerate(keep):
            rows.append(
                {
                    "model": model,
                    "row": int(row),
                    "observed": float(observed[i]),
                    "predicted": float(predicted[i, j]),
                }
            )
    return pd.DataFrame(rows)
