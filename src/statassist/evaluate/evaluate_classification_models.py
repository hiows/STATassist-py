"""Held-out classification model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.contracts.performance import sa_new_performance
from statassist.utils.evaluate_utils import (
    sa_check_model_agreement,
    sa_check_model_family,
    sa_collect_predictions,
    sa_evaluate_newdata,
    sa_prediction_table,
    sa_resolve_answer,
    sa_resolve_models,
)
from statassist.utils.performance_kernel import (
    sa_auc_delong,
    sa_brier,
    sa_delong_test,
    sa_idi,
    sa_nri,
    sa_roc_points,
    sa_threshold_scores,
)
from statassist.utils.validate import sa_check_scalar_num


def sa_response_code(answer: np.ndarray, outcome_lv: list[str]) -> np.ndarray:
    labels = np.asarray(answer, dtype=str)
    present = set(labels[~pd.isna(labels)])
    extra = present - set(outcome_lv)
    if extra:
        raise ValueError(
            f"`answer` holds class(es) the models were not fitted on: {', '.join(sorted(extra))}."
        )
    return (labels == outcome_lv[1]).astype(float)


def sa_evaluate_levels(
    model_lv: list[str],
    outcome_lv: list[str] | None,
    control_label: str | None,
) -> list[str]:
    if outcome_lv is not None:
        outcome_lv = [str(v) for v in outcome_lv]
        if len(outcome_lv) != 2 or len(set(outcome_lv)) != 2:
            raise ValueError("`outcome_lv` must be two distinct level names, the reference first.")
        if outcome_lv != model_lv:
            raise ValueError(
                f"`outcome_lv` is {', '.join(outcome_lv)} but the models were fitted with "
                f"{', '.join(model_lv)}."
            )
    if control_label is not None:
        if str(control_label) != model_lv[0]:
            raise ValueError(
                f"`control_label` is {control_label} but the models were fitted with "
                f"{model_lv[0]} as the reference."
            )
    return model_lv


def evaluate_classification_models(
    baseline_model: dict[str, Any],
    new_models: dict[str, Any] | None = None,
    newdata: pd.DataFrame | None = None,
    *,
    answer: Any = None,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    threshold: float = 0.5,
    conf_level: float = 0.95,
    baseline_label: str = "baseline",
) -> dict[str, Any]:
    if newdata is None:
        raise ValueError("`newdata` is required.")
    sa_check_scalar_num(threshold, "threshold", 0, 1)
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)

    newdata = sa_evaluate_newdata(newdata)
    models = sa_resolve_models(baseline_model, new_models, baseline_label)
    sa_check_model_family(models, "two classes", "evaluate_regression_models()")
    sa_check_model_agreement(models)

    model_lv = sa_evaluate_levels(
        baseline_model["design"]["outcome_lv"], outcome_lv, control_label
    )
    resolved = sa_resolve_answer(answer, newdata, baseline_model)
    response = sa_response_code(resolved["value"], model_lv)

    collected = sa_collect_predictions(models, newdata, response)
    response = response[collected["keep"] - 1]
    predicted = collected["predicted"]
    model_names = collected["model_names"]

    n_events = int(response.sum())
    if n_events == 0 or n_events == len(response):
        raise ValueError(
            f"the {len(response)} scored row(s) hold a single class, so there is nothing to "
            "discriminate between."
        )
    z = stats.norm.ppf(1 - (1 - conf_level) / 2)

    metric_rows = []
    for i, model in enumerate(model_names):
        p = predicted[:, i]
        area = sa_auc_delong(response, p)
        thr = sa_threshold_scores(response, p, threshold)
        metric_rows.append(
            {
                "model": model,
                "n_used": len(response),
                "n_events": n_events,
                "auc": area["auc"],
                "auc_lower_conf": area["auc"] - z * area["se"],
                "auc_upper_conf": area["auc"] + z * area["se"],
                "brier": sa_brier(response, p),
                **thr,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics["n_used"] = metrics["n_used"].astype(int)
    metrics["n_events"] = metrics["n_events"].astype(int)

    curve_parts = []
    for i, model in enumerate(model_names):
        pts = sa_roc_points(response, predicted[:, i])
        pts.insert(0, "model", model)
        curve_parts.append(pts)
    curves = pd.concat(curve_parts, ignore_index=True)

    comparisons = None
    if len(model_names) > 1:
        comp_rows = []
        for i in range(1, len(model_names)):
            new = predicted[:, i]
            old = predicted[:, 0]
            area = sa_delong_test(response, new, old)
            discrimination = sa_idi(response, old, new)
            reclassification = sa_nri(response, old, new)
            comp_rows.append(
                {
                    "model": model_names[i],
                    "delta_auc": area["delta"],
                    "delta_auc_lower_conf": area["delta"] - z * area["se"]
                    if np.isfinite(area["se"])
                    else np.nan,
                    "delta_auc_upper_conf": area["delta"] + z * area["se"]
                    if np.isfinite(area["se"])
                    else np.nan,
                    "delta_auc_pval": area["pval"],
                    "idi": discrimination["idi"],
                    "idi_lower_conf": discrimination["idi"] - z * discrimination["se"]
                    if np.isfinite(discrimination["se"])
                    else np.nan,
                    "idi_upper_conf": discrimination["idi"] + z * discrimination["se"]
                    if np.isfinite(discrimination["se"])
                    else np.nan,
                    "idi_pval": discrimination["pval"],
                    "nri": reclassification["nri"],
                    "nri_event": reclassification["nri_event"],
                    "nri_nonevent": reclassification["nri_other"],
                    "nri_lower_conf": reclassification["nri"] - z * reclassification["se"]
                    if np.isfinite(reclassification["se"])
                    else np.nan,
                    "nri_upper_conf": reclassification["nri"] + z * reclassification["se"]
                    if np.isfinite(reclassification["se"])
                    else np.nan,
                    "nri_pval": reclassification["pval"],
                }
            )
        comparisons = pd.DataFrame(comp_rows)

    return sa_new_performance(
        analysis="classification_performance",
        models=model_names,
        design={
            "outcome": resolved["label"],
            "outcome_type": "two classes",
            "outcome_lv": model_lv,
            "baseline": baseline_label,
            "n_obs": collected["n_obs"],
            "n_used": len(collected["keep"]),
            "n_dropped": collected["n_dropped"],
            "n_events": n_events,
        },
        parameters={"threshold": threshold, "conf_level": conf_level},
        predictions=sa_prediction_table(predicted, collected["keep"], response, model_names),
        metrics=metrics,
        comparisons=comparisons,
        curves=curves,
    )
