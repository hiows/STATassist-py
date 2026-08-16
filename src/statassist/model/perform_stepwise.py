"""Stepwise feature selection by AIC/BIC."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit

from statassist.contracts.selection import sa_new_selection
from statassist.utils.model_utils import sa_design_lv, sa_outcome_levels, sa_resolve_model_input


def perform_stepwise(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    model: str = "linear",
    criterion: str = "AIC",
    direction: str = "backward",
) -> dict[str, Any]:
    input_ = sa_resolve_model_input(data, outcome, predictors)
    classify = (
        model == "logistic"
        or outcome_lv is not None
        or control_label is not None
        or not np.issubdtype(input_["y"].dtype, np.number)
    )

    if model == "linear" and classify:
        raise ValueError(
            '`model = "linear"` searches for the predictors of a number, and '
            "`outcome` is a set of class labels."
        )
    if model == "logistic" and np.issubdtype(input_["y"].dtype, np.number) and len(np.unique(input_["y"])) > 2:
        raise ValueError('`model = "logistic"` classifies two classes on a multi-value numeric outcome.')

    if classify:
        y = sa_outcome_levels(input_["y"], outcome_lv, control_label, model="a stepwise selection")
        outcome_lv = list(y.categories)
        y_fit = np.asarray((y == outcome_lv[1]).astype(int))
    else:
        if not np.all(np.isfinite(input_["y"])):
            raise ValueError("`outcome` holds non-finite value(s).")
        y_fit = input_["y"].astype(float)

    k = np.log(input_["n_used"]) if criterion == "BIC" else 2.0
    x_df = sm.add_constant(input_["x"], has_constant="add")

    def fit_subset(cols: list[str]) -> float:
        if not cols:
            if model == "linear":
                m = OLS(y_fit, np.ones((len(y_fit), 1))).fit()
            else:
                m = Logit(y_fit, np.ones((len(y_fit), 1))).fit(disp=0)
        else:
            sub = sm.add_constant(input_["x"][cols], has_constant="add")
            if model == "linear":
                m = OLS(y_fit, sub).fit()
            else:
                m = Logit(y_fit, sub).fit(disp=0)
        return m.aic if criterion == "AIC" else m.bic

    current = list(input_["predictors"])
    path = [{"n_vars": len(current), "AIC": np.nan, "BIC": np.nan, "step": "Start"}]

    if direction in ("backward", "both"):
        improved = True
        while improved and current:
            improved = False
            base_score = fit_subset(current)
            best_drop, best_score = None, base_score
            for col in current:
                trial = [c for c in current if c != col]
                score = fit_subset(trial)
                if score < best_score:
                    best_score = score
                    best_drop = col
            if best_drop is not None:
                current.remove(best_drop)
                path.append(
                    {
                        "n_vars": len(current),
                        "AIC": fit_subset(current) if criterion == "AIC" else np.nan,
                        "BIC": fit_subset(current) if criterion == "BIC" else np.nan,
                        "step": f"- {best_drop}",
                    }
                )
                improved = True

    if direction in ("forward", "both") and not current:
        current = []
        remaining = list(input_["predictors"])
        improved = True
        while improved and remaining:
            improved = False
            base_score = fit_subset(current)
            best_add, best_score = None, base_score
            for col in remaining:
                trial = current + [col]
                score = fit_subset(trial)
                if score < best_score:
                    best_score = score
                    best_add = col
            if best_add is not None:
                current.append(best_add)
                remaining.remove(best_add)
                path.append(
                    {
                        "n_vars": len(current),
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "step": f"+ {best_add}",
                    }
                )
                improved = True

    if not current:
        raise ValueError(
            f"the search walked back to the intercept at charge {k:.3f} per parameter; "
            "no candidate lowers the criterion enough."
        )

    kept = current
    estimate = {}
    full_score = fit_subset(kept)
    for col in input_["predictors"]:
        if col in kept:
            trial = [c for c in kept if c != col]
            estimate[col] = fit_subset(trial) - full_score
        else:
            trial = kept + [col]
            estimate[col] = full_score - fit_subset(trial)

    candidates = sorted(input_["predictors"])
    at = sorted(candidates, key=lambda c: estimate.get(c, np.nan), reverse=True)
    ranking = pd.DataFrame(
        {
            "candidates": at,
            "estimate": [estimate[c] for c in at],
            "rank": list(range(1, len(at) + 1)),
            "selected": [c in kept for c in at],
        }
    )

    profile = pd.DataFrame(path)
    for i in range(len(profile)):
        cols = kept if i == len(profile) - 1 else list(input_["predictors"])[: profile.loc[i, "n_vars"]]
        try:
            profile.loc[i, "AIC"] = fit_subset(cols) if criterion == "AIC" else profile.loc[i, "AIC"]
            profile.loc[i, "BIC"] = fit_subset(cols) if criterion == "BIC" else profile.loc[i, "BIC"]
        except Exception:
            pass
    profile["chosen"] = False
    profile.loc[len(profile) - 1, "chosen"] = True

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
    }[model]

    return sa_new_selection(
        analysis="stepwise",
        candidates=ranking["candidates"].tolist(),
        design=design,
        parameters={
            "model": model,
            "criterion": criterion,
            "maximize": False,
            "k": k,
            "direction": direction,
        },
        selected=ranking.loc[ranking["selected"], "candidates"].tolist(),
        ranking=ranking,
        profile=profile,
        resampling=None,
        engine={
            "package": "statsmodels",
            "method": "step",
            "label": label,
            "metrics": ["AIC", "BIC"],
            "importance": f"{criterion} increase when the predictor is left out",
        },
        fit={"selected": kept, "path": profile},
    )
