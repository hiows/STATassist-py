"""Classification performance kernels (DeLong, IDI, NRI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sa_check_response(response: np.ndarray, predictor: np.ndarray) -> None:
    if len(response) != len(predictor):
        raise ValueError("internal error: `response` and `predictor` differ in length.")
    if not set(np.unique(response)).issubset({0, 1}):
        raise ValueError("internal error: `response` must be 0/1 with 1 for the event.")
    n_event = int((response == 1).sum())
    if n_event == 0 or n_event == len(response):
        raise ValueError("the scored rows hold a single class.")


def sa_roc_points(response: np.ndarray, predictor: np.ndarray) -> pd.DataFrame:
    sa_check_response(response, predictor)
    n_event = int((response == 1).sum())
    n_other = len(response) - n_event
    at = np.argsort(-predictor)
    sorted_pred = predictor[at]
    hit = np.cumsum(response[at])
    miss = np.cumsum(1 - response[at])
    last = np.concatenate([sorted_pred[:-1] != sorted_pred[1:], [True]])
    return pd.DataFrame(
        {
            "threshold": np.concatenate([[np.inf], sorted_pred[last]]),
            "sensitivity": np.concatenate([[0.0], hit[last] / n_event]),
            "specificity": 1 - np.concatenate([[0.0], miss[last] / n_other]),
        }
    )


def sa_placement_values(response: np.ndarray, predictor: np.ndarray) -> dict[str, np.ndarray]:
    is_event = response == 1
    x = predictor[is_event]
    y = predictor[~is_event]
    n_event = len(x)
    n_other = len(y)
    pooled = stats.rankdata(np.concatenate([x, y]))
    event = (pooled[:n_event] - stats.rankdata(x)) / n_other
    other = 1 - (pooled[n_event:] - stats.rankdata(y)) / n_event
    return {"event": event, "other": other}


def sa_auc_delong(response: np.ndarray, predictor: np.ndarray) -> dict[str, float]:
    sa_check_response(response, predictor)
    placement = sa_placement_values(response, predictor)
    n_event = len(placement["event"])
    n_other = len(placement["other"])
    auc = float(np.mean(placement["event"]))
    if n_event > 1 and n_other > 1:
        variance = np.var(placement["event"], ddof=1) / n_event + np.var(placement["other"], ddof=1) / n_other
    else:
        variance = np.nan
    return {"auc": auc, "se": float(np.sqrt(variance)) if np.isfinite(variance) else np.nan}


def sa_delong_test(
    response: np.ndarray,
    predictor_1: np.ndarray,
    predictor_2: np.ndarray,
) -> dict[str, float]:
    first = sa_placement_values(response, predictor_1)
    second = sa_placement_values(response, predictor_2)
    n_event = len(first["event"])
    n_other = len(first["other"])
    delta = float(np.mean(first["event"]) - np.mean(second["event"]))
    if n_event < 2 or n_other < 2:
        return {"delta": delta, "se": np.nan, "statistic": np.nan, "pval": np.nan}
    s_event = np.cov(np.column_stack([first["event"], second["event"]]).T)
    s_other = np.cov(np.column_stack([first["other"], second["other"]]).T)
    s = s_event / n_event + s_other / n_other
    variance = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if not np.isfinite(variance) or variance <= 0:
        return {"delta": delta, "se": 0.0, "statistic": np.nan, "pval": np.nan}
    se = float(np.sqrt(variance))
    statistic = delta / se
    pval = 2 * stats.norm.sf(abs(statistic))
    return {"delta": delta, "se": se, "statistic": statistic, "pval": pval}


def sa_idi(response: np.ndarray, predictor_old: np.ndarray, predictor_new: np.ndarray) -> dict[str, float]:
    is_event = response == 1
    moved_event = (predictor_new - predictor_old)[is_event]
    moved_other = (predictor_new - predictor_old)[~is_event]
    idi = float(moved_event.mean() - moved_other.mean())
    n_event = len(moved_event)
    n_other = len(moved_other)
    if n_event < 2 or n_other < 2:
        return {"idi": idi, "se": np.nan, "statistic": np.nan, "pval": np.nan}
    variance = np.var(moved_event, ddof=1) / n_event + np.var(moved_other, ddof=1) / n_other
    if not np.isfinite(variance) or variance <= 0:
        return {"idi": idi, "se": 0.0, "statistic": np.nan, "pval": np.nan}
    se = float(np.sqrt(variance))
    statistic = idi / se
    return {"idi": idi, "se": se, "statistic": statistic, "pval": 2 * stats.norm.sf(abs(statistic))}


def sa_nri(response: np.ndarray, predictor_old: np.ndarray, predictor_new: np.ndarray) -> dict[str, float]:
    is_event = response == 1
    moved = predictor_new - predictor_old
    moved_event = moved[is_event]
    moved_other = moved[~is_event]
    up_event = float((moved_event > 0).mean())
    down_event = float((moved_event < 0).mean())
    up_other = float((moved_other > 0).mean())
    down_other = float((moved_other < 0).mean())
    nri_event = up_event - down_event
    nri_other = down_other - up_other
    nri = nri_event + nri_other
    n_event = len(moved_event)
    n_other = len(moved_other)
    variance = (up_event + down_event - nri_event**2) / n_event + (up_other + down_other - nri_other**2) / n_other
    if not np.isfinite(variance) or variance <= 0:
        return {
            "nri": nri,
            "nri_event": nri_event,
            "nri_other": nri_other,
            "se": 0.0,
            "statistic": np.nan,
            "pval": np.nan,
        }
    se = float(np.sqrt(variance))
    statistic = nri / se
    return {
        "nri": nri,
        "nri_event": nri_event,
        "nri_other": nri_other,
        "se": se,
        "statistic": statistic,
        "pval": 2 * stats.norm.sf(abs(statistic)),
    }


def sa_brier(response: np.ndarray, predictor: np.ndarray) -> float:
    sa_check_response(response, predictor)
    return float(np.mean((predictor - response) ** 2))


def sa_threshold_scores(response: np.ndarray, predictor: np.ndarray, threshold: float) -> dict[str, float]:
    called = (predictor >= threshold).astype(float)
    is_event = response == 1
    return {
        "accuracy": float(np.mean(called == response)),
        "sensitivity": float(np.mean(called[is_event] == 1)),
        "specificity": float(np.mean(called[~is_event] == 0)),
    }


def sa_regression_scores(observed: np.ndarray, predicted: np.ndarray, var_observed: float) -> dict[str, float]:
    n = len(observed)
    residual = predicted - observed
    sse = float(np.sum(residual**2))
    sst = var_observed * (n - 1)
    if not np.isfinite(var_observed) or var_observed <= 0:
        return {
            "n_used": n,
            "cor": np.nan,
            "r_squared": np.nan,
            "rmse": float(np.sqrt(sse / n)),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "calib_slope": np.nan,
            "calib_intercept": np.nan,
        }
    var_predicted = float(np.var(predicted, ddof=1))
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if var_predicted > 0 else np.nan
    r_squared = 1 - sse / sst
    slope = float(np.cov(observed, predicted)[0, 1] / var_observed)
    intercept = float(predicted.mean() - slope * observed.mean())
    return {
        "n_used": n,
        "cor": correlation,
        "r_squared": r_squared,
        "rmse": float(np.sqrt(sse / n)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "calib_slope": slope,
        "calib_intercept": intercept,
    }
