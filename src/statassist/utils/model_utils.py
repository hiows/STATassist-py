"""Model fitting helpers shared across fit_* functions."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    cross_validate,
)

from statassist.utils.validate import (
    sa_check_count,
    sa_check_flag,
    sa_check_num_vector,
    sa_check_scalar_num,
    sa_preserve_seed,
    sa_resolve_row_vector,
)


def sa_design_lv(predictor_lv: dict[str, list[str]]) -> dict[str, Any]:
    if not predictor_lv:
        return {}
    return {"predictor_lv": predictor_lv}


def sa_resolve_model_input(
    data: pd.DataFrame | np.ndarray,
    outcome: Any,
    predictors: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data.frame or a matrix.")
    n_obs = len(data)
    if n_obs == 0:
        raise ValueError("`data` has zero rows.")

    resolved = sa_resolve_row_vector(outcome, "outcome", data, allow_na=True)
    if resolved is None or resolved["value"] is None:
        raise ValueError(
            "`outcome` must name a column of `data` or hold one entry per row of it."
        )
    y = np.asarray(resolved["value"])
    outcome_label = resolved["label"]

    if predictors is None:
        predictors = [c for c in data.columns if c != outcome_label]
    if not predictors:
        raise ValueError(
            "`predictors` must be a non-empty list of column names, or NULL for "
            "every column except `outcome`."
        )
    dup = sorted({p for p in predictors if predictors.count(p) > 1})
    if dup:
        raise ValueError(f"`predictors` contains duplicated names: {', '.join(dup)}.")
    unknown = set(predictors) - set(data.columns)
    if unknown:
        raise ValueError(f"`predictors` not found in `data`: {', '.join(sorted(unknown))}.")
    if outcome_label in predictors:
        raise ValueError(
            f"`predictors` contains the outcome column `{outcome_label}`, which would "
            "let the model predict from the answer."
        )

    x = data[predictors].copy()
    unsupported = [
        c
        for c in predictors
        if not (np.issubdtype(x[c].dtype, np.number) or pd.api.types.is_bool_dtype(x[c]))
        and not pd.api.types.is_categorical_dtype(x[c])
        and not pd.api.types.is_object_dtype(x[c])
    ]
    if unsupported:
        raise ValueError(
            "`predictors` must be numeric, logical, factor or character columns. "
            f"Not usable: {', '.join(unsupported)}."
        )

    keep = x.notna().all(axis=1).to_numpy() & ~pd.isna(y)
    n_used = int(keep.sum())
    if n_used < 2:
        raise ValueError(
            f"only {n_used} row(s) of `data` are complete across `outcome` and "
            "`predictors`; at least 2 are needed."
        )
    x = x.loc[keep].reset_index(drop=True)
    y = y[keep]

    predictor_lv: dict[str, list[str]] = {}
    for nm in predictors:
        if pd.api.types.is_object_dtype(x[nm]) or pd.api.types.is_categorical_dtype(x[nm]):
            x[nm] = pd.Categorical(x[nm])
            predictor_lv[nm] = x[nm].cat.categories.astype(str).tolist()
        elif pd.api.types.is_bool_dtype(x[nm]):
            x[nm] = x[nm].astype(str)
            predictor_lv[nm] = sorted(x[nm].unique().tolist())

    constant = [c for c in predictors if x[c].nunique(dropna=True) < 2]
    if constant:
        warnings.warn(
            "predictor(s) with a single value cannot contribute and were left out: "
            f"{', '.join(constant)}.",
            stacklevel=2,
        )
        x = x.drop(columns=constant)
        predictors = [p for p in predictors if p not in constant]
        for c in constant:
            predictor_lv.pop(c, None)
    if not predictors:
        raise ValueError(
            "every predictor takes a single value over the usable rows, so there is "
            "nothing to fit."
        )

    return {
        "x": x,
        "y": y,
        "outcome": outcome_label,
        "predictors": predictors,
        "dropped_predictors": constant,
        "predictor_lv": predictor_lv,
        "n_obs": n_obs,
        "n_used": n_used,
        "n_dropped": n_obs - n_used,
    }


def sa_outcome_levels(
    y: np.ndarray,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    *,
    model: str = "a logistic regression",
) -> pd.Categorical:
    y = pd.Series(y)
    if isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_categorical_dtype(y):
        y = y.astype(str)
    elif pd.api.types.is_bool_dtype(y):
        y = y.astype(str)
    elif pd.api.types.is_numeric_dtype(y):
        y = y.astype(str)
    else:
        y = y.astype(str)

    present = sorted(y.dropna().unique().tolist())
    if len(present) < 2:
        raise ValueError(
            "`outcome` takes a single value over the usable rows, so there is nothing to classify."
        )

    if outcome_lv is None:
        if len(present) > 2:
            raise ValueError(
                f"`outcome` holds {len(present)} classes, but {model} models two: "
                f"{', '.join(present)}. Name the two to model with `outcome_lv`."
            )
        outcome_lv = sorted(present)
    else:
        outcome_lv = [str(v) for v in outcome_lv]
        if len(outcome_lv) != 2 or len(set(outcome_lv)) != 2:
            raise ValueError("`outcome_lv` must be two distinct level names, the reference first.")
        absent = set(outcome_lv) - set(present)
        if absent:
            raise ValueError(
                f"`outcome_lv` level(s) absent from `outcome`: {', '.join(sorted(absent))}."
            )
        extra = set(present) - set(outcome_lv)
        if extra:
            raise ValueError(
                f"`outcome` holds {len(present)} classes and `outcome_lv` names two of them, "
                f"so {len(extra)} would be silently left out: {', '.join(sorted(extra))}."
            )

    if control_label is not None:
        control_label = str(control_label)
        if control_label not in outcome_lv:
            raise ValueError(
                f"`control_label` names a class `outcome` does not hold: {control_label}."
            )
        if outcome_lv[0] != control_label:
            if outcome_lv is not None and outcome_lv[0] != control_label:
                if control_label != outcome_lv[0]:
                    other = [lv for lv in outcome_lv if lv != control_label][0]
                    outcome_lv = [control_label, other]

    return pd.Categorical(y, categories=outcome_lv, ordered=True)


def sa_design_matrix(
    x: pd.DataFrame,
    xlev: dict[str, list[str]] | None = None,
    want: list[str] | None = None,
) -> pd.DataFrame:
    formula = "~ " + " + ".join(f"Q('{c}')" for c in x.columns)
    if xlev:
        for col, levels in xlev.items():
            if col in x.columns:
                x = x.copy()
                x[col] = pd.Categorical(x[col].astype(str), categories=levels)
    mat = dmatrix(formula, data=x, return_type="dataframe")
    mat = mat.drop(columns=["Intercept"], errors="ignore")
    if want is not None:
        absent = set(want) - set(mat.columns)
        if absent:
            raise ValueError(
                "internal error: the coding of `newdata` is missing term(s) the model has: "
                f"{', '.join(absent)}."
            )
        mat = mat[list(want)]
    return mat.astype(float)


def sa_predict_frame(newdata: pd.DataFrame | np.ndarray, design: dict[str, Any]) -> pd.DataFrame:
    if isinstance(newdata, np.ndarray):
        newdata = pd.DataFrame(newdata)
    if not isinstance(newdata, pd.DataFrame):
        raise ValueError("`newdata` must be a data.frame or a matrix.")
    if newdata.empty:
        raise ValueError("`newdata` has zero rows.")

    predictors = design["predictors"]
    absent = set(predictors) - set(newdata.columns)
    if absent:
        raise ValueError(
            "`newdata` is missing predictor column(s) the model was fitted on: "
            f"{', '.join(sorted(absent))}."
        )

    x = newdata[predictors].copy()
    predictor_lv = design.get("predictor_lv", {})
    for nm in predictors:
        lv = predictor_lv.get(nm)
        if lv is None:
            if not np.issubdtype(x[nm].dtype, np.number) and not pd.api.types.is_bool_dtype(x[nm]):
                raise ValueError(
                    f"`{nm}` was a numeric predictor when the model was fitted, and "
                    f"`newdata` holds it as {x[nm].dtype}."
                )
            continue
        v = x[nm]
        if pd.api.types.is_categorical_dtype(v) or pd.api.types.is_bool_dtype(v) or np.issubdtype(
            v.dtype, np.number
        ):
            v = v.astype(str)
        unseen = set(v.dropna().unique()) - set(lv)
        if unseen:
            raise ValueError(
                f"`newdata` holds level(s) of `{nm}` the model was not fitted on: "
                f"{', '.join(sorted(unseen))}. Fitted on: {', '.join(lv)}."
            )
        x[nm] = pd.Categorical(v.astype(str), categories=lv)
    return x.reset_index(drop=True)


def sa_wald_interval(
    coef_matrix: np.ndarray,
    conf_level: float,
    df: float | None = None,
) -> np.ndarray:
    from scipy import stats

    estimate = coef_matrix[:, 0]
    stderr = coef_matrix[:, 1]
    alpha = (1 - conf_level) / 2
    if df is None or not np.isfinite(df):
        crit = stats.norm.ppf(1 - alpha)
    else:
        crit = stats.t.ppf(1 - alpha, df)
    return np.column_stack([estimate - crit * stderr, estimate + crit * stderr])


def sa_coef_table(
    model: Any,
    interval: np.ndarray,
    df: float,
) -> pd.DataFrame:
    estimate = model.params
    summ = model.summary2().tables[1]
    terms = list(estimate.index)
    at = [terms.index(t) if t in terms else None for t in terms]
    stderr = summ["Std.Err."].to_numpy()
    statistic = summ["t"].to_numpy() if "t" in summ.columns else summ["z"].to_numpy()
    pval = summ["P>|t|"].to_numpy() if "P>|t|" in summ.columns else summ["P>|z|"].to_numpy()
    limits = interval
    return pd.DataFrame(
        {
            "terms": terms,
            "estimate": estimate.to_numpy(),
            "stderr": stderr,
            "statistic": statistic,
            "df": df,
            "pval": pval,
            "lower_conf": limits[:, 0],
            "upper_conf": limits[:, 1],
        }
    )


def sa_resample_scheme(
    cv_method: str,
    n_fold: int,
    n_repeat: int,
    n_obs: int,
) -> dict[str, Any]:
    n_fold = sa_check_count(n_fold, "n_fold", 2)
    n_repeat = sa_check_count(n_repeat, "n_repeat", 1)
    if cv_method != "loocv" and n_fold > n_obs:
        raise ValueError(
            f"`n_fold` = {n_fold} exceeds the {n_obs} usable observation(s), so a fold "
            "would be empty."
        )
    if cv_method == "repeated_kfold":
        return {
            "cv": RepeatedKFold(n_splits=n_fold, n_repeats=n_repeat, random_state=None),
            "cv_method": cv_method,
            "n_fold": n_fold,
            "n_repeat": n_repeat,
        }
    if cv_method == "kfold":
        return {
            "cv": KFold(n_splits=n_fold, shuffle=True),
            "cv_method": cv_method,
            "n_fold": n_fold,
            "n_repeat": None,
        }
    if cv_method == "loocv":
        return {
            "cv": LeaveOneOut(),
            "cv_method": cv_method,
            "n_fold": None,
            "n_repeat": None,
        }
    raise ValueError(f"internal error: unhandled `cv_method` {cv_method}.")


def sa_train_control(
    cv: bool,
    cv_method: str,
    n_fold: int,
    n_repeat: int,
    n_obs: int,
) -> dict[str, Any]:
    sa_check_flag(cv, "cv")
    if not cv:
        return {"cv": None, "cv_method": None, "n_fold": None, "n_repeat": None}
    scheme = sa_resample_scheme(cv_method, n_fold, n_repeat, n_obs)
    return scheme


def run_cv_scores(
    estimator: Any,
    x: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    cv: Any,
    *,
    scoring: dict[str, str],
    seed: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if cv is None:
        return None, None
    with sa_preserve_seed(seed):
        scores = cross_validate(
            estimator,
            x,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )
    metrics = {k.replace("test_", ""): [np.mean(scores[k])] for k in scores if k.startswith("test_")}
    for k in list(metrics.keys()):
        sd_key = f"{k}SD"
        raw = scores.get(f"test_{k}")
        if raw is not None:
            metrics[sd_key] = [np.std(raw, ddof=1)]
    perf = pd.DataFrame(metrics)
    resample_rows = []
    for i in range(len(scores["test_score"] if "test_score" in scores else scores[list(scores)[0]])):
        row = {"Resample": i + 1}
        for k, v in scores.items():
            if k.startswith("test_"):
                row[k.replace("test_", "")] = v[i]
        resample_rows.append(row)
    resampling = pd.DataFrame(resample_rows) if resample_rows else None
    return perf, resampling


def sa_enet_grid(
    penalty: str,
    alpha: np.ndarray,
    lambdas: np.ndarray,
    cv: bool,
) -> pd.DataFrame:
    sa_check_num_vector(alpha, "alpha", 0, 1)
    sa_check_num_vector(lambdas, "lambda", 0)
    if penalty == "lasso":
        alpha_vals = [1.0]
    elif penalty == "ridge":
        alpha_vals = [0.0]
    elif penalty == "elastic_net":
        alpha_vals = sorted(set(float(a) for a in alpha))
    else:
        raise ValueError(f"internal error: unhandled `penalty` {penalty}.")
    rows = []
    for a in alpha_vals:
        for lam in sorted(set(float(l) for l in lambdas)):
            rows.append({"alpha": a, "lambda": lam})
    grid = pd.DataFrame(rows)
    if not cv and len(grid) > 1:
        raise ValueError(
            "`cv = FALSE` fits one model, so the grid must hold one candidate, "
            f"but `alpha` and `lambda` give {len(grid)}."
        )
    return grid


def sa_rf_grid(
    mtry: np.ndarray | None,
    p: int,
    classify: bool,
    cv: bool,
) -> pd.DataFrame:
    if mtry is None:
        val = max(1, int(np.floor(np.sqrt(p) if classify else p / 3)))
        mtry = np.array([val])
    else:
        mtry = sa_check_num_vector(mtry, "mtry", 1)
        if np.any(mtry != np.trunc(mtry)):
            raise ValueError("`mtry` counts predictors, so it must hold whole numbers.")
        if np.any(mtry > p):
            raise ValueError(f"`mtry` cannot exceed the {p} predictor(s) the model has.")
    grid = pd.DataFrame({"mtry": sorted(set(int(v) for v in mtry))})
    if not cv and len(grid) > 1:
        raise ValueError(
            f"`cv = FALSE` fits one forest, so the grid must hold one candidate, "
            f"but `mtry` gives {len(grid)}."
        )
    return grid


def sa_svm_grid(c: np.ndarray, sigma: np.ndarray, cv: bool) -> pd.DataFrame:
    sa_check_num_vector(c, "C", 0)
    sa_check_num_vector(sigma, "sigma", 0)
    if np.any(c == 0) or np.any(sigma == 0):
        raise ValueError("`C` and `sigma` must be above 0.")
    grid = pd.DataFrame(
        {"C": sorted(set(float(v) for v in c)), "sigma": sorted(set(float(v) for v in sigma))}
    ).explode(["C", "sigma"]).drop_duplicates()
    grid = pd.DataFrame(
        [{"C": ci, "sigma": si} for ci in sorted(set(c)) for si in sorted(set(sigma))]
    )
    if not cv and len(grid) > 1:
        raise ValueError(
            f"`cv = FALSE` fits one machine, so the grid must hold one candidate, "
            f"but `C` and `sigma` give {len(grid)}."
        )
    return grid


def sa_svm_sigma(sigma: np.ndarray | None, x: np.ndarray) -> np.ndarray:
    if sigma is not None:
        return np.asarray(sigma, dtype=float)
    from sklearn.metrics import pairwise_distances

    d = pairwise_distances(x, metric="euclidean")
    d = d[np.triu_indices_from(d, k=1)]
    if d.size == 0:
        return np.array([1.0])
    q10, q50, q90 = np.quantile(d, [0.1, 0.5, 0.9])
    gamma = 1.0 / (2 * q50**2) if q50 > 0 else 1.0
    return np.array([q50 if q50 > 0 else 1.0])
