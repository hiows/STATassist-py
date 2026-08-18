"""Model fitting helpers shared across fit_* functions."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold

from statassist.utils.caret_resample import RFoldSplitter, sa_caret_resample_index
from statassist.utils.glmnet_r import sa_post_resample, sa_post_resample_class
from statassist.utils.validate import (
    sa_check_count,
    sa_check_flag,
    sa_check_num_vector,
    sa_check_scalar_num,
    sa_preserve_seed,
    sa_resolve_row_vector,
)


def sa_forest_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Predictors in the form a scikit-learn forest will accept.

    `randomForest()` splits a factor on a subset of its levels, and no
    scikit-learn tree does; a factor arrives here as its level codes, which is
    the closest a numeric splitter gets. Ordering levels that have no order is
    what that costs, and is why a forest's importances are not compared against
    R's number for number.
    """
    out = x.copy()
    for nm in out.columns:
        if isinstance(out[nm].dtype, pd.CategoricalDtype):
            out[nm] = out[nm].cat.codes.astype(float)
        elif pd.api.types.is_object_dtype(out[nm]):
            out[nm] = pd.Categorical(out[nm]).codes.astype(float)
    return out


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
        if not (
            isinstance(x[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(x[c])
            or pd.api.types.is_bool_dtype(x[c])
            or pd.api.types.is_numeric_dtype(x[c])
        )
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
        if isinstance(x[nm].dtype, pd.CategoricalDtype):
            # A column that already carries a level order keeps it, the way an R
            # factor does. Sorting it again would move the reference level, and
            # with it every coefficient the level is measured against.
            x[nm] = x[nm].cat.remove_unused_categories()
            predictor_lv[nm] = x[nm].cat.categories.astype(str).tolist()
        elif pd.api.types.is_object_dtype(x[nm]):
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
    if isinstance(y.dtype, pd.CategoricalDtype):
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


def sa_r_term_name(name: str) -> str:
    """A patsy column name written the way `model.matrix()` writes it.

    The quoting that lets a column called `x 1` through the formula is patsy's
    and belongs to the formula rather than to the term, and a factor level
    arrives as `[T.male]` where R appends a bare `male`. Names are what a
    coefficient table is read by, so they are R's.
    """
    if name in ("const", "Intercept"):
        return "(Intercept)"
    out = re.sub(r"Q\('([^']*)'\)", r"\1", name)
    return re.sub(r"\[T\.([^\]]*)\]", r"\1", out)


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
    mat.columns = [sa_r_term_name(c) for c in mat.columns]
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
    terms = [sa_r_term_name(t) for t in estimate.index]
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
            "method": "repeatedcv",
            "cv_method": cv_method,
            "n_fold": n_fold,
            "n_repeat": n_repeat,
        }
    if cv_method == "kfold":
        return {
            "cv": KFold(n_splits=n_fold, shuffle=True),
            "method": "cv",
            "cv_method": cv_method,
            "n_fold": n_fold,
            "n_repeat": None,
        }
    if cv_method == "loocv":
        return {
            "cv": LeaveOneOut(),
            "method": "LOOCV",
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
    sa_check_count(n_fold, "n_fold", 2)
    sa_check_count(n_repeat, "n_repeat", 1)
    if not cv:
        return {
            "cv": None,
            "method": "none",
            "cv_method": None,
            "n_fold": None,
            "n_repeat": None,
        }
    return sa_resample_scheme(cv_method, n_fold, n_repeat, n_obs)


def sa_resample_sets(
    y: np.ndarray,
    ctrl: dict[str, Any],
    seed: int | None,
) -> tuple[list[np.ndarray], list[str]]:
    """The training rows of each resample, and what caret calls them.

    With a seed these are the folds an R session would have drawn; without one
    there is nothing to line up with and the scikit-learn splitter is used.
    """
    if seed is None:
        train_sets = [np.asarray(tr) for tr, _ in ctrl["cv"].split(np.zeros(len(y)), y)]
        return train_sets, [f"Resample{i + 1}" for i in range(len(train_sets))]
    return sa_caret_resample_index(
        y, ctrl["method"], ctrl["n_fold"] or len(y), ctrl["n_repeat"] or 1, seed
    )


def sa_bind_folds(
    ctrl: dict[str, Any],
    strata: np.ndarray,
    seed: int | None,
) -> dict[str, Any]:
    """Swap the scikit-learn splitter for the fold index caret would have drawn.

    For the engines that still tune through `GridSearchCV`, this is as far as
    parity goes: the folds become the R ones, while the numbers scored on them
    stay whatever the Python engine produces.
    """
    if ctrl.get("cv") is None or seed is None:
        return ctrl
    train_sets, _ = sa_resample_sets(strata, ctrl, seed)
    return {**ctrl, "cv": RFoldSplitter(train_sets, len(strata))}


def run_cv_scores(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    ctrl: dict[str, Any],
    *,
    classify: bool = False,
    strata: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """caret's resampled performance for a model with one tuning candidate.

    `fit_predict` is handed the training and test row indices and returns the
    predictions for the test rows, which is the only part that differs between
    one engine and the next. `strata` is what the folds are balanced on, which
    for a classification is the class labels rather than the zero/one coding the
    engine is fitted against: caret sees a factor there and stratifies on its
    levels, and a numeric column would be cut into quantiles instead.
    """
    if ctrl.get("cv") is None:
        return None, None

    train_sets, names = sa_resample_sets(y if strata is None else strata, ctrl, seed)
    summarise = sa_post_resample_class if classify else sa_post_resample
    metric_names = ["Accuracy", "Kappa"] if classify else ["RMSE", "Rsquared", "MAE"]

    all_rows = np.arange(len(y))
    rows = []
    for train in train_sets:
        test = np.setdiff1d(all_rows, train)
        rows.append(summarise(fit_predict(train, test), y[test]))

    resampling = pd.DataFrame(rows)[metric_names]
    resampling["Resample"] = names

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        perf = pd.DataFrame(
            {m: [np.nanmean(resampling[m].to_numpy(dtype=float))] for m in metric_names}
            | {
                f"{m}SD": [np.nanstd(resampling[m].to_numpy(dtype=float), ddof=1)]
                for m in metric_names
            }
        )
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
