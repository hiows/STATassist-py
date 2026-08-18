"""Recursive feature elimination."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from statassist.contracts.selection import sa_new_selection
from statassist.utils.caret_resample import sa_caret_resample_index
from statassist.utils.glmnet_r import sa_post_resample, sa_post_resample_class
from statassist.utils.model_utils import (
    sa_design_lv,
    sa_forest_frame,
    sa_outcome_levels,
    sa_resolve_model_input,
    sa_train_control,
)
from statassist.utils.validate import sa_check_count, sa_preserve_seed


# `glm.control()`: the deviance tolerance and the iteration cap R stops at.
R_GLM_EPS = 1e-8
R_GLM_MAXIT = 25


def _rfe_design(x: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Model matrix with an intercept, and the term each column came from.

    A factor is several columns and only the whole factor can be eliminated, so
    the map back from a dummy column to the predictor that produced it is what
    the ranking needs. This is R's ``attr(model.matrix(object), "assign")``.
    """
    blocks = [np.ones((len(x), 1))]
    assign: list[str | None] = [None]
    for nm in x.columns:
        col = x[nm]
        if pd.api.types.is_numeric_dtype(col) and not isinstance(
            col.dtype, pd.CategoricalDtype
        ):
            blocks.append(col.to_numpy(dtype=float)[:, None])
            assign.append(nm)
            continue
        levels = (
            list(col.cat.categories)
            if isinstance(col.dtype, pd.CategoricalDtype)
            else sorted(col.astype(str).unique())
        )
        values = col.astype(str).to_numpy()
        # Treatment contrasts, the first level as the reference, as R codes an
        # unordered factor.
        for lv in levels[1:]:
            blocks.append((values == str(lv)).astype(float)[:, None])
            assign.append(nm)
    return np.hstack(blocks), assign


def _binomial_fit(y: np.ndarray, mat: np.ndarray):
    """Iteratively reweighted least squares, stopped where `glm()` stops.

    Both engines take the same steps from the same start, and differ only in
    when they call it done: `glm()` compares the change in deviance against the
    deviance itself, while statsmodels compares it against a fixed tolerance.
    The standard errors come from the weights of the final step, so one step of
    disagreement moves a Wald statistic in its fifth decimal — enough to reorder
    two predictors that are nearly equally useful. So the steps are counted
    first, under R's rule, and the fit is then stopped at that count.
    """
    model = sm.GLM(y, mat, family=sm.families.Binomial())
    probe = model.fit(tol=0.0, maxiter=R_GLM_MAXIT)
    deviance = np.asarray(probe.fit_history["deviance"][1:], dtype=float)

    steps = R_GLM_MAXIT
    for j in range(1, len(deviance)):
        change = abs(deviance[j] - deviance[j - 1])
        if change / (abs(deviance[j]) + 0.1) < R_GLM_EPS:
            steps = j
            break
    return sm.GLM(y, mat, family=sm.families.Binomial()).fit(
        tol=0.0, maxiter=steps
    )


def _wald_rank(
    x: pd.DataFrame, y: np.ndarray, classify: bool
) -> list[tuple[str, float]]:
    """Rank predictors by their own Wald statistic, largest first.

    The statistic rather than the estimate, since the estimate carries the units
    of its predictor and the statistic has divided them out. The largest
    statistic among a factor's levels stands for the factor, so a factor is kept
    as long as one of its levels is worth keeping. A term the fit could not
    estimate ranks at zero: it is a column the others already span, so dropping
    it costs nothing.
    """
    mat, assign = _rfe_design(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if classify:
            fit = _binomial_fit(y, mat)
        else:
            fit = sm.OLS(y, mat).fit()
        statistic = np.abs(np.asarray(fit.tvalues, dtype=float))
    statistic = np.nan_to_num(statistic, nan=0.0)

    overall = []
    for nm in x.columns:
        at = [i for i, term in enumerate(assign) if term == nm]
        overall.append((nm, max((statistic[i] for i in at), default=0.0)))
    overall.sort(key=lambda pair: -pair[1])
    return overall


def _forest_rank(
    x: pd.DataFrame, y: np.ndarray, classify: bool, ntree: int, nodesize: int, seed
) -> list[tuple[str, float]]:
    """Rank predictors by what shuffling them costs the forest."""
    frame = sa_forest_frame(x)
    model = _forest(classify, ntree, nodesize, seed, x.shape[1]).fit(frame, y)
    imp = permutation_importance(model, frame, y, n_repeats=5, random_state=seed)
    overall = list(zip(x.columns, imp.importances_mean))
    overall.sort(key=lambda pair: -pair[1])
    return overall


def _forest(classify: bool, ntree: int, nodesize: int, seed, n_pred: int):
    # `randomForest()`'s own rule, recomputed at each subset size: a fixed
    # `mtry` would exceed the predictor count at the small end of the profile,
    # where the whole question is what a handful of predictors can do.
    mtry = int(np.sqrt(n_pred)) if classify else max(n_pred // 3, 1)
    cls = RandomForestClassifier if classify else RandomForestRegressor
    return cls(
        n_estimators=ntree,
        min_samples_leaf=nodesize,
        max_features=max(min(mtry, n_pred), 1),
        random_state=seed,
    )


def _rfe_fit_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    model: str,
    classify: bool,
    ntree: int,
    nodesize: int,
    seed,
) -> np.ndarray:
    if model == "rf":
        forest = _forest(classify, ntree, nodesize, seed, x_train.shape[1])
        return forest.fit(sa_forest_frame(x_train), y_train).predict(
            sa_forest_frame(x_test)
        )

    mat, _ = _rfe_design(x_train)
    new, _ = _rfe_design(x_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if classify:
            fit = _binomial_fit(y_train, mat)
            # glm() models the probability of the last level, so the cut is at
            # the second one and nothing is reversed afterwards.
            return (np.asarray(fit.predict(new)) > 0.5).astype(int)
        fit = sm.OLS(y_train, mat).fit()
    return np.asarray(fit.predict(new))


def _rfe_iter(
    x: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    sizes: list[int],
    rank_fn,
    predict_fn,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[str, float, int]]]:
    """One resample of caret's elimination, with `rerank = FALSE`.

    The full set is fitted and ranked once, and every smaller subset is the top
    of that one ranking. Re-ranking after every drop refits the model once per
    remaining predictor per fold, which is a different and far slower procedure.
    """
    p = x.shape[1]
    size_values = sorted(set(list(sizes) + [p]), reverse=True)
    retained = list(x.columns)
    ranked: list[tuple[str, float]] = []
    scores: list[tuple[int, np.ndarray]] = []
    variables: list[tuple[str, float, int]] = []

    x_train, x_test = x.iloc[train], x.iloc[test]
    for k, size in enumerate(size_values):
        pred = predict_fn(x_train[retained], y[train], x_test[retained])
        scores.append((size, np.asarray(pred)))
        if k == 0:
            ranked = rank_fn(x_train[retained], y[train])
        kept = set(retained)
        variables.extend((v, o, size) for v, o in ranked if v in kept)
        if k + 1 < len(size_values):
            retained = [v for v, _ in ranked][: size_values[k + 1]]
    return scores, variables


def _rfe_sizes(subset_sizes: list[int] | None, p: int) -> list[int]:
    if subset_sizes is None:
        ladder = list(range(1, min(11, p + 1))) + [15, 20, 30, 50, 100]
        return sorted(set(min(p, s) for s in ladder))
    sizes = [int(s) for s in subset_sizes]
    return sorted(set(sizes))


def _rfe_metric(metric: str | None, classify: bool) -> tuple[str, bool]:
    available = ["Accuracy", "Kappa"] if classify else ["RMSE", "Rsquared", "MAE"]
    if metric is None:
        metric = available[0]
    if metric not in available:
        raise ValueError(
            f"`metric` must be one of {', '.join(available)} for a "
            f"{'classification' if classify else 'regression'}."
        )
    maximize = metric not in ("RMSE", "MAE")
    return metric, maximize


def perform_rfe(
    data: pd.DataFrame,
    outcome: Any,
    predictors: list[str] | None = None,
    *,
    outcome_lv: list[str] | None = None,
    control_label: str | None = None,
    model: str = "linear",
    subset_sizes: list[int] | None = None,
    metric: str | None = None,
    ntree: int = 500,
    nodesize: int | None = None,
    cv_method: str = "repeated_kfold",
    n_fold: int = 5,
    n_repeat: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    ntree = sa_check_count(ntree, "ntree", 1)
    input_ = sa_resolve_model_input(data, outcome, predictors)
    classify = (
        model == "logistic"
        or outcome_lv is not None
        or control_label is not None
        or not np.issubdtype(input_["y"].dtype, np.number)
    )

    if model == "linear" and classify:
        raise ValueError(
            '`model = "linear"` ranks by the coefficients of a straight line through a '
            "number, and `outcome` is a set of class labels."
        )
    if model == "logistic" and np.issubdtype(input_["y"].dtype, np.number) and len(np.unique(input_["y"])) > 2:
        raise ValueError('`model = "logistic"` classifies two classes on a multi-value numeric outcome.')

    if classify:
        y = sa_outcome_levels(input_["y"], outcome_lv, control_label, model="a recursive feature elimination")
        outcome_lv = list(y.categories)
        y_fit = y.astype(str).to_numpy() if model == "rf" else np.asarray((y == outcome_lv[1]).astype(int))
    else:
        if not np.all(np.isfinite(input_["y"])):
            raise ValueError("`outcome` holds non-finite value(s).")
        y_fit = input_["y"].astype(float)

    if nodesize is None:
        nodesize = 1 if classify else 5
    nodesize = sa_check_count(nodesize, "nodesize", 1)

    sizes = _rfe_sizes(subset_sizes, len(input_["predictors"]))
    scoring, maximize = _rfe_metric(metric, classify)
    metrics = ["Accuracy", "Kappa"] if classify else ["RMSE", "Rsquared", "MAE"]
    importance = {
        "linear": "absolute t statistic",
        "logistic": "absolute Wald z",
        "rf": "permutation importance",
    }[model]

    x = input_["x"]
    if model == "rf":
        def rank_fn(x_train, y_train):
            return _forest_rank(x_train, y_train, classify, ntree, nodesize, seed)
    else:
        def rank_fn(x_train, y_train):
            return _wald_rank(x_train, y_train, classify)

    def predict_fn(x_train, y_train, x_test):
        return _rfe_fit_predict(
            x_train, y_train, x_test, model, classify, ntree, nodesize, seed
        )

    def score_fn(pred, obs):
        if not classify:
            return sa_post_resample(pred, obs)
        pred = np.asarray(pred).astype(str)
        obs = np.asarray(obs).astype(str)
        values = sa_post_resample_class(pred, obs)
        # caret keeps the confusion table of every resample beside the two
        # metrics, so that a fold that got everything wrong one way can be told
        # from one that split its errors.
        levels = np.unique(np.asarray(y_fit).astype(str))
        cells = [
            int(np.sum((pred == p) & (obs == o))) for o in levels for p in levels
        ]
        values.update({f".cell{i + 1}": c for i, c in enumerate(cells)})
        return values

    ctrl = sa_train_control(True, cv_method, n_fold, n_repeat, input_["n_used"])
    rows = np.arange(input_["n_used"])
    # An elimination draws its index straight from the seed, unlike `train()`,
    # which spends a draw on a seed for the folds first.
    if seed is None:
        train_sets = [np.asarray(tr) for tr, _ in ctrl["cv"].split(rows, y_fit)]
        fold_names = [f"Resample{i + 1}" for i in range(len(train_sets))]
    else:
        train_sets, fold_names = sa_caret_resample_index(
            np.asarray(y) if classify else y_fit,
            ctrl["method"],
            ctrl["n_fold"] or input_["n_used"],
            ctrl["n_repeat"] or 1,
            seed,
            chain=False,
        )

    with sa_preserve_seed(seed):
        held: list[dict[str, Any]] = []
        variables: list[dict[str, Any]] = []
        for train, name in zip(train_sets, fold_names):
            test = np.setdiff1d(rows, train)
            fold_scores, fold_vars = _rfe_iter(
                x, y_fit, train, test, sizes, rank_fn, predict_fn
            )
            held.extend(
                {"Variables": size, "pred": pred, "obs": y_fit[test], "Resample": name}
                for size, pred in fold_scores
            )
            variables.extend(
                {"var": v, "Overall": o, "Variables": size, "Resample": name}
                for v, o, size in fold_vars
            )

    variables_df = pd.DataFrame(variables)
    loo = ctrl["method"] == "LOOCV"

    if loo:
        # One row held out is one prediction, and a single prediction has no
        # error to spread and no correlation to square. caret pools every
        # held-out row instead and scores them once, which is why a
        # leave-one-out profile has no standard deviations and no per-resample
        # table under it.
        profile_rows = []
        for size in sorted({row["Variables"] for row in held}):
            at = [row for row in held if row["Variables"] == size]
            pred = np.concatenate([row["pred"] for row in at])
            obs = np.concatenate([row["obs"] for row in at])
            profile_rows.append({"n_vars": size, **score_fn(pred, obs)})
        profile = pd.DataFrame(profile_rows)[["n_vars", *metrics]]
        scored_df = None
    else:
        scored_df = pd.DataFrame(
            {
                "Variables": row["Variables"],
                **score_fn(row["pred"], row["obs"]),
                "Resample": row["Resample"],
            }
            for row in held
        )
        grouped = scored_df.groupby("Variables")[metrics]
        profile = grouped.mean().reset_index()
        spread = grouped.std().reset_index(drop=True)
        for m in metrics:
            profile[f"{m}SD"] = spread[m]
        profile = profile.rename(columns={"Variables": "n_vars"})

    # `caret::pickSizeBest()`: the best score, and the smallest size that reaches
    # it, since a tie between two sizes is an argument for the smaller one.
    best = profile[scoring].max() if maximize else profile[scoring].min()
    best_subset = int(profile.loc[profile[scoring] == best, "n_vars"].min())
    profile["chosen"] = profile["n_vars"] == best_subset

    # `caret::lmFuncs$selectVar()`: every fold and every size that a predictor
    # survived to, averaged, so the ranking and the selection are one answer.
    averaged = variables_df.groupby("var")["Overall"].mean()
    candidates = sorted(input_["predictors"])
    estimate = np.array([averaged.get(c, np.nan) for c in candidates], dtype=float)
    at = np.argsort(np.where(np.isnan(estimate), -np.inf, -estimate), kind="stable")
    candidates = [candidates[i] for i in at]
    estimate = estimate[at]
    selected = candidates[:best_subset]

    ranking = pd.DataFrame(
        {
            "candidates": candidates,
            "estimate": estimate,
            "rank": list(range(1, len(candidates) + 1)),
            "selected": [c in selected for c in candidates],
        }
    )

    resampling = (
        None
        if scored_df is None
        else scored_df[scored_df["Variables"] == best_subset].reset_index(drop=True)
    )

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
        "rf": f"Random forest {'classification' if classify else 'regression'}",
    }[model]

    return sa_new_selection(
        analysis="rfe",
        candidates=ranking["candidates"].tolist(),
        design=design,
        parameters={
            "model": model,
            "metric": scoring,
            "maximize": maximize,
            **({"ntree": ntree, "nodesize": nodesize} if model == "rf" else {}),
            "cv_method": ctrl["cv_method"],
            "n_fold": ctrl["n_fold"],
            "n_repeat": ctrl["n_repeat"],
            "seed": seed,
        },
        selected=selected,
        ranking=ranking,
        profile=profile,
        resampling=resampling,
        engine={
            "package": "statsmodels",
            "method": "rfe",
            "label": label,
            "metrics": metrics,
            "importance": importance,
        },
        fit={
            "variables": variables_df,
            "resampling": scored_df,
            "best_subset": best_subset,
        },
    )
