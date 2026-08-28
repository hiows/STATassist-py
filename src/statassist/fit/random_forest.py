"""A forest of trees, scored on the rows each tree did not see.

The port of ``R/fit_rf.R``. Two things about the engine differ from R's, and both
are declared in ``engine["overridden"]`` rather than smoothed over.

R's ``randomForest`` splits on a factor directly, so one predictor is one term
throughout. ``scikit-learn``'s trees take a numeric matrix, so a factor arrives
here as one dummy column per level and the engine reports one number per column.
Those numbers are summed back onto the predictor they came from, which is what
makes ``coefficients`` the same shape on both sides: one row per predictor.
Summed rather than averaged, since what a predictor was worth does not depend on
how many columns it was spread over.

R's permutation importance is measured on the rows each tree left out. The engine
here permutes the rows the forest was fitted to, which is the routine it has. It
is the same measurement in spirit - shuffle a column, see what the score loses -
and it is optimistic in a way the out-of-bag version is not, so it is reported
beside ``impurity`` and the out-of-bag scores rather than on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core.result import SaModel, new_model
from ..core.validate import check_count
from ._shared import (
    CV_METHODS,
    EngineFit,
    ModelInput,
    Outcome,
    check_cv_method,
    class_scores,
    design_matrix,
    design_source,
    importance_table,
    model_design,
    model_frame,
    numeric_scores,
    permutation_scores,
    quiet_engine,
    resample_grid,
    resample_params,
    resolve_model_input,
    resolve_outcome,
    rf_grid,
    rollup,
    train_control,
)

__all__ = ["fit_rf"]

#: What this model is called where a message names it.
_MODEL = "a random forest"

#: Rows a leaf may hold, when the caller names none.
#:
#: ``randomForest``'s own defaults, and they differ by outcome type for a reason:
#: a classification leaf is asked which class, and one row answers that, while a
#: regression leaf is asked for an average, and an average of one row is that row.
_NODESIZE = {True: 1, False: 5}

#: What the port changed about the engine, for ``engine["overridden"]``.
_OVERRIDDEN = (
    "oob_score = True",
    "factor terms summed back into predictors",
    "permutation importance on the fitted rows",
)

#: Shuffles per term when measuring permutation importance.
#:
#: R's engine permutes once per tree and averages over the forest, which is not
#: an argument its caller sets. The routine here permutes the whole data set, so
#: the count is set rather than inherited: one shuffle of one column is a draw
#: rather than a measurement, and ten is enough for the ordering to settle.
_N_PERMUTE = 10

#: Where an out-of-bag probability becomes an out-of-bag class.
#:
#: A half, which is the cutoff that minimises the error count when the two
#: mistakes cost the same. They often do not, which is why
#: :func:`~statassist.fit._shared.class_scores` reports sensitivity and
#: specificity beside the accuracy rather than the accuracy alone.
_CUTOFF = 0.5


def fit_rf(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    mtry: Any = None,
    ntree: Any = 500,
    nodesize: Any = None,
    cv: bool = True,
    cv_method: str = CV_METHODS[0],
    n_fold: Any = 5,
    n_repeat: Any = 5,
    seed: int | None = None,
) -> SaModel:
    """Fit a random forest.

    Fits many trees, each on a bootstrap sample of the rows and each choosing
    among a random subset of the terms at every split, and averages them. The
    randomness is the method rather than a concession to speed: trees grown on
    the same data are nearly the same tree and averaging them gains nothing,
    while trees forced to disagree average into something better than any of
    them.

    What a forest answers with is importance rather than coefficients, and
    ``coefficients`` holds two kinds of it. ``estimate`` is permutation
    importance: a column is shuffled and the model is scored again, so it is what
    the model would lose without that predictor. ``impurity`` is how much the
    splits on it reduced the error over the whole forest. The two disagree
    usefully - impurity favours a predictor with many distinct values, since it
    offers more places to split - so a predictor high on one and low on the other
    is worth a second look.

    ``fit_stats`` is out-of-bag rather than in-sample, and that is the property
    that makes a forest worth fitting without a test set. Each tree is fitted on
    about two thirds of the rows, so each row can be predicted by the third of
    the forest that never saw it, and the score of those predictions is honest
    the way a held-out score is. ``performance`` is still the resampled score, so
    a forest can be compared with a model that has no out-of-bag notion at all.

    ``mtry`` is the one argument that is tuned, and it is the argument the method
    turns on: at the number of terms the forest stops disagreeing with itself and
    becomes an average of very similar trees. ``ntree`` and ``nodesize`` are the
    same for every candidate, since more trees is never worse than fewer and a
    leaf size is a statement about the data rather than something to search.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The outcome, either the name of a column of ``data`` or a vector
            with one entry per row.
        predictors: Column names to fit on, or ``None`` for every column of
            ``data`` except the outcome.
        outcome_lv: The two classes, reference first, to fit a classification.
        control_label: The reference class on its own. Either argument asks for a
            classification.
        mtry: Terms to choose among at each split, or a set of values to search
            over. ``None`` takes the rule of thumb: the square root of the term
            count for a classification and a third of it for a regression. The
            count is of model terms rather than of predictors, since a factor
            reaches the engine as one column per level;
            ``engine["x_names"]`` lists them.
        ntree: Trees in the forest. More is never worse, only slower.
        nodesize: Rows a leaf may hold. ``None`` takes the engine's default of 1
            for a classification and 5 for a regression. Larger values grow
            shallower trees, which is the way to fight overfitting here.
        cv: Whether to cross-validate. ``False`` fits one forest, so ``mtry``
            must then name one value.
        cv_method: Resampling scheme: ``"repeated_kfold"``, ``"kfold"`` or
            ``"loocv"``.
        n_fold: Folds per run, used by ``"repeated_kfold"`` and ``"kfold"``.
        n_repeat: Number of runs, used by ``"repeated_kfold"``.
        seed: Seed for the bootstrap samples, the split subsets, the shuffling
            and the fold assignment. Unlike the unpenalized fits, a forest
            depends on it: pass one for a result that can be reproduced.

    Returns:
        A :class:`~statassist.core.SaModel` whose ``analysis`` is
        ``"random_forest"``.

        * ``terms`` - the predictors, most important first, which is the order
          ``coefficients`` follows.
        * ``coefficients`` - ``terms``, ``estimate`` (permutation) and
          ``impurity``. No intercept row, since a forest has no intercept.
        * ``fit_stats`` - out-of-bag: ``oob_r_squared``, ``oob_rmse``,
          ``oob_mae`` for a regression; ``oob_accuracy``, ``oob_error``,
          ``oob_kappa``, ``oob_sensitivity``, ``oob_specificity`` for a
          classification. ``n_oob`` is how many rows had an out-of-bag
          prediction at all.
        * ``parameters`` - the ``mtry`` that was chosen, with ``ntree``,
          ``nodesize`` and ``n_candidates``.
        * ``engine["overridden"]`` - what this port changed about the engine.

    Raises:
        SaValueError: If ``mtry`` exceeds the term count, if ``cv=False`` is
            given more than one candidate, or if an argument is unusable.

    Examples:
        A small forest, fitted once and scored on what each tree left out.

        >>> from statassist import simulate_regression
        >>> sim = simulate_regression(n_samples=120, n_pred=5, n_factor_pred=0,
        ...                           p_missing=0, seed=7)
        >>> fit = fit_rf(**sim.args, ntree=60, cv=False, seed=1)
        >>> list(fit.coefficients)
        ['terms', 'estimate', 'impurity']
        >>> fit.parameters["mtry"]
        1
        >>> 0 < fit.fit_stats["oob_r_squared"] < 1
        True

        The term the simulator planted the largest coefficient on is near the top
        of the importance table.

        >>> planted = sim.truth.loc[sim.truth["beta"].abs().idxmax(), "predictors"]
        >>> fit.terms.index(planted) < 2
        True
    """
    cv_method = check_cv_method(cv_method)
    trees = check_count(ntree, "ntree", 1)

    input_ = resolve_model_input(data, outcome, predictors)
    resolved = resolve_outcome(
        input_.y,
        outcome_lv,
        control_label,
        _MODEL,
        "which a leaf of a regression tree cannot average.",
    )
    leaf = check_count(
        _NODESIZE[resolved.classify] if nodesize is None else nodesize, "nodesize", 1
    )

    matrix = design_matrix(input_.x)
    grid = rf_grid(mtry, len(matrix.columns), resolved.classify, cv)
    control = train_control(
        cv, cv_method, n_fold, n_repeat, input_.n_used, classify=resolved.classify, seed=seed
    )
    label = "Random forest " + ("classification" if resolved.classify else "regression")

    def build(params: Mapping[str, Any]) -> Any:
        return _engine(int(params["mtry"]), trees, leaf, resolved.classify, seed)

    resampled = resample_grid(
        build, matrix, resolved.y, grid, control, resolved.classify, label=label
    )

    chosen = resampled.best
    forest = build(chosen)
    with quiet_engine(label):
        forest.fit(np.asarray(matrix, dtype=float), resolved.y)

    coefficients = _importance(forest, matrix, input_, resolved, seed)
    return new_model(
        analysis="random_forest",
        terms=coefficients["terms"].tolist(),
        design=model_design(input_, resolved),
        # The `mtry` that ran, not the grid that was asked for: `performance`
        # holds every candidate, so recording the grid here as well would say the
        # same thing twice and leave two places for it to be wrong.
        parameters={
            "mtry": int(chosen["mtry"]),
            "ntree": trees,
            "nodesize": leaf,
            "n_candidates": len(grid.index),
            **resample_params(cv, control, seed),
        },
        coefficients=coefficients,
        fit_stats=_fit_stats(forest, resolved),
        performance=model_frame(resampled.results),
        resampling=model_frame(resampled.resampling),
        engine={
            "package": "scikit-learn",
            "method": "RandomForestClassifier" if resolved.classify else "RandomForestRegressor",
            "label": label,
            "metrics": resampled.metrics,
            "x_names": list(matrix.columns),
            "overridden": list(_OVERRIDDEN),
        },
        fit=EngineFit(
            estimator=forest,
            x=matrix,
            y=resolved.y,
            classify=resolved.classify,
            outcome_lv=resolved.levels,
        ),
    )


def _engine(mtry: int, ntree: int, nodesize: int, classify: bool, seed: int | None) -> Any:
    """An unfitted forest at one ``mtry``.

    ``oob_score`` is on because ``fit_stats`` is out-of-bag; the engine leaves it
    off, since a forest that is only going to predict has no use for it.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    maker = RandomForestClassifier if classify else RandomForestRegressor
    return maker(
        n_estimators=ntree,
        max_features=mtry,
        min_samples_leaf=nodesize,
        oob_score=True,
        random_state=seed,
    )


def _importance(
    forest: Any,
    matrix: pd.DataFrame,
    input_: ModelInput,
    resolved: Outcome,
    seed: int | None,
) -> pd.DataFrame:
    """The importance table, on the predictor axis rather than the term axis."""
    source = design_source(input_.x)
    terms = list(matrix.columns)
    permuted = permutation_scores(forest, matrix, resolved.y, resolved.classify, _N_PERMUTE, seed)
    impurity = np.asarray(forest.feature_importances_, dtype=float)
    return importance_table(
        input_.predictors,
        rollup(dict(zip(terms, permuted, strict=True)), source, input_.predictors),
        rollup(dict(zip(terms, impurity, strict=True)), source, input_.predictors),
    )


def _fit_stats(forest: Any, resolved: Outcome) -> dict[str, float]:
    """How the forest did on the rows each tree did not see.

    A row that no tree left out has no out-of-bag prediction, which the engine
    reports as missing. Those rows are left out of the score and counted, rather
    than read as a prediction of zero: with enough trees it does not happen, and
    where it does the number has to say so.
    """
    if resolved.classify:
        probability = np.asarray(forest.oob_decision_function_, dtype=float)
        usable = np.isfinite(probability).all(axis=1)
        at = list(forest.classes_).index(1)
        predicted = (probability[usable, at] > _CUTOFF).astype(int)
        scored = class_scores(resolved.y[usable], predicted)
    else:
        prediction = np.asarray(forest.oob_prediction_, dtype=float)
        usable = np.isfinite(prediction)
        scored = numeric_scores(resolved.y[usable], prediction[usable])

    stats = {f"oob_{name}": value for name, value in scored.items()}
    stats["n_oob"] = float(int(usable.sum()))
    return stats
