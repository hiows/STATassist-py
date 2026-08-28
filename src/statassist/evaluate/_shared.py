"""What the two ``evaluate_*`` functions share, up to the point of scoring.

Port of ``R/utils_evaluate.R``. Both take a baseline and a set of models to hold
against it, both read the rows through :meth:`SaModel.predict` rather than
through any engine object, and both are only meaningful if every model was scored
on the same rows.

That last one is why this module exists rather than the two functions each
resolving their own input. A prediction is missing for a row that is incomplete
across *that model's* predictors, so models fitted on different predictor sets
come back with different rows filled in. Scoring each model on whatever it
happened to manage would put two AUCs on two samples and call their difference an
improvement, and DeLong's test, the IDI and the NRI are all paired statistics
that have no meaning at all across different rows. The intersection is taken
once, here, for both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.result import SaModel
from ..core.validate import RowVector, resolve_row_vector

__all__ = [
    "MIN_SCORED_ROWS",
    "Collected",
    "check_model_agreement",
    "check_model_family",
    "collect_predictions",
    "evaluate_newdata",
    "prediction_table",
    "resolve_answer",
    "resolve_models",
]

#: How many scored rows an evaluation needs before it will report anything.
#:
#: Two, because every quantity in either result is measured across rows: a
#: correlation, a variance and a calibration slope are all undefined on one row,
#: and an AUC on one row has nothing to rank it against.
MIN_SCORED_ROWS = 2

#: What a model reports as its ``outcome`` when it was fitted from a vector.
#:
#: :func:`~statassist.core.validate.resolve_row_vector` records this in place of
#: a column name, and it is the one value of ``design["outcome"]`` that cannot be
#: looked up in ``newdata``.
_VECTOR_LABEL = "<vector>"


def resolve_models(
    baseline_model: Any,
    new_models: Any,
    baseline_label: Any,
) -> dict[str, SaModel]:
    """Resolve the baseline and the models held against it into one mapping.

    Port of ``sa_resolve_models()``. The names are not decoration: they are what
    every table is keyed on and what the legend of a plot reads, so an unnamed
    collection is refused rather than given positions for names. The baseline
    comes first, which is the order ``models`` fixes for the whole result.
    """
    if not isinstance(baseline_model, SaModel):
        raise SaValueError(
            "`baseline_model` must be a fitted model, as returned by "
            "fit_linear_regression(), fit_logistic_regression(), fit_elastic_net(), "
            "fit_rf() or fit_svm()."
        )
    if not isinstance(baseline_label, str) or not baseline_label:
        raise SaValueError("`baseline_label` must be a single non-empty name.")

    # An empty mapping is a call that named no comparison, which is the same
    # thing `None` says and reads more naturally out of a comprehension that
    # found nothing.
    if new_models is None or (not isinstance(new_models, SaModel) and len(new_models) == 0):
        return {baseline_label: baseline_model}

    if not isinstance(new_models, Mapping):
        raise SaValueError(
            "`new_models` must be a mapping of name to fitted model, such as "
            '{"selected": fit_1, "penalized": fit_2}, or None to score the baseline '
            "on its own."
        )
    labels = [str(name) for name in new_models]
    if any(not name for name in labels):
        raise SaValueError(
            "every entry of `new_models` must be named: the names are what the result "
            "tables and the plot legend call the models."
        )
    if baseline_label in labels:
        raise SaValueError(
            f"`new_models` holds a model called `{baseline_label}`, which is what the "
            "baseline is called. Rename it, or pass a different `baseline_label`."
        )
    not_models = [name for name, model in new_models.items() if not isinstance(model, SaModel)]
    if not_models:
        raise SaValueError(
            "every entry of `new_models` must be a fitted model. Not a model: "
            + ", ".join(str(name) for name in not_models)
            + "."
        )

    resolved = {baseline_label: baseline_model}
    resolved.update({str(name): model for name, model in new_models.items()})
    return resolved


def check_model_family(models: Mapping[str, SaModel], want: str, other: str) -> None:
    """Refuse a model that answers a different kind of question.

    Port of ``sa_check_model_family()``. A classification handed to the
    regression function would be scored by correlating a probability against a
    class label, which produces a number rather than an error and is the reason
    this is checked rather than left to fail downstream.
    """
    types = {name: str(model["design"]["outcome_type"]) for name, model in models.items()}
    wrong = [name for name, kind in types.items() if kind != want]
    if wrong:
        raise SaValueError(
            f"every model must have been fitted to {want} outcome. Not {want}: "
            + ", ".join(f"{name} ({types[name]})" for name in wrong)
            + f". Use {other} for those."
        )


def check_model_agreement(models: Mapping[str, SaModel]) -> None:
    """Refuse a set of models that are not describing the same question.

    Port of ``sa_check_model_agreement()``. Two models of different outcomes can
    both be scored, and the scores can be put in one table, and the table means
    nothing. The same goes for a pair of classifications whose ``outcome_lv``
    point at different classes: both answer a probability from
    ``type="response"`` and one of them is the probability of the other class, so
    every comparison between them is reversed. Neither is quietly repaired,
    because a re-pointed level order is a different model from the one the caller
    fitted and printed.
    """
    outcomes = {name: str(model["design"]["outcome"]) for name, model in models.items()}
    if len(set(outcomes.values())) > 1:
        raise SaValueError(
            "every model must have been fitted to the same outcome, since the scores "
            "are put side by side. Got "
            + ", ".join(f"{name} = {label}" for name, label in outcomes.items())
            + "."
        )

    named = {
        name: [str(level) for level in model["design"]["outcome_lv"]]
        for name, model in models.items()
        if model["design"].get("outcome_lv") is not None
    }
    if not named:
        return
    first = next(iter(named.values()))
    disagree = [name for name, levels in named.items() if levels != first]
    if disagree:
        raise SaValueError(
            "every model must hold the same `outcome_lv`, in the same order: the "
            'second level is the class `type="response"` reports the probability of, '
            "so a model that names them the other way round predicts the other class. "
            "Expected " + ", ".join(first) + ", but " + ", ".join(disagree) + " disagree(s). "
            "Refit with a matching `outcome_lv`."
        )


def resolve_answer(answer: Any, newdata: pd.DataFrame, baseline_model: SaModel) -> RowVector:
    """Resolve the observed outcome of the rows being scored.

    Port of ``sa_resolve_answer()``. ``answer=None`` reads the column the models
    were fitted to, which is the usual case: the held-out half of a
    :func:`~statassist.split_data` result carries the outcome under the same name
    as the half the models were fitted on. A model fitted from a vector remembers
    no name to look up, and says so.
    """
    if answer is not None:
        return resolve_row_vector(answer, "answer", newdata, allow_na=True)

    label = baseline_model["design"]["outcome"]
    if label is None or label == _VECTOR_LABEL:
        raise SaValueError(
            "`answer` is required: `baseline_model` was fitted to an outcome passed as "
            "a vector, so it remembers no column name to read from `newdata`."
        )
    if label not in newdata.columns:
        raise SaValueError(
            f"`answer` is None, so the outcome is read from the `{label}` column the "
            "models were fitted to, and `newdata` has no such column. Name the "
            "observed values with `answer`."
        )
    return RowVector(value=newdata[label], label=str(label))


class Collected:
    """Every model's predictions on the rows every model could predict.

    Attributes:
        predicted: One column per model, in the order ``models`` is in, holding
            only the kept rows.
        keep: The positions in ``newdata`` that survived.
        n_obs: How many rows were offered.
        n_dropped: How many of them were left out.
    """

    __slots__ = ("keep", "n_dropped", "n_obs", "predicted")

    def __init__(
        self,
        predicted: np.ndarray,
        keep: np.ndarray,
        n_obs: int,
        n_dropped: int,
    ) -> None:
        self.predicted = predicted
        self.keep = keep
        self.n_obs = n_obs
        self.n_dropped = n_dropped


def collect_predictions(
    models: Mapping[str, SaModel],
    newdata: pd.DataFrame,
    observed: np.ndarray,
) -> Collected:
    """Predict every model on the same rows, and say which rows those are.

    Port of ``sa_collect_predictions()``. The intersection rather than the union,
    and one message rather than one per model. Which rows a model can predict is
    a property of its predictors, so a baseline fitted on nine columns and a
    reduced model fitted on four disagree on any row that is missing one of the
    extra five. Scoring each on what it managed would compare two models on two
    samples.

    Args:
        models: The models, the baseline first.
        newdata: The rows to score.
        observed: The observed outcome, one entry per row of ``newdata``.
    """
    n_obs = len(newdata.index)
    names = list(models)

    columns = []
    for name in names:
        try:
            answered = models[name].predict(newdata, type="response")
        except Exception as error:  # noqa: BLE001
            # The engine's message names `newdata` and the predictor at fault,
            # which is the useful half. Which of several models asked for it is
            # the half only this loop knows.
            raise SaValueError(f"model `{name}` cannot be scored on `newdata`: {error}") from error
        columns.append(np.asarray(answered, dtype=float))
    predicted = np.column_stack(columns) if columns else np.empty((n_obs, 0))

    has_answer = ~np.isnan(np.asarray(observed, dtype=float))
    predictable = ~np.isnan(predicted).any(axis=1)
    keep = np.flatnonzero(has_answer & predictable)
    n_dropped = n_obs - int(keep.size)

    if n_dropped > 0:
        # Named per reason, since the two are fixed by different things: a
        # missing answer is a row that cannot be scored at all, while a row no
        # model could be given is a row whose predictors are incomplete.
        reasons = []
        no_answer = int((~has_answer).sum())
        if no_answer > 0:
            reasons.append(f"{no_answer} with no observed outcome")
        lost = [
            name
            for position, name in enumerate(names)
            if np.isnan(predicted[has_answer, position]).any()
        ]
        if lost:
            reasons.append("some incomplete across the predictors of " + ", ".join(lost))
        notify(
            f"{n_dropped} of {n_obs} row(s) of `newdata` were left out: "
            + ", ".join(reasons)
            + ". Every model is scored on the same rows, so a row one model cannot "
            "predict is left out of all of them."
        )

    if keep.size < MIN_SCORED_ROWS:
        raise SaValueError(
            f"only {keep.size} row(s) of `newdata` have an observed outcome and a "
            f"prediction from every model; at least {MIN_SCORED_ROWS} are needed."
        )

    return Collected(
        predicted=predicted[keep, :],
        keep=keep,
        n_obs=n_obs,
        n_dropped=n_dropped,
    )


def prediction_table(
    models: Sequence[str],
    predicted: np.ndarray,
    keep: np.ndarray,
    observed: np.ndarray,
) -> pd.DataFrame:
    """Fold the per-model predictions into the long table the contract holds.

    Port of ``sa_prediction_table()``. ``row`` is the position in ``newdata``
    rather than a running count, since the rows that survive are an intersection
    and are therefore not the first however many.
    """
    n_model = len(models)
    return pd.DataFrame(
        {
            "model": np.repeat(np.asarray(models, dtype=object), keep.size),
            "row": np.tile(np.asarray(keep, dtype=int), n_model),
            "observed": np.tile(np.asarray(observed, dtype=float), n_model),
            # Column-major, so the block of rows for a model is that model's
            # column, which is the order `model` above repeats in.
            "predicted": np.asarray(predicted, dtype=float).ravel(order="F"),
        }
    )


def evaluate_newdata(newdata: Any) -> pd.DataFrame:
    """Read ``newdata`` into the frame the rest of the evaluation works on.

    Port of ``sa_evaluate_newdata()``.
    """
    if isinstance(newdata, np.ndarray):
        newdata = pd.DataFrame(newdata)
    if not isinstance(newdata, pd.DataFrame):
        raise SaValueError("`newdata` must be a DataFrame or a 2-d array.")
    if len(newdata.index) == 0:
        raise SaValueError("`newdata` has zero rows, so there is nothing to score.")
    return newdata
