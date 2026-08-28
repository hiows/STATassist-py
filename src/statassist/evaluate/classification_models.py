"""Scoring a set of fitted classifications on rows none of them was fitted on.

Port of ``R/evaluate_classification_models.R``. The regression counterpart
reports differences without tests, because a difference of held-out errors has no
null this package is in a position to state. This side does carry tests, and the
reason is that its quantities are functions of a class label and a probability:
DeLong's variance, the IDI and the NRI are all built from per-row terms whose
sampling distribution follows from the two classes being what they are, not from
where the rows came from.

All three are paired, and all three are against the baseline rather than across
every pair of models. A model is proposed as an improvement on something, which
is one comparison per model and the one the caller named by passing a baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import SaValueError
from ..core.result import SaPerformance, new_performance
from ..core.validate import check_scalar_num
from ..kernel.performance import (
    auc_delong,
    brier,
    delong_test,
    idi,
    nri,
    roc_points,
    threshold_scores,
)
from ._shared import (
    check_model_agreement,
    check_model_family,
    collect_predictions,
    evaluate_newdata,
    prediction_table,
    resolve_answer,
    resolve_models,
)

__all__ = ["evaluate_classification_models"]

#: What a model has to have been fitted to before it can be scored here.
_WANTED_FAMILY = "two classes"

#: Where a model of the other kind belongs.
_OTHER_FUNCTION = "evaluate_regression_models()"

#: How many distinct classes an evaluated outcome has.
#:
#: Two. Every quantity in the result - the AUC, the Brier score, the IDI, the NRI
#: - is defined for an event indicator, so a third class is not something the
#: arithmetic could be widened to cover.
_N_CLASSES = 2


def evaluate_classification_models(
    baseline_model: Any,
    new_models: Any = None,
    newdata: Any = None,
    answer: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    threshold: float = 0.5,
    conf_level: float = 0.95,
    baseline_label: str = "baseline",
) -> SaPerformance:
    """Score fitted classifications on held-out rows.

    Port of ``evaluate_classification_models()``. Predicts one or more fitted
    two-class models on the same rows and reports how each one discriminated,
    with three tests of each model against a baseline where more than one was
    passed. Every model is read through
    :meth:`~statassist.core.result.SaModel.predict` with ``type="response"``, so
    the fitting functions are interchangeable here.

    The rows are the intersection rather than the union: all three comparisons
    are **paired** statistics that have no meaning at all across different rows,
    so a row any model cannot predict is left out of all of them.

    The direction is the fits'. ``outcome_lv[1]`` is the class
    ``predict(type="response")`` reports the probability of, so it is the class
    every number here is about. ``outcome_lv`` and ``control_label`` are read as a
    statement to be checked rather than as an instruction, since a fitted model
    cannot be re-pointed after the fact.

    ``comparisons`` asks three different questions of the same pair of models,
    which is why all three are reported rather than one being chosen.
    ``delta_auc`` is whether the ranking improved, tested by DeLong's paired test,
    and it is blind to any change that does not reorder rows. ``idi`` is how much
    further apart the two classes' predicted probabilities moved, which is
    exactly the change an AUC does not see. ``nri`` is how often a probability
    moved the right way, counting direction only. Every one of them is
    ``new - baseline`` and positive for a new model that did better.

    Args:
        baseline_model: The reference model, fitted to a two-class outcome. It is
            the first row of every table and what ``comparisons`` is measured
            against.
        new_models: Mapping of name to further model to hold against it, or
            ``None`` to score the baseline on its own.
        newdata: The rows to score, typically the test half of a
            :func:`~statassist.split_data` result.
        answer: The observed classes, either the name of a column of ``newdata``
            or a vector with one entry per row. ``None`` reads the column the
            models were fitted to.
        outcome_lv: The two classes, reference first, checked against the order
            the models were fitted with rather than used to change it. ``None``
            takes theirs.
        control_label: The reference class on its own, checked the same way.
        threshold: Where to cut the predicted probability for ``accuracy``,
            ``sensitivity`` and ``specificity``. A row is called an event when
            its probability is greater than or equal to this.
        conf_level: Confidence level of every interval in the result.
        baseline_label: What to call the baseline in the tables and the legend.

    Returns:
        A :class:`~statassist.core.result.SaPerformance` whose ``analysis`` is
        ``"classification_performance"``.

    Examples:
        >>> from statassist import fit_logistic_regression, simulate_classification
        >>> sim = simulate_classification(n_samples=120, n_pred=3, seed=1)
        >>> frame = sim.args["data"]
        >>> train, test = frame.iloc[:90], frame.iloc[90:]
        >>> full = fit_logistic_regression(
        ...     train,
        ...     outcome=sim.args["outcome"],
        ...     outcome_lv=sim.args["outcome_lv"],
        ...     cv=False,
        ... )
        >>> res = evaluate_classification_models(full, newdata=test)
        >>> res["analysis"]
        'classification_performance'
        >>> bool(0 <= res["metrics"]["auc"].iloc[0] <= 1)
        True
        >>> sorted(res["curves"].columns.tolist())
        ['model', 'sensitivity', 'specificity', 'threshold']
    """
    threshold = check_scalar_num(threshold, "threshold", 0, 1)
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)

    newdata = evaluate_newdata(newdata)
    models = resolve_models(baseline_model, new_models, baseline_label)
    check_model_family(models, _WANTED_FAMILY, _OTHER_FUNCTION)
    check_model_agreement(models)

    model_lv = _evaluate_levels(
        [str(level) for level in models[baseline_label]["design"]["outcome_lv"]],
        outcome_lv,
        control_label,
    )

    resolved = resolve_answer(answer, newdata, models[baseline_label])
    response = _response_code(resolved.value, model_lv)

    collected = collect_predictions(models, newdata, response)
    response = response[collected.keep]
    predicted = collected.predicted

    n_events = int(np.sum(response == 1))
    if n_events == 0 or n_events == response.size:
        single = model_lv[0] if n_events == 0 else model_lv[1]
        raise SaValueError(
            f"the {response.size} scored row(s) hold a single class, {single}, so there "
            "is nothing to discriminate between. Both classes have to be present among "
            "the rows every model could predict."
        )
    z = float(stats.norm.ppf(1 - (1 - conf_level) / 2))

    names = list(models)
    metrics = pd.DataFrame(
        [
            {
                "model": name,
                **_scores(response, predicted[:, position], n_events, threshold, z),
            }
            for position, name in enumerate(names)
        ]
    )
    metrics["n_used"] = metrics["n_used"].astype(int)
    metrics["n_events"] = metrics["n_events"].astype(int)

    curves = pd.concat(
        [
            roc_points(response, predicted[:, position]).assign(model=name)
            for position, name in enumerate(names)
        ],
        ignore_index=True,
    )
    curves = curves[["model", "threshold", "sensitivity", "specificity"]]

    comparisons = _comparisons(response, predicted, names, z)

    return new_performance(
        analysis="classification_performance",
        models=names,
        design={
            "outcome": resolved.label,
            "outcome_type": _WANTED_FAMILY,
            "outcome_lv": model_lv,
            "baseline": baseline_label,
            "n_obs": collected.n_obs,
            "n_used": int(collected.keep.size),
            "n_dropped": collected.n_dropped,
            "n_events": n_events,
        },
        parameters={"threshold": threshold, "conf_level": conf_level},
        predictions=prediction_table(names, predicted, collected.keep, response),
        metrics=metrics,
        comparisons=comparisons,
        curves=curves,
    )


def _response_code(answer: Any, outcome_lv: list[str]) -> np.ndarray:
    """Read the observed classes as the event indicator the kernels take.

    Port of ``sa_response_code()``. The direction is the fit's, not this call's:
    ``outcome_lv[1]`` is the class ``predict(type="response")`` reports the
    probability of, so it is the class a 1 has to mean here for the two to line
    up. A row with no observed class is missing rather than a non-event, which is
    what lets :func:`~statassist.evaluate._shared.collect_predictions` count it
    as a row that cannot be scored.
    """
    labels = pd.Series(answer).reset_index(drop=True).astype("object")
    present = [str(value) for value in labels.dropna().unique()]
    extra = sorted(set(present) - set(outcome_lv))
    if extra:
        raise SaValueError(
            "`answer` holds class(es) the models were not fitted on: "
            + ", ".join(extra)
            + ". The models classify "
            + " and ".join(outcome_lv)
            + f". Reduce `newdata` to those {_N_CLASSES} classes first."
        )
    coded = np.where(
        labels.isna().to_numpy(),
        np.nan,
        (labels.astype(str) == outcome_lv[1]).to_numpy(dtype=float),
    )
    return np.asarray(coded, dtype=float)


def _evaluate_levels(
    model_lv: list[str],
    outcome_lv: Any,
    control_label: Any,
) -> list[str]:
    """Settle the class order, which the fits have already fixed.

    Port of ``sa_evaluate_levels()``. Both arguments are read as a statement to
    be checked rather than as an instruction: a fitted classification predicts
    the probability of one particular class and cannot be re-pointed after the
    fact, so naming the other one is a disagreement to report rather than a
    request to carry out.
    """
    if outcome_lv is not None:
        wanted = [str(level) for level in outcome_lv]
        if len(wanted) != _N_CLASSES or len(set(wanted)) != _N_CLASSES:
            raise SaValueError(
                f"`outcome_lv` must be {_N_CLASSES} distinct level names, the reference first."
            )
        if wanted != model_lv:
            raise SaValueError(
                "`outcome_lv` is "
                + ", ".join(wanted)
                + " but the models were fitted with "
                + ", ".join(model_lv)
                + ". A fitted classification predicts the probability of its own second "
                "level and cannot be re-pointed here; refit to change it."
            )
    if control_label is not None:
        if not isinstance(control_label, str) or not control_label:
            raise SaValueError(
                "`control_label` must be a single level name, the one the models hold "
                "as the reference."
            )
        if control_label != model_lv[0]:
            raise SaValueError(
                f"`control_label` is {control_label} but the models were fitted with "
                f"{model_lv[0]} as the reference. A fitted classification cannot be "
                "re-pointed here; refit to change it."
            )
    return model_lv


def _scores(
    response: np.ndarray,
    predicted: np.ndarray,
    n_events: int,
    threshold: float,
    z: float,
) -> dict[str, Any]:
    """Per-model scores, threshold-free first.

    The AUC interval is a Wald one on its own DeLong standard error, which is an
    interval on a bounded quantity and can therefore run past 1 for a strong
    classifier. That is what R reports and it is left as it is: clamping it would
    hide the fact that the normal approximation is what the interval rests on.
    """
    area = auc_delong(response, predicted)
    return {
        "n_used": int(response.size),
        "n_events": n_events,
        "auc": area["auc"],
        "auc_lower_conf": area["auc"] - z * area["se"],
        "auc_upper_conf": area["auc"] + z * area["se"],
        "brier": brier(response, predicted),
        **threshold_scores(response, predicted, threshold),
    }


def _comparisons(
    response: np.ndarray,
    predicted: np.ndarray,
    names: list[str],
    z: float,
) -> pd.DataFrame | None:
    """The three paired tests of each model against the baseline.

    ``None`` where there is nothing to compare, which is what leaves the slot out
    of the result rather than filling it with an empty table.
    """
    if len(names) < 2:
        return None
    baseline = predicted[:, 0]
    rows = []
    for position in range(1, len(names)):
        new = predicted[:, position]
        area = delong_test(response, new, baseline)
        discrimination = idi(response, baseline, new)
        reclassification = nri(response, baseline, new)
        rows.append(
            {
                "model": names[position],
                "delta_auc": area["delta"],
                "delta_auc_lower_conf": area["delta"] - z * area["se"],
                "delta_auc_upper_conf": area["delta"] + z * area["se"],
                "delta_auc_pval": area["pval"],
                "idi": discrimination["idi"],
                "idi_lower_conf": discrimination["idi"] - z * discrimination["se"],
                "idi_upper_conf": discrimination["idi"] + z * discrimination["se"],
                "idi_pval": discrimination["pval"],
                "nri": reclassification["nri"],
                "nri_event": reclassification["nri_event"],
                # The kernel calls the non-event half `nri_other`, since it
                # computes the same quantity for whichever group is not the one
                # being asked about. The contract names the group.
                "nri_nonevent": reclassification["nri_other"],
                "nri_lower_conf": reclassification["nri"] - z * reclassification["se"],
                "nri_upper_conf": reclassification["nri"] + z * reclassification["se"],
                "nri_pval": reclassification["pval"],
            }
        )
    return pd.DataFrame(rows)
