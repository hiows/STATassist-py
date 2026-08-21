"""A two-class outcome whose coefficients are known.

The port of ``R/simulate_classification.R``. The two supervised simulators are
deliberately near-identical below the documentation: the same predictors, the
same planted coefficients, the same subjects. What is specific to this one is
that the outcome is a draw rather than a sum, and the two consequences of that.

The first is the intercept. In a regression it is a number the caller picks and
nothing depends on it; here it is what decides how many events there are, and how
many events there are is what decides whether a split has to be stratified. So it
is not asked for. ``event_rate`` is asked for and the intercept is solved from it.

The second is that a class is drawn once per subject rather than once per row. A
subject is a case or a control as a whole, which is the shape of the real design
this stands in for and also the reason a row-wise split of it is worthless: the
same subject's other rows carry its label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from ..core.errors import SaValueError
from ..core.random import SaRandom
from ..core.result import SaSimulation
from ..core.validate import UNSET, check_scalar_num
from ._supervised import add_intercept, solve_intercept, split_args, supervised_design

__all__ = ["simulate_classification"]

#: Rows generated when the caller names no count.
_DEFAULT_N_SAMPLES = 200
#: Numeric predictors generated when the caller names no count.
_DEFAULT_N_PRED = 8


def simulate_classification(
    n_samples: Any = UNSET,
    n_pred: Any = UNSET,
    n_pos: int | None = None,
    n_neg: int | None = None,
    beta: Any = None,
    beta_range: tuple[float, float] = (0.5, 2),
    event_rate: float = 0.3,
    outcome_lv: Sequence[str] = ("control", "case"),
    value_mean: Any = 0,
    value_sd: Any = 1,
    cor_mat: Any = None,
    n_factor_pred: int = 1,
    factor_lv: Any = ("low", "mid", "high"),
    n_constant_pred: int = 0,
    p_missing: float = 0,
    n_per_subject: Any = None,
    subject_sd: float = 1,
    subject_share: float = 0.5,
    pred_prefix: str = "x",
    seed: int | None = None,
) -> SaSimulation:
    """Simulate a two-class outcome whose coefficients are known.

    Generates a two-class outcome from a logistic model of its predictors and
    returns the coefficients that were planted alongside the data, so that a
    fitted model can be scored against what was actually there. The same design
    arguments :func:`simulate_regression` takes are available, together with the
    class balance, which is the reason a split of a classification has to be
    stratified.

    A classification differs from a regression in what can go wrong with it, and
    the defaults are set so that all of it can be seen. Classes are imbalanced,
    so an unstratified split can hand a fold too few events to fit on. A subject
    is a case or a control as a whole, so a split that does not respect ``id``
    scores the model on rows whose label it already holds. And a predictor with a
    coefficient of exactly zero is null in the strict sense, so an odds ratio
    away from 1 on one is a mistake by definition rather than by judgement.

    The linear predictor is
    ``intercept + sum(beta * x) + factor offsets + subject offset``, the class
    probability is its logistic transform, and the class is a Bernoulli draw from
    that probability. There is no noise argument: the draw is the noise, which is
    why a logistic regression recovers less from the same number of rows than a
    linear one does.

    The intercept is not an argument. It is solved for so that the mean class
    probability over the rows that were actually drawn equals ``event_rate``,
    which means the balance of the data is what was asked for rather than
    whatever the coefficients happened to imply. ``truth_model["intercept"]``
    reports the value it took and ``truth_model["achieved_event_rate"]`` the
    proportion the Bernoulli draw then produced.

    ``outcome_lv`` fixes the direction by the rule the rest of the package
    follows: the first level is the reference, so a planted positive coefficient
    raises the chance of ``outcome_lv[1]`` - the second label, zero-based here
    where R writes ``outcome_lv[2]`` - and its odds ratio comes back above 1. It
    is carried in ``args`` rather than left out, because a fit sorts the classes
    when it is not told them, and sorting ``case`` and ``control`` puts ``case``
    first, which would report the odds of the wrong class.

    Args:
        n_samples: Rows to generate. Ignored when ``n_per_subject`` gives a row
            count per subject.
        n_pred: Number of numeric predictors. Columns are named ``x_1`` upwards,
            or whatever ``pred_prefix`` asks for.
        n_pos: How many numeric predictors are given a positive coefficient.
        n_neg: How many are given a negative one. ``None`` takes a fraction of
            ``n_pred`` rather than a fixed count.
        beta: The coefficients themselves, one per numeric predictor, or ``None``
            to plant ``n_pos`` and ``n_neg`` of them. Supplying it together with
            either count is refused.
        beta_range: Range the magnitude of a planted coefficient is drawn from,
            on the log odds scale. The factor offsets are drawn from it too.
        event_rate: Proportion of rows in the second entry of ``outcome_lv``, the
            class being modelled. The default is deliberately away from a half,
            since a balanced outcome makes stratification look unnecessary.
        outcome_lv: The two class labels, the reference first, so that the
            coefficients describe the odds of the second one.
        value_mean: Mean of each numeric predictor, once for all or once each.
        value_sd: Standard deviation of each numeric predictor, the same way.
        cor_mat: Correlation matrix of the numeric predictors, as built by
            :func:`make_block_cor`, or ``None`` to leave them independent.
        n_factor_pred: Number of categorical predictors, named ``x_cat_1``
            upwards.
        factor_lv: Levels of each categorical predictor, the reference first.
        n_constant_pred: Number of predictors that take a single value.
        p_missing: Proportion of numeric predictor cells to replace with a
            missing value, drawn after the outcome has been computed.
        n_per_subject: Rows measured on each subject, one entry per subject.
            ``None``, the default, gives no ``subject`` column. With subjects the
            class is drawn once per subject from the mean of its rows'
            probabilities, so a subject is a case or a control as a whole and the
            outcome can still stratify a split taken over subjects.
        subject_sd: Standard deviation of the per-subject offset on the linear
            predictor. Ignored without ``n_per_subject``.
        subject_share: Share of each numeric predictor's variance that lies
            between subjects rather than within one. Ignored without
            ``n_per_subject``.
        pred_prefix: Prefix for the generated predictor names.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy.

    Returns:
        A :class:`~statassist.core.SaSimulation` of six slots, the same shape
        :func:`simulate_regression` returns, with these differences.

        * ``args`` also carries ``outcome_lv``, since the direction of every
          coefficient depends on it and the default would sort the labels the
          other way round.
        * ``split_args`` has the outcome as ``stratified`` always. Unlike a
          continuous one it is constant within a subject, so it stratifies a
          split over subjects as readily as one over rows.
        * ``truth_model`` holds ``intercept`` as solved, the ``event_rate`` asked
          for and the ``achieved_event_rate`` the draw produced, ``signal_var``,
          ``subject_var``, ``n_samples``, ``n_subject`` and ``subject_sd``. There
          is no ``r_squared``: the outcome is a draw, so no share of its variance
          is recoverable in that sense.
        * ``truth_row`` holds ``prob``, the class probability of the row, and
          ``draw_prob``, the probability the Bernoulli draw actually used, which
          is the subject's mean when there are subjects and ``prob`` itself when
          there are not.

    Raises:
        SaValueError: If ``event_rate`` is not strictly between 0 and 1, if
            ``outcome_lv`` is not two distinct labels, if an argument of the
            design is unusable, or if no intercept can reach ``event_rate`` on
            the predictors that were drawn.

    Examples:
        The intercept was not asked for, it was solved for, so the balance of the
        data is the balance that was requested.

        >>> sim = simulate_classification(seed=1)
        >>> round(sim.truth_model["achieved_event_rate"], 2)
        0.28
        >>> sorted(set(sim.args["data"]["y"]))
        ['case', 'control']
        >>> list(sim.args)
        ['data', 'outcome', 'predictors', 'outcome_lv']

        A subject is a case or a control as a whole, so the outcome can still be
        the stratifier of a split over subjects.

        >>> rep_sim = simulate_classification(n_per_subject=[2] * 100, seed=2)
        >>> rep_sim.split_args["stratified"], rep_sim.split_args["id"]
        ('y', 'subject')
        >>> labels = rep_sim.args["data"].groupby("subject", observed=True)["y"].nunique()
        >>> int(labels.max())
        1
    """
    event_rate = check_scalar_num(event_rate, "event_rate", 0, 1, lower_open=True, upper_open=True)
    labels = (
        list(outcome_lv)
        if isinstance(outcome_lv, Sequence) and not isinstance(outcome_lv, str)
        else []
    )
    if (
        len(labels) != 2
        or not all(isinstance(label, str) for label in labels)
        or labels[0] == labels[1]
    ):
        raise SaValueError(
            "`outcome_lv` must be two distinct non-missing class labels, the reference first."
        )
    labels = [str(label) for label in labels]
    explicit = [
        name
        for name, given in (
            ("n_pred", n_pred is not UNSET),
            ("n_pos", n_pos is not None),
            ("n_neg", n_neg is not None),
        )
        if given
    ]

    rng = SaRandom(seed).rng

    design = supervised_design(
        n_samples=_DEFAULT_N_SAMPLES if n_samples is UNSET else n_samples,
        n_pred=_DEFAULT_N_PRED if n_pred is UNSET else n_pred,
        beta=beta,
        n_pos=n_pos,
        n_neg=n_neg,
        beta_range=beta_range,
        value_mean=value_mean,
        value_sd=value_sd,
        cor_mat=cor_mat,
        n_factor_pred=n_factor_pred,
        factor_lv=factor_lv,
        n_constant_pred=n_constant_pred,
        p_missing=p_missing,
        n_per_subject=n_per_subject,
        subject_sd=subject_sd,
        subject_share=subject_share,
        pred_prefix=pred_prefix,
        explicit=explicit,
        use_default_n=n_samples is UNSET,
        rng=rng,
    )

    intercept = solve_intercept(design.eta, event_rate)
    eta = intercept + design.eta
    prob = expit(eta)

    if design.sizes is None:
        draw_prob = prob
        event = rng.binomial(1, prob, design.n_samples)
    else:
        # One draw per subject, from the mean of the probabilities of its rows. A
        # draw per row would let a subject be a case in one sample and a control
        # in the next, which is not the design `id` exists to protect.
        edges = np.cumsum([0] + design.sizes)
        unit_prob = np.array(
            [prob[start:stop].mean() for start, stop in zip(edges[:-1], edges[1:], strict=True)]
        )
        draw_prob = np.repeat(unit_prob, design.sizes)
        event = np.repeat(rng.binomial(1, unit_prob, unit_prob.size), design.sizes)

    data = pd.concat([pd.DataFrame({"y": [labels[value] for value in event]}), design.x], axis=1)
    if design.subject is not None:
        data["subject"] = design.subject

    return SaSimulation(
        {
            "args": {
                "data": data,
                "outcome": "y",
                "predictors": design.predictors,
                "outcome_lv": labels,
            },
            "split_args": split_args(data, design, stratify_outcome=True),
            "truth": design.truth,
            "truth_term": add_intercept(design.truth_term, intercept),
            "truth_model": {
                "intercept": intercept,
                "event_rate": event_rate,
                "achieved_event_rate": float(np.mean(event)),
                # `ddof=1` is R's `var()`; NumPy's default of 0 would divide by
                # the row count instead.
                "signal_var": float(np.var(design.eta - design.subject_offset, ddof=1)),
                "subject_var": 0.0
                if design.sizes is None
                else float(np.var(design.subject_offset, ddof=1)),
                "n_samples": design.n_samples,
                "n_subject": None if design.sizes is None else len(design.sizes),
                "subject_sd": None if design.sizes is None else subject_sd,
            },
            "truth_row": pd.DataFrame(
                {
                    "subject": [None] * design.n_samples
                    if design.subject is None
                    else design.subject,
                    "subject_offset": design.subject_offset,
                    "eta": eta,
                    "prob": prob,
                    "draw_prob": draw_prob,
                }
            ),
        }
    )
