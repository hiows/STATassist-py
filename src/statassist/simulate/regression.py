"""A regression whose coefficients are known.

The port of ``R/simulate_regression.R``. The expression simulators plant an
effect in some features and leave the rest strictly null, so that both kinds of
mistake a comparison can make are defined. A regression is scored the same way,
on the axis it has: a coefficient of exactly zero is a predictor that a p-value
below the cutoff is wrong about.

What is new here is that the two mistakes are not independent. A null predictor
correlated with a planted one carries real information about the outcome, so its
estimate is pulled away from the zero it truly has, and no amount of data fixes
it. ``cor_mat`` is how that is put into the data on purpose rather than met by
accident.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.random import SaRandom
from ..core.result import SaSimulation
from ..core.validate import UNSET, check_scalar_num
from ._supervised import add_intercept, split_args, supervised_design

__all__ = ["simulate_regression"]

#: Rows generated when the caller names no count.
_DEFAULT_N_SAMPLES = 200
#: Numeric predictors generated when the caller names no count.
_DEFAULT_N_PRED = 8


def simulate_regression(
    n_samples: Any = UNSET,
    n_pred: Any = UNSET,
    n_pos: int | None = None,
    n_neg: int | None = None,
    beta: Any = None,
    beta_range: tuple[float, float] = (0.5, 2),
    intercept: float = 0,
    value_mean: Any = 0,
    value_sd: Any = 1,
    noise_sd: float = 3,
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
    """Simulate a regression whose coefficients are known.

    Generates a continuous outcome from a linear combination of predictors and
    returns the coefficients that were planted alongside the data, so that a
    fitted model can be scored against what was actually there. Everything a
    model has to survive can be asked for: predictors that correlate, a
    categorical predictor, a predictor that takes one value, missing cells, and
    repeated measurements of the same subject.

    The point of the exercise is the gap between the coefficient table and the
    truth, and the three things that open it. A planted coefficient can be too
    small for the noise to let it through, a null predictor correlated with a
    planted one is estimated away from zero however much data there is, and a
    subject measured repeatedly makes a row-wise split score the model on rows it
    half knows already.

    Each row is drawn as
    ``y = intercept + sum(beta * x) + factor offsets + subject offset + noise``,
    with the numeric predictors drawn from a multivariate normal whose
    correlations are ``cor_mat`` and the noise normal with standard deviation
    ``noise_sd``. Because the outcome is built from the coefficients and nothing
    else, a predictor whose coefficient is zero is null in the strict sense, and
    a p-value below the cutoff on one is a false positive by definition.

    Which predictors carry a planted coefficient is drawn at random, but how many
    carry a positive one and how many a negative one is not: ``n_pos`` and
    ``n_neg`` are counts, so they do not move with the seed. ``beta`` states
    every coefficient instead, in which case nothing is planted and its length is
    how many numeric predictors there are.

    Args:
        n_samples: Rows to generate. Ignored when ``n_per_subject`` gives a row
            count per subject, since those already say how many rows there are.
        n_pred: Number of numeric predictors. Columns are named ``x_1`` upwards,
            or whatever ``pred_prefix`` asks for.
        n_pos: How many numeric predictors are given a positive coefficient.
        n_neg: How many are given a negative one. Their sum cannot exceed
            ``n_pred``, and every other numeric predictor is left with a
            coefficient of exactly zero. ``None`` takes a fraction of ``n_pred``
            rather than a fixed count, so that asking for fewer predictors plants
            fewer coefficients instead of failing.
        beta: The coefficients themselves, one per numeric predictor and no
            intercept among them, or ``None`` to plant ``n_pos`` and ``n_neg`` of
            them. Its length is then how many numeric predictors there are, so
            ``n_pred`` need not be given as well; naming both and disagreeing is
            an error rather than a guess. Supplying it together with ``n_pos`` or
            ``n_neg`` is refused, since the two are different ways of saying the
            same thing.
        beta_range: Range the magnitude of a planted coefficient is drawn from.
            The offsets of the factor predictors are drawn from it too, so it is
            read whether or not ``beta`` was given.
        intercept: The intercept. It is not part of ``beta``.
        value_mean: Mean of each numeric predictor, given once for all of them or
            once each.
        value_sd: Standard deviation of each numeric predictor, the same way. A
            coefficient is a change in the outcome per unit of its predictor, so
            these two fix what ``beta_range`` means; that is why they are given
            rather than drawn from a range as the expression simulators draw
            their spreads.
        noise_sd: Standard deviation of the residual noise. This and
            ``beta_range`` together decide how much of the outcome is recoverable
            at all; ``truth_model["r_squared"]`` reports how much, on the design
            that was drawn.
        cor_mat: Correlation matrix of the numeric predictors, as built by
            :func:`make_block_cor`, or ``None`` to leave them independent. A null
            predictor correlated with a planted one is the case a coefficient
            table gets wrong no matter how many rows it is given.
        n_factor_pred: Number of categorical predictors, named ``x_cat_1``
            upwards. Each becomes ``len(factor_lv) - 1`` terms in the model
            rather than one, which is why ``truth_term`` exists beside ``truth``.
            Levels are handed out in balanced counts, and each level beyond the
            first carries a planted offset.
        factor_lv: Levels of each categorical predictor, the first being the
            reference that carries no offset.
        n_constant_pred: Number of predictors that take a single value, named
            ``x_const_1`` upwards. They cannot contribute, so a fit leaves them
            out and names them among its dropped predictors. The default plants
            none.
        p_missing: Proportion of numeric predictor cells to replace with a
            missing value, drawn after the outcome has been computed from the
            complete values. The rows they fall in are the ones a model drops
            before its folds are laid out. Only the numeric predictors are holed,
            since a hole in the categorical one would stop it being able to
            stratify a split and a hole in a constant one would stop it being
            constant.
        n_per_subject: Rows measured on each subject, one entry per subject, so
            that its length is how many subjects there are. A single number is
            spread over ``n_samples`` rows and must divide them. ``None``, the
            default, gives no ``subject`` column and one row per sampling unit.
            This is what :func:`split_data`'s ``id`` argument exists for: a
            subject partly seen in training is partly known before its test rows
            are read.
        subject_sd: Standard deviation of the per-subject offset on the outcome,
            which is variation no predictor accounts for. Ignored without
            ``n_per_subject``.
        subject_share: Share of each numeric predictor's variance that lies
            between subjects rather than within one, so its intraclass
            correlation. This is what makes two rows of one subject resemble each
            other, and it is that resemblance a row-wise split gives away: at
            ``0`` the rows of a subject are independent draws that happen to
            share an outcome offset, and no model could tell one subject's rows
            from another's. The distribution of each column is the same whatever
            it is set to. Ignored without ``n_per_subject``.
        pred_prefix: Prefix for the generated predictor names. ``"x"`` gives
            ``x_1``, ``x_cat_1`` and ``x_const_1``.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy.

    Returns:
        A :class:`~statassist.core.SaSimulation` of six slots.

        * ``args`` - ``data``, ``outcome`` and ``predictors``, named after the
          arguments of the linear fit so that
          ``fit_linear_regression(**sim.args)`` fits the model. ``predictors`` is
          given explicitly rather than left to its default, which would take the
          ``subject`` column as a predictor and let the model fit on which
          subject a row came from.
        * ``split_args`` - ``data``, ``stratified`` and ``id``, named after the
          arguments of :func:`split_data`. The outcome is the stratifier when
          there are no subjects; with subjects it varies within a subject and so
          cannot stratify a split taken over them, and the first categorical
          predictor, which is drawn per subject, is used instead.
        * ``truth`` - one row per predictor, in the column order of ``data``,
          holding ``predictors``, ``role`` (``"signal"``, ``"null"``,
          ``"factor"`` or ``"constant"``), ``beta``, ``direction``,
          ``value_mean``, ``value_sd`` and ``max_cor_signal``, the largest
          correlation this predictor has with a planted one. The last is what
          accounts for a null predictor that came back significant.
        * ``truth_term`` - one row per model term, in the row order a coefficient
          table follows, holding ``terms``, the ``predictors`` each term came
          from, and ``beta``. This is the table that scores the coefficients,
          since a categorical predictor is several terms and a constant one is
          none.
        * ``truth_model`` - the model as a whole: ``intercept``, ``noise_sd``,
          ``signal_var``, ``subject_var`` and ``r_squared``, the share of the
          variance of the outcome the predictors account for. Also
          ``n_samples``, ``n_subject`` and ``subject_sd``.
        * ``truth_row`` - one row per observation, holding ``subject``,
          ``subject_offset``, ``eta`` (the whole linear predictor, intercept
          included) and ``noise``, so that ``y`` is exactly ``eta + noise``.

    Raises:
        SaValueError: If ``beta`` is given together with ``n_pos`` or ``n_neg``,
            if a count, range or proportion is unusable, or if
            ``n_per_subject`` does not divide ``n_samples``.

    Examples:
        Four of the eight numeric predictors carry a coefficient and the rest are
        null in the strict sense, so both kinds of mistake are defined.

        >>> sim = simulate_regression(seed=1)
        >>> {r: int((sim.truth["role"] == r).sum()) for r in ("signal", "null", "factor")}
        {'signal': 4, 'null': 4, 'factor': 1}
        >>> list(sim.args)
        ['data', 'outcome', 'predictors']
        >>> sim.truth_term["terms"].tolist()[:3]
        ['(Intercept)', 'x_1', 'x_2']

        The outcome is exactly the linear predictor plus the noise, which is what
        makes the planted answer an answer.

        >>> y = sim.args["data"]["y"].to_numpy()
        >>> eta = sim.truth_row["eta"].to_numpy() + sim.truth_row["noise"].to_numpy()
        >>> bool(abs(y - eta).max() < 1e-12)
        True

        Repeated measurements name the subject column, so the split can be taken
        over subjects rather than over rows.

        >>> rep_sim = simulate_regression(n_per_subject=[3] * 40, seed=3)
        >>> rep_sim.split_args["id"], rep_sim.split_args["stratified"]
        ('subject', 'x_cat_1')
    """
    intercept = check_scalar_num(intercept, "intercept")
    noise_sd = check_scalar_num(noise_sd, "noise_sd", 0)
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

    eta = intercept + design.eta
    noise = rng.normal(0, noise_sd, design.n_samples)

    data = pd.concat([pd.DataFrame({"y": eta + noise}), design.x], axis=1)
    if design.subject is not None:
        data["subject"] = design.subject

    # The subject offset is variance the predictors cannot account for, so it sits
    # with the noise in the denominator rather than with the signal. Counting it
    # as signal would make `r_squared` a number no model could reach. `ddof=1` is
    # R's `var()`; NumPy's default of 0 would give a slightly different share.
    signal_var = float(np.var(design.eta - design.subject_offset, ddof=1))
    subject_var = 0.0 if design.sizes is None else float(np.var(design.subject_offset, ddof=1))

    return SaSimulation(
        {
            "args": {
                "data": data,
                "outcome": "y",
                "predictors": design.predictors,
            },
            "split_args": split_args(data, design, stratify_outcome=False),
            "truth": design.truth,
            "truth_term": add_intercept(design.truth_term, intercept),
            "truth_model": {
                "intercept": intercept,
                "noise_sd": noise_sd,
                "signal_var": signal_var,
                "subject_var": subject_var,
                "r_squared": signal_var / (signal_var + subject_var + noise_sd**2),
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
                    "noise": noise,
                }
            ),
        }
    )
