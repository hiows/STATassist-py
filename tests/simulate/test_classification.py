"""A two-class outcome whose coefficients are known.

The design arguments are the regression's and are tested there. What is new is
the outcome: the intercept is solved for rather than asked for, so the balance of
the data is what was requested; the class is a Bernoulli draw, so the label is
not a function of the predictors alone; and a subject is a case or a control as a
whole, which is what keeps the outcome usable as a stratifier of a split taken
over subjects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from statassist import simulate_classification, split_data
from statassist.core.errors import SaValueError

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_the_six_the_result_promises() -> None:
    sim = simulate_classification(seed=1)
    assert list(sim) == [
        "args",
        "split_args",
        "truth",
        "truth_term",
        "truth_model",
        "truth_row",
    ]


def test_args_carries_the_class_labels_as_well() -> None:
    """A fit left to sort them would put `case` first and report the wrong odds."""
    sim = simulate_classification(seed=1)
    assert list(sim.args) == ["data", "outcome", "predictors", "outcome_lv"]
    assert sim.args["outcome_lv"] == ["control", "case"]


def test_the_row_answer_holds_its_columns_in_order() -> None:
    sim = simulate_classification(seed=1)
    assert sim.truth_row.columns.tolist() == [
        "subject",
        "subject_offset",
        "eta",
        "prob",
        "draw_prob",
    ]


def test_the_model_answer_holds_its_keys_in_order_and_no_r_squared() -> None:
    """The outcome is a draw, so no share of its variance is recoverable."""
    model = simulate_classification(seed=1).truth_model
    assert list(model) == [
        "intercept",
        "event_rate",
        "achieved_event_rate",
        "signal_var",
        "subject_var",
        "n_samples",
        "n_subject",
        "subject_sd",
    ]


def test_the_outcome_is_a_string_and_not_a_category() -> None:
    """Only the factor predictors are categorical; the outcome is left as labels."""
    sim = simulate_classification(seed=1)
    column = sim.args["data"]["y"]
    assert not isinstance(column.dtype, pd.CategoricalDtype)
    assert all(isinstance(value, str) for value in column)
    assert isinstance(sim.args["data"]["x_cat_1"].dtype, pd.CategoricalDtype)


def test_the_data_holds_the_two_labels_that_were_asked_for() -> None:
    sim = simulate_classification(outcome_lv=["healthy", "sick"], seed=1)
    assert sim.args["outcome_lv"] == ["healthy", "sick"]
    assert set(sim.args["data"]["y"]) == {"healthy", "sick"}


def test_the_columns_are_the_outcome_then_the_predictors() -> None:
    sim = simulate_classification(n_pred=3, seed=1)
    assert sim.args["data"].columns.tolist() == ["y", "x_1", "x_2", "x_3", "x_cat_1"]
    assert sim.args["predictors"] == ["x_1", "x_2", "x_3", "x_cat_1"]


# --------------------------------------------------------------------------- #
# The balance that was asked for
# --------------------------------------------------------------------------- #


def test_the_intercept_is_solved_so_the_mean_probability_is_the_event_rate() -> None:
    for rate in (0.1, 0.3, 0.5, 0.85):
        sim = simulate_classification(n_samples=500, event_rate=rate, seed=1)
        assert sim.truth_row["prob"].mean() == pytest.approx(rate)


def test_the_solved_intercept_is_the_one_the_answer_reports() -> None:
    sim = simulate_classification(seed=1)
    assert sim.truth_model["intercept"] == sim.truth_term.iloc[0]["beta"]
    assert sim.truth_term.iloc[0]["terms"] == "(Intercept)"


def test_the_probability_is_the_logistic_transform_of_the_linear_predictor() -> None:
    sim = simulate_classification(seed=1)
    assert np.allclose(sim.truth_row["prob"], expit(sim.truth_row["eta"]))


def test_a_rarer_event_asks_for_a_lower_intercept() -> None:
    rare = simulate_classification(event_rate=0.1, seed=1).truth_model["intercept"]
    common = simulate_classification(event_rate=0.6, seed=1).truth_model["intercept"]
    assert rare < common


def test_the_draw_lands_near_the_rate_it_was_asked_for() -> None:
    """The draw is the noise, so the achieved rate is close rather than exact."""
    achieved = [
        simulate_classification(n_samples=1000, event_rate=0.3, seed=seed).truth_model[
            "achieved_event_rate"
        ]
        for seed in range(10)
    ]
    assert abs(float(np.mean(achieved)) - 0.3) < 0.02
    assert max(abs(value - 0.3) for value in achieved) < 0.06


def test_the_achieved_rate_is_the_share_of_the_second_label() -> None:
    sim = simulate_classification(n_samples=400, seed=1)
    observed = (sim.args["data"]["y"] == sim.args["outcome_lv"][1]).mean()
    assert sim.truth_model["achieved_event_rate"] == pytest.approx(observed)


# --------------------------------------------------------------------------- #
# What was planted, and which way it points
# --------------------------------------------------------------------------- #


def test_the_planted_counts_are_a_function_of_the_arguments() -> None:
    for seed in range(5):
        truth = simulate_classification(n_pred=10, n_pos=3, n_neg=2, seed=seed).truth
        assert truth["direction"].tolist().count("up") == 3
        assert truth["direction"].tolist().count("down") == 2


def test_a_null_predictor_carries_exactly_zero() -> None:
    truth = simulate_classification(n_pred=10, seed=1).truth
    assert (truth.loc[truth["role"] == "null", "beta"] == 0).all()


def test_a_positive_coefficient_raises_the_chance_of_the_second_label() -> None:
    """The second label is the class modelled, so this is the direction of every
    odds ratio a fit will report."""
    sim = simulate_classification(n_samples=4000, n_pred=6, seed=1)
    data = sim.args["data"]
    event = data["y"] == sim.args["outcome_lv"][1]
    for name, beta in zip(sim.truth["predictors"], sim.truth["beta"], strict=False):
        if not isinstance(beta, float) or beta == 0 or np.isnan(beta):
            continue
        gap = data.loc[event, name].mean() - data.loc[~event, name].mean()
        assert np.sign(gap) == np.sign(beta)


def test_the_planted_coefficients_are_recovered_on_the_log_odds_scale() -> None:
    """The recovery is loose on purpose: the draw is the noise here."""
    sim = simulate_classification(n_samples=6000, n_pred=4, n_factor_pred=0, event_rate=0.5, seed=1)
    beta = sim.truth.set_index("predictors")["beta"]
    event = (sim.args["data"]["y"] == sim.args["outcome_lv"][1]).to_numpy(dtype=float)
    fitted = _fit_logistic(sim.args["data"][beta.index].to_numpy(), event)
    assert np.abs(fitted[1:] - beta.to_numpy()).max() < 0.25
    assert fitted[0] == pytest.approx(sim.truth_model["intercept"], abs=0.25)


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Newton-Raphson on the log likelihood, so the test needs no model class."""
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        mu = expit(design @ beta)
        weight = mu * (1 - mu)
        step = np.linalg.solve(design.T @ (design * weight[:, None]), design.T @ (y - mu))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def test_the_factor_terms_are_the_levels_beyond_the_reference() -> None:
    sim = simulate_classification(n_pred=2, factor_lv=["low", "mid", "high"], seed=1)
    rows = sim.truth_term[sim.truth_term["predictors"] == "x_cat_1"]
    assert rows["terms"].tolist() == ["x_cat_1mid", "x_cat_1high"]


def test_stated_coefficients_are_used_as_they_stand() -> None:
    sim = simulate_classification(beta=[1.5, 0, -2], seed=1)
    numeric = sim.truth[sim.truth["role"].isin(["signal", "null"])]
    assert numeric["beta"].tolist() == [1.5, 0, -2]


# --------------------------------------------------------------------------- #
# Repeated measurements
# --------------------------------------------------------------------------- #


def test_a_subject_is_a_case_or_a_control_as_a_whole() -> None:
    sim = simulate_classification(n_per_subject=[3] * 40, seed=1)
    labels = sim.args["data"].groupby("subject", observed=True)["y"].nunique()
    assert labels.max() == 1


def test_the_draw_uses_the_mean_probability_of_the_subject_rows() -> None:
    sim = simulate_classification(n_per_subject=[3] * 40, seed=1)
    rows = sim.truth_row
    means = rows.groupby("subject", observed=True)["prob"].transform("mean")
    assert np.allclose(rows["draw_prob"], means)


def test_without_subjects_the_row_is_drawn_from_its_own_probability() -> None:
    sim = simulate_classification(seed=1)
    assert np.allclose(sim.truth_row["draw_prob"], sim.truth_row["prob"])
    assert sim.truth_row["subject"].isna().all()
    assert sim.truth_model["n_subject"] is None


def test_the_subject_offset_is_shared_by_every_row_of_that_subject() -> None:
    sim = simulate_classification(n_per_subject=[3] * 30, subject_sd=2, seed=1)
    grouped = sim.truth_row.groupby("subject", observed=True)["subject_offset"]
    assert (grouped.nunique() == 1).all()
    assert sim.truth_model["subject_var"] > 0
    assert sim.truth_model["subject_sd"] == 2


def test_a_single_count_is_spread_over_the_rows() -> None:
    sim = simulate_classification(n_samples=120, n_per_subject=3, seed=1)
    assert sim.truth_model["n_subject"] == 40
    assert sim.args["data"]["subject"].value_counts().unique().tolist() == [3]


# --------------------------------------------------------------------------- #
# The split arguments
# --------------------------------------------------------------------------- #


def test_the_outcome_stratifies_the_split_either_way() -> None:
    """Unlike a continuous outcome it is constant within a subject."""
    assert simulate_classification(seed=1).split_args["stratified"] == "y"
    repeated = simulate_classification(n_per_subject=[2] * 40, seed=1)
    assert repeated.split_args["stratified"] == "y"
    assert repeated.split_args["id"] == "subject"


def test_the_split_arguments_can_be_handed_straight_to_the_split() -> None:
    sim = simulate_classification(n_per_subject=[2] * 60, seed=1)
    split = split_data(**sim.split_args, seed=1)
    resample = split.datasets["Resample1"]
    assert not set(resample["train_data"]["subject"]) & set(resample["test_data"]["subject"])
    # The stratifier is the point: both sides carry events to fit and score on.
    for side in ("train_data", "test_data"):
        assert set(resample[side]["y"]) == set(sim.args["outcome_lv"])


def test_an_imbalanced_outcome_survives_the_split_it_asks_to_be_stratified_by() -> None:
    sim = simulate_classification(n_samples=200, event_rate=0.1, seed=1)
    split = split_data(**sim.split_args, seed=1)
    whole = (sim.args["data"]["y"] == "case").mean()
    for side in ("train_data", "test_data"):
        share = (split.datasets["Resample1"][side]["y"] == "case").mean()
        assert abs(share - whole) < 0.05


# --------------------------------------------------------------------------- #
# Reproducibility and refusals
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_data() -> None:
    first = simulate_classification(n_pred=4, seed=11)
    second = simulate_classification(n_pred=4, seed=11)
    assert first.args["data"].equals(second.args["data"])
    assert first.truth_model == second.truth_model


def test_a_different_seed_gives_different_data() -> None:
    first = simulate_classification(n_pred=4, seed=11)
    second = simulate_classification(n_pred=4, seed=12)
    assert not first.args["data"].equals(second.args["data"])


def test_a_certain_outcome_is_refused() -> None:
    for rate in (0, 1):
        with pytest.raises(SaValueError, match="`event_rate` must be in"):
            simulate_classification(event_rate=rate)


def test_two_labels_that_are_the_same_are_refused() -> None:
    with pytest.raises(SaValueError, match="two distinct non-missing class labels"):
        simulate_classification(outcome_lv=["case", "case"])


def test_one_label_is_refused() -> None:
    with pytest.raises(SaValueError, match="two distinct non-missing class labels"):
        simulate_classification(outcome_lv=["case"])


def test_stating_the_coefficients_and_the_counts_at_once_is_refused() -> None:
    with pytest.raises(SaValueError, match="`beta` states every coefficient"):
        simulate_classification(beta=[1, 0, 2], n_neg=1)


def test_a_rate_no_intercept_can_reach_is_refused() -> None:
    """A coefficient large enough to saturate the logistic on one row puts a rate
    below one row's worth out of reach, since that row is an event whatever the
    intercept is."""
    with pytest.raises(SaValueError, match="no intercept gives an event rate"):
        simulate_classification(n_samples=200, beta=[1e5], n_factor_pred=0, event_rate=1e-6, seed=1)
