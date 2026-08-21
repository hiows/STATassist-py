"""A regression whose coefficients are known.

Three claims carry the weight here. The outcome is exactly the linear predictor
plus the noise, so the planted answer is an answer rather than an approximation.
A predictor whose coefficient is zero contributes nothing at all, so a p-value
below the cutoff on one is a false positive by definition. And ``r_squared`` is
the share a model could reach, which means the subject offset counts against it
rather than for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statassist import make_block_cor, simulate_regression, split_data
from statassist.core.errors import SaValueError

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_the_six_the_result_promises() -> None:
    sim = simulate_regression(seed=1)
    assert list(sim) == [
        "args",
        "split_args",
        "truth",
        "truth_term",
        "truth_model",
        "truth_row",
    ]


def test_args_is_named_after_the_fit_that_consumes_it() -> None:
    sim = simulate_regression(seed=1)
    assert list(sim.args) == ["data", "outcome", "predictors"]
    assert sim.args["outcome"] == "y"


def test_the_two_truth_tables_hold_their_columns_in_order() -> None:
    sim = simulate_regression(seed=1)
    assert sim.truth.columns.tolist() == [
        "predictors",
        "role",
        "beta",
        "direction",
        "value_mean",
        "value_sd",
        "max_cor_signal",
    ]
    assert sim.truth_term.columns.tolist() == ["terms", "predictors", "beta"]
    assert sim.truth_row.columns.tolist() == ["subject", "subject_offset", "eta", "noise"]


def test_the_model_answer_holds_its_keys_in_order() -> None:
    assert list(simulate_regression(seed=1).truth_model) == [
        "intercept",
        "noise_sd",
        "signal_var",
        "subject_var",
        "r_squared",
        "n_samples",
        "n_subject",
        "subject_sd",
    ]


def test_the_columns_are_the_outcome_then_the_predictors() -> None:
    sim = simulate_regression(n_pred=3, n_constant_pred=2, seed=1)
    assert sim.args["data"].columns.tolist() == [
        "y",
        "x_1",
        "x_2",
        "x_3",
        "x_cat_1",
        "x_const_1",
        "x_const_2",
    ]
    assert sim.args["predictors"] == sim.args["data"].columns.tolist()[1:]


def test_the_predictor_prefix_names_every_generated_column() -> None:
    sim = simulate_regression(n_pred=2, n_constant_pred=1, pred_prefix="v", seed=1)
    assert sim.args["predictors"] == ["v_1", "v_2", "v_cat_1", "v_const_1"]


def test_truth_is_in_the_column_order_of_the_data() -> None:
    sim = simulate_regression(n_pred=4, n_constant_pred=1, seed=1)
    assert sim.truth["predictors"].tolist() == sim.args["predictors"]


def test_the_subject_column_is_kept_out_of_the_predictors() -> None:
    """A model told to fit on it would fit on which subject a row came from."""
    sim = simulate_regression(n_per_subject=[3] * 20, seed=1)
    assert "subject" in sim.args["data"].columns
    assert "subject" not in sim.args["predictors"]


def test_the_categorical_predictor_carries_its_levels_in_the_order_given() -> None:
    sim = simulate_regression(factor_lv=["a", "b", "c", "d"], seed=1)
    column = sim.args["data"]["x_cat_1"]
    assert isinstance(column.dtype, pd.CategoricalDtype)
    assert column.cat.categories.tolist() == ["a", "b", "c", "d"]


def test_a_constant_predictor_takes_a_single_value() -> None:
    sim = simulate_regression(n_constant_pred=2, seed=1)
    for name in ("x_const_1", "x_const_2"):
        assert sim.args["data"][name].nunique() == 1


# --------------------------------------------------------------------------- #
# The term axis
# --------------------------------------------------------------------------- #


def test_the_intercept_heads_the_term_answer() -> None:
    sim = simulate_regression(intercept=2.5, seed=1)
    head = sim.truth_term.iloc[0]
    assert head["terms"] == "(Intercept)"
    assert head["predictors"] is None
    assert head["beta"] == 2.5


def test_a_factor_becomes_one_term_per_level_beyond_the_reference() -> None:
    sim = simulate_regression(n_pred=2, factor_lv=["low", "mid", "high"], seed=1)
    rows = sim.truth_term[sim.truth_term["predictors"] == "x_cat_1"]
    assert rows["terms"].tolist() == ["x_cat_1mid", "x_cat_1high"]


def test_a_constant_predictor_becomes_no_term_at_all() -> None:
    sim = simulate_regression(n_constant_pred=2, seed=1)
    assert "x_const_1" in sim.truth["predictors"].tolist()
    assert "x_const_1" not in sim.truth_term["predictors"].tolist()


def test_the_numeric_coefficients_agree_across_the_two_axes() -> None:
    sim = simulate_regression(n_pred=6, seed=1)
    numeric = sim.truth[sim.truth["role"].isin(["signal", "null"])]
    terms = sim.truth_term.set_index("predictors")["beta"]
    assert np.allclose(numeric["beta"], terms[numeric["predictors"]])


# --------------------------------------------------------------------------- #
# What was planted
# --------------------------------------------------------------------------- #


def test_the_planted_counts_are_a_function_of_the_arguments() -> None:
    for seed in range(5):
        truth = simulate_regression(n_pred=10, n_pos=3, n_neg=2, seed=seed).truth
        assert truth["direction"].tolist().count("up") == 3
        assert truth["direction"].tolist().count("down") == 2
        assert int((truth["role"] == "null").sum()) == 5


def test_the_default_counts_are_a_share_of_the_predictors() -> None:
    truth = simulate_regression(n_pred=20, seed=1).truth
    assert int((truth["role"] == "signal").sum()) == 10


def test_a_null_predictor_carries_exactly_zero() -> None:
    truth = simulate_regression(n_pred=10, seed=1).truth
    assert (truth.loc[truth["role"] == "null", "beta"] == 0).all()


def test_the_planted_coefficient_points_the_way_the_direction_says() -> None:
    truth = simulate_regression(n_pred=12, seed=1).truth
    assert (truth.loc[truth["direction"] == "up", "beta"] > 0).all()
    assert (truth.loc[truth["direction"] == "down", "beta"] < 0).all()


def test_the_planted_magnitude_lies_in_the_range_it_was_drawn_from() -> None:
    truth = simulate_regression(n_pred=12, beta_range=(1.5, 2.0), seed=1).truth
    planted = truth.loc[truth["role"] == "signal", "beta"].abs()
    assert planted.between(1.5, 2.0).all()


def test_stated_coefficients_are_used_as_they_stand_and_say_how_many_there_are() -> None:
    sim = simulate_regression(beta=[1.5, 0, -2, 0.25], seed=1)
    numeric = sim.truth[sim.truth["role"].isin(["signal", "null"])]
    assert numeric["beta"].tolist() == [1.5, 0, -2, 0.25]
    assert numeric["predictors"].tolist() == ["x_1", "x_2", "x_3", "x_4"]


def test_a_predictor_is_drawn_where_it_was_asked_to_be() -> None:
    sim = simulate_regression(
        n_samples=4000, n_pred=2, value_mean=[3, -5], value_sd=[0.5, 2], seed=1
    )
    data = sim.args["data"]
    assert abs(data["x_1"].mean() - 3) < 0.1
    assert abs(data["x_2"].mean() + 5) < 0.2
    assert abs(data["x_1"].std() - 0.5) < 0.05
    assert abs(data["x_2"].std() - 2) < 0.1


def test_a_correlation_matrix_is_honoured_by_the_draw() -> None:
    cor_mat = make_block_cor(4, [{"features": [0, 1], "cor": 0.8}], default_cor=0.0)
    sim = simulate_regression(n_samples=5000, n_pred=4, cor_mat=cor_mat, seed=1)
    observed = sim.args["data"][["x_1", "x_2", "x_3", "x_4"]].corr().to_numpy()
    assert np.abs(observed - cor_mat).max() < 0.05


def test_the_correlation_with_a_planted_predictor_is_reported() -> None:
    """This column is why a null predictor can come back significant."""
    cor_mat = make_block_cor(4, [{"features": [0, 1], "cor": 0.9}], default_cor=0.0)
    sim = simulate_regression(n_pred=4, n_pos=1, n_neg=0, beta=None, cor_mat=cor_mat, seed=1)
    reported = sim.truth.set_index("predictors")["max_cor_signal"]
    signal = sim.truth.loc[sim.truth["role"] == "signal", "predictors"].tolist()
    if signal == ["x_1"]:
        assert reported["x_2"] == pytest.approx(0.9)
    assert reported[signal[0]] == 0


def test_the_factor_offsets_move_the_outcome_between_the_levels() -> None:
    sim = simulate_regression(
        n_samples=4000, n_pred=1, beta=[0], noise_sd=0.5, factor_lv=["a", "b"], seed=1
    )
    means = sim.args["data"].groupby("x_cat_1", observed=True)["y"].mean()
    offset = sim.truth_term.set_index("terms")["beta"]["x_cat_1b"]
    assert means["b"] - means["a"] == pytest.approx(offset, abs=0.1)


# --------------------------------------------------------------------------- #
# The outcome
# --------------------------------------------------------------------------- #


def test_the_outcome_is_exactly_the_linear_predictor_plus_the_noise() -> None:
    sim = simulate_regression(n_per_subject=[4] * 25, seed=1)
    rebuilt = sim.truth_row["eta"] + sim.truth_row["noise"]
    assert np.allclose(sim.args["data"]["y"], rebuilt)


def test_the_linear_predictor_is_rebuilt_from_the_coefficients_alone() -> None:
    """Nothing enters the outcome that the answer does not name."""
    sim = simulate_regression(n_pred=5, intercept=1.5, n_factor_pred=0, seed=1)
    data = sim.args["data"]
    beta = sim.truth.set_index("predictors")["beta"]
    eta = 1.5 + data[beta.index].to_numpy() @ beta.to_numpy()
    assert np.allclose(sim.truth_row["eta"], eta)


def test_the_noise_is_drawn_at_the_spread_that_was_asked_for() -> None:
    sim = simulate_regression(n_samples=4000, noise_sd=2.5, seed=1)
    assert sim.truth_row["noise"].std() == pytest.approx(2.5, abs=0.1)


def test_no_noise_leaves_the_outcome_at_the_linear_predictor() -> None:
    sim = simulate_regression(noise_sd=0, seed=1)
    assert (sim.truth_row["noise"] == 0).all()
    assert np.allclose(sim.args["data"]["y"], sim.truth_row["eta"])
    assert sim.truth_model["r_squared"] == 1


def test_the_planted_coefficients_are_recovered_by_least_squares() -> None:
    sim = simulate_regression(
        n_samples=4000, n_pred=6, n_factor_pred=0, noise_sd=1, intercept=2, seed=1
    )
    data = sim.args["data"]
    beta = sim.truth.set_index("predictors")["beta"]
    design = np.column_stack([np.ones(len(data.index)), data[beta.index].to_numpy()])
    fitted, *_ = np.linalg.lstsq(design, data["y"].to_numpy(), rcond=None)
    assert fitted[0] == pytest.approx(2, abs=0.15)
    assert np.abs(fitted[1:] - beta.to_numpy()).max() < 0.15


# --------------------------------------------------------------------------- #
# The share of the variance a model could reach
# --------------------------------------------------------------------------- #


def test_r_squared_is_the_signal_share_computed_with_the_sample_variance() -> None:
    sim = simulate_regression(seed=1)
    model = sim.truth_model
    signal = float(np.var(sim.truth_row["eta"] - sim.truth_row["subject_offset"], ddof=1))
    assert model["signal_var"] == pytest.approx(signal)
    assert model["r_squared"] == pytest.approx(
        signal / (signal + model["subject_var"] + model["noise_sd"] ** 2)
    )


def test_the_subject_offset_counts_against_the_share_rather_than_for_it() -> None:
    """No predictor accounts for it, so a model cannot reach it."""
    plain = simulate_regression(n_samples=120, subject_sd=4, seed=1)
    repeated = simulate_regression(n_per_subject=[4] * 30, subject_sd=4, seed=1)
    assert plain.truth_model["subject_var"] == 0
    assert repeated.truth_model["subject_var"] > 0
    assert repeated.truth_model["r_squared"] < plain.truth_model["r_squared"]


def test_more_noise_leaves_less_of_the_outcome_recoverable() -> None:
    quiet = simulate_regression(noise_sd=1, seed=1).truth_model["r_squared"]
    loud = simulate_regression(noise_sd=6, seed=1).truth_model["r_squared"]
    assert quiet > loud


# --------------------------------------------------------------------------- #
# Repeated measurements
# --------------------------------------------------------------------------- #


def test_a_single_count_is_spread_over_the_rows() -> None:
    sim = simulate_regression(n_samples=120, n_per_subject=4, seed=1)
    assert sim.truth_model["n_samples"] == 120
    assert sim.truth_model["n_subject"] == 30
    assert sim.args["data"]["subject"].value_counts().unique().tolist() == [4]


def test_explicit_counts_say_how_many_rows_there_are() -> None:
    sim = simulate_regression(n_per_subject=[2, 3, 5], seed=1)
    assert sim.truth_model["n_samples"] == 10
    assert sim.args["data"]["subject"].tolist() == (
        ["subject_1"] * 2 + ["subject_2"] * 3 + ["subject_3"] * 5
    )


def test_a_subject_offset_is_shared_by_every_row_of_that_subject() -> None:
    sim = simulate_regression(n_per_subject=[3] * 20, subject_sd=2, seed=1)
    grouped = sim.truth_row.groupby("subject", observed=True)["subject_offset"]
    assert (grouped.nunique() == 1).all()


def test_without_subjects_there_is_no_offset_and_no_column() -> None:
    sim = simulate_regression(seed=1)
    assert "subject" not in sim.args["data"].columns
    assert (sim.truth_row["subject_offset"] == 0).all()
    assert sim.truth_row["subject"].isna().all()
    assert sim.truth_model["n_subject"] is None
    assert sim.truth_model["subject_sd"] is None


def test_the_rows_of_a_subject_resemble_each_other_when_asked_to() -> None:
    """That resemblance is what a row-wise split gives away."""
    shared = simulate_regression(n_pred=1, n_per_subject=[2] * 400, subject_share=0.9, seed=1).args[
        "data"
    ]
    independent = simulate_regression(
        n_pred=1, n_per_subject=[2] * 400, subject_share=0.0, seed=1
    ).args["data"]
    assert _within_subject_cor(shared) > 0.8
    assert abs(_within_subject_cor(independent)) < 0.15


def test_the_column_looks_the_same_however_its_variance_is_split() -> None:
    for share in (0.0, 0.5, 1.0):
        column = simulate_regression(
            n_samples=4000, n_pred=1, n_per_subject=4, value_sd=2, subject_share=share, seed=1
        ).args["data"]["x_1"]
        assert column.std() == pytest.approx(2, abs=0.15)


def _within_subject_cor(data: pd.DataFrame) -> float:
    values = data["x_1"].to_numpy().reshape(-1, 2)
    return float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])


# --------------------------------------------------------------------------- #
# Missing cells
# --------------------------------------------------------------------------- #


def test_the_holes_are_drawn_at_the_proportion_that_was_asked_for() -> None:
    sim = simulate_regression(n_samples=200, n_pred=5, p_missing=0.1, seed=1)
    numeric = sim.args["data"][["x_1", "x_2", "x_3", "x_4", "x_5"]]
    assert int(numeric.isna().to_numpy().sum()) == 100


def test_only_the_numeric_predictors_are_holed() -> None:
    """A holed factor could not stratify a split and a holed constant is not constant."""
    sim = simulate_regression(n_constant_pred=1, p_missing=0.2, seed=1)
    assert not sim.args["data"]["x_cat_1"].isna().any()
    assert not sim.args["data"]["x_const_1"].isna().any()
    assert not sim.args["data"]["y"].isna().any()


def test_the_outcome_is_computed_before_the_holes_are_made() -> None:
    """Otherwise the answer would describe values the data no longer holds."""
    holed = simulate_regression(n_pred=4, p_missing=0.3, seed=1)
    whole = simulate_regression(n_pred=4, p_missing=0, seed=1)
    assert np.allclose(holed.truth_row["eta"], whole.truth_row["eta"])


# --------------------------------------------------------------------------- #
# The split arguments
# --------------------------------------------------------------------------- #


def test_the_split_arguments_are_named_after_the_split() -> None:
    sim = simulate_regression(seed=1)
    assert list(sim.split_args) == ["data", "stratified", "id"]
    assert sim.split_args["data"] is sim.args["data"]


def test_the_outcome_stratifies_a_split_over_rows() -> None:
    sim = simulate_regression(seed=1)
    assert sim.split_args["stratified"] == "y"
    assert sim.split_args["id"] is None


def test_a_continuous_outcome_cannot_stratify_a_split_over_subjects() -> None:
    """It varies within a subject, so the subject-level factor is used instead."""
    sim = simulate_regression(n_per_subject=[3] * 20, seed=1)
    assert sim.split_args["stratified"] == "x_cat_1"
    assert sim.split_args["id"] == "subject"


def test_with_no_factor_to_fall_back_on_the_split_is_left_unstratified() -> None:
    sim = simulate_regression(n_per_subject=[3] * 20, n_factor_pred=0, seed=1)
    assert sim.split_args["stratified"] is None


def test_the_split_arguments_can_be_handed_straight_to_the_split() -> None:
    sim = simulate_regression(n_per_subject=[3] * 20, seed=1)
    split = split_data(**sim.split_args, seed=1)
    resample = split.datasets["Resample1"]
    assert not set(resample["train_data"]["subject"]) & set(resample["test_data"]["subject"])


# --------------------------------------------------------------------------- #
# Reproducibility and refusals
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_data() -> None:
    first = simulate_regression(n_pred=4, seed=7)
    second = simulate_regression(n_pred=4, seed=7)
    assert first.args["data"].equals(second.args["data"])
    assert first.truth.equals(second.truth)


def test_a_different_seed_gives_different_data() -> None:
    first = simulate_regression(n_pred=4, seed=7)
    second = simulate_regression(n_pred=4, seed=8)
    assert not first.args["data"].equals(second.args["data"])


def test_stating_the_coefficients_and_the_counts_at_once_is_refused() -> None:
    with pytest.raises(SaValueError, match="`beta` states every coefficient"):
        simulate_regression(beta=[1, 0, 2], n_pos=1)


def test_a_coefficient_count_that_disagrees_with_the_predictor_count_is_refused() -> None:
    with pytest.raises(SaValueError, match="`beta` gives 3 coefficient"):
        simulate_regression(n_pred=5, beta=[1, 0, 2])


def test_more_planted_predictors_than_predictors_is_refused() -> None:
    with pytest.raises(SaValueError, match="more coefficients than the 4"):
        simulate_regression(n_pred=4, n_pos=3, n_neg=2)


def test_a_row_count_a_subject_count_cannot_divide_is_refused() -> None:
    with pytest.raises(SaValueError, match="does not divide"):
        simulate_regression(n_samples=100, n_per_subject=3)


def test_a_negative_noise_spread_is_refused() -> None:
    with pytest.raises(SaValueError, match="`noise_sd` must be in"):
        simulate_regression(noise_sd=-1)


def test_a_share_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(SaValueError, match="`subject_share` must be in"):
        simulate_regression(n_per_subject=3, n_samples=120, subject_share=1.5)


def test_a_proportion_of_holes_above_one_is_refused() -> None:
    with pytest.raises(SaValueError, match="`p_missing` must be in"):
        simulate_regression(p_missing=1.5)


def test_fewer_than_two_factor_levels_is_refused() -> None:
    with pytest.raises(SaValueError, match="`factor_lv`"):
        simulate_regression(factor_lv=["only"])


def test_a_correlation_matrix_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(SaValueError, match="`cor_mat`"):
        simulate_regression(n_pred=5, cor_mat=make_block_cor(3))
