"""The helpers the two supervised simulators share.

The numerical claims here are the ones a wrong port would break silently rather
than loudly: the Cholesky factor being the upper one, so that the draw has the
covariance that was asked for, and the intercept being solved on the linear
predictor that was actually drawn.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from statassist.core.errors import SaValueError
from statassist.simulate._supervised import (
    PredSpec,
    balanced_levels,
    chol_or_none,
    cor_root,
    factor_offsets,
    mask_missing,
    mvnorm,
    plant_beta,
    pred_spec,
    recycle,
    solve_intercept,
    subject_sizes,
    truth_term,
)

# --------------------------------------------------------------------------- #
# The correlated draw
# --------------------------------------------------------------------------- #


def test_the_factor_is_the_upper_one() -> None:
    """R's ``chol()`` is upper and NumPy's is lower, and ``mvnorm`` needs the upper."""
    cor_mat = np.array([[1.0, 0.5], [0.5, 1.0]])
    root = chol_or_none(cor_mat)
    assert root is not None
    assert np.allclose(root, np.triu(root))
    assert np.allclose(root.T @ root, cor_mat)


def test_a_matrix_no_data_could_have_comes_back_as_none() -> None:
    """Three predictors each correlated 0.9 with the other two."""
    cor_mat = np.full((3, 3), 0.9)
    np.fill_diagonal(cor_mat, 1.0)
    cor_mat[0, 1] = cor_mat[1, 0] = -0.9
    assert chol_or_none(cor_mat) is None


def test_no_correlation_matrix_means_independence() -> None:
    assert cor_root(None, 3).tolist() == np.eye(3).tolist()


def test_the_draw_reproduces_the_correlations_it_was_given() -> None:
    cor_mat = np.array([[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]])
    root = cor_root(cor_mat, 3)
    drawn = mvnorm(200_000, np.zeros(3), np.ones(3), root, np.random.default_rng(1))
    assert np.allclose(np.corrcoef(drawn, rowvar=False), cor_mat, atol=0.01)


def test_the_draw_scales_and_shifts_each_column_on_its_own() -> None:
    drawn = mvnorm(
        200_000,
        np.array([5.0, -2.0]),
        np.array([2.0, 0.5]),
        np.eye(2),
        np.random.default_rng(1),
    )
    assert np.allclose(drawn.mean(axis=0), [5.0, -2.0], atol=0.05)
    assert np.allclose(drawn.std(axis=0, ddof=1), [2.0, 0.5], atol=0.05)


def test_an_asymmetric_correlation_matrix_is_refused() -> None:
    cor_mat = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(SaValueError, match="must be symmetric"):
        cor_root(cor_mat, 2)


def test_a_diagonal_that_is_not_one_is_refused() -> None:
    with pytest.raises(SaValueError, match="must have 1 on its diagonal"):
        cor_root(np.array([[0.9, 0.5], [0.5, 0.9]]), 2)


def test_a_correlation_matrix_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(SaValueError, match="one row and column per numeric predictor"):
        cor_root(np.eye(3), 2)


def test_a_correlation_matrix_no_data_could_have_is_refused_by_name() -> None:
    cor_mat = np.full((3, 3), 0.9)
    np.fill_diagonal(cor_mat, 1.0)
    cor_mat[0, 1] = cor_mat[1, 0] = -0.9
    with pytest.raises(SaValueError, match="not positive definite"):
        cor_root(cor_mat, 3)


# --------------------------------------------------------------------------- #
# recycle
# --------------------------------------------------------------------------- #


def test_one_value_covers_every_predictor() -> None:
    assert recycle(2.0, 3, "value_sd").tolist() == [2.0, 2.0, 2.0]


def test_one_value_per_predictor_is_kept_as_given() -> None:
    assert recycle([1.0, 2.0, 3.0], 3, "value_mean").tolist() == [1.0, 2.0, 3.0]


def test_a_length_between_one_and_n_is_refused() -> None:
    with pytest.raises(SaValueError, match="length 1 or 3"):
        recycle([1.0, 2.0], 3, "value_mean")


def test_a_value_below_the_bound_is_refused() -> None:
    with pytest.raises(SaValueError, match="must not go below 0"):
        recycle([-1.0], 3, "value_sd", 0)


# --------------------------------------------------------------------------- #
# pred_spec and plant_beta
# --------------------------------------------------------------------------- #


def test_the_counts_default_to_a_share_of_the_checked_predictor_count() -> None:
    spec = pred_spec(8, None, None, None, 0, 1, explicit=[])
    assert (spec.n_pred, spec.n_pos, spec.n_neg) == (8, 2, 2)


def test_beta_states_how_many_predictors_there_are() -> None:
    spec = pred_spec(8, [1.0, 0.0, -1.0], None, None, 0, 1, explicit=[])
    assert spec.n_pred == 3
    assert (spec.n_pos, spec.n_neg) == (0, 0)


def test_beta_together_with_a_planted_count_is_refused() -> None:
    with pytest.raises(SaValueError, match="`beta` states every coefficient"):
        pred_spec(8, [1.0, 0.0], None, 2, 0, 1, explicit=["n_neg"])


def test_beta_disagreeing_with_an_explicit_n_pred_is_refused() -> None:
    with pytest.raises(SaValueError, match="but `beta` gives 2 coefficient"):
        pred_spec(4, [1.0, 0.0], None, None, 0, 1, explicit=["n_pred"])


def test_beta_disagreeing_with_a_default_n_pred_is_accepted() -> None:
    assert pred_spec(8, [1.0, 0.0], None, None, 0, 1, explicit=[]).n_pred == 2


def test_more_coefficients_than_predictors_is_refused() -> None:
    with pytest.raises(SaValueError, match="more coefficients than the 3"):
        pred_spec(3, None, 2, 2, 0, 1, explicit=["n_pos", "n_neg"])


def test_the_signs_are_counts_rather_than_draws() -> None:
    spec = pred_spec(10, None, 3, 2, 0, 1, explicit=["n_pos", "n_neg"])
    for seed in range(5):
        planted = plant_beta(spec, (0.5, 2.0), np.random.default_rng(seed))
        assert planted.direction.count("up") == 3
        assert planted.direction.count("down") == 2
        assert planted.direction.count("none") == 5
        assert (planted.beta[np.array(planted.direction) == "up"] > 0).all()
        assert (planted.beta[np.array(planted.direction) == "down"] < 0).all()


def test_a_stated_beta_is_planted_exactly() -> None:
    spec = PredSpec(3, 0, 0, np.array([2.0, 0.0, -1.5]), np.zeros(3), np.ones(3))
    planted = plant_beta(spec, (0.5, 2.0), np.random.default_rng(1))
    assert planted.beta.tolist() == [2.0, 0.0, -1.5]
    assert planted.direction == ["up", "none", "down"]


# --------------------------------------------------------------------------- #
# Subjects, levels and holes
# --------------------------------------------------------------------------- #


def test_no_subjects_leaves_the_row_count_as_the_unit_count() -> None:
    assert subject_sizes(200, None, use_default_n=True) == (None, 200)


def test_one_row_count_is_spread_over_the_rows_asked_for() -> None:
    sizes, n_samples = subject_sizes(200, 4, use_default_n=True)
    assert n_samples == 200
    assert sizes is not None
    assert len(sizes) == 50
    assert set(sizes) == {4}


def test_a_row_count_that_does_not_divide_the_rows_is_refused() -> None:
    with pytest.raises(SaValueError, match="does not divide the 200 row"):
        subject_sizes(200, 3, use_default_n=True)


def test_a_count_per_subject_says_how_many_rows_there_are() -> None:
    sizes, n_samples = subject_sizes(200, [3, 2, 5], use_default_n=True)
    assert (sizes, n_samples) == ([3, 2, 5], 10)


def test_a_row_total_that_disagrees_with_the_counts_is_refused() -> None:
    with pytest.raises(SaValueError, match="but `n_samples` asks for 200"):
        subject_sizes(200, [3, 2, 5], use_default_n=False)


def test_a_split_over_subjects_needs_at_least_two_of_them() -> None:
    with pytest.raises(SaValueError, match="needs at least 2"):
        subject_sizes(10, [10], use_default_n=True)


def test_levels_are_handed_out_in_balanced_counts() -> None:
    levels = balanced_levels(10, ["low", "mid", "high"], np.random.default_rng(1))
    counts = pd.Series(levels).value_counts()
    assert sorted(counts.tolist()) == [3, 3, 4]
    assert list(levels.categories) == ["low", "mid", "high"]


def test_the_reference_level_carries_no_offset() -> None:
    offsets = factor_offsets(["low", "mid", "high"], (0.5, 2.0), np.random.default_rng(1))
    assert list(offsets) == ["low", "mid", "high"]
    assert offsets["low"] == 0.0
    assert offsets["mid"] > 0
    assert offsets["high"] < 0


def test_the_number_of_holes_is_a_function_of_the_proportion() -> None:
    frame = pd.DataFrame(np.ones((20, 5)))
    holed = mask_missing(frame, 0.1, np.random.default_rng(1))
    assert int(holed.isna().to_numpy().sum()) == 10


def test_no_holes_leaves_the_frame_alone() -> None:
    frame = pd.DataFrame(np.ones((20, 5)))
    assert mask_missing(frame, 0.0, np.random.default_rng(1)) is frame


# --------------------------------------------------------------------------- #
# The intercept that hits a requested event rate
# --------------------------------------------------------------------------- #


def test_the_intercept_hits_the_rate_on_the_predictor_that_was_drawn() -> None:
    eta = np.random.default_rng(1).normal(0, 2, 500)
    for rate in (0.1, 0.3, 0.5, 0.9):
        intercept = solve_intercept(eta, rate)
        assert expit(intercept + eta).mean() == pytest.approx(rate, abs=1e-6)


def test_a_rate_of_a_half_on_a_symmetric_predictor_needs_no_intercept() -> None:
    eta = np.array([-1.0, 1.0])
    assert solve_intercept(eta, 0.5) == pytest.approx(0.0, abs=1e-6)


def test_a_rate_the_predictors_cannot_reach_is_reported_as_such() -> None:
    """The bracket widens to 1e4 and stops; an unreachable rate is an argument error."""
    eta = np.array([-1e5, 1e5])
    with pytest.raises(SaValueError, match="no intercept gives an event rate"):
        solve_intercept(eta, 0.9)


# --------------------------------------------------------------------------- #
# The term axis
# --------------------------------------------------------------------------- #


def test_a_factor_becomes_one_term_per_level_beyond_the_reference() -> None:
    terms = truth_term(
        np.array([1.0, 0.0]),
        ["x_1", "x_2"],
        {"x_cat_1": {"low": 0.0, "mid": 1.5, "high": -0.5}},
    )
    assert terms["terms"].tolist() == ["x_1", "x_2", "x_cat_1mid", "x_cat_1high"]
    assert terms["predictors"].tolist() == ["x_1", "x_2", "x_cat_1", "x_cat_1"]
    assert terms["beta"].tolist() == [1.0, 0.0, 1.5, -0.5]


def test_a_design_with_no_factor_has_one_term_per_predictor() -> None:
    terms = truth_term(np.array([1.0, -2.0]), ["x_1", "x_2"], {})
    assert terms["terms"].tolist() == ["x_1", "x_2"]


def test_the_root_tolerance_is_the_one_r_solves_to() -> None:
    from statassist.simulate._supervised import _ROOT_TOL

    assert _ROOT_TOL == math.sqrt(np.finfo(float).eps)
