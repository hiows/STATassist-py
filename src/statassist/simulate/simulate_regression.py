"""Simulate regression data with known coefficients."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.simulate.make_block_cor import make_block_cor
from statassist.utils.validate import sa_check_count, sa_check_range, sa_preserve_seed


def simulate_regression(
    n_obs: int = 200,
    n_pred: int = 10,
    beta: list[float] | None = None,
    intercept: float = 1.0,
    sigma: float = 1.0,
    cor_mat: np.ndarray | None = None,
    seed: float | None = None,
) -> dict:
    n_obs = sa_check_count(n_obs, "n_obs", 10)
    n_pred = sa_check_count(n_pred, "n_pred", 1)
    sa_check_range((sigma, sigma * 2), "sigma", 0)

    if beta is None:
        beta = [2.0] + [0.0] * (n_pred - 1)
    beta = np.asarray(beta, dtype=float)
    if beta.size != n_pred:
        raise ValueError("`beta` length must equal `n_pred`.")

    if cor_mat is None:
        cor_mat = np.eye(n_pred)
    if cor_mat.shape != (n_pred, n_pred):
        raise ValueError("`cor_mat` must be n_pred by n_pred.")

    with sa_preserve_seed(seed):
        preds = [f"x_{i+1}" for i in range(n_pred)]
        x = np.random.multivariate_normal(np.zeros(n_pred), cor_mat, size=n_obs)
        y = intercept + x @ beta + np.random.normal(0, sigma, n_obs)
        data = pd.DataFrame(x, columns=preds)
        data["y"] = y

    truth = pd.DataFrame({"predictors": preds, "beta": beta})
    return {
        "args": {"data": data, "outcome": "y", "predictors": preds},
        "truth": truth,
        "truth_term": truth.copy(),
        "split_args": {"data": data, "outcome": "y", "test_frac": 0.3, "seed": seed},
    }
