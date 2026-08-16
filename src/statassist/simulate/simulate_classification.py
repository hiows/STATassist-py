"""Simulate classification data with known class separation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_check_count, sa_preserve_seed


def simulate_classification(
    n_obs: int = 200,
    n_pred: int = 10,
    n_signal: int = 3,
    class_sep: float = 1.5,
    outcome_lv: list[str] | None = None,
    seed: float | None = None,
) -> dict:
    n_obs = sa_check_count(n_obs, "n_obs", 20)
    n_pred = sa_check_count(n_pred, "n_pred", 1)
    n_signal = sa_check_count(n_signal, "n_signal")
    outcome_lv = outcome_lv or ["control", "case"]

    with sa_preserve_seed(seed):
        preds = [f"x_{i+1}" for i in range(n_pred)]
        x = np.random.normal(0, 1, size=(n_obs, n_pred))
        signal = np.random.choice(n_pred, min(n_signal, n_pred), replace=False)
        y_num = np.random.binomial(1, 0.5, n_obs)
        for j in signal:
            x[:, j] += class_sep * (2 * y_num - 1)
        y = np.where(y_num == 0, outcome_lv[0], outcome_lv[1])
        data = pd.DataFrame(x, columns=preds)
        data["y"] = y

    truth = pd.DataFrame(
        {
            "predictors": preds,
            "role": ["signal" if i in signal else "null" for i in range(n_pred)],
        }
    )
    return {
        "args": {
            "data": data,
            "outcome": "y",
            "predictors": preds,
            "outcome_lv": outcome_lv,
        },
        "truth": truth,
        "split_args": {"data": data, "outcome": "y", "test_frac": 0.3, "seed": seed},
    }
