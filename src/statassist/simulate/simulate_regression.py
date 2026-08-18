"""Simulate regression data with known coefficients (R simulate_regression.R)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.rng_r import get_rng, sa_r_seed
from statassist.utils.simulate_utils import (
    sa_sim_add_intercept,
    sa_sim_split_args,
    sa_sim_supervised_design,
)
from statassist.utils.validate import sa_check_scalar_num

_MISSING = object()


def simulate_regression(
    n_samples: int | object = _MISSING,
    n_pred: int | object = _MISSING,
    n_pos: int | object = _MISSING,
    n_neg: int | object = _MISSING,
    beta: list[float] | np.ndarray | None = None,
    beta_range: tuple[float, float] = (0.5, 2),
    intercept: float = 0,
    value_mean: float | list[float] = 0,
    value_sd: float | list[float] = 1,
    noise_sd: float = 3,
    cor_mat: np.ndarray | None = None,
    n_factor_pred: int = 1,
    factor_lv: list[str] | None = None,
    n_constant_pred: int = 0,
    p_missing: float = 0,
    n_per_subject: list[int] | None = None,
    subject_sd: float = 1,
    subject_share: float = 0.5,
    pred_prefix: str = "x",
    seed: float | None = None,
) -> dict[str, Any]:
    explicit = {
        name
        for name, val in (("n_pred", n_pred), ("n_pos", n_pos), ("n_neg", n_neg))
        if val is not _MISSING
    }
    use_default_n = n_samples is _MISSING
    if n_samples is _MISSING:
        n_samples = 200
    if n_pred is _MISSING:
        n_pred = 8
    if n_pos is _MISSING:
        n_pos = round(0.25 * int(n_pred))
    if n_neg is _MISSING:
        n_neg = round(0.25 * int(n_pred))

    sa_check_scalar_num(intercept, "intercept")
    sa_check_scalar_num(noise_sd, "noise_sd", 0)
    if factor_lv is None:
        factor_lv = ["low", "mid", "high"]

    with sa_r_seed(seed):
        rng = get_rng()
        design = sa_sim_supervised_design(
            int(n_samples),
            int(n_pred),
            None if beta is None else np.asarray(beta, dtype=float),
            int(n_pos),
            int(n_neg),
            beta_range,
            value_mean,
            value_sd,
            cor_mat,
            n_factor_pred,
            factor_lv,
            n_constant_pred,
            p_missing,
            n_per_subject,
            subject_sd,
            subject_share,
            pred_prefix,
            explicit,
            use_default_n,
        )

        eta = intercept + design["eta"]
        noise = rng.rnorm(design["n_samples"], 0, noise_sd)

        data = pd.DataFrame({"y": eta + noise})
        for col in design["x"].columns:
            data[col] = design["x"][col].to_numpy()
        if design["subject"] is not None:
            data["subject"] = design["subject"]

        signal_var = float(np.var(design["eta"] - design["subject_offset"], ddof=1))
        subject_var = (
            0.0
            if design["sizes"] is None
            else float(np.var(design["subject_offset"], ddof=1))
        )

    return {
        "args": {
            "data": data,
            "outcome": "y",
            "predictors": design["predictors"],
        },
        "split_args": sa_sim_split_args(data, design, stratify_outcome=False),
        "truth": design["truth"],
        "truth_term": sa_sim_add_intercept(design["truth_term"], intercept),
        "truth_model": {
            "intercept": intercept,
            "noise_sd": noise_sd,
            "signal_var": signal_var,
            "subject_var": subject_var,
            "r_squared": signal_var / (signal_var + subject_var + noise_sd**2),
            "n_samples": design["n_samples"],
            "n_subject": (
                np.nan if design["sizes"] is None else len(design["sizes"])
            ),
            "subject_sd": np.nan if design["sizes"] is None else subject_sd,
        },
        "truth_row": pd.DataFrame(
            {
                "subject": (
                    design["subject"]
                    if design["subject"] is not None
                    else [None] * design["n_samples"]
                ),
                "subject_offset": design["subject_offset"],
                "eta": eta,
                "noise": noise,
            }
        ),
    }
