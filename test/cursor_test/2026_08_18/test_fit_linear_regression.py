"""Golden parity tests for fit_linear_regression against caret (2026-08-18)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statassist as sa

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RTOL = 1e-9


@pytest.fixture(scope="module")
def linear() -> tuple[dict, dict]:
    sim = sa.simulate_regression(
        n_samples=60, n_pred=6, n_pos=2, n_neg=1, n_factor_pred=0, seed=2026
    )
    res = sa.fit_linear_regression(
        sim["args"]["data"],
        "y",
        cv=True,
        cv_method="repeated_kfold",
        n_fold=5,
        n_repeat=2,
        seed=2026,
    )
    with open(FIXTURES / "fit_linear_regression.json", encoding="utf-8") as f:
        return res, json.load(f)


def test_linear_coefficients_and_fit_stats_match_r(linear) -> None:
    res, golden = linear

    assert res["terms"] == golden["terms"]
    assert res["coefficients"]["terms"].tolist() == golden["coefficients"]["terms"]
    for column in ["estimate", "stderr", "statistic", "pval", "lower_conf", "upper_conf"]:
        np.testing.assert_allclose(
            res["coefficients"][column].to_numpy(dtype=float),
            np.asarray(golden["coefficients"][column], dtype=float),
            rtol=RTOL,
            atol=1e-12,
        )

    for name, want in golden["fit_stats"].items():
        assert res["fit_stats"][name] == pytest.approx(want, rel=RTOL, abs=1e-12)


def test_linear_resampled_performance_matches_caret(linear) -> None:
    res, golden = linear

    perf = res["performance"]
    assert list(perf.columns) == list(golden["performance"].keys())
    for metric in ["RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD"]:
        np.testing.assert_allclose(
            perf[metric].to_numpy(dtype=float),
            np.asarray(golden["performance"][metric], dtype=float),
            rtol=RTOL,
            atol=1e-12,
        )

    got = res["resampling"].sort_values("Resample").reset_index(drop=True)
    want = (
        pd.DataFrame(golden["resampling"])
        .sort_values("Resample")
        .reset_index(drop=True)
    )
    assert got["Resample"].tolist() == want["Resample"].tolist()
    for metric in ["RMSE", "Rsquared", "MAE"]:
        np.testing.assert_allclose(
            got[metric].to_numpy(dtype=float),
            want[metric].to_numpy(dtype=float),
            rtol=RTOL,
            atol=1e-12,
        )
