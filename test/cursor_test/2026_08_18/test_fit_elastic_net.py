"""Golden parity tests for fit_elastic_net against glmnet (2026-08-18)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statassist as sa

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# glmnet stops its coordinate descent at `thresh = 1e-7` by default, which
# leaves it a few parts in a hundred thousand short of the solution the Python
# path runs all the way down to. Running R at `thresh = 1e-16` closes the gap to
# 1e-10, so this is the engine's stopping rule rather than a difference in what
# is being solved, and the fixtures record R at its own default.
GLMNET_RTOL = 1e-3
GLMNET_ATOL = 5e-4

# Two summaries magnify that gap. A squared correlation is one: where the
# penalty has flattened the predictions towards their mean there is barely any
# variation left to correlate, so a coefficient differing in its fifth decimal
# moves the R squared in its third. A standard deviation across folds is the
# other, being a difference of numbers that are nearly equal.
R2_ATOL = 1e-2
SD_RTOL = 1e-2


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def lasso() -> tuple[dict, dict]:
    sim = sa.simulate_regression(
        n_samples=60, n_pred=6, n_pos=2, n_neg=1, n_factor_pred=0, seed=2026
    )
    res = sa.fit_elastic_net(
        sim["args"]["data"],
        "y",
        penalty="lasso",
        cv=True,
        cv_method="repeated_kfold",
        n_fold=5,
        n_repeat=2,
        seed=2026,
    )
    return res, _load("fit_elastic_net")


def test_elastic_net_penalty_and_coefficients_match_glmnet(lasso) -> None:
    res, golden = lasso

    # The two packages name these the other way round: glmnet's `alpha` mixes
    # the two penalties and its `lambda` sets the strength, while scikit-learn's
    # `alpha` is the strength. The parameters reported here are glmnet's.
    assert res["parameters"]["alpha"] == pytest.approx(golden["parameters"]["alpha"])
    assert res["parameters"]["lambda"] == pytest.approx(
        golden["parameters"]["lambda"], rel=1e-10
    )

    assert res["coefficients"]["terms"].tolist() == golden["coefficients"]["terms"]
    np.testing.assert_allclose(
        res["coefficients"]["estimate"].to_numpy(dtype=float),
        np.asarray(golden["coefficients"]["estimate"], dtype=float),
        rtol=GLMNET_RTOL,
        atol=GLMNET_ATOL,
    )
    assert int((res["coefficients"]["estimate"] == 0).sum()) == int(
        np.sum(np.asarray(golden["coefficients"]["estimate"]) == 0)
    )


def test_elastic_net_resampled_performance_matches_caret(lasso) -> None:
    res, golden = lasso

    perf = res["performance"]
    assert list(perf.columns) == list(golden["performance"].keys())
    for metric in ["RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD"]:
        got = perf[metric].to_numpy(dtype=float)
        want = np.asarray(
            [np.nan if v is None else v for v in golden["performance"][metric]],
            dtype=float,
        )
        # A fold whose penalty zeroed every coefficient predicts one number, and
        # a correlation with a constant is undefined; caret leaves the R squared
        # missing there rather than calling it zero.
        assert np.array_equal(np.isnan(got), np.isnan(want))
        atol = R2_ATOL if metric.startswith("Rsquared") else GLMNET_ATOL
        rtol = SD_RTOL if metric.endswith("SD") else GLMNET_RTOL
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)


def test_elastic_net_resampling_folds_are_the_r_folds(lasso) -> None:
    res, golden = lasso

    got = res["resampling"].sort_values("Resample").reset_index(drop=True)
    want = pd.DataFrame(golden["resampling"]).sort_values("Resample").reset_index(drop=True)

    assert got["Resample"].tolist() == want["Resample"].tolist()
    for metric in ["RMSE", "Rsquared", "MAE"]:
        np.testing.assert_allclose(
            got[metric].to_numpy(dtype=float),
            want[metric].to_numpy(dtype=float),
            rtol=GLMNET_RTOL,
            atol=GLMNET_ATOL,
            equal_nan=True,
        )
