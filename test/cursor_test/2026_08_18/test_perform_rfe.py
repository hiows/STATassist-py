"""Golden parity and behaviour tests for perform_rfe against caret (2026-08-18)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import statassist as sa

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def linear() -> tuple[dict, dict]:
    sim = sa.simulate_regression(
        n_samples=60, n_pred=6, n_pos=2, n_neg=1, n_factor_pred=0, seed=2026
    )
    res = sa.perform_rfe(
        sim["args"]["data"],
        "y",
        model="linear",
        cv_method="repeated_kfold",
        n_fold=5,
        n_repeat=2,
        seed=2026,
    )
    return res, _load("perform_rfe")


def test_rfe_ranking_and_selection_match_caret(linear) -> None:
    res, golden = linear

    assert res["candidates"] == golden["candidates"]
    assert res["selected"] == golden["selected"]

    ranking = res["ranking"]
    assert ranking["candidates"].tolist() == golden["ranking"]["candidates"]
    assert ranking["rank"].tolist() == golden["ranking"]["rank"]
    assert ranking["selected"].tolist() == golden["ranking"]["selected"]
    np.testing.assert_allclose(
        ranking["estimate"].to_numpy(dtype=float),
        np.asarray(golden["ranking"]["estimate"], dtype=float),
        rtol=1e-9,
        atol=1e-10,
    )


def test_rfe_profile_matches_caret(linear) -> None:
    res, golden = linear

    profile = res["profile"]
    assert list(profile.columns) == list(golden["profile"].keys())
    assert profile["n_vars"].tolist() == golden["profile"]["n_vars"]
    assert profile["chosen"].tolist() == golden["profile"]["chosen"]
    assert int(profile["chosen"].sum()) == 1

    for metric in ["RMSE", "Rsquared", "MAE", "RMSESD", "RsquaredSD", "MAESD"]:
        np.testing.assert_allclose(
            profile[metric].to_numpy(dtype=float),
            np.asarray(golden["profile"][metric], dtype=float),
            rtol=1e-9,
            atol=1e-10,
        )


def test_rfe_resampling_is_the_chosen_size_on_the_r_folds(linear) -> None:
    res, golden = linear

    resampling = res["resampling"]
    assert list(resampling.columns) == list(golden["resampling"].keys())
    assert resampling["Resample"].tolist() == golden["resampling"]["Resample"]
    assert resampling["Variables"].tolist() == golden["resampling"]["Variables"]
    for metric in ["RMSE", "Rsquared", "MAE"]:
        np.testing.assert_allclose(
            resampling[metric].to_numpy(dtype=float),
            np.asarray(golden["resampling"][metric], dtype=float),
            rtol=1e-9,
            atol=1e-10,
        )


def test_rfe_scores_every_requested_size_and_the_full_set() -> None:
    """`subset_sizes` is scored as asked, and the full set always with it.

    Keeping everything is the option a selection is being compared against, so
    a profile that left it out could not say whether the search had helped.
    """
    sim = sa.simulate_regression(
        n_samples=50, n_pred=5, n_pos=2, n_neg=1, n_factor_pred=0, seed=11
    )
    res = sa.perform_rfe(
        sim["args"]["data"],
        "y",
        model="linear",
        subset_sizes=[2, 4],
        cv_method="kfold",
        n_fold=3,
        seed=3,
    )
    assert res["profile"]["n_vars"].tolist() == [2, 4, 5]
    assert res["selected"] == res["candidates"][: len(res["selected"])]
    assert len(res["resampling"]) == 3


def test_rfe_leave_one_out_pools_its_predictions() -> None:
    """One held-out row has no spread and no correlation of its own.

    caret pools every held-out prediction and scores them once instead, so a
    leave-one-out profile carries no standard deviations and nothing underneath
    it per resample.
    """
    sim = sa.simulate_regression(
        n_samples=40, n_pred=3, n_pos=1, n_neg=1, n_factor_pred=0, seed=4
    )
    res = sa.perform_rfe(
        sim["args"]["data"], "y", model="linear", cv_method="loocv"
    )
    assert list(res["profile"].columns) == ["n_vars", "RMSE", "Rsquared", "MAE", "chosen"]
    assert res["resampling"] is None
    assert res["profile"]["Rsquared"].notna().all()


def test_rfe_linear_on_a_class_outcome_is_refused() -> None:
    sim = sa.simulate_classification(
        n_samples=40, n_pred=4, n_pos=2, n_neg=1, n_factor_pred=0, seed=5
    )
    with pytest.raises(ValueError, match="class labels"):
        sa.perform_rfe(sim["args"]["data"], "y", model="linear", cv_method="kfold", n_fold=3)
