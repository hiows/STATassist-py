"""Golden parity and workflow tests for simulate_* functions (2026-08-18)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import statassist as sa

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RTOL = 1e-12


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _assert_frame_close(got: pd.DataFrame, golden: dict, rtol: float = RTOL) -> None:
    golden_cols = [c for c in golden.keys() if c != "_row"]
    assert list(got.columns) == golden_cols

    def _norm_missing(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, str) and v == "NA":
            return None
        return v

    for col in got.columns:
        a = got[col].to_numpy()
        raw = golden[col]
        if pd.api.types.is_numeric_dtype(got[col]):
            b = np.asarray(raw, dtype=float)
            np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol, equal_nan=True)
        else:
            assert [_norm_missing(x) for x in a] == [_norm_missing(x) for x in raw]


def _assert_data_close(got: pd.DataFrame, golden: dict) -> None:
    assert list(got.columns) == list(golden.keys())
    for col in golden:
        if pd.api.types.is_numeric_dtype(got[col]):
            np.testing.assert_allclose(
                got[col].to_numpy(),
                np.asarray(golden[col], dtype=float),
                rtol=RTOL,
                atol=RTOL,
                equal_nan=True,
            )
        else:
            assert got[col].tolist() == golden[col]


# --- simulate_two_groups ---


def test_simulate_two_groups_golden_parity() -> None:
    golden = _load("simulate_two_groups")
    sim = sa.simulate_two_groups(n_feats=10, n_up=2, n_down=2, seed=2026)
    _assert_data_close(sim["args"]["data"], golden["args"]["data"])
    _assert_frame_close(sim["truth"], golden["truth"])


def test_simulate_two_groups_downstream() -> None:
    sim = sa.simulate_two_groups(n_feats=10, n_up=2, n_down=2, seed=2026)
    res = sa.compare_two_groups(**sim["args"], diagnose=False)
    planted = sim["truth"]["direction"] != "none"
    sig = sa.estimate_significance(res, test="t_test")
    called = sig.significance["is_signif"].fillna(False)
    assert res.tests["t_test"].shape[0] == 10
    assert planted.sum() == 4
    assert called.any()


def test_simulate_two_groups_invalid_n_up_down() -> None:
    with pytest.raises(ValueError, match="n_up"):
        sa.simulate_two_groups(n_feats=5, n_up=3, n_down=3, seed=1)


# --- simulate_multiple_groups ---


def test_simulate_multiple_groups_golden_parity() -> None:
    golden = _load("simulate_multiple_groups")
    sim = sa.simulate_multiple_groups(
        n_feats=10, n_up=2, n_down=2, n_control=12, n_treat=[12, 12], seed=2026
    )
    _assert_data_close(sim["args"]["data"], golden["args"]["data"])
    _assert_frame_close(sim["truth"], golden["truth"])


def test_simulate_multiple_groups_downstream() -> None:
    sim = sa.simulate_multiple_groups(
        n_feats=8, n_control=15, n_treat=[15, 15], n_up=2, n_down=1, seed=1
    )
    res = sa.compare_multiple_groups(**sim["args"], diagnose=False)
    assert "anova_test" in res.tests
    assert sim["truth"]["direction"].isin(["up", "down", "none"]).all()


def test_simulate_multiple_groups_paired_requires_equal_sizes() -> None:
    with pytest.raises(ValueError, match="paired"):
        sa.simulate_multiple_groups(
            n_feats=5, n_control=10, n_treat=[12, 10], paired=True, seed=1
        )


# --- simulate_factorial_groups ---


def test_simulate_factorial_groups_golden_parity() -> None:
    golden = _load("simulate_factorial_groups")
    sim = sa.simulate_factorial_groups(
        n_feats=8, n_up=2, n_down=2, n_per_cell=10, seed=2026
    )
    _assert_data_close(sim["args"]["data"], golden["args"]["data"])
    _assert_frame_close(sim["truth"], golden["truth"])


def test_simulate_factorial_groups_downstream() -> None:
    sim = sa.simulate_factorial_groups(n_feats=6, n_up=1, n_down=1, seed=1)
    res = sa.compare_factorial_groups(**sim["args"], diagnose=False)
    assert res.effect.shape[0] == 6
    assert "truth_term" in sim


def test_simulate_factorial_groups_unplanted_extreme_na() -> None:
    sim = sa.simulate_factorial_groups(n_feats=5, n_up=0, n_down=0, seed=1)
    assert sim["truth"]["extreme_cell"].isna().all()


# --- simulate_categorical_groups ---


def test_simulate_categorical_groups_golden_parity() -> None:
    golden = _load("simulate_categorical_groups")
    sim = sa.simulate_categorical_groups(n_samples=100, seed=2026)
    assert list(sim["args"]["data"].columns) == list(golden["args"]["data"].keys())
    for col in sim["args"]["data"].columns:
        assert sim["args"]["data"][col].tolist() == golden["args"]["data"][col]
    _assert_frame_close(sim["truth"], golden["truth"])


def test_simulate_categorical_groups_downstream() -> None:
    sim = sa.simulate_categorical_groups(n_samples=80, seed=1)
    res = sa.compare_categorical_groups(**sim["args"])
    assert res.cells.shape[0] > 0


def test_simulate_categorical_groups_matched_binary_only() -> None:
    with pytest.raises(ValueError, match="binary"):
        sa.simulate_categorical_groups(
            category_lv={"before": ["a", "b", "c"], "after": ["a", "b", "c"]},
            paired=True,
            seed=1,
        )


# --- simulate_regression / classification / make_block_cor ---


def test_simulate_regression_golden_parity() -> None:
    golden = _load("simulate_regression")
    sim = sa.simulate_regression(n_samples=50, n_pred=6, n_pos=2, n_neg=1, seed=2026)
    _assert_data_close(sim["args"]["data"], golden["args"]["data"])
    _assert_frame_close(sim["truth"], golden["truth"])


def test_simulate_regression_downstream() -> None:
    sim = sa.simulate_regression(n_samples=40, n_pred=5, n_factor_pred=0, seed=1)
    fit = sa.fit_linear_regression(**sim["args"], cv=False)
    assert fit["coefficients"].shape[0] >= 1


def test_simulate_regression_beta_clash() -> None:
    with pytest.raises(ValueError, match="beta"):
        sa.simulate_regression(n_pred=4, beta=[1, 0, 0, 0], n_pos=2, seed=1)


def test_simulate_classification_golden_parity() -> None:
    golden = _load("simulate_classification")
    sim = sa.simulate_classification(n_samples=50, n_pred=6, n_pos=2, n_neg=1, seed=2026)
    _assert_data_close(sim["args"]["data"], golden["args"]["data"])
    _assert_frame_close(sim["truth"], golden["truth"])


def test_simulate_classification_downstream() -> None:
    sim = sa.simulate_classification(n_samples=40, n_pred=5, n_factor_pred=0, seed=1)
    fit = sa.fit_logistic_regression(**sim["args"], cv=False)
    assert "y" in sim["args"]["data"].columns


def test_simulate_classification_invalid_event_rate() -> None:
    with pytest.raises(ValueError, match="event_rate"):
        sa.simulate_classification(event_rate=1.0, seed=1)


def test_make_block_cor_golden_and_non_pd() -> None:
    golden = _load("make_block_cor")
    mat = sa.make_block_cor(4, blocks=[{"features": [1, 2], "cor": 0.9}])
    np.testing.assert_allclose(mat, golden["cor_mat"], rtol=RTOL, atol=RTOL)
    with pytest.raises(ValueError, match="default_cor"):
        sa.make_block_cor(3, default_cor=-0.6)
