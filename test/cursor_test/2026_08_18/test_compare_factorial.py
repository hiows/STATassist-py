"""Golden parity and behaviour tests for compare_factorial_groups (2026-08-18)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import statassist as sa

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RTOL = 1e-8


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _norm(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _assert_frame_close(got: pd.DataFrame, golden: dict) -> None:
    cols = [c for c in golden if c != "_row"]
    assert list(got.columns) == cols

    for col in cols:
        a = got[col].to_numpy()
        raw = golden[col]
        if not pd.api.types.is_numeric_dtype(got[col]):
            assert [_norm(v) for v in a] == [_norm(v) for v in raw]
            continue
        b = np.asarray([np.nan if v is None else v for v in raw], dtype=float)
        a = a.astype(float)
        if col == "log2_effect":
            # A two-level term is two exactly opposite components, so which one
            # `which.max()` lands on is decided by the last bit of a mean. The
            # size of the effect is the claim; the sign of the tie is not.
            a, b = np.abs(a), np.abs(b)
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=1e-8, equal_nan=True)


def _compare(fixture: str, sim_kwargs: dict, cmp_kwargs: dict, edit=None):
    golden = _load(fixture)
    sim = sa.simulate_factorial_groups(**sim_kwargs)
    args = dict(sim["args"])
    if edit is not None:
        args = edit(args)
    res = sa.compare_factorial_groups(**args, diagnose=False, **cmp_kwargs)
    return res, golden


def _drop_rows(args: dict) -> dict:
    """Unbalance the design by taking a few rows out of three cells."""
    drop = {0, 1, 2, 8, 9, 19, 20, 21}
    keep = np.array([i not in drop for i in range(len(args["data"]))])
    args = dict(args)
    args["data"] = args["data"].loc[keep].reset_index(drop=True)
    args["factors"] = {k: np.asarray(v)[keep] for k, v in args["factors"].items()}
    return args


def test_factorial_two_way_golden_parity() -> None:
    res, golden = _compare(
        "compare_factorial_2way",
        dict(n_feats=4, n_up=1, n_down=1, n_per_cell=8, seed=2026),
        {},
    )
    _assert_frame_close(res.effect, golden["effect"])
    _assert_frame_close(res.tests["anova_test"], golden["tests"]["anova_test"])
    _assert_frame_close(res.terms, golden["terms"])
    _assert_frame_close(res.cells, golden["cells"])


def test_factorial_three_way_golden_parity() -> None:
    res, golden = _compare(
        "compare_factorial_3way",
        dict(
            n_feats=3,
            n_up=1,
            n_down=0,
            n_per_cell=5,
            factor_lv={"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]},
            seed=2026,
        ),
        {},
    )
    _assert_frame_close(res.terms, golden["terms"])
    _assert_frame_close(res.cells, golden["cells"])
    assert res.parameters["n_posthoc"] == golden["parameters"]["n_posthoc"]


def test_factorial_three_way_posthoc_golden_parity() -> None:
    res, golden = _compare(
        "compare_factorial_3way",
        dict(
            n_feats=3,
            n_up=1,
            n_down=0,
            n_per_cell=5,
            factor_lv={"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]},
            seed=2026,
        ),
        {},
    )
    assert golden["posthoc"] is not None
    _assert_frame_close(res.posthoc["anova_test"], golden["posthoc"]["anova_test"])


def test_factorial_unbalanced_type_two_golden_parity() -> None:
    res, golden = _compare(
        "compare_factorial_unbalanced",
        dict(n_feats=4, n_up=1, n_down=1, n_per_cell=8, seed=2026),
        {"ss_type": "II"},
        _drop_rows,
    )
    # With unequal cells the sums of squares stop being independent, so the type
    # is the whole of what this fixture is testing.
    assert res.parameters["ss_type"] == "II"
    _assert_frame_close(res.terms, golden["terms"])
    _assert_frame_close(res.cells, golden["cells"])
    _assert_frame_close(res.tests["anova_test"], golden["tests"]["anova_test"])


def test_factorial_within_subject_is_refused() -> None:
    """A repeated factor is an error here, as it is in R.

    A within-subject term needs its own error stratum, and this function fits
    one; refusing is what keeps a repeated design from being reported as though
    the measurements were independent.
    """
    sim = sa.simulate_factorial_groups(
        n_feats=2, n_up=1, n_down=0, n_per_cell=6, seed=2026
    )
    args = dict(sim["args"])
    within = [next(iter(args["factors"]))]
    with pytest.raises(ValueError, match="compare_multiple_groups"):
        sa.compare_factorial_groups(**args, within=within, diagnose=False)


def test_factorial_id_without_within_is_ignored_with_a_warning() -> None:
    sim = sa.simulate_factorial_groups(
        n_feats=2, n_up=1, n_down=0, n_per_cell=6, seed=2026
    )
    args = dict(sim["args"])
    n = len(args["data"])
    with pytest.warns(UserWarning, match="ignored"):
        res = sa.compare_factorial_groups(
            **args, id=np.arange(n), diagnose=False
        )
    assert res.design["paired"] is False
    assert res.design["within"] == []
