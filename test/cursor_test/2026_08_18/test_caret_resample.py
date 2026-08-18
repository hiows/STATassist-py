"""Fold parity with caret's resampling (2026-08-18).

The fold a model is scored on comes out of R's own random number stream, so if
these do not match then no cross-validated number below them can be compared
with an R one at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from statassist.utils.caret_resample import (
    sa_create_folds,
    sa_create_multi_folds,
    sa_fold_names,
)
from statassist.utils.rng_r import sa_r_seed

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _as_sets(folds: list[np.ndarray]) -> list[list[int]]:
    """1-based, the way R hands its indices back."""
    return [sorted(int(i) + 1 for i in fold) for fold in folds]


def test_create_folds_match_r() -> None:
    golden = _load("caret_folds")

    with sa_r_seed(2026):
        cls = sa_create_folds(np.asarray(golden["y_cls"]), k=5)
    assert _as_sets(cls) == [sorted(v) for v in golden["folds_cls"].values()]

    with sa_r_seed(2026):
        reg = sa_create_folds(np.asarray(golden["y_reg"], dtype=float), k=5)
    assert _as_sets(reg) == [sorted(v) for v in golden["folds_reg"].values()]

    # A count that does not divide evenly by k, which is the branch that spends
    # a draw on where the spare rows go.
    with sa_r_seed(7):
        odd = sa_create_folds(np.asarray(golden["y_odd"]), k=4)
    assert _as_sets(odd) == [sorted(v) for v in golden["folds_odd"].values()]


def test_create_multi_folds_match_r() -> None:
    golden = _load("caret_folds")

    with sa_r_seed(2026):
        multi = sa_create_multi_folds(np.asarray(golden["y_reg"], dtype=float), k=5, times=3)

    names = sa_fold_names(5, 3)
    assert names == golden["multi_names"]
    assert _as_sets(multi) == [sorted(golden["multi_reg"][nm]) for nm in names]
