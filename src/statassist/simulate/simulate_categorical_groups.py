"""Simulate a categorical table with planted association."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_check_count, sa_preserve_seed


def simulate_categorical_groups(
    n_samples: int = 400,
    row_lv: list[str] | None = None,
    col_lv: list[str] | None = None,
    assoc: float = 0.3,
    seed: float | None = None,
) -> dict:
    n_samples = sa_check_count(n_samples, "n_samples", 10)
    row_lv = row_lv or ["n", "y"]
    col_lv = col_lv or ["low", "mid", "high"]

    with sa_preserve_seed(seed):
        p_row = np.array([0.5, 0.5][: len(row_lv)])
        p_row /= p_row.sum()
        rows = np.random.choice(row_lv, n_samples, p=p_row)
        cols = []
        for r in rows:
            if r == row_lv[0]:
                p = np.array([0.5 - assoc / 2, 0.3, 0.2 + assoc / 2][: len(col_lv)])
            else:
                p = np.array([0.2 + assoc / 2, 0.3, 0.5 - assoc / 2][: len(col_lv)])
            p = np.clip(p, 0.01, None)
            p /= p.sum()
            cols.append(np.random.choice(col_lv, p=p))
        data = pd.DataFrame({"smoker": rows, "grade": cols})

    return {
        "args": {
            "data": data,
            "category_lv": {"smoker": row_lv, "grade": col_lv},
        },
        "truth": {"cramers_v": abs(assoc)},
    }
