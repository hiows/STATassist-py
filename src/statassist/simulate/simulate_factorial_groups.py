"""Simulate a factorial design with planted effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_check_count, sa_preserve_seed


def simulate_factorial_groups(
    n_obs_per_cell: int = 20,
    feats: list[str] | None = None,
    factor_lv: dict[str, list[str]] | None = None,
    main_effects: dict[str, float] | None = None,
    interaction: float = 0.0,
    sigma: float = 1.0,
    seed: float | None = None,
) -> dict:
    n_obs_per_cell = sa_check_count(n_obs_per_cell, "n_obs_per_cell", 2)
    factor_lv = factor_lv or {"A": ["a0", "a1"], "B": ["b0", "b1"]}
    feats = feats or ["y1"]
    main_effects = main_effects or {"A": 1.0, "B": 0.5}

    from itertools import product

    cells = list(product(*[factor_lv[k] for k in factor_lv]))
    rows = []
    with sa_preserve_seed(seed):
        for cell in cells:
            mu = 10.0
            for i, (fac, lv) in enumerate(zip(factor_lv.keys(), cell)):
                if lv != factor_lv[fac][0]:
                    mu += main_effects.get(fac, 0.0)
            if cell != tuple(factor_lv[f][0] for f in factor_lv) and interaction:
                mu += interaction
            for _ in range(n_obs_per_cell):
                row = {f: lv for f, lv in zip(factor_lv.keys(), cell)}
                for feat in feats:
                    row[feat] = np.random.normal(mu, sigma)
                rows.append(row)

    data = pd.DataFrame(rows)
    return {
        "args": {
            "data": data,
            "feats": feats,
            "factors": {k: k for k in factor_lv},
            "factor_lv": factor_lv,
        },
        "truth_term": pd.DataFrame(
            [{"terms": k, "max_abs_delta": abs(v)} for k, v in main_effects.items()]
        ),
    }
