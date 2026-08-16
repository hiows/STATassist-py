"""Simulate a multi-group experiment with known truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_check_count, sa_check_range, sa_preserve_seed


def simulate_multiple_groups(
    n_feats: int = 100,
    n_per_group: int = 30,
    n_groups: int = 3,
    n_signal: int = 20,
    expr_range: tuple[float, float] = (2, 12),
    sd_range: tuple[float, float] = (1.2, 2.8),
    deg_log2fc: tuple[float, float] = (0.8, 2.0),
    group_lv: list[str] | None = None,
    seed: float | None = None,
) -> dict:
    n_feats = sa_check_count(n_feats, "n_feats", 1)
    n_per_group = sa_check_count(n_per_group, "n_per_group", 2)
    n_groups = sa_check_count(n_groups, "n_groups", 3)
    n_signal = sa_check_count(n_signal, "n_signal")
    sa_check_range(expr_range, "expr_range")
    sa_check_range(sd_range, "sd_range", 0)
    sa_check_range(deg_log2fc, "deg_log2fc", 0)

    if group_lv is None:
        group_lv = [f"g{i}" for i in range(n_groups)]
    if len(group_lv) != n_groups:
        raise ValueError("`group_lv` length must match `n_groups`.")

    with sa_preserve_seed(seed):
        feats = [f"gene_{i}" for i in range(1, n_feats + 1)]
        baseline = np.random.uniform(expr_range[0], expr_range[1], n_feats)
        sds = np.random.uniform(sd_range[0], sd_range[1], n_feats)
        signal_idx = np.random.choice(n_feats, min(n_signal, n_feats), replace=False)
        effects = np.zeros((n_feats, n_groups))
        for i in signal_idx:
            shift = np.random.uniform(deg_log2fc[0], deg_log2fc[1], n_groups - 1)
            shift = np.insert(shift, 0, 0.0)
            effects[i, :] = shift - shift[0]

        blocks = []
        for g, lv in enumerate(group_lv):
            mat = np.column_stack(
                [
                    np.random.normal(baseline[j] + effects[j, g], sds[j], n_per_group)
                    for j in range(n_feats)
                ]
            )
            blocks.append(mat)
        values = np.vstack(blocks)
        data = pd.DataFrame(values, columns=feats)
        group = np.repeat(group_lv, n_per_group)

        extreme = np.max(np.abs(effects[:, 1:]), axis=1)
        truth = pd.DataFrame(
            {
                "features": feats,
                "signal": ["yes" if i in signal_idx else "no" for i in range(n_feats)],
                "max_abs_log2fc": extreme,
                "baseline": baseline,
            }
        )

    return {
        "args": {
            "data": data,
            "feats": feats,
            "group": group,
            "group_lv": group_lv,
            "input_scale": "log2",
        },
        "truth": truth,
    }
