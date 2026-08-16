"""Descriptive summary of several features, optionally split by group."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.describe import sa_describe_vector
from statassist.utils.validate import sa_validate_wide_input


def summarize_descriptive_stats(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: Any | None = None,
    group_lv: list[str] | None = None,
) -> pd.DataFrame:
    grouped = group is not None

    if not grouped:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a data.frame or a matrix.")
        group = np.repeat("all", len(data))
        group_lv = ["all"]
    elif group_lv is None:
        if isinstance(group, pd.Categorical):
            group_lv = list(group.categories)
        elif hasattr(group, "cat"):
            group_lv = list(group.cat.categories)
        else:
            group_lv = sorted(set(str(g) for g in group))

    input_data = sa_validate_wide_input(
        data, feats, group, group_lv, min_levels=1
    )
    data = input_data["data"]
    feats = input_data["feats"]
    group = input_data["group"]
    group_lv = list(group.categories)

    if input_data["n_dropped"] > 0:
        print(
            f"Dropped {input_data['n_dropped']} row(s) belonging to a level "
            "outside `group_lv`."
        )

    rows: list[dict[str, float]] = []
    for f in feats:
        for lv in group_lv:
            mask = group == lv
            rows.append(sa_describe_vector(data.loc[mask, f].to_numpy()))

    out = pd.DataFrame(rows)
    out.insert(0, "features", np.repeat(feats, len(group_lv)))
    if grouped:
        out.insert(1, "group", np.tile(group_lv, len(feats)))
    return out.reset_index(drop=True)
