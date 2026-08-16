"""Correlation between every pair of features, with p-values."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.associate import sa_association_matrices
from statassist.utils.validate import sa_check_p_adjust, sa_validate_wide_input

KNOWN_METHODS = ("pearson", "spearman", "kendall")


def summarize_association_stats(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    methods: list[str] | None = None,
    adj_type: str = "BH",
    use: str = "pairwise.complete.obs",
) -> dict[str, Any]:
    if methods is None:
        methods = list(KNOWN_METHODS)
    if use not in ("pairwise.complete.obs", "complete.obs"):
        raise ValueError(
            "`use` must be 'pairwise.complete.obs' or 'complete.obs'."
        )
    sa_check_p_adjust(adj_type, "adj_type")

    if (
        not isinstance(methods, (list, tuple))
        or len(methods) == 0
        or any(pd.isna(m) for m in methods)
    ):
        raise ValueError(
            "`methods` must be a non-empty character vector drawn from: "
            + ", ".join(KNOWN_METHODS)
            + "."
        )
    unknown = [m for m in methods if m not in KNOWN_METHODS]
    if unknown:
        raise ValueError(
            "`methods` must be drawn from: "
            + ", ".join(KNOWN_METHODS)
            + f". Not recognised: {', '.join(unknown)}."
        )
    dup = [m for m in methods if methods.count(m) > 1]
    dup = list(dict.fromkeys(dup))
    if dup:
        raise ValueError(
            f"`methods` contains duplicated names: {', '.join(dup)}"
        )

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data.frame or a matrix.")

    if feats is None:
        feats = [
            c
            for c in data.columns
            if pd.api.types.is_numeric_dtype(data[c])
        ]
        if len(feats) == 0:
            raise ValueError("`data` holds no numeric column to correlate.")

    input_data = sa_validate_wide_input(data, list(feats), None, None)
    data = input_data["data"]
    feats = input_data["feats"]

    if len(feats) < 2:
        raise ValueError(
            f"`summarize_association_stats()` needs at least 2 features to "
            f"correlate, but got {len(feats)}."
        )

    x = data[feats].to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan

    if use == "complete.obs":
        keep = np.all(np.isfinite(x), axis=1)
        if not keep.any():
            raise ValueError(
                '`use = "complete.obs"` leaves no row: every row has a missing '
                "value in at least one of `feats`."
            )
        n_drop = int(np.sum(~keep))
        if n_drop > 0:
            print(
                f"Dropped {n_drop} row(s) with a missing value, as "
                '`use = "complete.obs"` asks.'
            )
        x = x[keep, :]

    spread = np.nanstd(x, axis=0, ddof=1)
    flat = ~np.isfinite(spread) | (spread == 0)
    if flat.any():
        flat_feats = [feats[i] for i, f in enumerate(flat) if f]
        print(
            f"{int(flat.sum())} feature(s) have no variance to correlate and "
            f"come back as NA: {', '.join(flat_feats)}."
        )

    out: dict[str, Any] = {}
    for m in methods:
        out[m] = sa_association_matrices(x, m, adj_type, feats=feats)

    out["design"] = {
        "feats": feats,
        "n_obs": x.shape[0],
        "methods": list(methods),
        "adj_type": adj_type,
        "use": use,
    }
    return out
