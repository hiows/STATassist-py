"""SplitResult (sa_split) contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statassist.utils.metadata import sa_metadata


def sa_new_split(
    *,
    full_data: pd.DataFrame,
    datasets: list[dict[str, Any]],
    train_idx: list[list[int]],
    design: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(full_data, pd.DataFrame):
        raise ValueError("internal error: `full_data` must be a DataFrame.")
    if not datasets:
        raise ValueError("internal error: `datasets` must be non-empty.")
    return {
        "full_data": full_data,
        "datasets": datasets,
        "train_idx": train_idx,
        "design": design,
        "parameters": parameters,
        "metadata": sa_metadata(),
        "__class__": ("sa_split", "sa_result"),
    }
