"""SelectionResult (sa_selection) contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statassist.utils.metadata import sa_metadata

SELECTION_ANALYSES = ("rfe", "stepwise")


def sa_new_selection(
    *,
    analysis: str,
    candidates: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    selected: list[str],
    ranking: pd.DataFrame,
    profile: pd.DataFrame,
    resampling: pd.DataFrame | None = None,
    engine: dict[str, Any],
    fit: Any,
) -> dict[str, Any]:
    if analysis not in SELECTION_ANALYSES:
        raise ValueError(
            f"internal error: `analysis` must be one of {', '.join(SELECTION_ANALYSES)}."
        )
    if not candidates:
        raise ValueError("internal error: `candidates` must be a non-empty list.")
    if not isinstance(ranking, pd.DataFrame) or ranking["candidates"].tolist() != list(
        candidates
    ):
        raise ValueError("internal error: `ranking` is not aligned with `candidates`.")
    if not selected:
        raise ValueError("internal error: `selected` must be a non-empty list.")
    unknown = set(selected) - set(candidates)
    if unknown:
        raise ValueError(
            f"internal error: `selected` holds name(s) that are not candidates: "
            f"{', '.join(sorted(unknown))}."
        )
    if not ranking["selected"].equals(pd.Series([c in selected for c in candidates])):
        raise ValueError("internal error: `ranking$selected` disagrees with `selected`.")
    if (
        not isinstance(profile, pd.DataFrame)
        or profile.empty
        or "n_vars" not in profile.columns
    ):
        raise ValueError(
            "internal error: `profile` must be a non-empty DataFrame with an `n_vars` column."
        )
    if int(profile["chosen"].sum()) != 1:
        raise ValueError(
            f"internal error: exactly one row of `profile` must be chosen, "
            f"but {int(profile['chosen'].sum())} are marked."
        )
    chosen_n = int(profile.loc[profile["chosen"], "n_vars"].iloc[0])
    if chosen_n != len(selected):
        raise ValueError(
            f"internal error: the chosen row of `profile` is size {chosen_n} "
            f"but `selected` holds {len(selected)}."
        )
    for key in ("package", "method", "label", "metrics", "importance"):
        if engine.get(key) is None:
            raise ValueError(f"internal error: `engine` is missing `{key}`.")
    if resampling is not None and not isinstance(resampling, pd.DataFrame):
        raise ValueError("internal error: `resampling` must be a DataFrame or None.")

    return {
        "analysis": analysis,
        "candidates": list(candidates),
        "design": design,
        "parameters": parameters,
        "selected": list(selected),
        "ranking": ranking.reset_index(drop=True),
        "profile": profile.reset_index(drop=True),
        "resampling": resampling,
        "engine": engine,
        "fit": fit,
        "metadata": sa_metadata(),
        "__class__": ("sa_selection", "sa_result"),
    }
