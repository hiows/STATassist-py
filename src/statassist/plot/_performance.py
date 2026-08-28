"""What the two evaluation plots share: the input check, the models, the colours.

Port of the ``sa_performance_*`` helpers in ``R/performance.R``. The two pictures
are not interchangeable - a regression is drawn against the outcome it predicted
and a classification against the two classes it ranked - and the object that
carries one also carries the names of the other's slots, so a classification
handed to the scatter would otherwise fail on a missing column rather than on the
mistake that was made.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.errors import SaValueError
from ..core.result import SaModel, SaPerformance
from ._theme import group_colors

__all__ = ["performance_colors", "performance_input", "performance_metrics", "performance_models"]


def performance_input(x: Any, want: str, arg: str, other: str) -> SaPerformance:
    """Refuse anything that is not the evaluation this plot draws.

    Port of ``sa_performance_input()``. A fitted model is named separately from
    everything else, because handing one over is the natural mistake and the fix
    is a step rather than a different argument.
    """
    if not isinstance(x, SaPerformance):
        if isinstance(x, SaModel):
            raise SaValueError(
                f"`{arg}` is a fitted model rather than an evaluation of one. Score it "
                "on held-out rows with evaluate_regression_models() or "
                "evaluate_classification_models() first."
            )
        raise SaValueError(
            f"`{arg}` must be an evaluation result, as returned by "
            "evaluate_regression_models() or evaluate_classification_models()."
        )
    if x["analysis"] != want:
        raise SaValueError(f"`{arg}` is a {x['analysis']} result. Use {other} for that one.")
    return x


def performance_models(x: SaPerformance, models: Any) -> list[str]:
    """Choose which models to draw, and in what order.

    Port of ``sa_performance_models()``. ``None`` draws all of them in the order
    the evaluation holds, which puts the baseline first.
    """
    held = [str(name) for name in x["models"]]
    if models is None:
        return held

    wanted = [models] if isinstance(models, str) else [str(name) for name in models]
    if not wanted:
        raise SaValueError(
            "`models` must be a non-empty sequence of model names, or None for every "
            "model in the order the evaluation holds them."
        )
    unknown = [name for name in wanted if name not in held]
    if unknown:
        raise SaValueError(
            "`models` names model(s) the evaluation does not hold: "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(held)
            + "."
        )
    seen = {name for name in wanted if wanted.count(name) > 1}
    if seen:
        raise SaValueError("`models` contains duplicated names: " + ", ".join(sorted(seen)) + ".")
    return wanted


def performance_colors(n: int, col: Any) -> list[Any]:
    """One colour per drawn model.

    Port of ``sa_performance_colours()``. The palette is the one every other
    ``draw_*`` colours a level with, so a model here and a group there are drawn
    from the same set. A named colour must be one or one per model: recycling a
    shorter set would give two models the same colour, which is the one thing an
    overlaid plot cannot survive.
    """
    if col is None:
        return group_colors(None, n)
    held = [col] if isinstance(col, str) else list(col)
    if len(held) not in (1, n):
        raise SaValueError(f"`col` must hold one colour, or one per drawn model ({n}).")
    return [held[index % len(held)] for index in range(n)]


def performance_metrics(x: SaPerformance, drawn: list[str]) -> pd.DataFrame:
    """The rows of ``metrics`` that were drawn, in the order they were drawn."""
    table: pd.DataFrame = x["metrics"]
    indexed = table.set_index(table["model"].astype(str))
    return indexed.loc[drawn].reset_index(drop=True)
