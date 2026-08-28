"""Overlaid ROC curves, the picture of a classification evaluation.

Port of ``R/draw_roc_curve.R``. Unlike the predicted-against-observed scatter,
this one overlays rather than panels: a curve is a line rather than a cloud,
several of them share the unit square without obscuring each other, and the whole
question the plot is asked is which of them is above the others.

The legend goes inside the panel at the bottom right, which is the one corner of a
ROC plot that no useful curve passes through.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.validate import check_flag, check_margin, check_scalar_num, fmt_est
from ._performance import (
    performance_colors,
    performance_input,
    performance_metrics,
    performance_models,
)
from ._theme import figure, font, linestyle, set_margin, theme

__all__ = ["draw_roc_curve"]

#: Where a legend may be placed, as matplotlib spells it.
#:
#: R's ``legend()`` keywords and matplotlib's ``loc`` strings are the same idea
#: under different spellings, so R's are answered here and anything matplotlib
#: already understands is passed straight through.
_LEGEND_POSITIONS = {
    "bottomright": "lower right",
    "bottomleft": "lower left",
    "topright": "upper right",
    "topleft": "upper left",
    "bottom": "lower center",
    "top": "upper center",
    "left": "center left",
    "right": "center right",
    "center": "center",
}

#: The unit square a ROC curve lives in.
_UNIT = (0.0, 1.0)


def draw_roc_curve(
    performance_result: Any,
    models: Any = None,
    anno_auc: bool = False,
    chance: bool = True,
    dark: bool = False,
    col: Any = None,
    lwd: float = 2.0,
    lty: Any = 1,
    legend_pos: Any = "bottomright",
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    cex_axis: float = 1.2,
    cex_lab: float = 1.3,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
    cex_anno: float | None = None,
    margin: Any = (5, 5, 4, 2),
) -> pd.DataFrame:
    """Draw the ROC curves of an evaluated classification.

    The picture of an
    :func:`~statassist.evaluate_classification_models` result: one curve per
    model, all on the rows every model was scored on, with the chance diagonal to
    read them against.

    The curves are ``performance_result["curves"]``, the operating points the
    evaluation already computed, rather than anything recomputed here, so the
    picture and the ``auc`` column of ``metrics`` describe the same curve.
    Consecutive points are joined by straight lines, which is what makes the area
    under the drawn curve the ``auc`` beside it: a run of tied predictions cannot
    be separated by any threshold and the curve crosses it diagonally.

    Args:
        performance_result: A classification evaluation.
        models: Which models to draw and in what order, or ``None`` for all of
            them in the order the evaluation holds, which puts the baseline
            first.
        anno_auc: Whether to add each model's AUC to its legend entry.
        chance: Whether to draw the chance diagonal.
        dark: Whether to draw on a dark background.
        col: One colour, or one per drawn model. ``None`` takes the package
            palette.
        lwd: Width of the curves.
        lty: One R line type, or one per drawn model.
        legend_pos: Where to put the legend, or ``None`` for no legend.
        xlab, ylab, main: Axis and figure labels. ``None`` builds them from the
            evaluation.
        cex_axis, cex_lab, cex_main, cex_legend, cex_anno: Relative text sizes.
            ``cex_anno`` sizes the legend once each entry carries an AUC;
            ``None`` matches ``cex_legend``.
        margin: Plot margins in lines of text: bottom, left, top, right.

    Returns:
        The rows of ``metrics`` that were drawn, in the order they were drawn.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import (
        ...     evaluate_classification_models,
        ...     fit_logistic_regression,
        ...     simulate_classification,
        ... )
        >>> sim = simulate_classification(n_samples=120, n_pred=3, seed=1)
        >>> frame = sim.args["data"]
        >>> fit = fit_logistic_regression(
        ...     frame.iloc[:90],
        ...     outcome=sim.args["outcome"],
        ...     outcome_lv=sim.args["outcome_lv"],
        ...     cv=False,
        ... )
        >>> res = evaluate_classification_models(fit, newdata=frame.iloc[90:])
        >>> drawn = draw_roc_curve(res, anno_auc=True)
        >>> drawn["model"].tolist()
        ['baseline']
    """
    result = performance_input(
        performance_result,
        "classification_performance",
        "performance_result",
        "draw_prediction_plot()",
    )
    anno_auc = check_flag(anno_auc, "anno_auc")
    chance = check_flag(chance, "chance")
    dark = check_flag(dark, "dark")
    lwd = check_scalar_num(lwd, "lwd", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    if cex_anno is None:
        cex_anno = cex_legend
    else:
        cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    margins = check_margin(margin)

    drawn_models = performance_models(result, models)
    n_model = len(drawn_models)
    colors = performance_colors(n_model, col)
    styles = _linestyles(lty, n_model)
    palette = theme(dark)
    metrics = performance_metrics(result, drawn_models)

    fig = figure()
    fig.set_facecolor(palette.bg)
    ax = fig.add_subplot()
    set_margin(fig, margins)
    ax.set_facecolor(palette.bg)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(palette.fg)
    ax.tick_params(colors=palette.fg, labelsize=font(cex_axis))

    if chance:
        ax.plot(_UNIT, _UNIT, color=palette.guide, linewidth=2, linestyle=":")

    curves: pd.DataFrame = result["curves"]
    for position, name in enumerate(drawn_models):
        points = curves.loc[curves["model"].astype(str) == name]
        # Sorted the way the curve is walked rather than the way the thresholds
        # descend, so that a run of tied predictions is crossed once.
        x = 1 - points["specificity"].to_numpy(dtype=float)
        y = points["sensitivity"].to_numpy(dtype=float)
        order = np.lexsort((y, x))
        label = name
        if anno_auc:
            label = f"{name}  ({fmt_est(metrics['auc'].iloc[position])})"
        ax.plot(
            x[order],
            y[order],
            color=colors[position],
            linewidth=lwd,
            linestyle=styles[position],
            label=label,
        )

    ax.set_xlim(_UNIT)
    ax.set_ylim(_UNIT)
    ax.set_xlabel(
        "1 - specificity" if xlab is None else xlab,
        fontsize=font(cex_lab),
        color=palette.fg,
    )
    ax.set_ylabel(
        "sensitivity" if ylab is None else ylab,
        fontsize=font(cex_lab),
        color=palette.fg,
    )
    levels = result["design"]["outcome_lv"]
    ax.set_title(
        f"ROC: {levels[1]} against {levels[0]}" if main is None else main,
        fontsize=font(cex_main),
        color=palette.fg,
    )

    if legend_pos is not None:
        legend = ax.legend(
            loc=_legend_loc(legend_pos),
            frameon=False,
            fontsize=font(cex_anno if anno_auc else cex_legend),
        )
        for text in legend.get_texts():
            text.set_color(palette.fg)

    return cast(pd.DataFrame, metrics)


def _linestyles(lty: Any, n_model: int) -> list[Any]:
    """One line style per drawn model, from one R line type or one each."""
    held = [lty] if isinstance(lty, (str, int, float)) else list(lty)
    if len(held) not in (1, n_model):
        raise SaValueError(f"`lty` must hold one line type, or one per drawn model ({n_model}).")
    return [linestyle(held[index % len(held)], "lty") for index in range(n_model)]


def _legend_loc(legend_pos: Any) -> Any:
    """R's legend keyword as matplotlib's, or whatever matplotlib was given."""
    if isinstance(legend_pos, str):
        return _LEGEND_POSITIONS.get(legend_pos, legend_pos)
    return legend_pos
