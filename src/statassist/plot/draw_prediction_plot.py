"""Predicted vs observed plot for regression evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_prediction_plot(
    performance_result: dict[str, Any],
    *,
    models: list[str] | None = None,
    type: str = "auto",
    points: bool = True,
    dark: bool = False,
    col: list[str] | None = None,
) -> pd.DataFrame:
    if performance_result.get("analysis") != "regression_performance":
        raise ValueError(
            "`performance_result` must be a regression evaluation from evaluate_regression_models()."
        )
    if models is None:
        models = performance_result["models"]
    preds = performance_result["predictions"]
    metrics = performance_result["metrics"].set_index("model")

    if type == "auto":
        layout = "overlay" if len(models) == 1 else "panel"
    else:
        layout = type

    bg = "#1a1a1a" if dark else "white"
    fg = "white" if dark else "black"
    guide = "#888888"

    all_obs = preds.loc[preds["model"].isin(models), "observed"]
    all_pred = preds.loc[preds["model"].isin(models), "predicted"]
    lim = (min(all_obs.min(), all_pred.min()), max(all_obs.max(), all_pred.max()))

    if col is None:
        colours = plt.cm.tab10(np.linspace(0, 1, max(len(models), 2)))[: len(models)]
    else:
        colours = col

    if layout == "panel":
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), facecolor=bg)
        if len(models) == 1:
            axes = [axes]
        for ax, model, colour in zip(axes, models, colours):
            ax.set_facecolor(bg)
            sub = preds[preds["model"] == model]
            row = metrics.loc[model]
            _panel(ax, sub["observed"], sub["predicted"], row, lim, colour, guide, fg, points)
            ax.set_title(model, color=fg)
    else:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=bg)
        ax.set_facecolor(bg)
        for model, colour in zip(models, colours):
            sub = preds[preds["model"] == model]
            row = metrics.loc[model]
            _panel(ax, sub["observed"], sub["predicted"], row, lim, colour, guide, fg, points)
        ax.set_title("Predicted vs observed", color=fg)

    fig.tight_layout()
    plt.show()
    return metrics.loc[models].reset_index()


def _panel(ax, observed, predicted, row, lim, colour, guide, fg, draw_points):
    ax.plot(lim, lim, linestyle=":", color=guide, linewidth=2)
    if draw_points:
        ax.scatter(observed, predicted, color=colour, s=20, alpha=0.8)
    if not np.isnan(row["calib_slope"]):
        xs = np.array(lim)
        ys = row["calib_intercept"] + row["calib_slope"] * xs
        ax.plot(xs, ys, color=colour, linewidth=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed", color=fg)
    ax.set_ylabel("Predicted", color=fg)
    ax.tick_params(colors=fg)
