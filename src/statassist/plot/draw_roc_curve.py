"""ROC curve plot for classification evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_roc_curve(
    performance_result: dict[str, Any],
    *,
    models: list[str] | None = None,
    anno_auc: bool = False,
    chance: bool = True,
    dark: bool = False,
    col: list[str] | None = None,
    lwd: float = 2,
    legend_pos: str | None = "lower right",
) -> pd.DataFrame:
    if performance_result.get("analysis") != "classification_performance":
        raise ValueError(
            "`performance_result` must be a classification evaluation from "
            "evaluate_classification_models()."
        )
    if models is None:
        models = performance_result["models"]
    curves = performance_result["curves"]
    metrics = performance_result["metrics"].set_index("model")

    bg = "#1a1a1a" if dark else "white"
    fg = "white" if dark else "black"

    if col is None:
        colours = plt.cm.tab10(np.linspace(0, 1, max(len(models), 2)))[: len(models)]
    else:
        colours = col

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=bg)
    ax.set_facecolor(bg)

    if chance:
        ax.plot([0, 1], [1, 0], linestyle=":", color="#888888", linewidth=1)

    labels = []
    for model, colour in zip(models, colours):
        sub = curves[curves["model"] == model]
        ax.plot(1 - sub["specificity"], sub["sensitivity"], color=colour, linewidth=lwd)
        label = model
        if anno_auc:
            label = f"{model} (AUC = {metrics.loc[model, 'auc']:.3f})"
        labels.append(label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity", color=fg)
    ax.set_ylabel("Sensitivity", color=fg)
    ax.set_title("ROC curve", color=fg)
    ax.tick_params(colors=fg)
    if legend_pos:
        ax.legend(labels, loc=legend_pos.replace("bottomright", "lower right"), facecolor=bg, labelcolor=fg)

    fig.tight_layout()
    plt.show()
    return metrics.loc[models].reset_index()
