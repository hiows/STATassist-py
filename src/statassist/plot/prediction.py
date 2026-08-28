"""Predicted against observed, the picture of a regression evaluation.

Port of ``R/draw_prediction_plot.R``. Two lines carry the whole reading: the
identity, where a perfect prediction would lie, and the calibration line, which
is where the predictions actually lie. The gap between them is the same thing
``metrics`` reports as ``calib_slope`` and ``calib_intercept``, so the plot and the
table cannot disagree.

Several models are panelled rather than overlaid by default. Overlaying two clouds
of points that share an axis produces a third cloud that belongs to neither.
Panelling answers it: each model keeps its own points, and the shared limits are
what makes the panels comparable.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.validate import (
    check_count,
    check_flag,
    check_lim,
    check_margin,
    check_scalar_num,
    fmt_est,
)
from ._performance import (
    performance_colors,
    performance_input,
    performance_metrics,
    performance_models,
)
from ._theme import Theme, figure, font, set_margin, theme

__all__ = ["PREDICTION_VIEWS", "draw_prediction_plot"]

#: The views a prediction plot can be drawn as, ``"auto"`` first.
#:
#: ``"auto"`` is ``"overlay"`` for a single model and ``"panel"`` beyond that,
#: because two clouds of points on shared axes make a third cloud that belongs to
#: neither.
PREDICTION_VIEWS = ("auto", "overlay", "panel")

#: How big a scatter point is, in points squared.
#:
#: matplotlib sizes a marker by area and R by diameter, so R's default ``cex = 1``
#: does not carry over as a number. This is the area that reads as the same dot.
_POINT_AREA = 20.0


def draw_prediction_plot(
    performance_result: Any,
    models: Any = None,
    type: str = "auto",  # noqa: A002
    panel_nrow: int | None = None,
    points: bool = True,
    anno_corr: bool = False,
    anno_rsq: bool = False,
    anno_lm: bool = False,
    dark: bool = False,
    lim: Any = None,
    col: Any = None,
    lwd: float = 2.0,
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
    """Draw predicted against observed for an evaluated regression.

    The picture of an :func:`~statassist.evaluate_regression_models` result: each
    model's predictions against the outcome they were predicting, with the
    identity line to read them against and the calibration line to read the
    identity against.

    Two lines are drawn in every panel. The dotted grey one is the identity, where
    a prediction that was exactly right would lie. The solid coloured one is the
    calibration line, taken from the ``calib_slope`` and ``calib_intercept`` of
    ``metrics`` rather than fitted again here, so the picture and the table cannot
    drift apart. A slope under one is a model whose predictions are squeezed
    towards their own mean, which is the usual shape of a fit scored on rows it
    has not seen.

    Both axes span the same range in every panel, taken over every model drawn, so
    the panels are comparable and the identity line is the diagonal of the square
    rather than an arbitrary chord.

    Args:
        performance_result: A regression evaluation.
        models: Which models to draw and in what order, or ``None`` for all of
            them in the order the evaluation holds.
        type: One of :data:`PREDICTION_VIEWS`.
        panel_nrow: Rows of panels under ``type="panel"``. ``None`` draws them in
            one row.
        points: Whether to draw the predictions themselves. ``False`` leaves the
            calibration lines, which is what makes a crowded overlay readable.
        anno_corr: Whether to report each model's correlation beside its name.
        anno_rsq: Whether to report each model's held-out ``r_squared``. This is
            ``1 - SSE/SST`` on the scored rows, not ``cor`` squared, and the two
            are annotated separately for the same reason the table carries both.
        anno_lm: Whether to report each model's calibration line as an equation.
        dark: Whether to draw on a dark background.
        lim: Range of both axes, or ``None`` to span every value drawn.
        col: One colour, or one per drawn model.
        lwd: Width of the calibration lines.
        xlab, ylab, main: Axis and figure labels. ``None`` builds them from the
            evaluation.
        cex_axis, cex_lab, cex_main, cex_legend, cex_anno: Relative text sizes.
            ``cex_anno`` sizes the labels drawn inside the plot area; ``None``
            matches ``cex_legend``.
        margin: Plot margins in lines of text: bottom, left, top, right.

    Returns:
        The rows of ``metrics`` that were drawn, carrying the resolved ``type`` in
        ``attrs["view"]`` because ``type="auto"`` is settled here.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import (
        ...     evaluate_regression_models,
        ...     fit_linear_regression,
        ...     simulate_regression,
        ... )
        >>> sim = simulate_regression(n_samples=100, n_pred=3, seed=1)
        >>> frame = sim.args["data"]
        >>> fit = fit_linear_regression(frame.iloc[:70], outcome=sim.args["outcome"], cv=False)
        >>> res = evaluate_regression_models(fit, newdata=frame.iloc[70:])
        >>> drawn = draw_prediction_plot(res, anno_lm=True)
        >>> drawn.attrs["view"]
        'overlay'
    """
    result = performance_input(
        performance_result,
        "regression_performance",
        "performance_result",
        "draw_roc_curve()",
    )
    if type not in PREDICTION_VIEWS:
        raise SaValueError("`type` must be one of: " + ", ".join(PREDICTION_VIEWS) + ".")
    points = check_flag(points, "points")
    anno_corr = check_flag(anno_corr, "anno_corr")
    anno_rsq = check_flag(anno_rsq, "anno_rsq")
    anno_lm = check_flag(anno_lm, "anno_lm")
    dark = check_flag(dark, "dark")
    lwd = check_scalar_num(lwd, "lwd", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    if cex_anno is None:
        cex_anno = cex_legend
    else:
        cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    limits = check_lim(lim, "lim")
    margins = check_margin(margin)
    if panel_nrow is not None:
        panel_nrow = check_count(panel_nrow, "panel_nrow", 1)

    drawn_models = performance_models(result, models)
    n_model = len(drawn_models)
    if type == "auto":
        type = "panel" if n_model > 1 else "overlay"  # noqa: A001
    colors = performance_colors(n_model, col)
    palette = theme(dark)
    metrics = performance_metrics(result, drawn_models)

    predictions: pd.DataFrame = result["predictions"]
    by_model = [predictions.loc[predictions["model"].astype(str) == name] for name in drawn_models]
    span = limits if limits is not None else _span(by_model)

    label_x = f"Observed {result['design']['outcome']}" if xlab is None else xlab
    label_y = "Predicted" if ylab is None else ylab
    notes = [
        _annotation(metrics.iloc[position], anno_corr, anno_rsq, anno_lm)
        for position in range(n_model)
    ]

    fig = figure()
    fig.set_facecolor(palette.bg)
    if type == "overlay":
        ax = fig.add_subplot()
        set_margin(fig, margins)
        _dress(ax, palette, span, label_x, label_y, cex_axis, cex_lab)
        ax.set_title(
            "Predicted against observed" if main is None else main,
            fontsize=font(cex_main),
            color=palette.fg,
        )
        for position, name in enumerate(drawn_models):
            _draw_model(
                ax,
                by_model[position],
                metrics.iloc[position],
                colors[position],
                points,
                lwd,
                label=name,
            )
        # One line per model, in that model's colour and in the order the legend
        # beside it lists them, so the two read together without the names being
        # written twice.
        if any(note for note in notes):
            _annotate(ax, [", ".join(note) for note in notes], colors, cex_anno)
        legend = ax.legend(loc="upper left", frameon=False, fontsize=font(cex_legend))
        for text in legend.get_texts():
            text.set_color(palette.fg)
    else:
        n_row = min(1 if panel_nrow is None else panel_nrow, n_model)
        n_col = math.ceil(n_model / n_row)
        axes = fig.subplots(n_row, n_col, squeeze=False)
        for position, name in enumerate(drawn_models):
            row, column = divmod(position, n_col)
            ax = axes[row][column]
            _dress(ax, palette, span, label_x, label_y, cex_axis, cex_lab)
            # A title belongs to the figure rather than to a panel of it, so the
            # panels keep their model names and `main` goes above all of them.
            ax.set_title(name, fontsize=font(cex_main), color=palette.fg)
            _draw_model(
                ax,
                by_model[position],
                metrics.iloc[position],
                colors[position],
                points,
                lwd,
                label=None,
            )
            if notes[position]:
                _annotate(ax, notes[position], [palette.fg] * len(notes[position]), cex_anno)
        for position in range(n_model, n_row * n_col):
            row, column = divmod(position, n_col)
            axes[row][column].set_visible(False)
        if main is not None:
            fig.suptitle(main, fontsize=font(cex_main), fontweight="bold", color=palette.fg)
        # Every panel carries both axis labels, as R's do, and a grid of them does
        # not fit the room `margin` reserves for a single square. The margins are
        # what the outer edge keeps; the space between panels is measured here.
        fig.tight_layout(rect=(0, 0, 1, 0.94 if main is not None else 1))

    # The view is carried on the result because `type="auto"` resolves it here and
    # the caller would otherwise have no way to find out which of the two it got,
    # short of counting the panels on the figure.
    metrics.attrs["view"] = type
    return cast(pd.DataFrame, metrics)


def _span(by_model: list[pd.DataFrame]) -> tuple[float, float]:
    """One range over every model drawn, for both axes.

    Shared so that the panels are comparable and the identity is the diagonal of
    the square rather than a chord across it.
    """
    values = np.concatenate(
        [
            np.concatenate(
                (
                    table["observed"].to_numpy(dtype=float),
                    table["predicted"].to_numpy(dtype=float),
                )
            )
            for table in by_model
        ]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SaValueError(
            "nothing can be plotted: no scored row has both a finite observed value "
            "and a finite prediction."
        )
    low, high = float(finite.min()), float(finite.max())
    # A single value would give the axes no extent, which matplotlib widens by an
    # arbitrary amount of its own; half a unit either side keeps the square square.
    return (low - 0.5, high + 0.5) if low == high else (low, high)


def _dress(
    ax: Any,
    palette: Theme,
    span: tuple[float, float],
    label_x: str,
    label_y: str,
    cex_axis: float,
    cex_lab: float,
) -> None:
    """The frame, the limits and the identity line every panel carries."""
    ax.set_facecolor(palette.bg)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(palette.fg)
    ax.tick_params(colors=palette.fg, labelsize=font(cex_axis))
    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_xlabel(label_x, fontsize=font(cex_lab), color=palette.fg)
    ax.set_ylabel(label_y, fontsize=font(cex_lab), color=palette.fg)
    # The identity first, so the points and the fitted line sit on top of it.
    ax.plot(span, span, color=palette.guide, linewidth=2, linestyle=":")


def _draw_model(
    ax: Any,
    table: pd.DataFrame,
    row: pd.Series,
    color: Any,
    points: bool,
    lwd: float,
    label: str | None,
) -> None:
    """One model's cloud and its calibration line."""
    if points:
        ax.scatter(
            table["observed"].to_numpy(dtype=float),
            table["predicted"].to_numpy(dtype=float),
            color=color,
            s=_POINT_AREA,
        )
    slope = float(row["calib_slope"])
    # Drawn from the two numbers the table holds rather than from a fresh fit, so
    # that the line and `metrics` are the same line. A model the rows defeated has
    # no line, and a legend entry is still owed for it.
    if math.isnan(slope):
        if label is not None:
            ax.plot([], [], color=color, linewidth=lwd, label=label)
        return
    intercept = float(row["calib_intercept"])
    edges = np.asarray(ax.get_xlim(), dtype=float)
    ax.plot(edges, intercept + slope * edges, color=color, linewidth=lwd, label=label)


def _annotation(row: pd.Series, anno_corr: bool, anno_rsq: bool, anno_lm: bool) -> list[str]:
    """What a model has written beside it, in the order R writes it."""
    note = []
    if anno_corr:
        note.append(f"Corr = {fmt_est(row['cor'])}")
    if anno_rsq:
        note.append(f"R-sq = {fmt_est(row['r_squared'])}")
    if anno_lm:
        note.append(_calibration_label(row))
    return note


def _calibration_label(row: pd.Series) -> str:
    """The equation of a calibration line, written the way it reads.

    Port of ``sa_calibration_label()``.
    """
    slope = float(row["calib_slope"])
    if math.isnan(slope):
        return "no calibration line"
    intercept = float(row["calib_intercept"])
    sign = "- " if intercept < 0 else "+ "
    return f"y = {fmt_est(slope)}x {sign}{fmt_est(abs(intercept))}"


def _annotate(ax: Any, lines: list[str], colors: list[Any], cex_anno: float) -> None:
    """Write the annotations into the bottom right of a panel.

    R puts them there with ``legend("bottomright")``. matplotlib has one legend
    per axes and the model names have already claimed it in the overlay view, so
    the lines are placed as text instead, which also lets each one keep its own
    colour.
    """
    size = font(cex_anno)
    for offset, (text, color) in enumerate(zip(lines, colors, strict=True)):
        if not text:
            continue
        ax.text(
            0.98,
            0.02 + offset * 0.06,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=color,
            fontsize=size,
        )
