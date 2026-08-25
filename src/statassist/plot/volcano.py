"""The volcano plot: effect size against significance.

Port of ``R/draw_volcano_plot.R``, for the readings this port can produce. A
term reading arrives with the factorial scenario, and with it the one-panel-per-
term figure R draws for it; a mapping of contrast tables is sent back to name
the one to draw, exactly as R does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.result import SaSignificance, verdict_effect_col
from ..core.validate import (
    check_feat_names,
    check_flag,
    check_lim,
    check_margin,
    check_pvalues,
    check_scalar_num,
)
from ._theme import figure, font, set_margin

__all__ = ["draw_volcano_plot"]

#: The colours a point is drawn in: up, down, neither, and a magnitude hit.
#: R's ``grey70`` and ``green3`` are given as hex, since matplotlib does not
#: know those names. Term panels have no direction, so their significant points
#: are black rather than red or blue.
UP_COLOR = "#D73027"
DOWN_COLOR = "#4575B4"
NS_COLOR = "#B3B3B3"
MAG_COLOR = "black"

# Labels sit on top of the points, so they use a purer and brighter shade than
# the points do rather than the same one, which would blend in. Under a term
# reading the labels stay red so they stay legible without being read as
# up-regulation.
_UP_LABEL = "red"
_DOWN_LABEL = "blue"
_GUIDE_COLOR = "#00CD00"


def draw_volcano_plot(
    significance_result: Any,
    terms: Any = None,
    panel_nrow: int | None = None,
    use_adjusted: bool = True,
    log2fc_cutoff: float | None = None,
    pval_cutoff: float | None = None,
    anno_feats: bool = True,
    anno_top: int = 10,
    cex_anno: float = 1.0,
    xlim: Any = None,
    ylim: Any = None,
    xlab: str | None = None,
    main: str | None = None,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    margin: Any = (5, 5, 4, 3),
) -> None:
    """Plot effect size against ``-log10(pvalue)`` and label the strongest features.

    A term reading (``estimate_significance(..., by="term")``) is drawn as one
    panel per term. A contrast reading still needs one table named explicitly.

    On an omnibus or contrast reading the effect is signed ``log2fc`` and points
    are coloured by direction. On a term panel the x axis is ``|log2_effect|``
    with a single significance colour: significant points are black and labels
    stay red so they are not read as up-regulation.

    Args:
        significance_result: The object returned by
            :func:`~statassist.estimate_significance`, or a bare verdict frame.
        terms: Term labels to draw under a term reading, or ``None`` for the
            default (first two main effects and their interaction).
        panel_nrow: Rows for the term-panel layout, or ``None`` for one row.
        use_adjusted: If ``True``, plot and threshold ``adj_pvalue``.
        log2fc_cutoff: Effect cutoff; ``None`` takes the recorded value. Under
            a term reading the magnitude cutoff is applied to ``abs(log2_effect)``.
        pval_cutoff: Significance cutoff; ``None`` takes the recorded value.
        anno_feats: If ``True``, label the strongest significant features.
        anno_top: How many features to label in each direction (up to
            ``2 * anno_top`` in total), or how many significant features to
            label under a term reading.
        cex_anno: Character expansion for those labels.
        xlim: Length-2 x axis range, or ``None`` to derive it.
        ylim: The same for the y axis.
        xlab: X axis label, or ``None`` to derive it.
        main: Plot title (figure title when drawing term panels).
        cex_lab, cex_axis, cex_main: Character expansion multipliers.
        margin: Plot margins in lines of text: bottom, left, top, right.
    """
    use_adjusted = check_flag(use_adjusted, "use_adjusted")
    anno_feats = check_flag(anno_feats, "anno_feats")
    anno_top = int(check_scalar_num(anno_top, "anno_top", 0))
    cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    cex_lab = check_scalar_num(cex_lab, "cex_lab", 0, lower_open=True)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    margins = check_margin(margin)
    x_limits = check_lim(xlim, "xlim")
    y_limits = check_lim(ylim, "ylim")
    if panel_nrow is not None:
        from ..core.validate import check_count

        panel_nrow = check_count(panel_nrow, "panel_nrow", 1)
    if terms is not None:
        if not isinstance(terms, (list, tuple)) and not (
            isinstance(terms, np.ndarray) and terms.dtype.kind in "UO"
        ):
            if isinstance(terms, str):
                terms = [terms]
            else:
                raise SaValueError("`terms` must be NULL or a character vector of term labels.")
        terms = [str(name) for name in terms]
        if not terms or any(name == "" for name in terms):
            raise SaValueError("`terms` must be NULL or a character vector of term labels.")

    if isinstance(significance_result, SaSignificance):
        significance_result = significance_result["significance"]

    if isinstance(significance_result, Mapping) and not isinstance(
        significance_result, pd.DataFrame
    ):
        tables = _volcano_term_tables(significance_result)
        selected = {name: tables[name] for name in _volcano_terms(tables, terms)}
        _volcano_panels(
            selected,
            panel_nrow,
            use_adjusted,
            log2fc_cutoff,
            pval_cutoff,
            anno_feats,
            anno_top,
            cex_anno,
            x_limits,
            y_limits,
            xlab,
            main,
            cex_lab,
            cex_axis,
            cex_main,
            margins,
        )
        return

    if not isinstance(significance_result, pd.DataFrame):
        raise SaValueError(_naming_message(significance_result))

    _volcano_one(
        significance_result,
        use_adjusted,
        log2fc_cutoff,
        pval_cutoff,
        anno_feats,
        anno_top,
        cex_anno,
        x_limits,
        y_limits,
        xlab,
        main,
        cex_lab,
        cex_axis,
        cex_main,
        margins,
        ax=None,
    )


def _volcano_one(
    table: pd.DataFrame,
    use_adjusted: bool,
    log2fc_cutoff: float | None,
    pval_cutoff: float | None,
    anno_feats: bool,
    anno_top: int,
    cex_anno: float,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
    xlab: str | None,
    main: str | None,
    cex_lab: float,
    cex_axis: float,
    cex_main: float,
    margins: tuple[float, float, float, float],
    ax: Any = None,
) -> None:
    """Draw one volcano panel on ``ax``, or on a fresh figure when ``ax`` is None."""
    p_col = "adj_pvalue" if use_adjusted else "pvalue"
    effect_col = verdict_effect_col(table)
    magnitude = effect_col == "log2_effect"
    _check_columns(table, p_col, effect_col)
    cut_fc, cut_p = _cutoffs(table, log2fc_cutoff, pval_cutoff)

    features = [str(name) for name in table["features"]]
    check_feat_names(features)
    effect = table[effect_col].to_numpy(dtype=float)
    mag = np.abs(effect) if magnitude else effect
    pvalue = table[p_col].to_numpy(dtype=float)
    check_pvalues(pvalue, p_col)

    with np.errstate(divide="ignore", invalid="ignore"):
        neglog_p = -np.log10(pvalue)
    x_limits, y_limits = _limits(
        mag, neglog_p, cut_fc, cut_p, x_limits, y_limits, magnitude, effect_col, p_col
    )
    label_offset = (y_limits[1] - y_limits[0]) * 0.05

    plot_y = neglog_p.copy()
    plot_x = mag.copy()
    capped_y = np.isinf(plot_y)
    capped_x = np.isinf(plot_x)
    plot_y[capped_y] = y_limits[1] - 2 * label_offset
    span_x = x_limits[1] - x_limits[0]
    if magnitude:
        plot_x[capped_x] = x_limits[1] - span_x * 0.02
    else:
        plot_x[capped_x] = np.where(
            plot_x[capped_x] > 0, x_limits[1] - span_x * 0.02, x_limits[0] + span_x * 0.02
        )
    n_capped_y = int(capped_y.sum())
    n_capped_x = int(capped_x.sum())
    if n_capped_y or n_capped_x:
        detail = ""
        if n_capped_y:
            detail += f" ({n_capped_y} with p = 0)"
        if n_capped_x:
            detail += f" ({n_capped_x} with an infinite {effect_col})"
        notify(
            f"Drew {max(n_capped_y, n_capped_x)} point(s) at the edge of the plot"
            f"{detail}; their true position is off the axis."
        )

    with np.errstate(invalid="ignore"):
        passes_p = ~np.isnan(pvalue) & (pvalue <= cut_p)
        if magnitude:
            is_up = passes_p & ~np.isnan(mag) & (mag >= cut_fc)
            is_down = np.zeros(len(is_up), dtype=bool)
        else:
            is_up = passes_p & ~np.isnan(mag) & (mag >= cut_fc)
            is_down = passes_p & ~np.isnan(mag) & (mag <= -cut_fc)

    if ax is None:
        fig = figure()
        ax = fig.add_subplot()
        set_margin(fig, margins)

    ax.scatter(plot_x, plot_y, color=NS_COLOR, s=20)
    if magnitude:
        ax.scatter(plot_x[is_up], plot_y[is_up], color=MAG_COLOR, s=20)
    else:
        ax.scatter(plot_x[is_up], plot_y[is_up], color=UP_COLOR, s=20)
        ax.scatter(plot_x[is_down], plot_y[is_down], color=DOWN_COLOR, s=20)
    ax.axhline(-np.log10(cut_p), color=_GUIDE_COLOR, linewidth=2, linestyle=":")
    if magnitude:
        ax.axvline(cut_fc, color=_GUIDE_COLOR, linewidth=2, linestyle=":")
    else:
        for edge in (-cut_fc, cut_fc):
            ax.axvline(edge, color=_GUIDE_COLOR, linewidth=2, linestyle=":")

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel(_xlab(table) if xlab is None else xlab, fontsize=font(cex_lab))
    y_lab = r"$-\log_{10}$ adjusted $P$" if use_adjusted else r"$-\log_{10}\,P$"
    ax.set_ylabel(y_lab, fontsize=font(cex_lab))
    ax.tick_params(labelsize=font(cex_axis))
    if main is not None:
        ax.set_title(main, fontsize=font(cex_main))

    if anno_feats and anno_top >= 1:
        if magnitude:
            picked = _strongest(is_up, pvalue, mag, anno_top, up=True)
            picked_down = np.array([], dtype=int)
        else:
            picked = _strongest(is_up, pvalue, mag, anno_top, up=True)
            picked_down = _strongest(is_down, pvalue, mag, anno_top, up=False)
        if picked.size == 0 and picked_down.size == 0:
            notify("No feature clears both cutoffs, so nothing was labelled.")
        for index, colour in ((picked, _UP_LABEL), (picked_down, _DOWN_LABEL)):
            for i in index:
                ax.text(
                    plot_x[i],
                    plot_y[i] + label_offset,
                    features[i],
                    color=colour,
                    fontsize=font(cex_anno),
                    ha="center",
                )


def _volcano_term_tables(held: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """The verdict tables of a term reading, or a refusal for a contrast list."""
    if not held or not all(isinstance(table, pd.DataFrame) for table in held.values()):
        raise SaValueError(
            "`significance_result` must be the object returned by estimate_significance()."
        )
    labels: list[str] = []
    for table in held.values():
        term = table.attrs.get("term")
        labels.append(str(term) if term is not None else "")
    if any(label == "" for label in labels):
        raise SaValueError(_naming_message(held))
    return {label: table for label, table in zip(labels, held.values(), strict=True)}


def _volcano_terms(tables: Mapping[str, pd.DataFrame], terms: list[str] | None) -> list[str]:
    """Which terms earn a panel."""
    labels = list(tables)
    if terms is not None:
        unknown = [name for name in terms if name not in labels]
        if unknown:
            raise SaValueError(
                "`terms` names term(s) the verdict does not hold: "
                + ", ".join(unknown)
                + ". It holds "
                + ", ".join(labels)
                + "."
            )
        return list(dict.fromkeys(terms))

    orders = [int(tables[name].attrs.get("term_order", 0)) for name in labels]
    mains = [name for name, order in zip(labels, orders, strict=True) if order == 1]
    if len(mains) < 2:
        return labels
    pair = mains[:2]
    wanted = [name for name in [pair[0], pair[1], f"{pair[0]}:{pair[1]}"] if name in tables]
    left_out = [name for name in labels if name not in wanted]
    if left_out:
        notify(
            f"Drew the terms of {pair[0]} and {pair[1]}, leaving out "
            + ", ".join(left_out)
            + ". Name terms in `terms` to draw those instead."
        )
    return wanted


def _volcano_panels(
    tables: Mapping[str, pd.DataFrame],
    panel_nrow: int | None,
    use_adjusted: bool,
    log2fc_cutoff: float | None,
    pval_cutoff: float | None,
    anno_feats: bool,
    anno_top: int,
    cex_anno: float,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    xlab: str | None,
    main: str | None,
    cex_lab: float,
    cex_axis: float,
    cex_main: float,
    margins: tuple[float, float, float, float],
) -> None:
    """One volcano plot per term, on one figure."""
    p_col = "adj_pvalue" if use_adjusted else "pvalue"
    first = next(iter(tables.values()))
    effect_col = verdict_effect_col(first)
    magnitude = effect_col == "log2_effect"
    for table in tables.values():
        _check_columns(table, p_col, verdict_effect_col(table))

    cut_fc, cut_p = _cutoffs(first, log2fc_cutoff, pval_cutoff)

    # Shared axes across panels unless the caller fixed them.
    if xlim is None or ylim is None:
        all_mag: list[float] = []
        all_neg: list[float] = []
        for table in tables.values():
            col = verdict_effect_col(table)
            effect = table[col].to_numpy(dtype=float)
            mag = np.abs(effect) if col == "log2_effect" else effect
            pvalue = table[p_col].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                neglog_p = -np.log10(pvalue)
            all_mag.extend(mag[np.isfinite(mag)].tolist())
            all_neg.extend(neglog_p[np.isfinite(neglog_p)].tolist())
        shared_x, shared_y = _limits(
            np.asarray(all_mag, dtype=float),
            np.asarray(all_neg, dtype=float),
            cut_fc,
            cut_p,
            xlim,
            ylim,
            magnitude,
            effect_col,
            p_col,
        )
    else:
        shared_x, shared_y = xlim, ylim

    n_panel = len(tables)
    n_row = min(1 if panel_nrow is None else panel_nrow, n_panel)
    n_col = int(np.ceil(n_panel / n_row))

    fig = figure()
    axes = fig.subplots(n_row, n_col, squeeze=False)
    if main is not None:
        fig.suptitle(main, fontsize=font(cex_main), fontweight="bold")
        fig.subplots_adjust(top=0.88)

    for index, (name, table) in enumerate(tables.items()):
        row, column = divmod(index, n_col)
        _volcano_one(
            table,
            use_adjusted,
            cut_fc,
            cut_p,
            anno_feats,
            anno_top,
            cex_anno,
            shared_x,
            shared_y,
            xlab,
            name,
            cex_lab,
            cex_axis,
            cex_main,
            margins,
            ax=axes[row][column],
        )
    for index in range(n_panel, n_row * n_col):
        row, column = divmod(index, n_col)
        axes[row][column].set_visible(False)


def _naming_message(held: Any) -> str:
    """What to say to a caller who handed over a whole contrast reading."""
    if isinstance(held, Mapping) and held:
        first = next(iter(held))
        return (
            "`significance_result` holds one verdict table per contrast, and a volcano "
            f'plot draws one of them. Name it: `sig["significance"]["{first}"]`.'
        )
    return "`significance_result` must be the object returned by estimate_significance()."


def _check_columns(table: pd.DataFrame, p_col: str, effect_col: str) -> None:
    """The columns a verdict table has to carry to be plotted."""
    absent = [name for name in ("features", effect_col, p_col) if name not in table.columns]
    if absent:
        raise SaValueError(
            "`significance_result` is missing the column(s) "
            + ", ".join(absent)
            + ". Pass the table returned by estimate_significance()."
        )


def _cutoffs(
    table: pd.DataFrame, log2fc_cutoff: float | None, pval_cutoff: float | None
) -> tuple[float, float]:
    """The rule a plot draws its guides for."""
    if log2fc_cutoff is None:
        log2fc_cutoff = table.attrs.get("log2fc_cutoff")
    if pval_cutoff is None:
        pval_cutoff = table.attrs.get("pval_cutoff")
    if log2fc_cutoff is None or pval_cutoff is None:
        raise SaValueError(
            "`significance_result` does not carry the cutoffs estimate_significance() "
            "records, so `log2fc_cutoff` and `pval_cutoff` must be supplied. Selecting "
            "columns from the table drops them."
        )
    return (
        check_scalar_num(log2fc_cutoff, "log2fc_cutoff", 0),
        check_scalar_num(pval_cutoff, "pval_cutoff", 0, 1, lower_open=True),
    )


def _limits(
    mag: np.ndarray,
    neglog_p: np.ndarray,
    cut_fc: float,
    cut_p: float,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    magnitude: bool,
    effect_col: str,
    p_col: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis ranges wide enough for the points and the guides both."""
    if xlim is not None and ylim is not None:
        return xlim, ylim

    y_finite = neglog_p[np.isfinite(neglog_p)]
    x_finite = mag[np.isfinite(mag)]
    if y_finite.size == 0 or x_finite.size == 0:
        raise SaValueError(
            "nothing can be plotted: no feature has both a finite "
            f"`{effect_col}` and a finite -log10(`{p_col}`)."
        )

    y_top = max(float(y_finite.max()), float(-np.log10(cut_p)), 1.0)
    x_max = max(float(np.abs(x_finite).max()), cut_fc)
    if magnitude:
        derived_x = (0.0, x_max * 1.05)
    else:
        derived_x = (-x_max * 1.05, x_max * 1.05)
    return (
        xlim if xlim is not None else derived_x,
        ylim if ylim is not None else (0.0, y_top * 1.1),
    )


def _strongest(
    mask: np.ndarray, pvalue: np.ndarray, mag: np.ndarray, anno_top: int, up: bool
) -> np.ndarray:
    """The features to label on one side."""
    index = np.flatnonzero(mask)
    if index.size == 0:
        return index
    second = -mag[index] if up else mag[index]
    order = np.lexsort((second, pvalue[index]))
    return index[order][:anno_top]


def _xlab(table: pd.DataFrame) -> str:
    """The x axis label a verdict table earns."""
    plain = r"$\log_2$ FC"
    term = table.attrs.get("term")
    if term is not None:
        return rf"$|\log_2\,\mathrm{{effect}}|$ ({term})"
    if table.attrs.get("contrast") is not None:
        return plain
    analysis = table.attrs.get("analysis")
    if analysis == "multi_group_comparison":
        unit = "level"
    elif analysis == "factorial_comparison":
        unit = "cell"
    else:
        return plain
    levels = table.attrs.get("group_lv") or []
    reference = str(levels[0]) if len(levels) > 0 else "reference"
    return f"{plain} (most extreme {unit} vs {reference})"
