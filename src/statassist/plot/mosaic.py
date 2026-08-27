"""The mosaic plot: a contingency table drawn as area.

Port of ``R/draw_mosaic_plot.R``. The geometry, the colours, the cuts and what
comes back are R's. Two things are approximations of what base R does exactly,
and both are about measuring text: whether a tile has room for its annotation is
judged from :data:`~statassist.plot._theme.CHAR_WIDTH` rather than from
``strwidth()``, and the residual key is drawn in a gridspec panel at a fixed
character expansion rather than fitted to the panel ``layout()`` handed out. What
is drawn and what is reported are the same either way.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from ..core.errors import SaValueError
from ..core.result import SaCategorical, SaComparison
from ..core.validate import check_flag, check_scalar_num, fmt_num
from ._theme import CHAR_WIDTH, LINE_HEIGHT, figure, font, line_inches, set_margin, theme
from .volcano import DOWN_COLOR, UP_COLOR

__all__ = ["ANNO_MODES", "MOSAIC_COLORS", "RESIDUAL_BREAKS", "RESIDUALS", "draw_mosaic_plot"]

#: Which residual the shading can read, in the order R lists them.
RESIDUALS: tuple[str, ...] = ("pearson", "standardized")

#: What can be written on a tile, in the order R lists them.
ANNO_MODES: tuple[str, ...] = ("auto", "count", "percent", "both", "none")

#: The cuts the shading reads a residual at.
#:
#: 2 and 4 are what a residual referred to a standard normal would call
#: surprising and extreme. Pearson residuals grow with the sample, so a large
#: table lights up more tiles than a small one of the same shape - the same fact
#: the chi-square statistic reports - which is why the key names the residual it
#: is a scale for and not only the cuts.
RESIDUAL_BREAKS: tuple[float, ...] = (-np.inf, -4.0, -2.0, 2.0, 4.0, np.inf)

#: What each band of :data:`RESIDUAL_BREAKS` is called in the key.
RESIDUAL_LABELS: tuple[str, ...] = ("< -4", "-4 to -2", "-2 to 2", "2 to 4", "> 4")

#: The colour of each band, from the most negative residual to the most positive.
#:
#: The two extremes are the colours :func:`~statassist.draw_volcano_plot` already
#: draws a feature that moved down and one that moved up in, so "less than
#: expected" and "more than expected" are the pair of colours the rest of the
#: package reads as a direction. The middle is R's ``gray88``.
MOSAIC_COLORS: tuple[str, ...] = (DOWN_COLOR, "#A6C5DE", "#E0E0E0", "#F4A6A0", UP_COLOR)

#: The largest share of an axis every gap together may take.
#:
#: ``gap`` is one gap rather than all of them, so that a five-level axis is not
#: five times finer than a two-level one. This is what keeps the tiles from
#: vanishing when a wide gap meets many levels.
GAP_MAX_SHARE = 0.4

#: Largest ``gap`` a caller may ask for.
GAP_MAX = 0.2

#: How much of a tile an annotation may take up before it is dropped.
ANNO_FIT = 0.92

#: The largest share of the figure the residual key may reserve.
KEY_MAX_SHARE = 0.32

#: What share of the key panel is left free at the left and at the right.
KEY_PAD = (0.06, 0.02)

#: R's greys, as the hex matplotlib needs: ``gray20``, ``gray30``, ``gray92``
#: and ``grey15``, in that order. The values are R's own.
_TILE_BORDER = "#333333"
_KEY_BORDER = "#4D4D4D"
_EMPTY_FILL = "#EBEBEB"
_DARK_INK = "#262626"

#: The slate a dark mosaic draws an unshaded or an empty tile in.
_DARK_FILL = "#36454F"

#: Luminance below which a fill needs light ink on it.
_INK_CUTOFF = 0.5

#: How a fill's luminance is weighed out of its channels, ITU-R BT.601.
_LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)


class _Layout(NamedTuple):
    """A mosaic placed in the unit square.

    Attributes:
        cells: The cell table with ``x1``, ``x2``, ``y1`` and ``y2`` added.
        widths: Marginal share of each strip, before its gap is taken out.
        heights: Conditional share of each tile within its strip.
        expected_prop: The same shares the null hypothesis expects.
        expected_y: Where the null would have cut each strip, one boundary fewer
            than there are tiles.
        strip_x: Left and right edge of each strip.
        empty_levels: The row and column levels holding no observation.
        null: Which hypothesis ``expected_prop`` states.
        row_lv: Row levels, in the order the comparison settled.
        col_lv: Column levels, likewise.
        x_at: Where each strip's axis label goes.
        y_at: Where each column level's axis label goes, read off the reference
            strip.
    """

    cells: pd.DataFrame
    widths: pd.Series
    heights: pd.DataFrame
    expected_prop: pd.DataFrame
    expected_y: np.ndarray
    strip_x: np.ndarray
    empty_levels: dict[str, list[str]]
    null: str
    row_lv: list[str]
    col_lv: list[str]
    x_at: np.ndarray
    y_at: np.ndarray


def draw_mosaic_plot(
    categorical_comparison_result: Any,
    shade: bool = True,
    residual: str = RESIDUALS[0],
    expected_line: bool = True,
    anno_cells: Any = ANNO_MODES[0],
    gap: float = 0.015,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
    cex_anno: float = 1.0,
    dark: bool = False,
) -> dict[str, Any]:
    """Draw a mosaic plot of a contingency table.

    Splits the x axis by the first variable's marginal shares and each strip by
    the second variable's conditional shares, so the area of a tile is the cell's
    share of the table. Three things are drawn on top of that geometry, and each
    of them answers a question the geometry alone leaves open.

    The **shading** says which cells made the statistic what it is. It reads the
    Pearson residual, the quantity that squares and sums to that statistic, at the
    conventional cuts of 2 and 4. The two colours are the ones
    :func:`~statassist.draw_volcano_plot` already uses for a feature that moved up
    and one that moved down, so "more than expected" and "less than expected" are
    the same pair of colours the rest of the package reads as a direction.

    The **expected line** says what "expected" was. A dotted segment sits at each
    boundary the tiles of a strip would have had under the null hypothesis, so the
    departure is the distance between a tile edge and the line beside it rather
    than something to be inferred by comparing strips by eye. Under independence
    the lines fall at the same heights in every strip, which is what makes an
    association visible at a glance; under symmetry they do not, because there the
    expectation is a cell against its own transpose.

    The **annotation** says how many observations a tile stands for, which area
    cannot: a wide short tile and a narrow tall one can hold the same count.

    Which null hypothesis is drawn is the one the result was tested against,
    ``categorical_comparison_result.design["null"]``. That is the point of reading
    it off the result rather than recomputing it here: a matched design is tested
    for symmetry, so a mosaic of it shaded by departure from independence would be
    a picture of a hypothesis nothing in the result has a p-value for. A bare table
    carries no such hypothesis, which is why one is not accepted:
    :func:`~statassist.compare_categorical_groups` is what settles the null, the
    levels and their order, and this function draws what it settled.

    Args:
        categorical_comparison_result: A categorical comparison result, as
            :func:`~statassist.compare_categorical_groups` returns. The levels
            that take part, their order and the null hypothesis the shading is
            read under are all settled there, so there is no ``category_lv`` or
            ``control_label`` to restate here.
        shade: If ``False``, every tile is drawn in the background colour and the
            residual key is omitted.
        residual: Which residual the shading reads, one of :data:`RESIDUALS`.
            ``"pearson"`` squares and sums to the test statistic, so it says how
            the statistic was made up. ``"standardized"`` is referred to a
            standard normal, so it says which cells are individually surprising,
            and it is only defined when the null is a statement about the margins.
        expected_line: If ``True``, mark each strip at the tile boundaries the
            null hypothesis expects.
        anno_cells: What to write on a tile, one of :data:`ANNO_MODES`.
            ``"count"`` is the observed count, ``"percent"`` the tile's share of
            its own strip, ``"both"`` puts one over the other, and ``"none"``
            writes nothing. ``"auto"``, the default, writes as much as the tile
            has room for, measured against the label rather than against a fixed
            fraction of the plot. ``True`` and ``False`` are accepted as
            ``"auto"`` and ``"none"``.
        gap: Gap between neighbouring tiles, as a fraction of the axis, for each
            gap rather than for all of them together. Capped so that the gaps
            never take more than :data:`GAP_MAX_SHARE` of an axis however many
            levels there are.
        xlab: X axis label, or ``None`` to take the row variable's name.
        ylab: The same for the y axis and the column variable.
        main: Plot title, or ``None`` for none.
        cex_lab: Character expansion for the axis labels.
        cex_axis: The same for the level names.
        cex_main: The same for the title.
        cex_legend: The same for the residual key.
        cex_anno: The same for the tile annotation.
        dark: If ``True``, draw on a dark background with light annotation, the
            same palette :func:`~statassist.draw_grouped_boxplot` uses.

    Returns:
        The picture as it was drawn, in the shape R returns invisibly:

        * ``cells`` - the cell table with ``x1``, ``x2``, ``y1``, ``y2`` and
          ``fill`` added, in the order the tiles were painted.
        * ``widths`` - marginal share of each strip, indexed by row level. These
          are the widths before the gaps are taken out of them.
        * ``heights`` - conditional share of each tile within its strip, as a
          frame of row level by column level.
        * ``expected_prop`` - the same shares the null hypothesis expects, which
          is what the dotted segments were drawn from.
        * ``empty_levels`` - ``row`` and ``col`` levels that hold no observation,
          so drew no tile and took no axis label.
        * ``null`` - which hypothesis the shading and the segments are about.
        * ``residual``, ``residual_breaks``, ``colors`` - the scale the shading
          read.

    Raises:
        SaValueError: If an argument is unusable, if the input is not a
            categorical comparison result, or if ``residual="standardized"`` is
            asked of a result tested for symmetry, where that residual is missing
            throughout.

    Notes:
        The level names on the y axis are read off the **reference strip**, the
        first one, which the ``control_label`` and ``category_lv`` of
        :func:`~statassist.compare_categorical_groups` decide. No single set of
        positions can label every strip, since the whole content of a mosaic is
        that the strips are cut at different heights, so labelling one of them and
        saying which is the honest version of the choice. The first strip is the
        one the rest of the package already treats as the reference.

        A level holding no observation has no share to take and no conditional
        distribution to be cut into, so it draws nothing and is reported in
        ``empty_levels`` instead. Its gap is left in place: a space where a level
        should have been is the honest picture of a level that was named and never
        seen.

    Examples:
        >>> import pandas as pd
        >>> from statassist import compare_categorical_groups
        >>> smoking = pd.DataFrame(
        ...     {
        ...         "smoker": ["y"] * 60 + ["n"] * 60,
        ...         "grade": (
        ...             ["high"] * 10 + ["mid"] * 20 + ["low"] * 30
        ...             + ["high"] * 30 + ["mid"] * 20 + ["low"] * 10
        ...         ),
        ...     }
        ... )
        >>> drawn = draw_mosaic_plot(compare_categorical_groups(smoking))

        The tiles cover the unit square but for the gaps, and the dotted segments
        are what independence expected, so the departure is a distance rather
        than a comparison between strips.

        >>> float(drawn["widths"].sum())
        1.0
        >>> drawn["expected_prop"].round(3).to_numpy().tolist()
        [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]
        >>> drawn["null"]
        'independence'
    """
    mode = _anno_mode(anno_cells)
    if residual not in RESIDUALS:
        raise SaValueError("`residual` must be one of: " + ", ".join(RESIDUALS) + ".")

    shade = check_flag(shade, "shade")
    expected_line = check_flag(expected_line, "expected_line")
    dark = check_flag(dark, "dark")
    gap = check_scalar_num(gap, "gap", 0, GAP_MAX)
    cex_lab = check_scalar_num(cex_lab, "cex_lab", 0, lower_open=True)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)

    cells, null, row_var, col_var = _mosaic_input(categorical_comparison_result)
    if residual == RESIDUALS[1] and null == "symmetry":
        raise SaValueError(
            '`residual = "standardized"` has no value under symmetry, which is the '
            "null this result was tested against: the variance correction it "
            "divides by is derived for a table held against its own margins, so "
            "`cells['std_residual']` is NA here. Use `residual = \"pearson\"`, "
            "whose squares sum to McNemar's statistic."
        )

    layout = _layout(cells, gap, null)
    palette = theme(dark)
    if dark:
        border = key_border = ink = "white"
        empty_fill = plain_fill = _DARK_FILL
    else:
        border, key_border, ink = _TILE_BORDER, _KEY_BORDER, _DARK_INK
        empty_fill, plain_fill = _EMPTY_FILL, "white"

    tiles = layout.cells.copy()
    value = tiles["residual" if residual == RESIDUALS[0] else "std_residual"].to_numpy(dtype=float)
    tiles["fill"] = _fill(value, empty_fill) if shade else [plain_fill] * len(tiles.index)

    fig = figure()
    margins = (5, 5, 2 if main is None else 4, 1)
    if shade:
        # Measured against the room the margins leave rather than against the
        # whole figure, which is where R's `layout()` splits: the key is drawn
        # inside the same margins as the tiles here, so a share of the device
        # would reserve a panel its own text does not fit in.
        across = fig.get_size_inches()[0] - (margins[1] + margins[3]) * line_inches()
        key_share = min(KEY_MAX_SHARE, _key_inches(residual, cex_legend) / max(across, 1.0))
        grid = fig.add_gridspec(1, 2, width_ratios=[1 - key_share, key_share], wspace=0.0)
        ax = fig.add_subplot(grid[0, 0])
        key_ax = fig.add_subplot(grid[0, 1])
    else:
        ax = fig.add_subplot()
        key_ax = None
    set_margin(fig, margins)
    if dark:
        fig.patch.set_facecolor(palette.bg)
        ax.set_facecolor(palette.bg)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # The panel's size in inches is what turns a label's width into the axis
    # fractions the tiles are measured in, and it is only settled once the
    # margins are.
    position = ax.get_position()
    fig_width, fig_height = fig.get_size_inches()
    panel = (position.width * fig_width, position.height * fig_height)

    for position_in_table in range(len(tiles.index)):
        tile = tiles.iloc[position_in_table]
        width = float(tile["x2"]) - float(tile["x1"])
        height = float(tile["y2"]) - float(tile["y1"])
        if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
            continue
        ax.add_patch(
            Rectangle(
                (float(tile["x1"]), float(tile["y1"])),
                width,
                height,
                facecolor=str(tile["fill"]),
                edgecolor=border,
                linewidth=1.0,
            )
        )
        label = _anno_label(
            mode, tile["observed"], tile["prop_row"], width, height, cex_anno, panel
        )
        if label is not None:
            ax.text(
                float(tile["x1"]) + width / 2,
                float(tile["y1"]) + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=font(cex_anno),
                color=_ink(str(tile["fill"])),
            )

    if expected_line:
        _draw_expected(ax, layout, palette.guide)

    keep_row = [level not in layout.empty_levels["row"] for level in layout.row_lv]
    keep_col = [level not in layout.empty_levels["col"] for level in layout.col_lv]
    ax.set_xticks(
        [at for at, keep in zip(layout.x_at, keep_row, strict=True) if keep],
        labels=[level for level, keep in zip(layout.row_lv, keep_row, strict=True) if keep],
    )
    ax.set_yticks(
        [at for at, keep in zip(layout.y_at, keep_col, strict=True) if keep],
        labels=[level for level, keep in zip(layout.col_lv, keep_col, strict=True) if keep],
    )
    ax.tick_params(axis="both", length=0, labelsize=font(cex_axis), colors=palette.fg)
    ax.set_xlabel(row_var if xlab is None else xlab, fontsize=font(cex_lab), color=palette.fg)
    ax.set_ylabel(col_var if ylab is None else ylab, fontsize=font(cex_lab), color=palette.fg)
    if main is not None:
        ax.set_title(main, fontsize=font(cex_main), color=palette.fg)

    if key_ax is not None:
        _draw_key(key_ax, residual, cex_legend, ink, key_border, palette.bg if dark else None)

    return {
        "cells": tiles,
        "widths": layout.widths,
        "heights": layout.heights,
        "expected_prop": layout.expected_prop,
        "empty_levels": layout.empty_levels,
        "null": layout.null,
        "residual": residual,
        "residual_breaks": list(RESIDUAL_BREAKS),
        "colors": list(MOSAIC_COLORS),
    }


def _anno_mode(anno_cells: Any) -> str:
    """Read ``anno_cells`` as one of :data:`ANNO_MODES`.

    The argument used to be a flag, and the two flags still say what they said.
    """
    if isinstance(anno_cells, bool):
        return ANNO_MODES[0] if anno_cells else ANNO_MODES[-1]
    if anno_cells not in ANNO_MODES:
        raise SaValueError("`anno_cells` must be one of: " + ", ".join(ANNO_MODES) + ".")
    return str(anno_cells)


def _mosaic_input(res: Any) -> tuple[pd.DataFrame, str, str, str]:
    """The cell table of a categorical comparison, or the reason there is none.

    Port of ``sa_mosaic_input()``. ``null`` travels with the cells, because the
    cells alone do not say what their ``expected`` column is a statement about and
    the shading has to know.
    """
    if isinstance(res, SaCategorical):
        return (
            res["cells"],
            str(res["design"]["null"]),
            str(res["design"]["row_var"]),
            str(res["design"]["col_var"]),
        )
    if isinstance(res, SaComparison):
        raise SaValueError(
            "`categorical_comparison_result` is a numeric comparison result. "
            "draw_mosaic_plot() draws a contingency table; draw_grouped_boxplot() "
            "and draw_volcano_plot() are what read a comparison."
        )
    raise SaValueError(
        "`categorical_comparison_result` must be a categorical comparison result, "
        "as returned by compare_categorical_groups(). The shading and the expected "
        "lines are read under the null hypothesis that result was tested against, "
        "which a bare table or a data.frame does not carry. Cross the variables "
        "with compare_categorical_groups() first."
    )


def _layout(cells: pd.DataFrame, gap: float, null: str) -> _Layout:
    """Place the tiles of a mosaic in the unit square.

    Port of ``sa_mosaic_layout()``. The first variable takes the x axis, one strip
    per level, width the marginal share. Each strip is split by the second
    variable, height the share of that strip. The same arithmetic is run a second
    time on the expected counts, which is what puts the null hypothesis on the
    same scale as the tiles and lets it be drawn as a line rather than described
    in a caption.

    Args:
        cells: The cell table, one row per cell.
        gap: Fraction of the axis each single gap takes.
        null: Which hypothesis ``cells["expected"]`` states.
    """
    row_lv = list(dict.fromkeys(str(level) for level in cells["row_level"]))
    col_lv = list(dict.fromkeys(str(level) for level in cells["col_level"]))
    n_row, n_col = len(row_lv), len(col_lv)

    at_row = np.array([row_lv.index(str(level)) for level in cells["row_level"]], dtype=int)
    at_col = np.array([col_lv.index(str(level)) for level in cells["col_level"]], dtype=int)
    observed = np.zeros((n_row, n_col))
    expected = np.zeros((n_row, n_col))
    observed[at_row, at_col] = np.asarray(cells["observed"], dtype=float)
    held = np.asarray(cells["expected"], dtype=float)
    expected[at_row, at_col] = np.where(np.isfinite(held), held, 0.0)

    row_n = observed.sum(axis=1)
    col_n = observed.sum(axis=0)
    total = observed.sum()

    widths = np.full(n_row, 1 / n_row) if total == 0 else row_n / total
    heights = observed / np.maximum(row_n, 1)[:, None]
    heights[row_n == 0, :] = 0.0

    # Normalised within the strip rather than by `row_n`, because the row sums of
    # the expected table are the row sums of the observed one only under
    # independence. Under symmetry they are the average of the row and column
    # margins, and what is being compared is still one distribution to another.
    exp_row = expected.sum(axis=1)
    expected_prop = expected / np.maximum(exp_row, 1)[:, None]
    expected_prop[exp_row == 0, :] = 0.0

    gap_x = min(gap, GAP_MAX_SHARE / (n_row - 1)) if n_row > 1 else 0.0
    gap_y = min(gap, GAP_MAX_SHARE / (n_col - 1)) if n_col > 1 else 0.0
    usable_x = 1 - gap_x * (n_row - 1)
    usable_y = 1 - gap_y * (n_col - 1)

    left = np.concatenate([[0.0], np.cumsum(widths)])[:n_row] * usable_x + (
        np.arange(n_row) * gap_x
    )
    right = left + widths * usable_x

    bottom = np.zeros((n_row, n_col))
    top = np.zeros((n_row, n_col))
    # One boundary fewer than there are tiles: the top of the last one is the top
    # of the strip under every hypothesis, so there is nothing to mark there.
    expected_y = np.full((n_row, max(n_col - 1, 0)), np.nan)

    # The reference strip is the first, which the comparison settled through its
    # `control_label` and `category_lv`, unless it is empty and so has no
    # boundaries to label.
    filled = np.flatnonzero(row_n > 0)
    reference = int(filled[0]) if filled.size > 0 else 0
    y_at = np.zeros(n_col)

    for index in range(n_row):
        bottom[index] = np.concatenate([[0.0], np.cumsum(heights[index])])[:n_col] * usable_y + (
            np.arange(n_col) * gap_y
        )
        top[index] = bottom[index] + heights[index] * usable_y
        if index == reference:
            y_at = (bottom[index] + top[index]) / 2
        if n_col > 1:
            expected_y[index] = np.cumsum(expected_prop[index])[: n_col - 1] * usable_y + (
                np.arange(n_col - 1) * gap_y
            )

    tiles = cells.copy()
    tiles["x1"] = left[at_row]
    tiles["x2"] = right[at_row]
    tiles["y1"] = bottom[at_row, at_col]
    tiles["y2"] = top[at_row, at_col]

    return _Layout(
        cells=tiles,
        widths=pd.Series(widths, index=row_lv),
        heights=pd.DataFrame(heights, index=row_lv, columns=col_lv),
        expected_prop=pd.DataFrame(expected_prop, index=row_lv, columns=col_lv),
        expected_y=expected_y,
        strip_x=np.column_stack([left, right]),
        empty_levels={
            "row": [level for level, n in zip(row_lv, row_n, strict=True) if n == 0],
            "col": [level for level, n in zip(col_lv, col_n, strict=True) if n == 0],
        },
        null=null,
        row_lv=row_lv,
        col_lv=col_lv,
        x_at=(left + right) / 2,
        y_at=y_at,
    )


def _draw_expected(ax: Any, layout: _Layout, colour: str) -> None:
    """Mark each strip where the null hypothesis would have cut it.

    Port of ``sa_mosaic_draw_expected()``.
    """
    if layout.expected_y.shape[1] == 0:
        return
    for index, level in enumerate(layout.row_lv):
        x0, x1 = layout.strip_x[index]
        if level in layout.empty_levels["row"] or x1 <= x0:
            continue
        for height in layout.expected_y[index]:
            if np.isfinite(height):
                ax.plot([x0, x1], [height, height], color=colour, linestyle=":", linewidth=1.6)


def _fill(residual: np.ndarray, empty_fill: str) -> list[str]:
    """Map residuals onto the mosaic palette.

    Port of ``sa_mosaic_fill()``. A cell with no residual - an empty margin
    leaves none - takes the empty colour rather than the neutral one, so that "no
    departure to measure" and "no departure" are not drawn the same way.
    """
    breaks = np.asarray(RESIDUAL_BREAKS, dtype=float)
    # R's `findInterval(all.inside = TRUE)`: the two open ends fold into the
    # bands beside them rather than falling outside the palette.
    band = np.clip(np.searchsorted(breaks, residual, side="right"), 1, len(breaks) - 1) - 1
    return [
        empty_fill if not np.isfinite(value) else MOSAIC_COLORS[at]
        for value, at in zip(residual, band, strict=True)
    ]


def _ink(fill: str) -> str:
    """Ink that can be read on a given fill.

    Port of ``sa_mosaic_text_col()``. Decided from the fill's luminance rather
    than from the residual that chose it, so an unshaded tile, an empty one and a
    dark background are all covered by the same rule.
    """
    red, green, blue = to_rgb(fill)
    weights = _LUMINANCE_WEIGHTS
    luminance = weights[0] * red + weights[1] * green + weights[2] * blue
    return "white" if luminance < _INK_CUTOFF else _DARK_INK


def _anno_label(
    mode: str,
    observed: Any,
    prop_row: Any,
    width: float,
    height: float,
    cex: float,
    panel: tuple[float, float],
) -> str | None:
    """As much of a tile's numbers as the tile has room for.

    Port of ``sa_mosaic_anno()``. Measured against the label rather than against
    a fixed fraction of the plot, because a fraction that fits ``"7"`` does not
    fit ``"1284"`` and a mosaic of a large table holds both. R measures with
    ``strwidth()``; this side approximates a character as
    :data:`~statassist.plot._theme.CHAR_WIDTH` of the font size, which is what
    :mod:`~statassist.plot.heatmap` already reserves label room with.

    Args:
        mode: What was asked for. ``"auto"`` gives up one line at a time.
        observed: The count on the tile.
        prop_row: Its share of the strip.
        width: The tile's width, as a fraction of the axis.
        height: The same for its height.
        cex: Character expansion the label would be drawn at.
        panel: The panel's width and height in inches, which is what turns a
            label's size into the same fractions.

    Returns:
        The label, or ``None`` when nothing fits or nothing was asked for.
    """
    if mode == "none":
        return None

    count = fmt_num(observed)
    share = np.asarray(prop_row, dtype=float)
    percent = None if not np.isfinite(share) else f"{round(100 * float(share)):.0f}%"

    if mode == "count":
        wanted = [count]
    elif mode == "percent":
        wanted = [] if percent is None else [percent]
    elif mode == "both":
        wanted = [count if percent is None else f"{count}\n{percent}"]
    else:
        wanted = ([] if percent is None else [f"{count}\n{percent}"]) + [count]
    if not wanted:
        return None

    size = font(cex)
    for label in wanted:
        lines = label.split("\n")
        text_w = max(len(line) for line in lines) * size / 72 * CHAR_WIDTH / panel[0]
        text_h = len(lines) * size / 72 * LINE_HEIGHT / panel[1]
        if text_w <= ANNO_FIT * width and text_h <= ANNO_FIT * height:
            return label
    # An explicit request is honoured even where it overflows, since the caller
    # asked for the number rather than for a tidy picture. `"auto"` did not.
    return None if mode == "auto" else wanted[0]


def _key_inches(residual: str, cex_legend: float) -> float:
    """How wide a strip the residual key needs.

    Port of ``sa_mosaic_key_width()``, and modelled on the way
    :mod:`~statassist.plot.heatmap` sizes its own key: the width comes from what
    goes in the key rather than from a fraction of the figure, so a key of short
    band labels does not reserve room for long ones.
    """
    char = font(cex_legend) / 72
    swatch = 2.2 * char
    bands = swatch + 0.5 * char + max(len(label) for label in RESIDUAL_LABELS) * char * CHAR_WIDTH
    content = max(bands, len(_key_title(residual)) * char * CHAR_WIDTH)
    return line_inches() + content / (1 - KEY_PAD[0] - KEY_PAD[1])


def _key_title(residual: str) -> str:
    """What the key calls the residual it is a scale for."""
    return "Pearson residual" if residual == RESIDUALS[0] else "Std. residual"


def _draw_key(
    ax: Any,
    residual: str,
    cex_legend: float,
    ink: str,
    border: str,
    background: str | None,
) -> None:
    """Draw the residual key beside the tiles it explains.

    Port of ``sa_mosaic_draw_key()``. Highest band at the top, so the key runs the
    same way up as the colours do on the tiles: more than expected above, less
    than expected below.

    R fits the character expansion to the panel ``layout()`` handed out, since
    every width in the key is proportional to it. Here the panel is a gridspec
    cell reserved from the same measurement, so the expansion is taken as given
    and the panel is what was asked for.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if background is not None:
        ax.set_facecolor(background)

    left = KEY_PAD[0]
    n_band = len(MOSAIC_COLORS)
    # The bands share the panel below the title, so the key is as tall as the
    # tiles are however many bands there happen to be.
    top = 0.96
    title_gap = 0.09
    band_top = top - title_gap
    band_h = min(0.08, (band_top - 0.04) / n_band)

    ax.text(
        left,
        top,
        _key_title(residual),
        ha="left",
        va="top",
        fontsize=font(cex_legend),
        fontweight="bold",
        color=ink,
    )
    swatch = min(0.34, 1 - KEY_PAD[1] - left)
    for index in range(n_band):
        at = n_band - index - 1
        y1 = band_top - index * band_h
        ax.add_patch(
            Rectangle(
                (left, y1 - band_h),
                swatch,
                band_h,
                facecolor=MOSAIC_COLORS[at],
                edgecolor=border,
                linewidth=0.8,
            )
        )
        ax.text(
            left + swatch + 0.06,
            y1 - band_h / 2,
            RESIDUAL_LABELS[at],
            ha="left",
            va="center",
            fontsize=font(cex_legend),
            color=ink,
        )
