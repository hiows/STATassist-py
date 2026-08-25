"""What every ``draw_*`` function shares: a theme, a font size and a margin.

R's plots are sized in *lines of text* and coloured by ``par()``. matplotlib
sizes in points and inches and colours per artist. The three helpers here are
that translation, written once, so the four drawing functions read the same as
their R originals - ``cex.axis`` is still a multiplier and ``margin`` is still
four numbers of lines - without each of them working out how a line becomes an
inch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from matplotlib import colormaps, rcParams
from matplotlib.figure import Figure

from ..core.errors import SaValueError

__all__ = [
    "LINE_HEIGHT",
    "Theme",
    "estimate_column",
    "figure",
    "font",
    "group_colors",
    "line_inches",
    "linestyle",
    "set_margin",
    "theme",
    "tick_rotation",
]

#: A line of text as a multiple of the font size, which is what turns R's
#: margins in lines into the inches matplotlib lays a figure out in. R's
#: ``par("csi")`` is 0.2 inch at its default 12 point font, the same ratio.
LINE_HEIGHT = 1.2

#: The estimate columns a forest plot knows how to draw, in the order it looks
#: for them. Port of ``sa_estimate_column()``'s candidates: the tests report
#: their estimates under their own names, so the plot looks for the first one it
#: recognises rather than making every table agree on a single column name.
ESTIMATE_COLUMNS = (
    "mean_diff",
    "hl_shift",
    "trim_diff",
    "relative_effect",
    "diff",
    "estimate",
)

#: R's ``lty`` as matplotlib line styles, by code and by name.
#:
#: R names its line types both ways, ``lty = 2`` and ``lty = "dashed"``, and the
#: ``grid_lty`` arguments this package carries over are documented as R's, so both
#: spellings are answered here. ``0`` is R's ``"blank"``: a line that is not drawn
#: rather than a style, which is how the callers read a ``None``.
#: The two of R's line types matplotlib has no name for are given as dash patterns.
LINE_TYPES: dict[Any, Any] = {
    0: None,
    1: "-",
    2: "--",
    3: ":",
    4: "-.",
    5: (0, (8, 4)),
    6: (0, (2, 2, 6, 2)),
    "blank": None,
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dotdash": "-.",
    "longdash": (0, (8, 4)),
    "twodash": (0, (2, 2, 6, 2)),
}

#: The palette group levels are drawn in, R's ``hcl.colors(n, "Dark 2")``.
GROUP_PALETTE = "Dark2"

#: How wide a character is, as a fraction of the font size.
#:
#: The usual approximation for a proportional face, and the one
#: :mod:`~statassist.plot.heatmap` already reserves label room with. It is only
#: used to decide whether a label fits, never to place one.
CHAR_WIDTH = 0.6

#: How far a tick label is turned when it does not fit lying flat.
TILT_DEGREES = 30.0


class Theme(NamedTuple):
    """Foreground, background and guide colours for a plot."""

    bg: str
    fg: str
    guide: str


def theme(dark: bool) -> Theme:
    """The colours a plot draws itself in.

    Port of ``sa_plot_theme()``. R's greys are named there and given as hex here,
    since matplotlib knows ``"gray"`` but not R's ``"gray40"``; the values are
    R's own.
    """
    if dark:
        return Theme(bg="#2B2B2B", fg="white", guide="#B3B3B3")
    return Theme(bg="white", fg="black", guide="#666666")


def font(cex: float) -> float:
    """A ``cex`` multiplier as a font size in points.

    R's ``cex.*`` arguments are relative to the device's base font size, and so
    is this: the base is whatever the caller's ``rcParams["font.size"]`` says,
    so a figure drawn into a style sheet keeps that style's scale.
    """
    return float(rcParams["font.size"]) * float(cex)


def line_inches() -> float:
    """How tall one line of text is, in inches."""
    return float(rcParams["font.size"]) / 72.0 * LINE_HEIGHT


def linestyle(lty: Any, arg: str = "grid_lty") -> Any:
    """An R ``lty`` as a matplotlib line style, or ``None`` for a line not drawn.

    The ``grid_lty`` arguments are R's, so a caller who reaches for ``2`` or for
    ``"dashed"`` gets a dashed line either way. Anything matplotlib already
    understands is handed straight back, so a caller who would rather write a dash
    pattern is not made to look up R's numbering for it.

    Args:
        lty: An R line type by code or by name, a matplotlib line style, or
            ``None``.
        arg: Which argument is being read, for the message if it cannot be.
    """
    if lty is None:
        return None
    # `True` would otherwise be read as R's 1 and draw a solid line, since Python
    # counts a bool as an int.
    if isinstance(lty, bool):
        raise SaValueError(f"`{arg}` must be an R line type, not {lty!r}.")
    if isinstance(lty, (int, float)) and float(lty).is_integer():
        lty = int(lty)
    if lty in LINE_TYPES:
        return LINE_TYPES[lty]
    return lty


def tick_rotation(labels: Sequence[str], span_inches: float, size: float) -> float:
    """How far to turn a set of tick labels so that they do not run together.

    R's :func:`axis` drops the labels that would overlap, which answers the
    question by hiding some of them. matplotlib draws every label it is given, so
    the crowding has to be answered here instead, and turning them keeps the ones
    R would have dropped.

    Args:
        labels: The labels about to be drawn.
        span_inches: How much room one label has, which is the width of the axis
            divided by the number of ticks on it.
        size: Font size the labels are drawn at, in points.

    Returns:
        ``0`` while the longest label fits lying flat, :data:`TILT_DEGREES` once
        it does not.
    """
    if not labels or span_inches <= 0:
        return 0.0
    widest = max(len(str(label)) for label in labels) * size / 72.0 * CHAR_WIDTH
    return 0.0 if widest <= span_inches else TILT_DEGREES


def group_colors(col: Any, n_levels: int) -> list[Any]:
    """One colour per group level.

    R takes ``hcl.colors(n, "Dark 2")`` where nothing was named, which spreads the
    levels across the palette rather than always taking its first few, and
    recycles a named vector that is short. Both are here, so the boxes, the bars
    and the traces of an interaction all colour a group the same way.
    """
    if col is None:
        palette = colormaps[GROUP_PALETTE]
        return [palette(index / max(n_levels, 1)) for index in range(n_levels)]
    held = [col] if isinstance(col, str) else list(col)
    if not held:
        raise SaValueError("`col` must name at least one colour, or be `None`.")
    return [held[index % len(held)] for index in range(n_levels)]


def estimate_column(table: Any) -> str | None:
    """Which column of a test table holds the estimate to draw.

    Port of ``sa_estimate_column()``. A table with none of them, which is every
    omnibus table, gets ``None`` and falls through to the p-value view.
    """
    for name in ESTIMATE_COLUMNS:
        if name in table.columns:
            return name
    return None


def figure() -> Figure:
    """The figure a ``draw_*`` call draws on, cleared and ready.

    R draws on the current device and replaces what was on it. This does the
    same with the current figure, so a script that calls two of these functions
    in a row gets two plots rather than one plot drawn over another. A caller who
    wants a figure of a particular size or a second plot beside the first opens
    it with :func:`matplotlib.pyplot.figure` first, exactly as they would open a
    device in R.
    """
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.clear()
    return fig


def set_margin(fig: Figure, margin: tuple[float, float, float, float]) -> None:
    """Give the axes of ``fig`` the margins ``margin`` asks for, in lines.

    ``margin`` is R's ``mar``: bottom, left, top and right, in lines of text.
    matplotlib places axes as a fraction of the figure, so the lines are turned
    into inches first, which is what makes the same four numbers mean the same
    space on a figure of any size.
    """
    bottom, left, top, right = (value * line_inches() for value in margin)
    width, height = fig.get_size_inches()
    # A margin wider than the figure would put the right edge left of the left
    # one, which matplotlib refuses. The panel keeps a tenth of the figure.
    span_x = max(width * 0.1, width - left - right)
    span_y = max(height * 0.1, height - bottom - top)
    fig.subplots_adjust(
        left=left / width,
        right=(left + span_x) / width,
        bottom=bottom / height,
        top=(bottom + span_y) / height,
    )
