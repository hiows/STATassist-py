"""What every ``draw_*`` function shares: a theme, a font size and a margin.

R's plots are sized in *lines of text* and coloured by ``par()``. matplotlib
sizes in points and inches and colours per artist. The three helpers here are
that translation, written once, so the four drawing functions read the same as
their R originals - ``cex.axis`` is still a multiplier and ``margin`` is still
four numbers of lines - without each of them working out how a line becomes an
inch.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from matplotlib import rcParams
from matplotlib.figure import Figure

__all__ = [
    "LINE_HEIGHT",
    "Theme",
    "estimate_column",
    "figure",
    "font",
    "line_inches",
    "set_margin",
    "theme",
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
