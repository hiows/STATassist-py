"""Shared plot styling helpers."""

from __future__ import annotations

DARK2 = [
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
]


def dark2_colors(n: int) -> list[str]:
    return [DARK2[i % len(DARK2)] for i in range(n)]


def sa_plot_theme(dark: bool) -> dict[str, str]:
    if dark:
        return {"bg": "#2B2B2B", "fg": "white", "guide": "#CCCCCC"}
    return {"bg": "white", "fg": "black", "guide": "#666666"}
