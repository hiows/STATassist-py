"""Assembly helpers of a comparison of three or more levels.

Port of the private helpers at the bottom of ``R/compare_multiple_groups.R``.
They are here rather than in :mod:`statassist.transform` because none of them is
a transformation a caller would ask for on its own: each exists to keep the
three views of one pairwise stage - ``effect``, ``posthoc`` and ``pairwise`` -
from being able to disagree about what was divided by what.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.contracts import posthoc_stat_columns
from ..core.errors import SaValueError
from ..core.tables import feature_table, level_pairs
from ..transform._foldchange import INPUT_SCALES, fc_center

__all__ = [
    "Centers",
    "feature_samples",
    "group_centers",
    "multi_fold_change",
    "pairwise_tables",
    "require_groups",
]


def require_groups(samples: Mapping[str, np.ndarray], n_min: int) -> Mapping[str, np.ndarray]:
    """Reject a feature whose groups are too small before the kernel sees it.

    Port of ``sa_require_groups()``. The kernels raise their own errors, but they
    see the samples without knowing which level each came from. Checking here
    means the message names the level.
    """
    short = {name: sample.size for name, sample in samples.items() if sample.size < n_min}
    if short:
        detail = ", ".join(f"{name} = {size}" for name, size in short.items())
        raise SaValueError(f"needs at least {n_min} usable observation(s) per group; {detail}.")
    return samples


def feature_samples(
    held: Any,
    group_lv: Sequence[str],
    paired: bool,
) -> dict[str, np.ndarray]:
    """The samples of one feature as a mapping named by level.

    Port of ``sa_feature_samples()``. A repeated design stores a feature as a
    subjects-by-conditions matrix and an independent one as samples by level.
    Everything downstream wants the mapping, so the one place that knows about
    the matrix is here.
    """
    if not paired:
        return {str(level): np.asarray(held[str(level)], dtype=float) for level in group_lv}
    matrix = np.asarray(held, dtype=float)
    return {str(level): matrix[:, index] for index, level in enumerate(group_lv)}


class Centers(NamedTuple):
    """The centre of every level of every feature.

    Attributes:
        centers: Features by levels, in the row order ``feats`` fixes and the
            column order ``group_lv`` fixes. R returns a matrix with dimnames; a
            frame is what carries the same two labels in pandas.
        errors: One message per feature, or ``None`` where the centres were
            taken. Kept rather than raised, so the failure can be re-raised
            inside :func:`~statassist.core.feature_table` where it becomes a
            missing row and part of the aggregated warning.
        n_used: How many observations the feature was left with.
    """

    centers: pd.DataFrame
    errors: list[str | None]
    n_used: list[float]


def group_centers(
    per_feature: Mapping[str, Any],
    feats: Sequence[str],
    group_lv: Sequence[str],
    mean_type: str,
    paired: bool,
    input_scale: str = INPUT_SCALES[0],
) -> Centers:
    """The centre of every level of every feature.

    Port of ``sa_group_centers()``. One pass over the data producing the quantity
    both the ``effect`` table and the pairwise tables are ratios of. Computing it
    once is what keeps ``effect['log2fc']`` and ``pairwise[...]['log2fc']`` from
    being able to disagree.

    A centre that cannot be taken, which happens when a level is empty or when a
    geometric mean meets a value at or below zero, fails the whole feature rather
    than that one level: a ratio needs both sides.
    """
    names = [str(name) for name in feats]
    levels = [str(level) for level in group_lv]
    matrix = np.full((len(names), len(levels)), np.nan)
    errors: list[str | None] = [None] * len(names)
    n_used: list[float] = [float("nan")] * len(names)

    for index, name in enumerate(names):
        samples = feature_samples(per_feature[name], levels, paired)
        try:
            matrix[index, :] = [
                fc_center(samples[level], level, mean_type, input_scale) for level in levels
            ]
        except SaValueError as error:
            errors[index] = str(error)
        n_used[index] = float(
            np.asarray(per_feature[name], dtype=float).shape[0]
            if paired
            else sum(sample.size for sample in samples.values())
        )

    return Centers(
        centers=pd.DataFrame(matrix, index=names, columns=levels),
        errors=errors,
        n_used=n_used,
    )


def multi_fold_change(
    centers: Centers,
    feats: Sequence[str],
    group_lv: Sequence[str],
    mean_type: str,
) -> pd.DataFrame:
    """Fold change of the most extreme level against the reference.

    Port of ``sa_multi_fold_change()``, the multi-group counterpart of
    :func:`~statassist.transform._foldchange.fold_change`. Both reduce a
    comparison to one signed magnitude per feature so that the same volcano plot
    path works for either, but with three or more levels there is no single
    contrast to take the ratio of. The level furthest from the reference on the
    log2 scale is used, which is the largest change the comparison found.

    Args:
        centers: What :func:`group_centers` returned.
        feats: Feature names, one output row per entry.
        group_lv: Group levels, the first being the reference denominator.
        mean_type: ``"arith"`` or ``"geom"``, used for the label only.

    Returns:
        ``features``, ``n_used``, ``n_groups``, ``ref_center``,
        ``extreme_level``, ``extreme_center``, ``fold_change`` and ``log2fc``.
    """
    names = [str(name) for name in feats]
    levels = [str(level) for level in group_lv]
    label = ("Arithmetic" if mean_type == "arith" else "Geometric") + " mean fold change"
    values = centers.centers.to_numpy(dtype=float)

    def one(index: int) -> dict[str, float]:
        # Raised here rather than where it happened, so that the failure still
        # arrives inside `feature_table` and is reported in its warning
        # alongside every other feature that could not be computed.
        message = centers.errors[index]
        if message is not None:
            raise SaValueError(message)

        row = values[index, :]
        # Both divisions leave the domain in the ways R's do, silently: a zero
        # reference gives an infinite ratio and a negative one an undefined log.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = row / row[0]
            log_ratios = np.log2(ratios)

        # A level whose ratio left the domain of log2 cannot be ranked by
        # distance, so it is skipped rather than allowed to win by being NaN.
        rankable = np.isfinite(log_ratios)
        rankable[0] = False
        if rankable.any():
            candidates = np.flatnonzero(rankable)
            extreme = int(candidates[int(np.argmax(np.abs(log_ratios[candidates])))])
        else:
            extreme = 1

        return {
            "n_used": centers.n_used[index],
            "n_groups": float(row.size),
            "ref_center": float(row[0]),
            "extreme_index": float(extreme),
            "extreme_center": float(row[extreme]),
            "fold_change": float(ratios[extreme]),
            "log2fc": float(log_ratios[extreme]),
        }

    out = feature_table(
        names,
        [
            "n_used",
            "n_groups",
            "ref_center",
            "extreme_index",
            "extreme_center",
            "fold_change",
            "log2fc",
        ],
        label,
        fun=one,
        p_adjust_method=None,
    )
    # A feature whose centres could not be taken has no extreme level either, so
    # the label is missing rather than the reference by default.
    out["extreme_level"] = [
        None if not np.isfinite(index) else levels[int(index)] for index in out["extreme_index"]
    ]
    return out[
        [
            "features",
            "n_used",
            "n_groups",
            "ref_center",
            "extreme_level",
            "extreme_center",
            "fold_change",
            "log2fc",
        ]
    ]


def pairwise_tables(
    posthoc_tbl: pd.DataFrame,
    centers: pd.DataFrame,
    feats: Sequence[str],
    group_lv: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Split one post-hoc table into a rectangular table per contrast.

    Port of ``sa_pairwise_tables()``. The post-hoc table is the honest record of
    the pairwise stage: it holds a row only for the features that were actually
    asked. That shape is awkward to read one contrast at a time, which is what a
    reader who came for "treatment 1 against control" wants, so this builds the
    other view of the same numbers.

    Two things differ. Every table holds every feature in the order ``feats``
    fixes, so the contrasts can be lined up against each other and against the
    omnibus tables without matching by name. And ``fold_change`` and ``log2fc``
    are added, which the post-hoc procedures do not report: they are ratios of
    the group centres and so do not depend on which test was run.

    The two kinds of column therefore say different things where a feature did
    not qualify. ``log2fc`` is still there, because the data can always be
    divided; the inference columns are missing, because nothing was asked of
    them.
    """
    names = [str(name) for name in feats]
    pairs = level_pairs(group_lv)
    stat_columns = posthoc_stat_columns()
    values = centers.to_numpy(dtype=float)

    out: dict[str, pd.DataFrame] = {}
    for position in range(len(pairs.index)):
        pair = pairs.iloc[position]
        contrast = str(pair["contrast"])
        # group1 over group2, so that log2fc and estimate, which the contract
        # fixes as group1 - group2, always point the same way within a row.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = values[:, int(pair["i"])] / values[:, int(pair["j"])]
            log_ratios = np.log2(ratios)

        table = pd.DataFrame(
            {
                "features": names,
                "contrast": contrast,
                "group1": str(pair["group1"]),
                "group2": str(pair["group2"]),
                "fold_change": ratios,
                "log2fc": log_ratios,
            }
        )
        rows = posthoc_tbl[posthoc_tbl["contrast"] == contrast].set_index("features")
        for column in stat_columns:
            table[column] = rows[column].reindex(names).to_numpy(dtype=float)
        out[contrast] = table

    return out
