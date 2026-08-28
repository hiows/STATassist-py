"""What every reduction and every clustering shares: how a matrix is read.

Port of ``R/utils_reduce.R``. The reductions differ in what they do with a matrix
and not at all in how they read one, so a caller who moves from one to the next
finds the same rows dropped for the same reasons and the same ``design``
describing them. That is what makes them comparable: a cluster only one of them
finds is a fact about the method, and it can only be read that way if all of them
were handed the same numbers.

The four ``cluster_*`` functions read their input through here as well, for the
reason that extends the same sentence: a clustering drawn on top of a reduction of
the same frame has to be about the same rows, and it can only be if one function
decided which rows those are. The messages are worded for both, since by the time
one is printed the caller knows perfectly well which function they called.

``embedding_scale`` lives here for the same reason, and it is not a preprocessing
option: it names which margin of the input becomes a point. ``center`` and
``scale`` always apply to the features, whichever margin is being embedded,
because that is what scaling a data set means. Standardising the transpose instead
standardises samples and answers a third question that looks exactly like an
answer to this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.validate import check_count, check_scalar_num, validate_wide_input

__all__ = [
    "EMBEDDING_SCALES",
    "FEW_POINTS",
    "MIN_REDUCE_FEATS",
    "MIN_REDUCE_SAMPLES",
    "POINT_TYPES",
    "ReduceInput",
    "check_embedding_scale",
    "embedding_frame",
    "embedding_matrix",
    "reduce_few_points",
    "reduce_input",
    "reduce_points",
    "tsne_perplexity",
    "umap_neighbors",
]

#: Which margin of the input becomes a point, in the order R lists them.
#:
#: The first is the default everywhere. ``cluster_*`` calls the same argument
#: ``cluster_scale``; the values and their meaning are these.
EMBEDDING_SCALES = ("samples", "features")

#: What one point is, in the order :data:`EMBEDDING_SCALES` names them.
POINT_TYPES = ("sample", "feature")

#: Fewest rows a reduction or a clustering will work with.
#:
#: Two, on both margins. One point has nothing to be near or far from, and one
#: variable gives every point the same coordinate, so neither is a table anything
#: here can describe.
MIN_REDUCE_SAMPLES = 2
MIN_REDUCE_FEATS = 2

#: Below how many points a neighbourhood method is worth saying something about.
#:
#: Not a threshold that changes what is computed. A neighbourhood embedding of a
#: dozen points is still an embedding; there is just very little neighbourhood in
#: it, and the caller is told so once.
FEW_POINTS = 16

#: The largest neighbourhood the derived defaults will ask for.
#:
#: Both are the engines' own defaults: 30 for t-SNE's perplexity and 15 for UMAP's
#: neighbour count. They are named here because a small table cannot honour either
#: and the derived value is then the largest one it can.
_TSNE_PERPLEXITY_MAX = 30
_UMAP_NEIGHBORS_MAX = 15

#: How many rows t-SNE needs per unit of perplexity.
#:
#: Fixed by the method rather than chosen here: an implementation refuses
#: ``n - 1 < 3 * perplexity``, since a perplexity is a number of neighbours and
#: there have to be that many rows to be neighbours with.
_TSNE_ROWS_PER_PERPLEXITY = 3

#: The smallest neighbourhood there is, for UMAP.
_UMAP_NEIGHBORS_MIN = 2

#: The column a reduction's own scores table carries its point labels in.
#:
#: Read back here so that a reduction can be handed to a clustering: the labels
#: are in a column rather than in the index, and the index is then the default one.
_POINT_COLUMN = "points"

#: What a column of a nameless matrix is called.
_MATRIX_PREFIX = "V"


def check_embedding_scale(scale: Any, arg: str) -> str:
    """Resolve which margin becomes a point, R's ``match.arg()``."""
    if scale not in EMBEDDING_SCALES:
        raise SaValueError(f"`{arg}` must be one of: " + ", ".join(EMBEDDING_SCALES) + ".")
    return str(scale)


@dataclass(frozen=True)
class ReduceInput:
    """The matrix a reduction is computed on, and what it came from.

    Attributes:
        x: One row per usable sample and one column per kept feature.
        feats: The kept feature names, in the order they are columns of ``x``.
        samples: Row labels of ``x``.
        n_samples: How many rows arrived.
        n_dropped: How many of them could not be used.
        dropped_feats: Features left out for having no variance to rescale.
    """

    x: np.ndarray
    feats: list[str]
    samples: list[str]
    n_samples: int
    n_dropped: int
    dropped_feats: list[str]


def reduce_input(data: Any, feats: Any, scale: bool, fn: str) -> ReduceInput:
    """Read the matrix a reduction is computed on out of the caller's frame.

    Port of ``sa_reduce_input()``. Everything that decides which rows and which
    columns reach an engine happens here, so that no two of these functions can be
    given two different matrices.

    Args:
        data: The caller's ``data``, wide: one row per sample, one column per
            feature.
        feats: The caller's ``feats``, or ``None`` for every numeric column.
        scale: Whether the features will be divided by their standard deviation,
            which is what makes a feature of no variance impossible rather than
            merely useless.
        fn: Name of the calling function, so that the message about a table too
            small to reduce names the function the caller actually called.
    """
    frame, samples = _labelled_frame(data)
    names = _numeric_feats(frame, feats)

    # One synthetic level, as an ungrouped summarize_descriptive_stats() call
    # does: the validator is written in terms of group levels and a reduction has
    # none.
    input_ = validate_wide_input(
        frame,
        names,
        group=["all"] * len(frame.index),
        group_lv=["all"],
        id=samples,
        min_levels=1,
    )
    values = np.asarray(input_.data[input_.feats], dtype=float)
    kept_feats = list(input_.feats)
    labels = list(input_.id or [])

    # No engine here takes a hole, so the rows go before any of them is called. An
    # infinite value is dropped with the missing ones: it survives a completeness
    # check and then turns the whole column into NaN on the way through scaling.
    usable = np.isfinite(values).all(axis=1)
    n_dropped = int((~usable).sum())
    if n_dropped > 0:
        notify(
            f"Dropped {n_dropped} row(s) that are not complete and finite across the "
            "feature(s) in use."
        )
    values = values[usable, :]
    labels = [label for label, take in zip(labels, usable, strict=True) if take]

    dropped_feats: list[str] = []
    if scale and values.shape[0] > 1:
        # A feature that never moves would be divided by zero. It carries nothing
        # either way, so it is named and left out rather than allowed to turn the
        # whole matrix into NaN.
        spread = values.std(axis=0, ddof=1)
        flat = ~np.isfinite(spread) | (spread == 0)
        if flat.any():
            dropped_feats = [name for name, gone in zip(kept_feats, flat, strict=True) if gone]
            notify(
                f"Left out {len(dropped_feats)} feature(s) of no variance, which "
                "`scale=True` cannot rescale: " + ", ".join(dropped_feats) + "."
            )
            values = values[:, ~flat]
            kept_feats = [name for name, gone in zip(kept_feats, flat, strict=True) if not gone]

    if values.shape[0] < MIN_REDUCE_SAMPLES or values.shape[1] < MIN_REDUCE_FEATS:
        raise SaValueError(
            f"`{fn}()` needs at least {MIN_REDUCE_SAMPLES} samples and "
            f"{MIN_REDUCE_FEATS} features, but got {values.shape[0]} usable sample(s) "
            f"and {values.shape[1]} usable feature(s)."
        )

    return ReduceInput(
        x=values,
        feats=kept_feats,
        samples=labels,
        n_samples=len(frame.index),
        n_dropped=n_dropped,
        dropped_feats=dropped_feats,
    )


def _labelled_frame(data: Any) -> tuple[pd.DataFrame, list[str]]:
    """The caller's input as a frame, beside the label of every row of it.

    R reads the labels off ``rownames()`` and numbers the rows when there are
    none. A pandas frame always has an index, so the index is what is read, and a
    frame that arrived with the default one is labelled by position - from zero,
    the way every other position in this port is counted, rather than from one.

    A frame carrying a ``points`` column under a default index is a reduction's own
    scores table being handed back, which is how a clustering is drawn on top of an
    embedding of the same rows.
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise SaValueError("`data` must be a DataFrame or a 2-d array.")
        width = data.shape[1]
        data = pd.DataFrame(data, columns=[f"{_MATRIX_PREFIX}{i + 1}" for i in range(width)])
    if not isinstance(data, pd.DataFrame):
        raise SaValueError("`data` must be a DataFrame or a 2-d array.")

    default_index = isinstance(data.index, pd.RangeIndex)
    if default_index and _POINT_COLUMN in data.columns:
        column = data[_POINT_COLUMN]
        if not column.isna().any():
            return data, [str(value) for value in column]
    return data, [str(value) for value in data.index]


def _numeric_feats(frame: pd.DataFrame, feats: Any) -> list[str]:
    """Which columns are worked on, when the caller has not said.

    A non-numeric column being left out is not an error: a wide frame usually
    carries the grouping column beside the measurements, and asking the caller to
    strip it would be asking them to write ``feats`` out by hand.
    """
    if feats is not None:
        return [feats] if isinstance(feats, str) else [str(name) for name in feats]

    numeric = [
        str(name)
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name]) and frame[name].dtype != bool
    ]
    if not numeric:
        raise SaValueError("`data` holds no numeric column, so there is nothing to work with.")
    left_out = [str(name) for name in frame.columns if str(name) not in set(numeric)]
    if left_out:
        notify(f"Left out {len(left_out)} non-numeric column(s): " + ", ".join(left_out) + ".")
    return numeric


def reduce_points(
    feats: list[str],
    samples: list[str],
    embedding_scale: str,
) -> tuple[list[str], str]:
    """What one point is, and what it is called.

    Port of ``sa_reduce_points()``. The row axis of a reduction is the one axis in
    this package the caller chooses, so it is resolved in one place and every
    function reports it the same way.

    Returns:
        The point labels and the ``point_type`` that says which margin they are.
    """
    if embedding_scale == EMBEDDING_SCALES[1]:
        return list(feats), POINT_TYPES[1]
    return list(samples), POINT_TYPES[0]


def embedding_matrix(
    x: np.ndarray,
    embedding_scale: str,
    center: bool,
    scale: bool,
) -> np.ndarray:
    """The matrix an embedding engine is handed, one row per point.

    Port of ``sa_reduce_embedding_matrix()``. The features are standardised first
    and the transpose is then embedded as it stands, which is the definition under
    which the feature scale agrees with what a principal component analysis says
    about the same features. Standardising after the transpose would standardise
    samples.
    """
    standardised = np.asarray(x, dtype=float)
    if center:
        standardised = standardised - standardised.mean(axis=0)
    if scale:
        # R's `scale()` divides by the root mean square of the centred column,
        # which is the sample standard deviation once the column is centred and
        # is not it when `center = FALSE`. Written out rather than taken from
        # `std()` so that the uncentred case reads the same in both languages.
        size = np.sqrt((standardised**2).sum(axis=0) / (standardised.shape[0] - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            standardised = standardised / size
    if embedding_scale == EMBEDDING_SCALES[1]:
        return np.ascontiguousarray(standardised.T)
    return standardised


def reduce_few_points(n_points: int, point_type: str, size: str) -> None:
    """Say out loud how small the neighbourhood came out.

    Port of ``sa_reduce_few_points()``. Not a warning: the run goes through and its
    output is a picture like any other. It is said because the derived
    neighbourhood is the whole behaviour of these two methods, and someone
    embedding eight features is the last person who would think to check what it
    came out as.
    """
    if n_points >= FEW_POINTS:
        return
    notify(
        f"Only {n_points} {point_type}(s) to embed ({size}). This method describes a "
        f"neighbourhood, and below about {FEW_POINTS} points there is not much of one "
        "to describe. `perform_pca()` is not governed by one."
    )


def tsne_perplexity(perplexity: Any, n: int, point_type: str = POINT_TYPES[0]) -> float:
    """The neighbourhood size t-SNE is run at.

    Port of ``sa_tsne_perplexity()``. A value that was asked for and cannot be
    honoured is an error naming the limit, the way an ``mtry`` above the predictor
    count is in :func:`fit_rf`. A value this function derived and cannot honour is
    an error too, and says instead that there are too few points for the method at
    all.
    """
    upper = (n - 1) / _TSNE_ROWS_PER_PERPLEXITY
    if perplexity is None:
        derived = min(_TSNE_PERPLEXITY_MAX, int(np.floor(upper)))
        if derived < 1:
            raise SaValueError(
                f"`perform_tsne()` cannot embed {n} {point_type}(s): they admit no "
                "perplexity of 1 or more, since t-SNE requires "
                f"{_TSNE_ROWS_PER_PERPLEXITY} * perplexity <= n - 1. `perform_pca()` "
                "has no neighbourhood and is not limited this way."
            )
        return float(derived)
    value = check_scalar_num(perplexity, "perplexity", 1)
    if value > upper:
        raise SaValueError(
            f"`perplexity` must not exceed (n - 1) / {_TSNE_ROWS_PER_PERPLEXITY}, which "
            f"is {upper:.4g} for the {n} usable {point_type}(s), but is {value}. t-SNE "
            "keeps a neighbourhood rather than a pair, and there are not that many "
            "points to fill one."
        )
    return value


def umap_neighbors(n_neighbors: Any, n: int, point_type: str = POINT_TYPES[0]) -> int:
    """The neighbourhood size UMAP is run at.

    Port of ``sa_umap_neighbors()``. Two is the smallest neighbourhood there is and
    the number of points is the largest. The same split as :func:`tsne_perplexity`:
    an impossible request and an impossible derived value are both errors, and they
    say different things about whose mistake it was.
    """
    if n_neighbors is None:
        derived = min(_UMAP_NEIGHBORS_MAX, n)
        if derived < _UMAP_NEIGHBORS_MIN:
            raise SaValueError(
                f"`perform_umap()` cannot embed {n} {point_type}(s): they admit no "
                f"neighbourhood of {_UMAP_NEIGHBORS_MIN} or more. `perform_pca()` has "
                "no neighbourhood and is not limited this way."
            )
        return int(derived)
    value = check_count(n_neighbors, "n_neighbors", _UMAP_NEIGHBORS_MIN)
    if value > n:
        raise SaValueError(
            f"`n_neighbors` must not exceed the {n} usable {point_type}(s) being "
            f"embedded, but is {value}. A {point_type} cannot have more neighbours than "
            f"there are {point_type}s."
        )
    return value


def embedding_frame(m: np.ndarray, points: list[str], prefix: str) -> pd.DataFrame:
    """Put an embedding beside the points it describes.

    Port of ``sa_embedding_frame()``. The engines do not agree on what a
    coordinate matrix looks like, and the row a coordinate belongs to is its
    position, as it is everywhere else in this package, so the labels come from
    ``points`` rather than from whatever the engine happened to carry through.
    """
    values = np.asarray(m, dtype=float)
    frame = pd.DataFrame(
        values,
        columns=[f"{prefix}{index + 1}" for index in range(values.shape[1])],
    )
    frame.insert(0, _POINT_COLUMN, list(points))
    return frame
