"""The reduction that can say which features moved a point.

Port of ``R/perform_pca.R``. This is the only one of the three that answers on
both margins from one fit. A principal component analysis is a singular value
decomposition, so the sample coordinates and the feature directions are the same
fit read from either end. That is why ``embedding_scale`` does not transpose
anything here: the matrix goes in as it arrived and the argument only decides
which side of the decomposition becomes ``scores``. :func:`~statassist.perform_tsne`
and :func:`~statassist.perform_umap` have no such dual, and the transpose they are
handed is what makes the three answer about the same margin.

The feature scale reports the component directions rescaled from unit length to
variance-weighted length rather than the raw loadings. Unit-length loadings are the
same map on a scale nothing else in the result uses; rescaled, they are the
coordinates a feature would have if the features had been the rows, and they can be
read on the same axis proportions ``variance`` reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.result import SaReduction, new_reduction
from ..core.validate import check_flag
from ._shared import (
    EMBEDDING_SCALES,
    check_embedding_scale,
    embedding_frame,
    reduce_input,
    reduce_points,
)

__all__ = ["COMPONENT_PREFIX", "perform_pca"]

#: What a component column is called, before its number.
COMPONENT_PREFIX = "PC"

#: What the label column of the loadings table is called.
_VARIABLE_COLUMN = "variables"

#: Percentage points in a whole, for turning a share of the variance into one.
_PERCENT = 100.0


def perform_pca(
    data: Any,
    feats: Any = None,
    embedding_scale: str = EMBEDDING_SCALES[0],
    center: bool = True,
    scale: bool = True,
) -> SaReduction:
    """Reduce many features to a few components.

    Rotates a wide table of samples and features onto the axes that carry the most
    variance, so that a few coordinates stand in for all of them. The components
    are ordered by how much of the variance they carry, ``variance`` says how much
    that is, and ``loadings`` says which features built each one.

    The input is the wide format the comparison functions take: **one row per
    sample and one column per feature**. What comes back has one row per point, in
    one order every table follows, which is what makes a reduction plottable
    against anything else read from the same frame.

    ``embedding_scale="features"`` is how a map of the features is asked for, and
    hand-transposing the input is not the same thing: centring and scaling apply to
    **columns**, so on the transpose they standardise samples, and a hand-transposed
    call runs a third analysis that is neither of the two on offer. What the
    argument does instead is leave the matrix alone. One decomposition already
    answers on both margins, so nothing has to be recomputed, and ``variance`` does
    not change at all - the axis labels a plot carries are the same on both scales.

    Transposing by hand is right only when the rows of ``data`` really are features,
    which is how an expression matrix usually arrives. Then the transpose puts the
    data into this function's layout and ``embedding_scale`` chooses from there.

    Rows that are not complete and finite across ``feats`` are dropped before the
    rotation, and ``design["n_dropped"]`` reports how many. A feature that takes a
    single value cannot be scaled, so with ``scale=True`` it is left out with a
    message and named in ``design["dropped_feats"]``; with ``scale=False`` it stays
    and becomes a component of no variance.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and one
            column per feature. The index is kept as the sample labels, repeated
            ones included, since a sample name is a naming choice rather than a key.
        feats: Column names to reduce, or ``None`` for every numeric column of
            ``data``. A non-numeric column is left out with a message, so a frame
            that carries a grouping column alongside the measurements can be passed
            as it is.
        embedding_scale: Which margin becomes the points of the picture, one of
            :data:`EMBEDDING_SCALES`.
        center: Whether to centre each feature before rotating.
        scale: Whether to divide each feature by its standard deviation before
            rotating. On by default because features are not measured on a common
            scale, and without it the feature with the widest units decides where
            the first component points. Both flags always apply to the **columns
            of** ``data``, whatever ``embedding_scale`` is.

    Returns:
        A :class:`~statassist.core.result.SaReduction` with ``analysis`` ``"pca"``,
        ``variance`` one row per component, ``loadings`` the margin that was not
        embedded, and ``res.fit`` the fitted decomposition, always fitted to the
        samples.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, or if fewer
            than two samples or two features survive.

    Examples:
        >>> from statassist import perform_pca, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=3)
        >>> res = perform_pca(sim.args["data"])
        >>> res["analysis"], list(res["scores"].columns[:3])
        ('pca', ['points', 'PC1', 'PC2'])
        >>> round(float(res["variance"]["cum_var"].iloc[-1]), 6)
        100.0
        >>> by_feat = perform_pca(sim.args["data"], embedding_scale="features")
        >>> by_feat["design"]["point_type"], len(by_feat["points"])
        ('feature', 30)
    """
    scale_name = check_embedding_scale(embedding_scale, "embedding_scale")
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")

    input_ = reduce_input(data, feats, scale, "perform_pca")
    x = input_.x
    points, point_type = reduce_points(input_.feats, input_.samples, scale_name)

    from sklearn.decomposition import PCA

    # The unscaled columns are standardised here rather than by the estimator,
    # which has a `center` of its own and no `scale`. The centre and the spread are
    # kept on the result so that a row the fit has not seen can still be projected.
    centre = x.mean(axis=0) if center else np.zeros(x.shape[1])
    centred = x - centre
    spread = np.sqrt((centred**2).sum(axis=0) / (x.shape[0] - 1)) if scale else np.ones(x.shape[1])
    standardised = centred / spread

    fit = PCA().fit(standardised)
    sdev = np.sqrt(np.asarray(fit.explained_variance_, dtype=float))
    # One row per feature and one column per component: the directions, at unit
    # length, which is what a loading is.
    rotation = np.asarray(fit.components_, dtype=float).T
    names = [f"{COMPONENT_PREFIX}{index + 1}" for index in range(rotation.shape[1])]
    samples = standardised @ rotation

    if scale_name == EMBEDDING_SCALES[1]:
        # One decomposition, read from the other end: the same map, on the scale a
        # coordinate has rather than the unit length a direction has.
        coords = rotation * (sdev * np.sqrt(x.shape[0] - 1))
        other, other_labels = samples, input_.samples
    else:
        coords = samples
        other, other_labels = rotation, input_.feats

    spread_of = sdev**2
    variance = pd.DataFrame(
        {
            "component": names,
            "sdev": sdev,
            "prop_var": spread_of / spread_of.sum() * _PERCENT,
            "cum_var": np.cumsum(spread_of) / spread_of.sum() * _PERCENT,
        }
    )
    loadings = pd.DataFrame(other, columns=names)
    loadings.insert(0, _VARIABLE_COLUMN, list(other_labels))

    return new_reduction(
        analysis="pca",
        points=points,
        # `design` describes the input, so its counts do not turn with
        # `embedding_scale`: `n_samples` is always rows of `data` and `feats` always
        # the columns kept. `point_type` is what says which of the two became the
        # points.
        design={
            "point_type": point_type,
            "n_samples": input_.n_samples,
            "n_used": x.shape[0],
            "n_dropped": input_.n_dropped,
            "n_feats": x.shape[1],
            "feats": list(input_.feats),
            "dropped_feats": list(input_.dropped_feats),
        },
        parameters={
            "embedding_scale": scale_name,
            "center": center,
            "scale": scale,
        },
        scores=embedding_frame(coords, points, COMPONENT_PREFIX),
        engine={
            "package": "sklearn",
            "method": "PCA",
            "label": "Principal component analysis",
            # The estimator centres but does not scale, so the standardising is
            # done before the fit rather than by it. The numbers are the same
            # either way; what changes is that `res.fit.mean_` is zero, because
            # the matrix it saw was already centred.
            "overridden": ["standardised before the fit"],
        },
        variance=variance,
        loadings=loadings,
        fit=fit,
    )
