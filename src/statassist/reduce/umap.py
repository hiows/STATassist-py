"""The other embedding, and the one that does nothing to the input first.

Port of ``R/perform_umap.R``. :func:`~statassist.perform_pca` and
:func:`~statassist.perform_tsne` standardise the features by default because a
rotation and a Barnes-Hut gradient both answer in Euclidean distance and have no
other way to be told that a feature measured in thousands is not more important
than one measured in units. UMAP is handed a ``metric`` instead, and two of its four
choices answer that question themselves by comparing the shape of a row rather than
its size. Standardising first would be answering it twice, so the default here is
the engine's own and ``scale=True`` is one argument away for the cases that need it.

This is the only function in the package whose engine is not installed by default.
``umap-learn`` is an optional extra, and the import is inside the call so that
everything else in the package works without it; a caller who has not installed it
is told what to install rather than shown a traceback from an import.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import SaValueError
from ..core.result import SaReduction, new_reduction
from ..core.validate import check_count, check_flag, check_scalar_num
from ._shared import (
    EMBEDDING_SCALES,
    check_embedding_scale,
    embedding_frame,
    embedding_matrix,
    reduce_few_points,
    reduce_input,
    reduce_points,
    umap_neighbors,
)

__all__ = ["EMBEDDING_PREFIX", "UMAP_METRICS", "perform_umap"]

#: What an embedding column is called, before its number.
EMBEDDING_PREFIX = "UMAP"

#: The distances neighbours can be measured with, in the order R lists them.
#:
#: ``"cosine"`` and ``"correlation"`` compare the shape of a row rather than its
#: size, which is why the two scaling flags are off by default. R spells the last
#: one ``"pearson"``; the engine here spells it ``"correlation"`` and the engine's
#: spelling is the one that has to reach it.
UMAP_METRICS = ("euclidean", "manhattan", "cosine", "correlation")

#: What to say when the engine is not installed.
_MISSING_ENGINE = (
    "`perform_umap()` needs the `umap-learn` package, which is an optional extra "
    'of this one: install it with `pip install "statassist[umap]"`. '
    "`perform_pca()` and `perform_tsne()` need nothing beyond the core "
    "dependencies and answer about the same points."
)


def perform_umap(
    data: Any,
    feats: Any = None,
    embedding_scale: str = EMBEDDING_SCALES[0],
    center: bool = False,
    scale: bool = False,
    n_dim: int = 2,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
    metric: str = UMAP_METRICS[0],
    seed: int | None = None,
) -> SaReduction:
    """Embed samples or features with UMAP.

    Places each point in a few dimensions so that the neighbourhood structure of
    the full feature space survives, by uniform manifold approximation and
    projection. What comes back is a picture and the coordinates to draw it with:
    an axis is not a direction the way a principal component is, so there is
    nothing to read off one but position.

    ``center`` and ``scale`` are off here and on in
    :func:`~statassist.perform_pca` and :func:`~statassist.perform_tsne`. The
    difference is ``metric``: ``"cosine"`` and ``"correlation"`` compare the shape
    of a row rather than its size, so they have already answered the question
    standardising would answer. What that leaves the caller is one decision rather
    than none. With ``metric="euclidean"`` or ``"manhattan"`` on features that are
    not measured on a common scale, the feature with the widest units decides who
    is whose neighbour, and ``scale=True`` is what the picture needs - it is also
    what makes this embedding comparable with a rotation or a t-SNE of the same
    data, since those two standardise by default.

    One consequence of the default is that a feature of no variance is kept. It can
    be kept because nothing divides by its standard deviation; it contributes
    nothing to any distance either way. With ``scale=True`` it is left out with a
    message and named in ``design["dropped_feats"]``.

    ``embedding_scale="features"`` embeds the features instead, and unlike
    :func:`~statassist.perform_pca` this really does transpose: UMAP embeds the rows
    it is handed and has no second answer to read off the same fit. ``center`` and
    ``scale``, if they are turned on, still apply to the **features**, and the
    transpose is then embedded as it stands.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and one
            column per feature.
        feats: Column names to embed, or ``None`` for every numeric column of
            ``data``.
        embedding_scale: Which margin becomes the points of the picture, one of
            :data:`EMBEDDING_SCALES`.
        center: Whether to centre each feature before embedding.
        scale: Whether to divide each feature by its standard deviation before
            embedding. Both flags always apply to the **columns of** ``data``.
        n_dim: How many dimensions to embed into.
        n_neighbors: The neighbourhood size, or ``None`` to read one off the number
            of points. Small values follow local detail and large ones the overall
            shape.
        min_dist: How tightly points that belong together are allowed to be packed.
        metric: Distance neighbours are measured with, one of :data:`UMAP_METRICS`.
        seed: Seed for the embedding, or ``None`` to leave the engine to its own
            entropy. A seeded run is single-threaded, which is the engine's own
            trade: reproducibility costs the parallel gradient.

    Returns:
        A :class:`~statassist.core.result.SaReduction` with ``analysis`` ``"umap"``
        and no ``variance`` or ``loadings``: an embedding has no components.
        ``parameters`` holds the choices as they were used rather than as they were
        passed, so a derived ``n_neighbors`` is the value that was derived.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, if fewer than
            two samples or two features survive, if the points admit no
            neighbourhood, or if ``umap-learn`` is not installed.

    Examples:
        The engine is an optional extra, so this example is written to be read
        rather than run:

        >>> from statassist import perform_umap  # doctest: +SKIP
        >>> res = perform_umap(data, scale=True, seed=1)  # doctest: +SKIP
        >>> list(res["scores"].columns)  # doctest: +SKIP
        ['points', 'UMAP1', 'UMAP2']
    """
    scale_name = check_embedding_scale(embedding_scale, "embedding_scale")
    if metric not in UMAP_METRICS:
        raise SaValueError(
            "`metric` must be one of " + ", ".join(UMAP_METRICS) + f". Got {metric}."
        )
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")
    n_dim = check_count(n_dim, "n_dim", 1)
    min_dist = check_scalar_num(min_dist, "min_dist", 0, lower_open=True)
    seed_used = None if seed is None else check_count(seed, "seed")

    input_ = reduce_input(data, feats, scale, "perform_umap")
    x = input_.x
    points, point_type = reduce_points(input_.feats, input_.samples, scale_name)
    m = embedding_matrix(x, scale_name, center, scale)
    n_points = m.shape[0]

    # Read before the engine is called, so that a rejected neighbourhood is
    # rejected for what it is rather than as whatever the engine makes of it. It is
    # counted in points: on the feature margin it is the features that have
    # neighbours.
    n_neighbors = umap_neighbors(n_neighbors, n_points, point_type)
    reduce_few_points(n_points, point_type, f"n_neighbors = {n_neighbors}")

    try:
        from umap import UMAP
    except ImportError as missing:  # pragma: no cover - depends on the environment
        raise SaValueError(_MISSING_ENGINE) from missing

    fit = UMAP(
        n_components=n_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed_used,
    )
    coords = fit.fit_transform(m)

    return new_reduction(
        analysis="umap",
        points=points,
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
            "n_dim": n_dim,
            "n_neighbors": n_neighbors,
            "min_dist": float(min_dist),
            "metric": metric,
            "seed": seed_used,
        },
        scores=embedding_frame(coords, points, EMBEDDING_PREFIX),
        engine={
            "package": "umap-learn",
            "method": "UMAP",
            "label": "Uniform manifold approximation and projection",
            # R offers a choice of backend because its `umap` package carries a
            # pure-R implementation beside a bridge to this one. Here there is only
            # this one, so there is no `method` argument to record a choice of.
            "overridden": [],
        },
        fit=fit,
    )
