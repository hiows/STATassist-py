"""The stochastic counterpart of a rotation, and why there is more than one.

Port of ``R/perform_tsne.R``. A rotation can only draw straight structure. t-SNE
keeps whichever points were neighbours in the full feature space close in two
dimensions, so it finds structure that curves - and pays for it: it cannot say
which feature made the picture, its global distances mean nothing, and a different
perplexity is a different answer. Read beside a principal component analysis of the
same matrix, a cluster both of them find is a different fact from one only this
method sees.

The engine's own input reduction is not used. R has to turn two of ``Rtsne``'s
defaults off - a normalisation that would undo the ``center`` and ``scale`` this
function was asked for, and a principal component step that would show t-SNE the
output of a rotation rather than the matrix :func:`~statassist.perform_pca` sees.
scikit-learn's estimator does neither, so there is nothing to turn off and the two
functions see literally the same matrix without an override.

Where the two engines do part company is the start of the gradient. ``Rtsne``
starts from noise and scikit-learn from a rotation, so an unseeded run repeats here
and does not in R. The estimator's default is kept, since a deterministic start is
the reason it is the default and re-randomising it to match R would be trading a
better picture for a matching one.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import SaValueError
from ..core.result import SaReduction, new_reduction
from ..core.validate import check_count, check_flag, check_scalar_num, fmt_est
from ._shared import (
    EMBEDDING_SCALES,
    check_embedding_scale,
    embedding_frame,
    embedding_matrix,
    reduce_few_points,
    reduce_input,
    reduce_points,
    tsne_perplexity,
)

__all__ = ["EMBEDDING_PREFIX", "MAX_TSNE_DIM", "perform_tsne"]

#: What an embedding column is called, before its number.
EMBEDDING_PREFIX = "tSNE"

#: Most dimensions this function will embed into.
#:
#: The Barnes-Hut approximation every practical t-SNE uses is defined for one, two
#: or three dimensions. :func:`~statassist.perform_umap` embeds into more, and an
#: exact gradient in higher dimensions is not offered here because a picture of
#: four dimensions is not a picture.
MAX_TSNE_DIM = 3


def perform_tsne(
    data: Any,
    feats: Any = None,
    embedding_scale: str = EMBEDDING_SCALES[0],
    center: bool = True,
    scale: bool = True,
    n_dim: int = 2,
    perplexity: float | None = None,
    theta: float = 0.5,
    seed: int | None = None,
) -> SaReduction:
    """Embed samples or features with t-SNE.

    Places each point in two or three dimensions so that the points it was near in
    the full feature space stay near it, by t-distributed stochastic neighbour
    embedding. What comes back is a picture and the coordinates to draw it with: an
    axis is not a direction the way a principal component is, so there is nothing
    to read off one but position.

    Every distance this method measures is a distance across all of ``feats`` at
    once, so a feature measured in thousands would decide who is whose neighbour.
    ``center`` and ``scale`` are therefore on by default, as they are for
    :func:`~statassist.perform_pca`, and the two functions then see literally the
    same matrix - which is what lets the two pictures be attributed to the methods
    rather than to the preprocessing. :func:`~statassist.perform_umap` defaults the
    other way, since UMAP is more often run on coordinates that already mean
    something, such as the components of a rotation.

    ``embedding_scale="features"`` embeds the features instead, and unlike
    :func:`~statassist.perform_pca` this really does transpose: t-SNE embeds the
    rows it is handed and has no second answer to read off the same fit. The
    features are standardised first and the transpose is then embedded as it
    stands, which is what makes this the same margin
    ``perform_pca(embedding_scale="features")`` reports on.

    A feature margin needs enough features to be worth drawing. ``perplexity`` is
    read off the number of points, so a handful of features force a perplexity of
    two and a message says so; at that size the loadings of a rotation are the whole
    answer and an embedding has nothing to add.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and one
            column per feature.
        feats: Column names to embed, or ``None`` for every numeric column of
            ``data``.
        embedding_scale: Which margin becomes the points of the picture, one of
            :data:`EMBEDDING_SCALES`. On the feature scale the standardised matrix
            is transposed before it is embedded.
        center: Whether to centre each feature before embedding.
        scale: Whether to divide each feature by its standard deviation before
            embedding. Both flags always apply to the **columns of** ``data``.
        n_dim: How many dimensions to embed into, at most :data:`MAX_TSNE_DIM`.
        perplexity: The neighbourhood size, or ``None`` to read one off the number
            of points. It is roughly how many neighbours each point is asked to
            keep close, and the method requires ``3 * perplexity <= n - 1``.
        theta: Barnes-Hut approximation angle. ``0`` is the exact gradient and
            slow, and larger values trade accuracy for speed. This is the argument
            the estimator calls ``angle``; the name here is the one R's engine uses,
            so that the same call reads the same in both languages.
        seed: Seed for the embedding, or ``None`` to leave the engine to its own
            entropy. Pass one anyway when the picture is going into something that
            has to be reproduced: the gradient is started from a rotation rather
            than from noise, which makes an unseeded run repeat *here* but is the
            estimator's default rather than a promise this function makes.

    Returns:
        A :class:`~statassist.core.result.SaReduction` with ``analysis`` ``"tsne"``
        and no ``variance`` or ``loadings``: an embedding has no components.
        ``parameters`` holds the choices as they were used rather than as they were
        passed, so a derived ``perplexity`` is the value that was derived.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, if fewer than
            two samples or two features survive, or if the points admit no
            perplexity the method can run at.

    Examples:
        >>> from statassist import perform_tsne, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=3)
        >>> res = perform_tsne(sim.args["data"], seed=1)
        >>> res["analysis"], list(res["scores"].columns)
        ('tsne', ['points', 'tSNE1', 'tSNE2'])
        >>> res["parameters"]["perplexity"]
        30.0
        >>> "variance" in res
        False
    """
    scale_name = check_embedding_scale(embedding_scale, "embedding_scale")
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")
    n_dim = check_count(n_dim, "n_dim", 1)
    theta = check_scalar_num(theta, "theta", 0, 1)
    seed_used = None if seed is None else check_count(seed, "seed")
    if n_dim > MAX_TSNE_DIM:
        raise SaValueError(
            f"`n_dim` must be at most {MAX_TSNE_DIM}, but is {n_dim}. t-SNE embeds "
            "into 1, 2 or 3 dimensions; `perform_umap()` embeds into more."
        )

    input_ = reduce_input(data, feats, scale, "perform_tsne")
    x = input_.x
    points, point_type = reduce_points(input_.feats, input_.samples, scale_name)
    m = embedding_matrix(x, scale_name, center, scale)
    n_points = m.shape[0]

    # Read before the engine is called, so that a rejected neighbourhood is
    # rejected for what it is rather than as whatever the engine makes of it. It is
    # counted in points: on the feature margin it is the features that have
    # neighbours.
    perplexity = tsne_perplexity(perplexity, n_points, point_type)
    reduce_few_points(n_points, point_type, f"perplexity = {fmt_est(perplexity)}")

    from sklearn.manifold import TSNE

    fit = TSNE(
        n_components=n_dim,
        perplexity=perplexity,
        angle=theta,
        random_state=seed_used,
    )
    coords = fit.fit_transform(m)

    return new_reduction(
        analysis="tsne",
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
            "perplexity": float(perplexity),
            "theta": float(theta),
            "seed": seed_used,
        },
        scores=embedding_frame(coords, points, EMBEDDING_PREFIX),
        engine={
            "package": "sklearn",
            "method": "TSNE",
            "label": "t-distributed stochastic neighbour embedding",
            # Nothing. R turns off two of `Rtsne`'s preprocessing steps so that
            # this and `perform_pca()` see the same matrix; the estimator here does
            # neither of them, so the two already do.
            "overridden": [],
        },
        fit=fit,
    )
