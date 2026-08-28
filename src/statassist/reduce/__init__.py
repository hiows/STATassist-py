"""Describing a wide table in fewer coordinates than it arrived with.

Three functions rather than one, for the reason the comparison wrappers run several
tests in one call and this family does not: a comparison's tests answer the same
question and can be read down one table, while these three answer in coordinates
that share no scale. :func:`perform_pca` is a rotation, so it is reproducible,
invertible and readable as "which features moved this sample", but it can only ever
draw straight structure. :func:`perform_tsne` and :func:`perform_umap` find
structure that curves and cannot say which feature made it, and each pays for that
with an arbitrary global scale.

What they share - how a matrix is read out of the caller's frame, which margin
becomes a point, and how the features are standardised - is in
:mod:`statassist.reduce._shared`, which the four ``cluster_*`` functions read too: a
clustering drawn on top of a reduction of the same frame has to be about the same
rows, and it can only be if one function decided which rows those are.
"""

from __future__ import annotations

from .pca import COMPONENT_PREFIX, perform_pca
from .tsne import MAX_TSNE_DIM, perform_tsne
from .umap import UMAP_METRICS, perform_umap

__all__ = [
    "COMPONENT_PREFIX",
    "MAX_TSNE_DIM",
    "UMAP_METRICS",
    "perform_pca",
    "perform_tsne",
    "perform_umap",
]
