"""t-SNE embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from statassist.contracts.reduction import sa_new_reduction
from statassist.utils.reduce import (
    sa_embedding_frame,
    sa_reduce_embedding_matrix,
    sa_reduce_few_points,
    sa_reduce_input,
    sa_reduce_points,
    sa_tsne_perplexity,
)
from statassist.utils.validate import sa_check_flag, sa_preserve_seed


def perform_tsne(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    embedding_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
    n_dim: int = 2,
    perplexity: float | None = None,
    theta: float = 0.5,
    seed: int | None = None,
) -> dict:
    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")
    input_ = sa_reduce_input(data, feats, scale, "perform_tsne")
    pt = sa_reduce_points(input_["x"], input_["samples"], embedding_scale)
    m = sa_reduce_embedding_matrix(input_["x"], embedding_scale, center, scale)
    perp = sa_tsne_perplexity(perplexity, m.shape[0], pt["point_type"])
    sa_reduce_few_points(m.shape[0], pt["point_type"], f"perplexity = {perp}")

    with sa_preserve_seed(seed):
        fit = TSNE(
            n_components=n_dim,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        )
        coords = fit.fit_transform(m)

    return sa_new_reduction(
        analysis="tsne",
        points=pt["points"],
        design={
            "point_type": pt["point_type"],
            "n_samples": input_["n_samples"],
            "n_used": input_["x"].shape[0],
            "n_dropped": input_["n_dropped"],
            "n_feats": input_["x"].shape[1],
            "feats": list(input_["feats"]),
            "dropped_feats": input_["dropped_feats"],
        },
        parameters={
            "embedding_scale": embedding_scale,
            "center": center,
            "scale": scale,
            "n_dim": n_dim,
            "perplexity": perp,
            "theta": theta,
            "seed": seed,
        },
        scores=sa_embedding_frame(coords, pt["points"], "tSNE"),
        engine={
            "package": "sklearn",
            "method": "tsne",
            "label": "t-Distributed Stochastic Neighbor Embedding",
            "overridden": [],
        },
        fit=fit,
    )
