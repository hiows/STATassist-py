"""Dimensionality reduction input helpers (from utils_reduce.R)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_check_count, sa_check_scalar_num, sa_validate_wide_input


def sa_reduce_input(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None,
    scale: bool,
    fn: str,
) -> dict[str, Any]:
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("`data` must be a two-dimensional matrix.")
        if data.dtype.names is None and getattr(data, "columns", None) is None:
            data = pd.DataFrame(data, columns=[f"V{i}" for i in range(data.shape[1])])
        else:
            data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data.frame or a matrix.")

    sample_labels: list[str] | np.ndarray | None = None
    if isinstance(data, pd.DataFrame) and "points" in data.columns:
        pt = data["points"].astype(str)
        rn = data.index.astype(str).tolist()
        default_rn = [str(i + 1) for i in range(len(data))]
        if not pt.isna().any() and (rn == default_rn or len(set(rn)) != len(rn)):
            sample_labels = pt.tolist()
    if sample_labels is None:
        sample_labels = data.index.astype(str).tolist()
    if sample_labels is None or len(sample_labels) != len(data):
        sample_labels = [str(i + 1) for i in range(len(data))]

    work = data.copy()
    if isinstance(work, pd.DataFrame):
        work.index = pd.RangeIndex(len(work))

    if feats is None:
        numeric_col = [c for c in work.columns if np.issubdtype(work[c].dtype, np.number)]
        if not numeric_col:
            raise ValueError("`data` holds no numeric column, so there is nothing to work with.")
        non_numeric = [c for c in work.columns if c not in numeric_col]
        if non_numeric:
            warnings.warn(
                f"Left out {len(non_numeric)} non-numeric column(s): "
                f"{', '.join(non_numeric)}.",
                stacklevel=2,
            )
        feats = numeric_col

    n_rows = len(work)
    input_ = sa_validate_wide_input(
        work,
        list(feats),
        group=np.array(["all"] * n_rows),
        group_lv=["all"],
        id=sample_labels,
        min_levels=1,
    )
    x = input_["data"][input_["feats"]].to_numpy(dtype=float)
    samples = np.asarray(input_["id"], dtype=str)

    usable = np.all(np.isfinite(x), axis=1)
    n_dropped = int((~usable).sum())
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} row(s) that are not complete and finite "
            "across the feature(s) in use.",
            stacklevel=2,
        )
    x = x[usable]
    samples = samples[usable]

    dropped_feats: list[str] = []
    if scale and x.shape[0] > 1:
        sds = np.nanstd(x, axis=0, ddof=1)
        flat = ~np.isfinite(sds) | (sds == 0)
        if np.any(flat):
            dropped_feats = [input_["feats"][i] for i, f in enumerate(flat) if f]
            warnings.warn(
                f"Left out {len(dropped_feats)} feature(s) of no variance, which "
                f"`scale = True` cannot rescale: {', '.join(dropped_feats)}.",
                stacklevel=2,
            )
            x = x[:, ~flat]
            feats = [f for f, fflat in zip(input_["feats"], flat) if not fflat]

    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError(
            f"`{fn}()` needs at least 2 samples and 2 features, but got "
            f"{x.shape[0]} usable sample(s) and {x.shape[1]} usable feature(s)."
        )

    return {
        "x": x,
        "samples": samples.tolist(),
        "n_samples": n_rows,
        "n_dropped": n_dropped,
        "dropped_feats": dropped_feats,
        "feats": list(feats) if isinstance(feats, list) else feats,
    }


def sa_reduce_points(
    x: np.ndarray,
    samples: list[str],
    embedding_scale: str,
) -> dict[str, str | list[str]]:
    if embedding_scale == "features":
        cols = [str(c) for c in range(x.shape[1])]
        if hasattr(x, "columns"):
            cols = list(x.columns)  # type: ignore[attr-defined]
        return {"points": cols, "point_type": "feature"}
    return {"points": samples, "point_type": "sample"}


def sa_reduce_embedding_matrix(
    x: np.ndarray,
    embedding_scale: str,
    center: bool,
    scale: bool,
) -> np.ndarray:
    xs = x.astype(float)
    if center or scale:
        mean = xs.mean(axis=0) if center else 0.0
        std = xs.std(axis=0, ddof=1) if scale else 1.0
        std = np.where(std == 0, 1.0, std)
        xs = (xs - mean) / std if scale else (xs - mean)
    if embedding_scale == "features":
        return xs.T
    return xs


def sa_embedding_frame(m: np.ndarray, points: list[str], prefix: str) -> pd.DataFrame:
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    n_dim = m.shape[1]
    cols = [f"{prefix}{i + 1}" for i in range(n_dim)]
    out = pd.DataFrame({"points": points})
    for i, col in enumerate(cols):
        out[col] = m[:, i]
    return out


def sa_tsne_perplexity(
    perplexity: float | None,
    n: int,
    point_type: str = "sample",
) -> float:
    upper = (n - 1) / 3
    if perplexity is None:
        derived = min(30, int(np.floor(upper)))
        if derived < 1:
            raise ValueError(
                f"`perform_tsne()` cannot embed {n} {point_type}(s): they admit no "
                "perplexity of 1 or more, since Rtsne requires 3 * perplexity <= n - 1."
            )
        return float(derived)
    sa_check_scalar_num(perplexity, "perplexity", 1)
    if perplexity > upper:
        raise ValueError(
            f"`perplexity` must not exceed (n - 1) / 3, which is {upper:.4g} for "
            f"the {n} usable {point_type}(s), but is {perplexity}."
        )
    return float(perplexity)


def sa_umap_neighbors(
    n_neighbors: int | None,
    n: int,
    point_type: str = "sample",
) -> int:
    if n_neighbors is None:
        derived = min(15, n)
        if derived < 2:
            raise ValueError(
                f"`perform_umap()` cannot embed {n} {point_type}(s): they admit no "
                "neighbourhood of 2 or more."
            )
        return int(derived)
    n_neighbors = sa_check_count(n_neighbors, "n_neighbors", 2)
    if n_neighbors > n:
        raise ValueError(
            f"`n_neighbors` must not exceed the {n} usable {point_type}(s) being "
            f"embedded, but is {n_neighbors}."
        )
    return n_neighbors


def sa_reduce_few_points(n_points: int, point_type: str, size: str) -> None:
    if n_points >= 16:
        return
    warnings.warn(
        f"Only {n_points} {point_type}(s) to embed ({size}). This method describes "
        "a neighbourhood, and below about 16 points there is not much of one to "
        "describe. `perform_pca()` is not governed by one.",
        stacklevel=2,
    )
