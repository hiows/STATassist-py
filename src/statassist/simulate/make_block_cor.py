"""Build a block correlation matrix for simulators."""

from __future__ import annotations

from typing import Any

import numpy as np

from statassist.utils.validate import sa_check_count, sa_check_scalar_num


def _shared_bound(value: float, label: str, k: int, among: str) -> None:
    if value >= 1:
        raise ValueError(
            f"`{label}` of {value} puts {among} at perfect agreement; use a value below 1."
        )
    bound = -1 / (k - 1)
    if value <= bound:
        raise ValueError(
            f"`{label}` of {value} is not possible among {among}: must be above "
            f"{bound:.3g}. Use `against` in a block for strong negative correlation."
        )


def _block_index(idx: Any, label: str, n_features: int, min_len: int) -> list[int]:
    arr = np.asarray(idx, dtype=int)
    if arr.size < min_len or np.any(arr < 1) or np.any(arr > n_features) or np.unique(arr).size != arr.size:
        raise ValueError(f"`{label}` must be distinct integer indices within `n_features`.")
    return arr.tolist()


def make_block_cor(
    n_features: int,
    blocks: list[dict[str, Any]] | None = None,
    default_cor: float = 0.0,
) -> np.ndarray:
    n_features = sa_check_count(n_features, "n_features", 1)
    sa_check_scalar_num(default_cor, "default_cor", -1, 1)
    blocks = blocks or []

    if not blocks and n_features > 1:
        _shared_bound(default_cor, "default_cor", n_features, f"{n_features} predictors")

    cor_mat = np.full((n_features, n_features), default_cor)
    np.fill_diagonal(cor_mat, 1.0)
    claimed: set[int] = set()

    for k, block in enumerate(blocks):
        label = f"blocks[{k}]"
        if not isinstance(block, dict) or "features" not in block or "cor" not in block:
            raise ValueError(f"`{label}` must be a dict with `features` and `cor`.")
        two_sided = "against" in block and block["against"] is not None
        feat = _block_index(block["features"], f"{label}.features", n_features, 1 if two_sided else 2)
        agn = (
            _block_index(block["against"], f"{label}.against", n_features, 1)
            if two_sided
            else []
        )
        overlap = set(feat) & set(agn)
        if overlap:
            raise ValueError(f"`{label}` lists predictor(s) in both `features` and `against`.")
        idx = feat + agn
        signs = [1] * len(feat) + [-1] * len(agn)
        hit = set(idx) & claimed
        if hit:
            raise ValueError(f"`{label}` overlaps earlier block at {sorted(hit)}.")
        claimed |= set(idx)

        cor_val = float(block["cor"])
        sa_check_scalar_num(cor_val, f"{label}.cor", -1, 1)
        if two_sided:
            if cor_val <= 0:
                raise ValueError(f"`{label}.cor` must be > 0 when `against` is given.")
        else:
            _shared_bound(cor_val, f"{label}.cor", len(idx), f"the {len(idx)} predictors of the block")

        sub = np.array(signs, dtype=float)
        cor_mat[np.ix_(np.array(idx) - 1, np.array(idx) - 1)] = cor_val * np.outer(sub, sub)
        np.fill_diagonal(cor_mat[np.ix_(np.array(idx) - 1, np.array(idx) - 1)], 1.0)

    eigvals = np.linalg.eigvalsh(cor_mat)
    if eigvals.min() < -1e-8:
        raise ValueError(
            f"these blocks do not describe a positive-definite correlation matrix "
            f"(min eigenvalue {eigvals.min():.3g})."
        )
    return cor_mat
