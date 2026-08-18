"""Build a block correlation matrix (R make_block_cor.R)."""

from __future__ import annotations

from typing import Any

import numpy as np

from statassist.utils.simulate_utils import sa_sim_chol
from statassist.utils.validate import sa_check_count, sa_check_scalar_num


def sa_block_shared_bound(value: float, label: str, k: int, among: str) -> None:
    if value >= 1:
        raise ValueError(
            f"`{label}` of {value} puts {among} at perfect agreement, which is "
            "one variable repeated rather than several, so the matrix is singular "
            "rather than a correlation matrix. Use a value below 1."
        )
    bound = -1 / (k - 1)
    if value <= bound:
        raise ValueError(
            f"`{label}` of {value} is not possible among {among}: one value shared "
            f"by every pair holds only above {bound:.3g}, since they cannot all "
            "disagree with each other at once. Name the ones that move the other "
            "way as `against` in a block instead, which carries the sign and has "
            "no such limit."
        )


def sa_block_names(block: dict[str, Any], label: str) -> None:
    if len(block) == 0:
        return
    nm = list(block.keys())
    if any(n is None or n == "" for n in nm):
        raise ValueError(
            f"`{label}` must name every element it holds: `features`, `cor`, "
            "and `against` when its predictors do not all move the same way."
        )
    repeated = sorted({n for n in nm if nm.count(n) > 1})
    if repeated:
        rep = " and ".join(f"`{r}`" for r in repeated)
        raise ValueError(
            f"`{label}` names {rep} more than once. `$` reads the first of a "
            "repeated name and nothing else, so every later value would be "
            "dropped without a word. Several blocks are several `list()`s: "
            "blocks = [dict(features=[1,2,3], cor=0.9), "
            "dict(features=[4,5,6], cor=-0.4)]."
        )
    unknown = set(nm) - {"features", "cor", "against"}
    if unknown:
        unk = ", ".join(f"`{u}`" for u in sorted(unknown))
        raise ValueError(
            f"`{label}` holds {unk}, which a block has no use for. A block is "
            "`features` and `cor`, and `against` when its predictors do not all "
            "move the same way."
        )


def sa_block_index(
    idx: Any,
    label: str,
    n_features: int,
    min_len: int,
) -> list[int]:
    arr = np.asarray(idx, dtype=int)
    ok = (
        arr.ndim == 1
        and arr.size >= min_len
        and np.all(np.isfinite(arr))
        and np.all(arr == np.floor(arr))
        and np.unique(arr).size == arr.size
    )
    if not ok:
        detail = (
            "one or more distinct whole numbers, the indices of the predictors "
            "on that side of the block."
            if min_len == 1
            else "at least two distinct whole numbers, the indices of the "
            "predictors in the block."
        )
        raise ValueError(f"`{label}` must be {detail}")
    outside = sorted(set(int(x) for x in arr if x < 1 or x > n_features))
    if outside:
        raise ValueError(
            f"`{label}` indexes predictor(s) outside the {n_features} that "
            f"`n_features` asks for: {', '.join(str(x) for x in outside)}."
        )
    return arr.tolist()


def make_block_cor(
    n_features: int,
    blocks: list[dict[str, Any]] | None = None,
    default_cor: float = 0.0,
) -> np.ndarray:
    n_features = sa_check_count(n_features, "n_features", 1)
    sa_check_scalar_num(default_cor, "default_cor", -1, 1)
    blocks = blocks or []
    if not isinstance(blocks, list):
        raise ValueError(
            "`blocks` must be a list of blocks, each a list with `features` and "
            "`cor`."
        )

    if len(blocks) == 0 and n_features > 1:
        sa_block_shared_bound(
            default_cor, "default_cor", n_features, f"{n_features} predictors"
        )

    cor_mat = np.full((n_features, n_features), default_cor, dtype=float)
    np.fill_diagonal(cor_mat, 1.0)
    claimed: list[int] = []

    for k, block in enumerate(blocks):
        label = f"blocks[{k}]"
        if not isinstance(block, dict):
            raise ValueError(f"`{label}` must be a list with `features` and `cor`.")
        sa_block_names(block, label)
        if "features" not in block or "cor" not in block:
            raise ValueError(f"`{label}` must be a list with `features` and `cor`.")

        two_sided = block.get("against") is not None
        feat = sa_block_index(
            block["features"],
            f"{label}$features",
            n_features,
            1 if two_sided else 2,
        )
        agn = (
            sa_block_index(
                block["against"],
                f"{label}$against",
                n_features,
                1,
            )
            if two_sided
            else []
        )
        on_both = set(feat) & set(agn)
        if on_both:
            raise ValueError(
                f"`{label}` names predictor(s) {', '.join(str(x) for x in sorted(on_both))} "
                "in both `features` and `against`, and a predictor cannot move "
                "against itself."
            )

        idx = feat + agn
        signs = [1] * len(feat) + [-1] * len(agn)
        overlap = set(idx) & set(claimed)
        if overlap:
            raise ValueError(
                f"`{label}` overlaps an earlier block at predictor(s) "
                f"{', '.join(str(x) for x in sorted(overlap))}. A predictor can "
                "only carry one within-block correlation, so nest the smaller one "
                "as `default_cor`, or, when this block is what the earlier one "
                "moves against, name it as `against` in that block instead."
            )
        claimed.extend(idx)

        cor_label = f"{label}$cor"
        cor_val = sa_check_scalar_num(block["cor"], cor_label, -1, 1)
        if two_sided:
            if cor_val <= 0:
                raise ValueError(
                    f"`{cor_label}` must be above 0 when `against` is given, but "
                    f"is {cor_val}. Each side agrees at `cor` and disagrees with "
                    "the other side at -`cor`, so it is `against` that makes a "
                    "correlation negative."
                )
            if cor_val >= 1:
                raise ValueError(
                    f"`{cor_label}` of {cor_val} puts each side of the block at "
                    "perfect agreement, which is one variable repeated rather than "
                    "several, so the matrix is singular rather than a correlation "
                    "matrix. Use a value below 1."
                )
        else:
            sa_block_shared_bound(
                cor_val,
                cor_label,
                len(idx),
                f"the {len(idx)} predictors of the block",
            )

        sub = np.array(signs, dtype=float)
        ix = np.array(idx) - 1
        submat = cor_val * np.outer(sub, sub)
        np.fill_diagonal(submat, 1.0)
        cor_mat[np.ix_(ix, ix)] = submat

    if sa_sim_chol(cor_mat) is None:
        eigvals = np.linalg.eigvalsh(cor_mat)
        min_eigen = float(np.min(eigvals))
        raise ValueError(
            "these blocks do not describe a possible correlation matrix: its "
            f"smallest eigenvalue is {min_eigen:.3g}, where a correlation matrix "
            "has none below 0, so no data has these correlations. Every block "
            "holds on its own, so what does not is `default_cor` beside them, "
            "most often a value of the opposite sign to the blocks."
        )
    return cor_mat
