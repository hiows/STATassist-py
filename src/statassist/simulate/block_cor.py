"""The correlation matrix the supervised learning simulators take.

The port of ``R/make_block_cor.R``. It is a separate function rather than an
argument of the simulators because a matrix written out by hand is unreadable
past about four predictors, while the structure that matters is almost always
blocks: a handful of predictors that measure nearly the same thing, and the rest
unrelated to them.

It is checked here rather than at the draw. This is where the caller wrote the
blocks down, so it is where a message about them is worth anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core.errors import SaValueError
from ..core.validate import check_count, check_scalar_num, fmt_num
from ._supervised import chol_or_none

__all__ = ["make_block_cor"]

#: The only keys a block has a use for.
_BLOCK_KEYS = ("features", "cor", "against")


def _signif(value: float, digits: int = 3) -> str:
    """R's ``signif()`` followed by ``as.character()``.

    The bounds and eigenvalues quoted in these messages are exact fractions and
    floating point residues, and printing seventeen digits of either would bury
    the number that matters.
    """
    if value == 0 or not math.isfinite(value):
        return fmt_num(value)
    exponent = math.floor(math.log10(abs(value)))
    return fmt_num(round(value, -(exponent - digits + 1)))


def make_block_cor(
    n_features: int,
    blocks: Sequence[Mapping[str, Any]] = (),
    default_cor: float = 0,
) -> np.ndarray:
    """Build a block correlation matrix.

    Assembles a correlation matrix out of groups of predictors that correlate
    with each other, which is the structure worth simulating when the question
    is how a model behaves under collinearity. Every predictor inside a block
    correlates with every other one in it at the same value, and everything
    outside every block correlates at ``default_cor``. A block that names
    ``against`` is split in two instead: each side agrees within itself and
    disagrees with the other side.

    The matrix is what :func:`~statassist.simulate.simulate_regression` and
    :func:`~statassist.simulate.simulate_classification` take as ``cor_mat``.
    Its point there is that a null predictor correlated with a planted one is not
    distinguishable from the planted one by the data alone, so its estimated
    coefficient is drawn away from the zero it really has. That is the single
    most common reason a coefficient table names the wrong predictor, and it
    cannot be shown at all with independent predictors.

    **Indices are zero-based**, unlike R's, so they index the predictors the way
    Python indexes everything else: ``range(3)`` is the first three, where R
    writes ``1:3``. The predictor *names* the simulators give out are
    one-based - ``features=[0, 1]`` is ``x_1`` and ``x_2``.

    Blocks may not overlap. A predictor in two blocks would have two
    correlations with the same partner and only one of them could be written
    down, so the second block would silently win. Nest the smaller correlation
    as ``default_cor`` and name one block instead, or, when the second group is
    what the first moves against, name it as ``against`` in one block.

    How strong a negative correlation one block can hold depends on how many
    predictors are in it. A block of ``k`` predictors sharing one value is
    positive definite only above ``-1 / (k - 1)``: two predictors may disagree
    at -0.9, three at no more than -0.5, four at no more than -0.333. Three
    predictors cannot all disagree strongly, since whichever way the third moves
    it agrees with one of the first two.

    ``against`` is how a strong negative correlation is written down instead.
    ``{"features": range(3), "cor": 0.9, "against": range(3, 6)}`` puts 0.9 among
    the first three, 0.9 among the last three and -0.9 between the two sides.
    Splitting a block by which way its predictors move leaves it positive
    definite for any ``cor`` below 1 whatever its size, since it is then one
    factor with a sign per predictor rather than a demand that everything
    disagree at once.

    The result is checked for positive definiteness, which is the property that
    separates a matrix of correlations from a matrix of numbers between -1 and 1.
    The blocks and ``default_cor`` are checked one at a time first, so that a
    value no block of that size could hold is named as such, and the eigenvalue
    of the assembled matrix is what reports a ``default_cor`` the blocks cannot
    sit inside.

    Args:
        n_features: Number of predictors the matrix describes, so its size.
        blocks: Blocks, each a mapping with ``features``, the indices in the
            block, and ``cor``, the correlation they share. A block may also
            name ``against``, further indices that correlate at ``cor`` among
            themselves and at ``-cor`` with the ones in ``features``. No blocks
            at all gives a matrix with ``default_cor`` everywhere off the
            diagonal.
        default_cor: Correlation between any two predictors that are not in a
            block together. The default of ``0`` leaves them independent.

    Returns:
        A symmetric ``n_features`` by ``n_features`` array with 1 on the
        diagonal.

    Raises:
        SaValueError: If a block is malformed, holds a correlation no block of
            its size could hold, overlaps an earlier block, or if the assembled
            matrix is not positive definite.

    Examples:
        Six predictors: the first two nearly interchangeable, the next three
        moderately related, and the sixth on its own.

        >>> cor_mat = make_block_cor(
        ...     n_features=6,
        ...     blocks=[
        ...         {"features": [0, 1], "cor": 0.8},
        ...         {"features": [2, 3, 4], "cor": 0.5},
        ...     ],
        ... )
        >>> cor_mat[:3, :3].tolist()
        [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]]

        Three predictors moving one way and three the other. ``against`` carries
        the sign, so -0.9 between the sides is available at any block size.

        >>> opposed = make_block_cor(
        ...     6, [{"features": range(3), "cor": 0.9, "against": range(3, 6)}]
        ... )
        >>> [float(opposed[0, 1]), float(opposed[0, 3])]
        [0.9, -0.9]

        Three predictors cannot all disagree at -0.6, and the limit for a block
        of three is named rather than left to the matrix.

        >>> make_block_cor(6, [{"features": range(3), "cor": -0.6}])
        Traceback (most recent call last):
            ...
        statassist.core.errors.SaValueError: `blocks[0]['cor']` of -0.6 is not
        possible among the 3 predictors of the block: ...

        Four predictors cannot all disagree with each other, so a request for it
        is refused here rather than met later by a draw that quietly ignores it.

        >>> make_block_cor(4, default_cor=-0.5)
        Traceback (most recent call last):
            ...
        statassist.core.errors.SaValueError: `default_cor` of -0.5 is not
        possible among 4 predictors: ...
    """
    n_features = check_count(n_features, "n_features", 1)
    default_cor = check_scalar_num(default_cor, "default_cor", -1, 1)
    if isinstance(blocks, Mapping) or not isinstance(blocks, Sequence):
        raise SaValueError(
            "`blocks` must be a sequence of blocks, each a mapping with `features` and `cor`."
        )
    # With no block to break the pattern the whole matrix holds one value off the
    # diagonal, so the bound on it is exact and belongs beside the argument. With
    # blocks it is the combination that can fail, and the eigenvalue at the end
    # is what says so.
    if len(blocks) == 0 and n_features > 1:
        _shared_bound(default_cor, "default_cor", n_features, f"{n_features} predictors")

    cor_mat = np.full((n_features, n_features), float(default_cor))
    np.fill_diagonal(cor_mat, 1.0)

    claimed: list[int] = []
    for k, block in enumerate(blocks):
        label = f"blocks[{k}]"
        if not isinstance(block, Mapping):
            raise SaValueError(f"`{label}` must be a mapping with `features` and `cor`.")
        _block_names(block, label)
        if block.get("features") is None or block.get("cor") is None:
            raise SaValueError(f"`{label}` must be a mapping with `features` and `cor`.")

        # A side of one predictor is meaningful only against another side, so one
        # index is enough for a split block and two for `features` otherwise.
        two_sided = block.get("against") is not None
        feat = _block_index(
            block["features"], f"{label}['features']", n_features, 1 if two_sided else 2
        )
        agn = (
            _block_index(block["against"], f"{label}['against']", n_features, 1)
            if two_sided
            else []
        )
        on_both = sorted(set(feat) & set(agn))
        if on_both:
            raise SaValueError(
                f"`{label}` names predictor(s) " + ", ".join(str(i) for i in on_both) + " in both "
                "`features` and `against`, and a predictor cannot move against itself."
            )

        idx = feat + agn
        signs = np.array([1.0] * len(feat) + [-1.0] * len(agn))

        # A predictor in two blocks would need two correlations with the same
        # partner, and only the later one would survive being written down. Asked
        # before the value, since how negative the value may be is a question
        # about the size of the block, and the block is not settled until this
        # holds.
        overlap = sorted(set(idx) & set(claimed))
        if overlap:
            raise SaValueError(
                f"`{label}` overlaps an earlier block at predictor(s) "
                + ", ".join(str(i) for i in overlap)
                + ". A predictor can only carry one within-block correlation, so "
                "nest the smaller one as `default_cor`, or, when this block is "
                "what the earlier one moves against, name it as `against` in "
                "that block instead."
            )
        claimed.extend(idx)

        cor_label = f"{label}['cor']"
        cor = check_scalar_num(block["cor"], cor_label, -1, 1)
        if two_sided:
            # `against` is what carries the sign here. A negative `cor` would
            # make each side disagree within itself and agree across, which is
            # the limit this argument exists to lift, written backwards.
            if cor <= 0:
                raise SaValueError(
                    f"`{cor_label}` must be above 0 when `against` is given, but is "
                    f"{fmt_num(cor)}. Each side agrees at `cor` and disagrees with "
                    "the other side at -`cor`, so it is `against` that makes a "
                    "correlation negative."
                )
            if cor >= 1:
                raise SaValueError(
                    f"`{cor_label}` of {fmt_num(cor)} puts each side of the block at "
                    "perfect agreement, which is one variable repeated rather than "
                    "several, so the matrix is singular rather than a correlation "
                    "matrix. Use a value below 1."
                )
        else:
            _shared_bound(cor, cor_label, len(idx), f"the {len(idx)} predictors of the block")

        block_idx = np.ix_(idx, idx)
        cor_mat[block_idx] = cor * np.outer(signs, signs)
        # `outer()` puts `cor` where the diagonal is, since a sign times itself
        # is 1 rather than the correlation of a predictor with itself.
        cor_mat[idx, idx] = 1.0

    # Rejected here rather than by the factorisation inside a simulator, where
    # the message would be about a matrix the caller never wrote out. Every block
    # holds on its own by this point, so what is left to catch is how they meet.
    if chol_or_none(cor_mat) is None:
        min_eigen = float(np.linalg.eigvalsh(cor_mat).min())
        raise SaValueError(
            "these blocks do not describe a possible correlation matrix: its "
            f"smallest eigenvalue is {_signif(min_eigen)}, where a correlation "
            "matrix has none below 0, so no data has these correlations. Every "
            "block holds on its own, so what does not is `default_cor` beside "
            "them, most often a value of the opposite sign to the blocks."
        )
    return cor_mat


def _block_names(block: Mapping[str, Any], label: str) -> None:
    """Check the names a block holds.

    Port of ``sa_block_names()``. The trap it exists for in R is
    ``list(features = 1:3, cor = 0.9, features = 4:6, cor = -0.4)``, which reads
    as two blocks and is one: ``$`` returns the first of a repeated name, so the
    second pair is dropped without a word.

    A mapping cannot hold a repeated key, so that half of the check has nothing
    left to catch here - though the trap itself survives the port with its sign
    flipped, since a Python literal keeps the **last** of a repeated key where R
    keeps the first. What is checked is that every key is one a block has a use
    for, since a misspelled ``agianst`` would otherwise be a block that quietly
    is not split.
    """
    if not block:
        return
    if not all(isinstance(name, str) and name for name in block):
        raise SaValueError(
            f"`{label}` must name every element it holds: `features`, `cor`, and "
            "`against` when its predictors do not all move the same way."
        )
    unknown = [name for name in block if name not in _BLOCK_KEYS]
    if unknown:
        raise SaValueError(
            f"`{label}` holds "
            + ", ".join(f"`{name}`" for name in unknown)
            + ", which a block has no use for. A block is `features` and `cor`, "
            "and `against` when its predictors do not all move the same way. "
            "Several blocks are several mappings: blocks=[{'features': [0, 1, 2], "
            "'cor': 0.9}, {'features': [3, 4, 5], 'cor': -0.4}]."
        )


def _block_index(idx: Any, label: str, n_features: int, min_len: int) -> list[int]:
    """Check one side of a block's predictor indices.

    Port of ``sa_block_index()``.

    Args:
        idx: The indices as received.
        label: What to call them in an error message.
        n_features: How many predictors there are to index.
        min_len: Indices a side must hold. Two for a block that is not split,
            since a correlation needs a pair, and one for each side of a split
            block, since the pair is then across the two sides.

    Returns:
        The indices as a list of ints, in the order they were given.
    """
    # A bare number is a length-one vector in R, so `against=5` names one
    # predictor on that side rather than being a malformed side.
    array = np.atleast_1d(np.asarray(list(idx) if _is_vector(idx) else idx))
    numeric = array.ndim == 1 and array.dtype.kind in "iuf" and array.dtype != bool
    values = np.asarray(array, dtype=float) if numeric else np.empty(0)
    ok = (
        numeric
        and values.size >= min_len
        and bool(np.isfinite(values).all())
        and bool((values == np.trunc(values)).all())
        and len(set(values.tolist())) == values.size
    )
    if not ok:
        detail = (
            "one or more distinct whole numbers, the indices of the predictors "
            "on that side of the block."
            if min_len == 1
            else "at least two distinct whole numbers, the indices of the predictors in the block."
        )
        raise SaValueError(f"`{label}` must be {detail}")

    out = [int(value) for value in values]
    outside = sorted({i for i in out if i < 0 or i >= n_features})
    if outside:
        raise SaValueError(
            f"`{label}` indexes predictor(s) outside the {n_features} that "
            "`n_features` asks for: " + ", ".join(str(i) for i in outside) + "."
        )
    return out


def _is_vector(value: Any) -> bool:
    """Whether the value is an iterable of indices rather than a single one."""
    if isinstance(value, str | bytes | np.ndarray):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _shared_bound(value: float, label: str, k: int, among: str) -> None:
    """Check that one shared correlation could hold among ``k`` predictors.

    Port of ``sa_block_shared_bound()``. A ``k`` by ``k`` matrix with 1 on the
    diagonal and one value everywhere else has eigenvalues ``1 - value`` and
    ``1 + (k - 1) * value``, so it is a correlation matrix exactly for ``value``
    in ``(-1 / (k - 1), 1)``. Both ends are worth a sentence of their own: the
    top is one predictor written twice, and the bottom is the reason ``against``
    exists.

    Args:
        value: The shared correlation.
        label: What to call it in an error message.
        k: How many predictors share it.
        among: Noun phrase for the predictors sharing the value, since the same
            bound is what ``default_cor`` and an unsplit block are each held to.
    """
    if value >= 1:
        raise SaValueError(
            f"`{label}` of {fmt_num(value)} puts {among} at perfect agreement, which "
            "is one variable repeated rather than several, so the matrix is "
            "singular rather than a correlation matrix. Use a value below 1."
        )
    bound = -1 / (k - 1)
    if value <= bound:
        raise SaValueError(
            f"`{label}` of {fmt_num(value)} is not possible among {among}: one value "
            f"shared by every pair holds only above {_signif(bound)}, since they "
            "cannot all disagree with each other at once. Name the ones that move "
            "the other way as `against` in a block instead, which carries the sign "
            "and has no such limit."
        )
