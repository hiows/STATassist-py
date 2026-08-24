"""Facts about a crossed design, rather than about what was planted in one.

The port of ``R/utils_factorial.R``. How many cells there are, the order they are
counted in, which terms a fully crossed model has and which cells each pairwise
contrast averages are all answers that :func:`~statassist.simulate_factorial_groups`
and the factorial comparison have to agree on exactly, since the first writes the
answer key the second is scored against. Two copies of the enumeration would line
up on a balanced two-by-two design and drift apart on anything larger, and the
drift would show as a recall figure rather than as an error.

Two things are spelled differently here than in R, and both are the same decision
the rest of the port already made.

Every index is zero-based
    R stores one-based level positions in its grid and one-based cell numbers in
    :func:`fact_cell_index`. Here they are zero-based throughout, the way
    :func:`~statassist.core.level_pairs` already returns zero-based ``i`` and
    ``j``. Which cell is which does not change; only how it is counted does.

``cells`` is a :class:`~pandas.DataFrame`
    R's ``expand.grid()`` returns a data.frame and the callers read it as
    ``cells[[f]]``, so a frame keyed by factor name is the closest thing that
    reads the same. A design with no factors is a frame of one row and no
    columns, which is the answer wanted when nothing is crossed: one combination,
    holding no constraints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .errors import SaValueError
from .rstats import r_mean
from .tables import level_pairs
from .validate import control_first

__all__ = [
    "FACT_TOL",
    "ContrastSkeleton",
    "fact_cell_index",
    "fact_cell_labels",
    "fact_collapse",
    "fact_component",
    "fact_contrast_skeleton",
    "fact_control_first",
    "fact_grid",
    "fact_subsets",
    "fact_term_effect",
    "fact_term_labels",
    "fact_terms",
]


#: Anything below this is the rounding left over from averaging, not an effect.
#:
#: Port of ``sa_fact_tol()``. Averaging a set of cells divides, so a component or
#: a contrast that is zero by construction can come back a rounding away from it.
#: The ``is_effect`` and ``is_diff`` columns of a truth table are exact
#: comparisons against zero, so the rounding has to be cleared rather than
#: tolerated at the point it is read.
FACT_TOL = 1e-8


def fact_control_first(
    factor_lv: Mapping[str, Sequence[str]],
    control_label: Any,
    lv_source: str = "factor_lv",
) -> dict[str, list[str]]:
    """Point each factor at the reference level it was told to hold.

    Port of ``sa_fact_control_first()``: :func:`~statassist.core.control_first`
    once per factor named. A crossed design has one reference per factor rather
    than one in total, and the cell where every one of them lands is the
    reference cell the effects and the cell labels are read against. Naming a
    level here moves it to the front of its own factor and leaves the other
    factors in the order they arrived, so pointing one factor of three at its
    control is a sentence rather than a rewrite of all three.

    Args:
        factor_lv: Factors and their levels, already validated.
        control_label: A mapping from factor name to the level to hold as that
            factor's reference, or ``None`` to leave every factor as it arrived.
            R accepts a named list or a named character vector here; a
            :class:`dict` is both of those, so a value may be the level name
            itself or a one-element sequence holding it.
        lv_source: What the levels came from, ``"factor_lv"`` when the caller
            named them and ``"factors"`` when they were sorted out of the data,
            so that an error names the argument the missing level is missing
            from.

    Returns:
        ``factor_lv``, each named factor re-pointed and the rest untouched. A new
        mapping; the one handed in is not modified.

    Raises:
        SaValueError: If ``control_label`` is not a mapping of one level name per
            factor, or names a factor the design does not hold.
    """
    resolved = {name: [str(level) for level in levels] for name, levels in factor_lv.items()}
    if control_label is None:
        return resolved

    if not isinstance(control_label, Mapping):
        raise SaValueError(
            "`control_label` must be a named list or named character vector, one level "
            "name per factor it points, with the factor's name as the name. Naming a "
            "factor twice, or none, is not a direction."
        )

    labels: dict[str, Any] = {}
    unusable: list[str] = []
    for name, value in control_label.items():
        if isinstance(value, str):
            labels[str(name)] = value
            continue
        if isinstance(value, Iterable) and len(list(value)) == 1:
            labels[str(name)] = list(value)[0]
            continue
        unusable.append(str(name))
    if unusable or not labels:
        raise SaValueError(
            "`control_label` must hold one level name per factor. Entry/entries holding "
            "something else: " + ", ".join(unusable) + "."
        )

    unknown = [name for name in labels if name not in resolved]
    if unknown:
        raise SaValueError(
            "`control_label` names factor(s) the design does not hold: "
            + ", ".join(unknown)
            + ". Present: "
            + ", ".join(resolved)
            + "."
        )

    for name, label in labels.items():
        resolved[name] = control_first(
            resolved[name],
            label,
            arg=f"control_label['{name}']",
            lv_arg=f"{lv_source}['{name}']",
        )
    return resolved


def fact_grid(lv_list: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Cross a set of factors, allowing the empty set.

    Port of ``sa_fact_grid()``. R's ``expand.grid()`` varies its *first* argument
    fastest, and every truth table's row order rests on that, so the strides are
    written out here rather than left to a library whose default is the other way
    round: :func:`numpy.meshgrid` and :func:`itertools.product` both vary the last
    one fastest.

    ``expand.grid()`` of nothing is a frame of one row and no columns, which is
    the answer wanted when no factor is between or none is within: one
    combination, holding no constraints. Written out rather than relied on,
    because the empty case is the one every index built on it has to survive.

    Args:
        lv_list: Factors and their levels, possibly empty.

    Returns:
        One row per combination, one integer column per factor holding the
        **zero-based** position of that factor's level, with the first factor
        varying fastest.
    """
    names = list(lv_list)
    if not names:
        return pd.DataFrame(index=pd.RangeIndex(1))

    dims = [len(lv_list[name]) for name in names]
    n_cells = int(np.prod(dims))
    columns: dict[str, np.ndarray] = {}
    stride = 1
    for name, size in zip(names, dims, strict=True):
        block = np.repeat(np.arange(size, dtype=np.int64), stride)
        columns[name] = np.tile(block, n_cells // (stride * size))
        stride *= size
    return pd.DataFrame(columns, index=pd.RangeIndex(n_cells))


def fact_cell_labels(
    factor_lv: Mapping[str, Sequence[str]],
    cells: pd.DataFrame,
) -> list[str]:
    """The readable label of every cell of a grid.

    Port of ``sa_fact_cell_labels()``. The level names of each factor joined by a
    dot, in ``factor_lv`` order, which is what a cell truth table and a post-hoc
    stratum are keyed on. A grid of no factors labels its single cell ``""``,
    which is what R's ``paste(character(0), collapse = ".")`` gives.
    """
    names = list(factor_lv)
    if not names:
        return [""] * len(cells)

    parts = [
        [str(factor_lv[name][position]) for position in np.asarray(cells[name], dtype=np.int64)]
        for name in names
    ]
    return [".".join(pieces) for pieces in zip(*parts, strict=True)]


def fact_cell_index(level_idx: Any, dims: Sequence[int]) -> np.ndarray:
    """Which cell of the grid each row of level indices sits in.

    Port of ``sa_fact_cell_index()``. The grid counts the first factor fastest,
    so the cell number is the level indices read as a mixed-radix number. Doing
    it by arithmetic rather than by matching label strings keeps the numbering
    identical to :func:`fact_grid`'s row order by construction, and holds for the
    empty set of factors, where every row belongs to the single cell.

    Args:
        level_idx: One row per observation and one column per factor, in ``dims``
            order, holding **zero-based** level positions.
        dims: Number of levels of each factor.

    Returns:
        Zero-based cell position per row.
    """
    index = np.atleast_2d(np.asarray(level_idx, dtype=np.int64))
    if len(dims) == 0:
        return np.zeros(index.shape[0], dtype=np.int64)

    strides = np.cumprod(np.concatenate(([1], np.asarray(dims, dtype=np.int64))))[: len(dims)]
    return np.asarray(index.reshape(-1, len(dims)) @ strides, dtype=np.int64)


def fact_terms(fac_names: Sequence[str]) -> list[tuple[str, ...]]:
    """Every term a fully crossed model of these factors has.

    Port of ``sa_fact_terms()``. Main effects first, then the interactions in
    increasing order, and within an order in the order the factors were declared.
    That is the order an ANOVA table lists them in, so a term truth table needs no
    reordering to sit beside one.

    Returns:
        One tuple per term, each the factors that term is over.
    """
    names = [str(name) for name in fac_names]
    return [
        combination
        for size in range(1, len(names) + 1)
        for combination in combinations(names, size)
    ]


def fact_term_labels(terms: Sequence[Sequence[str]]) -> list[str]:
    """The readable label of every model term.

    Port of ``sa_fact_term_labels()``. ``a:b``, the way R's ``terms()`` writes an
    interaction, so a term row and a row of an ``aov()`` summary can be matched on
    the name alone.
    """
    return [":".join(term) for term in terms]


def fact_subsets(term: Sequence[str]) -> list[tuple[str, ...]]:
    """Every subset of a term, the empty one included.

    Port of ``sa_fact_subsets()``.
    """
    names = [str(name) for name in term]
    subsets: list[tuple[str, ...]] = [()]
    for size in range(1, len(names) + 1):
        subsets.extend(combinations(names, size))
    return subsets


def _mean(values: np.ndarray) -> float:
    """The correctly rounded mean, which is what R's ``mean()`` aims at.

    Delegates to :func:`~statassist.core.r_mean` so the same rounding is used
    here and in the cell centres that feed :func:`fact_term_effect`.
    """
    return r_mean(values)


def fact_collapse(eff: Any, cells: pd.DataFrame, keep: Sequence[str]) -> np.ndarray:
    """Average the cells down to the levels of a few factors.

    Port of ``sa_fact_collapse()``, which is R's ``stats::ave()`` over the kept
    columns of the grid. **Unweighted**: every cell counts once, whatever it
    holds. That is what makes the decomposition a statement about the levels
    rather than about how many observations happened to land in each, which is
    the same choice ``ss_type = "III"`` and the marginal means of the post-hoc
    stage make.

    Args:
        eff: One value per cell, in ``cells`` order.
        cells: Grid of level indices, as :func:`fact_grid` returns.
        keep: Factors to keep, possibly none, in which case this is the grand mean
            repeated.

    Returns:
        One value per cell, each holding the mean of the cells that agree with it
        on every kept factor.
    """
    values = np.asarray(eff, dtype=float).reshape(-1)
    if len(keep) == 0:
        return np.full(values.shape, _mean(values))

    key = np.column_stack([np.asarray(cells[name], dtype=np.int64) for name in keep])
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    out = np.empty(values.shape, dtype=float)
    for group in range(int(inverse.max()) + 1 if inverse.size else 0):
        at = np.flatnonzero(inverse == group)
        out[at] = _mean(values[at])
    return out


def fact_component(eff: Any, cells: pd.DataFrame, term: Sequence[str]) -> np.ndarray:
    """The ANOVA component of one term, cell by cell.

    Port of ``sa_fact_component()``. The inclusion-exclusion form of the
    decomposition: the component of a term is the cell values averaged down to
    that term, minus everything already accounted for by its sub-terms. Computing
    it rather than remembering what was planted is the point, and it is the same
    computation whether the values are the shifts a simulation put in the cells or
    the centres a comparison measured out of them.

    Args:
        eff: One value per cell, in ``cells`` order.
        cells: Grid of level indices, as :func:`fact_grid` returns.
        term: The factors the term is over.

    Returns:
        One value per cell. Anything within :data:`FACT_TOL` of zero is set to
        exactly zero, so a term that moved nothing reads as having moved nothing.
        A missing value stays missing, where R's subscripted assignment refuses
        one outright; the callers here never hand over a missing cell.
    """
    values = np.asarray(eff, dtype=float).reshape(-1)
    total = np.zeros(values.shape[0], dtype=float)
    for subset in fact_subsets(term):
        sign = (-1.0) ** (len(term) - len(subset))
        total = total + sign * fact_collapse(values, cells, subset)

    total[np.abs(total) < FACT_TOL] = 0.0
    return total


def fact_term_effect(
    eff: Any,
    cells: pd.DataFrame,
    terms: Sequence[Sequence[str]],
) -> np.ndarray:
    """The largest effect each term accounts for, with its sign.

    Port of ``sa_fact_term_effect()``. One number per term out of the whole
    component vector, so that a term has an effect size that can be put on an
    axis beside its p-value. The component of largest absolute value is the one
    taken, since it is the cell the term moved furthest and a term that moved
    nothing is exactly zero there.

    A component is a deviation from what the other terms already predict, not the
    difference between two levels: the components of a two-level factor whose
    levels differ by ``d`` are ``-d/2`` and ``+d/2``. That is the quantity
    :func:`~statassist.simulate_factorial_groups` records, and the reason to keep
    it rather than rescale to a pairwise difference is that the truth table and a
    measured one are then the same number. It is therefore not a fold-change
    up/down between two levels; cutoffs written for a fold change are stricter
    here than they look.

    Args:
        eff: One value per cell, in ``cells`` order, on whatever scale the caller
            wants the answer on.
        cells: Grid of level indices, as :func:`fact_grid` returns.
        terms: One entry per term, as :func:`fact_terms` returns.

    Returns:
        One entry per term, in ``terms`` order. Absolute values within
        :data:`FACT_TOL` of the maximum are treated as a tie, and the earlier
        cell wins (the earlier level of the first factor, which is the reference
        after ``control_label``). That keeps a two-level term from flipping sign
        when an ULP breaks an exact ``+/- d/2`` tie.
    """
    out = np.empty(len(terms), dtype=float)
    for position, term in enumerate(terms):
        component = fact_component(eff, cells, term)
        if np.all(np.isnan(component)):
            out[position] = np.nan
            continue
        out[position] = component[_first_max_abs(component)]
    return out


def _first_max_abs(component: np.ndarray) -> int:
    """Index of the largest ``|component|``, with near-ties taking the earlier cell.

    Values within :data:`FACT_TOL` of the running maximum are treated as equal, so
    a two-level factor's ``-d/2`` / ``+d/2`` pair keeps the earlier (reference)
    cell even when floating point has made one side a hair larger.
    """
    best = -np.inf
    chosen: int | None = None
    for index, value in enumerate(component):
        magnitude = abs(float(value))
        if not np.isfinite(magnitude):
            continue
        if chosen is None or magnitude > best + FACT_TOL:
            best = magnitude
            chosen = index
    return 0 if chosen is None else chosen


class ContrastSkeleton(NamedTuple):
    """The pairwise contrasts a crossed design has, and which cells each is over.

    Attributes:
        table: One row per contrast, holding ``factor``, ``stratum``,
            ``contrast``, ``group1`` and ``group2``. ``stratum`` is ``None`` on
            the marginal row, where R writes ``NA``.
        sel1: Per contrast, the zero-based cell positions whose mean is the left
            side of ``group1 - group2``.
        sel2: Per contrast, the cells whose mean is the right side.
    """

    table: pd.DataFrame
    sel1: list[np.ndarray]
    sel2: list[np.ndarray]


def fact_contrast_skeleton(
    factor_lv: Mapping[str, Sequence[str]],
    cells: pd.DataFrame,
) -> ContrastSkeleton:
    """The pairwise contrasts a crossed design has, and which cells each averages.

    Port of ``sa_fact_contrast_skeleton()``. Built once from the design and reused
    for every feature, since which cells a contrast averages is a fact about the
    layout rather than about the data in it. The simulator turns these selections
    into planted deltas and the comparison turns them into estimates, so the two
    tables merge on ``factor`` / ``stratum`` / ``contrast`` without either side
    sorting the other.

    A factorial design has two pairwise questions per factor and both are here.
    The marginal contrast averages the other factors away, which is what an
    estimated marginal mean does and what a main effect is a statement about. The
    simple effect holds them at one combination, which is the only one of the two
    that says anything when an interaction is real.

    R takes the whole design object and reads ``factor_lv`` and ``cells`` off it.
    The two are passed separately here, so that a caller holding a design of its
    own shape does not have to pretend to be R's list.

    Args:
        factor_lv: Factors and their levels, the reference first.
        cells: Grid of level indices, as :func:`fact_grid` returns.

    Returns:
        A :class:`ContrastSkeleton`. The marginal contrasts of a factor come
        first, then one block per combination of the other factors, so a table
        read top to bottom moves from the main effect to the simple effects it
        may be hiding.
    """
    names = list(factor_lv)
    factors: list[str] = []
    strata_label: list[str | None] = []
    contrast: list[str] = []
    group1: list[str] = []
    group2: list[str] = []
    sel1: list[np.ndarray] = []
    sel2: list[np.ndarray] = []

    for name in names:
        levels = [str(level) for level in factor_lv[name]]
        pairs = level_pairs(levels)
        pair_at = pairs[["i", "j"]].to_numpy(dtype=np.int64)
        pair_text = pairs[["contrast", "group1", "group2"]].astype(str).to_numpy()
        others = [other for other in names if other != name]
        strata = fact_grid({other: factor_lv[other] for other in others})
        own = np.asarray(cells[name], dtype=np.int64)

        # The marginal block first, then one block per combination of the other
        # factors, so a table read top to bottom moves from the main effect to the
        # simple effects it may be hiding.
        blocks: list[tuple[str | None, np.ndarray]] = [(None, np.ones(len(cells), dtype=bool))]
        for row in range(len(strata)):
            held = np.ones(len(cells), dtype=bool)
            for other in others:
                held &= np.asarray(cells[other], dtype=np.int64) == int(strata[other].iloc[row])
            label = ".".join(
                str(factor_lv[other][int(strata[other].iloc[row])]) for other in others
            )
            blocks.append((label, held))

        for stratum, held in blocks:
            at = [np.flatnonzero(held & (own == position)) for position in range(len(levels))]
            for (first, second), text in zip(pair_at, pair_text, strict=True):
                factors.append(name)
                strata_label.append(stratum)
                contrast.append(str(text[0]))
                group1.append(str(text[1]))
                group2.append(str(text[2]))
                sel1.append(at[first])
                sel2.append(at[second])

    table = pd.DataFrame(
        {
            "factor": pd.Series(factors, dtype=object),
            "stratum": pd.Series(strata_label, dtype=object),
            "contrast": pd.Series(contrast, dtype=object),
            "group1": pd.Series(group1, dtype=object),
            "group2": pd.Series(group2, dtype=object),
        }
    )
    return ContrastSkeleton(table=table, sel1=sel1, sel2=sel2)
