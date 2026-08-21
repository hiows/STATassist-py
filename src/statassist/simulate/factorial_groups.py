"""A crossed-factor experiment whose answer is known, term by term.

The port of ``R/simulate_factorial_groups.R``.

Two things this port settles differently from R, both of them the decision the
rest of the port already made.

The seed reproduces this package and not R
    ``seed=`` promises the same result from the same seed within Python. R's
    generators and NumPy's differ, so no draw here matches a draw there and
    nothing short of reimplementing R's RNG would change that. What the seed does
    not excuse is an undocumented **order** of consumption, since that is what
    decides whether two calls that differ in one argument share a prefix of their
    draws. The order is written out under :func:`simulate_factorial_groups`.

Every index is zero-based
    The reference cell is cell 0 rather than cell 1, and the level positions in
    ``design.cells`` count from zero, exactly as
    :mod:`statassist.core.factorial` returns them. Which cell is which does not
    change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.errors import SaInternalError, SaValueError
from ..core.factorial import (
    FACT_TOL,
    fact_cell_index,
    fact_cell_labels,
    fact_component,
    fact_contrast_skeleton,
    fact_grid,
    fact_term_labels,
    fact_terms,
)
from ..core.random import SaRandom
from ..core.result import SaSimulation
from ..core.validate import check_count, check_range, check_scalar_num
from ._patterns import allocate, pattern_delta, pick_up_down
from ._patterns import pattern_mix as check_pattern_mix

__all__ = ["FACT_SHAPES", "RESERVED_FACTOR_NAMES", "simulate_factorial_groups"]

#: The shapes an effect can take across the terms of a factorial model.
#:
#: Port of ``sa_fact_shapes()``. Named in one place so that the ``term_mix``
#: argument, its validation and the dispatch in :func:`_plant` cannot come to
#: disagree about what exists.
FACT_SHAPES: tuple[str, ...] = (
    "main_only",
    "additive",
    "interaction",
    "crossover",
    "nuisance_only",
)

#: Column names ``truth_cell`` uses for itself, which no factor may be named.
#:
#: That table gives each factor a column of its own beside these, so a factor
#: named after one of them would produce a table with two columns of one name.
RESERVED_FACTOR_NAMES: tuple[str, ...] = ("features", "is_ref", "delta", "center", "sd", "n")

#: The design used when the caller names no factors: four treatments by two sexes.
_DEFAULT_FACTOR_LV: dict[str, tuple[str, ...]] = {
    "treatment": ("control", "treat_A", "treat_B", "treat_C"),
    "sex": ("male", "female"),
}
#: Share of the features moved in each direction when the caller names no count.
_PLANT_SHARE = 0.15


class Design(NamedTuple):
    """Everything the layout of a crossed design fixes, before anything is drawn.

    Attributes:
        factor_lv: Factors and their levels, in declaration order.
        within: The factors measured within subjects, in ``factor_lv`` order.
        between: The rest, in ``factor_lv`` order.
        cells: One row per cell, one column per factor of zero-based level
            positions, the first factor varying fastest.
        cell_label: Readable label of every cell.
        ref_cell: Position of the cell where every factor is at its reference,
            which :func:`~statassist.core.fact_grid` puts first.
        cell_n: Observations in each cell.
        terms: Every term of the fully crossed model.
        cell_idx: Which cell each row of the data sits in.
        factors: One list of level labels per factor, as long as the data.
        subject: Subject label per row, or ``None`` in a wholly between design.
        subject_idx: Which unit each row belongs to.
        n_units: Subjects when there are within factors, observations otherwise.
        n_rows: Rows of data.
    """

    factor_lv: dict[str, list[str]]
    within: list[str]
    between: list[str]
    cells: pd.DataFrame
    cell_label: list[str]
    ref_cell: int
    cell_n: np.ndarray
    terms: list[tuple[str, ...]]
    cell_idx: np.ndarray
    factors: dict[str, list[str]]
    subject: list[str] | None
    subject_idx: np.ndarray
    n_units: int
    n_rows: int

    @property
    def n_cells(self) -> int:
        """How many cells the grid holds."""
        return len(self.cells.index)


def _check_factor_lv(factor_lv: Any) -> dict[str, list[str]]:
    """Read the factors and their levels, or say why they cannot be read."""
    if (
        not isinstance(factor_lv, Mapping)
        or len(factor_lv) < 2
        or not all(isinstance(name, str) and name for name in factor_lv)
    ):
        raise SaValueError(
            "`factor_lv` must be a named mapping of at least two crossed factors, "
            "each entry the levels of one factor with the reference level first. "
            "Use simulate_multiple_groups() for a single factor."
        )

    reserved = [name for name in factor_lv if name in RESERVED_FACTOR_NAMES]
    if reserved:
        raise SaValueError(
            "`factor_lv` names factor(s) that the answer tables already use as "
            "columns: " + ", ".join(reserved) + "."
        )

    out: dict[str, list[str]] = {}
    for name, levels in factor_lv.items():
        usable = isinstance(levels, Sequence) and not isinstance(levels, str)
        items = list(levels) if usable else []
        if (
            not usable
            or len(items) < 2
            or not all(isinstance(level, str) and level for level in items)
            or len(set(items)) != len(items)
        ):
            raise SaValueError(
                f"`factor_lv['{name}']` must be at least two distinct non-empty level "
                "names, the first being the reference."
            )
        out[str(name)] = [str(level) for level in items]
    return out


def _design(factor_lv: Any, within: Any, n_per_cell: Any) -> Design:
    """Work out the cells, the rows and the subjects the design implies.

    Port of ``sa_fact_design()``. The factors, which of them are repeated and how
    big a cell is all come from three arguments that have to agree, so they are
    settled together rather than in passes that could disagree. ``factor_lv``
    names the factors and their levels, which makes its length the number of
    crossed factors; ``within`` names a subset of them; and ``n_per_cell`` carries
    one size per combination of the factors that are left, which makes its length
    say how many of those there are.

    Nothing is drawn here.
    """
    resolved = _check_factor_lv(factor_lv)

    named = [] if within is None else list(np.atleast_1d(np.asarray(within, dtype=object)))
    if not all(isinstance(name, str) for name in named) or len(set(named)) != len(named):
        raise SaValueError(
            "`within` must be None, or the distinct names of the factors measured within subjects."
        )
    unknown = [name for name in named if name not in resolved]
    if unknown:
        raise SaValueError(
            "`within` names factor(s) that `factor_lv` does not hold: "
            + ", ".join(unknown)
            + ". Known factors are: "
            + ", ".join(resolved)
            + "."
        )
    # Put back into `factor_lv` order, so that every table built from either list
    # is in the order the factors were declared in rather than the order they were
    # named in here.
    repeated = [name for name in resolved if name in named]
    between = [name for name in resolved if name not in repeated]

    dims = [len(levels) for levels in resolved.values()]
    cells = fact_grid(resolved)
    between_cells = fact_grid({name: resolved[name] for name in between})
    within_cells = fact_grid({name: resolved[name] for name in repeated})
    n_between = len(between_cells.index)
    n_within = len(within_cells.index)

    array = np.atleast_1d(np.asarray(n_per_cell))
    numeric = array.ndim == 1 and array.dtype.kind in "iuf" and array.dtype != bool
    if not numeric or array.size not in (1, n_between):
        raise SaValueError(
            "`n_per_cell` must be one size, or one size per combination of the "
            f"between-subject factors, of which this design has {n_between}. The "
            "within-subject factors are crossed with every subject, so they cannot "
            "hold sizes of their own."
        )
    if array.size == 1:
        sizes = np.full(n_between, check_count(array[0], "n_per_cell", 2), dtype=np.int64)
    else:
        sizes = np.array(
            [check_count(value, f"n_per_cell[{k}]", 2) for k, value in enumerate(array)],
            dtype=np.int64,
        )

    # A unit is a subject when there are within factors and an observation when
    # there are not, which is the same statement either way: a unit sits in one
    # combination of the between factors and contributes one row to each
    # combination of the within ones.
    unit_between = np.repeat(np.arange(n_between, dtype=np.int64), sizes)
    n_units = int(unit_between.size)
    n_rows = n_units * n_within
    subject_idx = np.repeat(np.arange(n_units, dtype=np.int64), n_within)
    within_row = np.tile(np.arange(n_within, dtype=np.int64), n_units)

    level_idx = np.zeros((n_rows, len(resolved)), dtype=np.int64)
    position = {name: at for at, name in enumerate(resolved)}
    for name in between:
        level_idx[:, position[name]] = np.asarray(between_cells[name], dtype=np.int64)[
            unit_between[subject_idx]
        ]
    for name in repeated:
        level_idx[:, position[name]] = np.asarray(within_cells[name], dtype=np.int64)[within_row]

    # Which between-subject combination each cell belongs to, so that a cell can
    # say how many observations it holds without counting rows.
    cell_between = fact_cell_index(
        np.column_stack([np.asarray(cells[name], dtype=np.int64) for name in between])
        if between
        else np.zeros((len(cells.index), 0), dtype=np.int64),
        [len(resolved[name]) for name in between],
    )

    return Design(
        factor_lv=resolved,
        within=repeated,
        between=between,
        cells=cells,
        cell_label=fact_cell_labels(resolved, cells),
        ref_cell=0,
        cell_n=sizes[cell_between],
        terms=fact_terms(list(resolved)),
        cell_idx=fact_cell_index(level_idx, dims),
        factors={
            name: [resolved[name][at] for at in level_idx[:, position[name]]] for name in resolved
        },
        subject=[f"subject_{at + 1}" for at in subject_idx] if repeated else None,
        subject_idx=subject_idx,
        n_units=n_units,
        n_rows=n_rows,
    )


def _shuffle(values: list[str], rng: np.random.Generator) -> list[str]:
    """Permute a list, leaving a list of one alone.

    Port of ``sa_fact_shuffle()``. R's guard is there because ``sample()`` of a
    length-one vector reads its argument as a count; ``rng.permutation`` has no
    such trap. The branch is kept anyway, because the RNG stream is documented
    and R draws nothing here - so neither does this.
    """
    if len(values) <= 1:
        return list(values)
    return [values[at] for at in rng.permutation(len(values))]


def _partner(shape: str, fac_names: Sequence[str], rng: np.random.Generator) -> str | None:
    """Pick the factor an effect leans on besides the primary one.

    Port of ``sa_fact_partner()``. Drawn at random from the factors after the
    first, the way the ``"single"`` profile picks the level it moves, and reported
    in ``truth["partner"]`` so that it is looked up rather than inferred.
    ``"main_only"`` needs no partner and takes none, which is also why it draws
    nothing: the stream is a function of the shapes that were handed out.
    """
    if shape == "main_only":
        return None
    others = list(fac_names[1:])
    return others[int(rng.integers(len(others)))]


def _profile(d: float, spread: str, k: int, rng: np.random.Generator) -> np.ndarray:
    """A profile along one factor, centred so that it is a main effect.

    Port of ``sa_fact_profile()``. The uncentred profile is the one
    :func:`~statassist.simulate_multiple_groups` plants: zero at the reference
    level and the magnitude spread over the others by the shape. Centring it turns
    it into the main effect an ANOVA would report, and since the whole array is
    shifted to put the reference cell at zero afterwards, the uncentred profile is
    what ``truth_cell["delta"]`` comes back holding anyway.
    """
    raw = np.concatenate([[0.0], pattern_delta(d, spread, k - 1, rng)])
    return np.asarray(raw - raw.mean(), dtype=float)


def _flip(k: int) -> np.ndarray:
    """Signs that turn a profile into a pure interaction.

    Port of ``sa_fact_flip()``. A main effect along the primary factor becomes an
    interaction with the partner by being multiplied by a different number at each
    partner level. For the result to be a *pure* interaction, both of its main
    effects have to vanish: the primary one vanishes when these numbers average to
    zero, and the partner one when the profile they multiply is centred, which
    :func:`_profile` sees to.

    Alternating signs are the arrangement that makes the effect reverse rather
    than merely differ, which is what a crossover is. They are centred so that
    they average to exactly zero at an odd number of levels too, and then scaled
    so that the largest of them is one, which keeps the size of the interaction
    the size that was asked for rather than a multiple of it that depends on how
    many levels the partner has.
    """
    signs = np.resize(np.array([1.0, -1.0]), k)
    signs = signs - signs.mean()
    return np.asarray(signs / np.abs(signs).max(), dtype=float)


def _plant(
    d: float,
    shape: str,
    spread: str,
    mate: str | None,
    factor_lv: Mapping[str, Sequence[str]],
    cells: pd.DataFrame,
    interaction_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spread one magnitude over the terms according to its shape.

    Port of ``sa_fact_plant()``.

    Args:
        d: Signed magnitude of the effect, on the log2 scale.
        shape: One of :data:`FACT_SHAPES`.
        spread: ``"all"``, ``"gradient"`` or ``"single"``, the profile along a
            factor.
        mate: Partner factor name, or ``None`` for ``"main_only"``.
        factor_lv: Factors and their levels, the primary one first.
        cells: Grid of level indices, one row per cell.
        interaction_scale: Size of the interaction relative to the main effect
            under ``"interaction"``.
        rng: What the ``"single"`` profile picks its level from.

    Returns:
        One effect per cell, in ``cells`` order, summing to zero along every
        factor it does not use.
    """
    primary = list(factor_lv)[0]
    if shape not in FACT_SHAPES:
        raise SaInternalError(f"internal error: unknown effect shape `{shape}`.")
    if shape == "main_only":
        mate_name = primary
    elif mate is None:
        raise SaInternalError(f"internal error: shape `{shape}` needs a partner factor.")
    else:
        mate_name = mate

    def profile(name: str) -> np.ndarray:
        return _profile(d, spread, len(factor_lv[name]), rng)

    def main(name: str, values: np.ndarray) -> np.ndarray:
        return values[np.asarray(cells[name], dtype=np.int64)]

    def crossed(values: np.ndarray) -> np.ndarray:
        """The primary profile, reversed level by level along the partner.

        Both main effects of this are exactly zero, so it is interaction and
        nothing else. R indexes ``outer(v, flip)`` with a two-column matrix of
        one-based level positions; the zero-based positions index the same cells.
        """
        table = np.outer(values, _flip(len(factor_lv[mate_name])))
        return table[
            np.asarray(cells[primary], dtype=np.int64),
            np.asarray(cells[mate_name], dtype=np.int64),
        ]

    if shape == "main_only":
        return main(primary, profile(primary))
    if shape == "additive":
        together = main(primary, profile(primary)) + main(mate_name, profile(mate_name))
        return np.asarray(together, dtype=float)
    if shape == "nuisance_only":
        return main(mate_name, profile(mate_name))
    if shape == "interaction":
        primary_profile = profile(primary)
        return main(primary, primary_profile) + interaction_scale * crossed(primary_profile)
    return crossed(profile(primary))


def _truth(
    feats: list[str],
    delta: np.ndarray,
    design: Design,
    pattern: np.ndarray,
    spread: np.ndarray,
    direction: np.ndarray,
    partner: list[str | None],
    baseline: np.ndarray,
    sd_subject: np.ndarray,
) -> pd.DataFrame:
    """Feature-level answer.

    Port of ``sa_fact_truth()``. ``delta`` is features by cells, the reference
    cell being zero throughout.
    """
    abs_delta = np.abs(delta)
    largest = abs_delta.max(axis=1)
    # A shape that leaves several cells equally far from the reference leaves no
    # single cell furthest, so the tie is reported rather than broken silently and
    # scored on.
    tied = (abs_delta == largest[:, None]).sum(axis=1) > 1
    # R's `max.col(ties.method = "first")` and `argmax` both take the first
    # maximum, so the cell a tie names is the same in both.
    which_max = abs_delta.argmax(axis=1)

    return pd.DataFrame(
        {
            "features": feats,
            "pattern": pattern.astype(str),
            "spread": spread.astype(str),
            "direction": direction.astype(str),
            "partner": pd.Series(partner, dtype=object),
            "extreme_cell": pd.Series(
                [
                    None if size == 0 else design.cell_label[int(cell)]
                    for size, cell in zip(largest, which_max, strict=True)
                ],
                dtype=object,
            ),
            "extreme_tied": tied,
            "log2fc": delta[np.arange(len(feats)), which_max],
            "baseline": baseline,
            "sd_subject": sd_subject,
        }
    )


def _truth_term(
    feats: list[str],
    delta: np.ndarray,
    design: Design,
    planted: np.ndarray,
) -> pd.DataFrame:
    """Per feature and term answer, in the row order an ANOVA table follows.

    Port of ``sa_fact_truth_term()``. An unplanted feature is zero in every cell,
    so its every component is zero too and decomposing it would be work done to
    arrive back at the matrix it started as.
    """
    terms = design.terms
    n_terms = len(terms)

    components = np.zeros((len(feats), n_terms))
    for i in np.flatnonzero(planted):
        for k, term in enumerate(terms):
            components[i, k] = np.abs(fact_component(delta[i], design.cells, term)).max()
    # R flattens with `as.vector(t(comp))`, which is feature-major; `ravel()` is
    # the same order.
    flat = components.ravel()

    return pd.DataFrame(
        {
            "features": np.repeat(feats, n_terms),
            "terms": np.tile(fact_term_labels(terms), len(feats)),
            "term_order": np.tile([len(term) for term in terms], len(feats)),
            "is_within": np.tile(
                [any(name in design.within for name in term) for term in terms], len(feats)
            ),
            "max_abs_delta": flat,
            "is_effect": flat > 0,
        }
    )


def _truth_cell(
    feats: list[str],
    delta: np.ndarray,
    center: np.ndarray,
    sd_mat: np.ndarray,
    design: Design,
) -> pd.DataFrame:
    """Per feature and cell answer.

    Port of ``sa_fact_truth_cell()``.
    """
    n_cells = design.n_cells
    n_feats = len(feats)

    columns: dict[str, Any] = {"features": np.repeat(feats, n_cells)}
    for name, levels in design.factor_lv.items():
        labels = [levels[at] for at in np.asarray(design.cells[name], dtype=np.int64)]
        columns[name] = np.tile(labels, n_feats)
    columns["is_ref"] = np.tile(np.arange(n_cells) == design.ref_cell, n_feats)
    columns["delta"] = delta.ravel()
    columns["center"] = center.ravel()
    columns["sd"] = sd_mat.ravel()
    columns["n"] = np.tile(design.cell_n, n_feats)
    return pd.DataFrame(columns)


def _truth_contrast(feats: list[str], delta: np.ndarray, design: Design) -> pd.DataFrame:
    """Per feature and pair answer, in a post-hoc table's own direction.

    Port of ``sa_fact_truth_contrast()``. The rows and the cells each of them
    averages come from :func:`~statassist.core.fact_contrast_skeleton`, which the
    factorial comparison reads too, so the row order and the ``group1 - group2``
    direction cannot drift apart from the tables this is meant to score.
    """
    skeleton = fact_contrast_skeleton(design.factor_lv, design.cells)
    table = skeleton.table
    n_rows = len(table.index)

    values = np.zeros((len(feats), n_rows))
    for k in range(n_rows):
        values[:, k] = delta[:, skeleton.sel1[k]].mean(axis=1) - delta[:, skeleton.sel2[k]].mean(
            axis=1
        )
    # Averaging a set of cells divides, so a contrast that is zero by construction
    # can come back a rounding away from it. `is_diff` has to be exact.
    values[np.abs(values) < FACT_TOL] = 0.0
    flat = values.ravel()

    repeated = table.iloc[np.tile(np.arange(n_rows), len(feats))].reset_index(drop=True)
    return pd.concat(
        [
            pd.DataFrame({"features": np.repeat(feats, n_rows)}),
            repeated,
            pd.DataFrame({"delta": flat, "is_diff": flat != 0}),
        ],
        axis=1,
    )


def simulate_factorial_groups(
    n_feats: int = 100,
    factor_lv: Mapping[str, Sequence[str]] | None = None,
    within: str | Sequence[str] | None = None,
    n_per_cell: Any = 20,
    n_up: int | None = None,
    n_down: int | None = None,
    term_mix: Mapping[str, float] | None = None,
    pattern_mix: Mapping[str, float] | None = None,
    expr_range: tuple[float, float] = (2, 12),
    ref_sd: tuple[float, float] = (1.2, 2.4),
    cell_sd: tuple[float, float] = (1.8, 3.2),
    deg_log2fc: tuple[float, float] = (1, 2.5),
    interaction_scale: float = 0.8,
    subject_sd: tuple[float, float] = (2, 4),
    feat_prefix: str = "prot",
    seed: int | None = None,
) -> SaSimulation:
    """Simulate a crossed-factor experiment whose answer is known.

    The factorial counterpart of :func:`simulate_multiple_groups`. Crosses any
    number of factors, lets each one be measured between subjects or within them,
    and returns the planted answer alongside the data so that a two-way or an
    n-way analysis can be scored against what was actually put there.

    One factor asks one question: are the levels alike. Crossing a second one asks
    three, and they fail separately and for different reasons. Each factor has a
    main effect, the pair has an interaction, and a design that is read as though
    the second factor were not there answers none of them. So the effect is
    planted in five shapes rather than one, chosen to make the three questions
    come apart: a shape whose main effects are real and whose interaction is not,
    and a shape whose interaction is real and whose main effects are exactly zero,
    are both here, and no single test tells them apart.

    Args:
        n_feats: Number of features to generate. Columns are named ``prot_1``
            upwards, or whatever ``feat_prefix`` asks for.
        factor_lv: Mapping of factor name to the levels of that factor, the
            reference level first. Its length is how many factors are crossed, so
            there is no separate argument for that, and there have to be at least
            two: one factor is :func:`simulate_multiple_groups`. ``None`` gives a
            four-treatment design crossed with sex.

            The **first factor is the primary one**, the treatment the experiment
            is about, and the ones after it are the other factors the effect may
            or may not depend on. The shapes below are written in those terms. The
            cell in which every factor sits at its reference level is the
            reference cell, and it is what the planted effects are measured
            against.
        within: Names of the factors measured within subjects, or ``None`` for a
            design that is entirely between them. A subject belongs to one
            combination of the between factors and is measured under every
            combination of the within ones, so naming no factor gives a factorial
            ANOVA, naming all of them gives a fully repeated design, and naming
            some of them gives a mixed one. When any factor is named, ``args``
            gains ``id`` and ``within``.
        n_per_cell: Observations per cell. When there are within factors this is
            also the number of subjects per combination of the between factors,
            because each of those subjects contributes exactly one observation to
            each cell it is measured in. One number spreads over every cell; a
            sequence carries one size per combination of the between factors,
            which makes its length how many of those there are. The within factors
            are fully crossed with every subject, so they cannot hold different
            sizes.
        n_up: How many features are moved up.
        n_down: How many are moved down. Their sum cannot exceed ``n_feats``, and
            every other feature is left with a true effect of exactly zero in
            every cell and every term. ``None`` takes a fraction of ``n_feats``
            rather than a fixed count, so that asking for fewer features plants
            fewer effects instead of failing.
        term_mix: Relative weights over the five shapes of effect described under
            "The five shapes" below, which decide **which terms** of the model an
            effect is planted in. ``None`` weighs them equally. Set a weight to
            zero to leave that shape out. The planted features are split between
            the shapes by the largest remainder method rather than drawn at
            random, so the counts are exactly what the weights ask for and do not
            move with the seed.
        pattern_mix: Relative weights over ``"all"``, ``"gradient"`` and
            ``"single"``, which decide **how a factor spreads its main effect**
            over its non-reference levels, exactly as in
            :func:`simulate_multiple_groups`. The two mixes are on different axes
            and are crossed at random: ``term_mix`` says which terms move and
            ``pattern_mix`` says what the profile along a factor looks like. Split
            by largest remainder too, so both sets of counts are a function of the
            arguments alone.
        expr_range: Range the baseline log2 abundance of each feature is drawn
            from. Every cell shares the baseline, which is what makes an unplanted
            feature null in every term at once.
        ref_sd: Range the per-feature standard deviation of the reference cell is
            drawn from.
        cell_sd: The same for every other cell. Every cell draws its own, so the
            design is heteroscedastic and unbalanced variance is something the
            analysis has to survive rather than something it is spared. Pass the
            same range twice for equal variances.

            Keeping them apart costs something that is worth knowing about. An
            interaction test on a ``"main_only"`` feature, whose interaction is
            exactly zero, is called about eight times in a hundred rather than
            five, because the cells whose means differ are also the cells whose
            spread differs. Passing the same range twice brings it back to about
            three. The anticonservatism is the test's rather than the simulation's,
            and finding it is the sort of thing a simulation with a known answer is
            for.
        deg_log2fc: Range the magnitude of the planted effect is drawn from, on the
            log2 scale. One magnitude is drawn per planted feature and the shape
            decides how it is spread over the terms.
        interaction_scale: Size of the interaction relative to the main effect
            under the ``"interaction"`` shape, as a fraction of the magnitude drawn
            from ``deg_log2fc``. It has no bearing on ``"crossover"``, where the
            interaction carries the whole effect because there is nothing else to
            carry it. Must be above zero: a scale of zero would leave a feature
            whose ``pattern`` says ``"interaction"`` with no interaction in it.

            The default puts the ``"interaction"`` shape between the other two
            things an interaction row can be, so the row is neither trivially
            recovered nor indistinguishable from noise. Raising it to 1 takes the
            shape to the rate ``"crossover"`` reaches and the two stop differing.
        subject_sd: Range the per-feature subject standard deviation is drawn
            from. A subject's offset is drawn once per feature and reused across
            every condition it is measured under, which is what a within-subject
            test exists to remove. Ignored when ``within`` is ``None``.
        feat_prefix: Prefix for the generated feature names. ``"prot"`` gives
            ``prot_1``, ``prot_2`` and so on.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy.

    Returns:
        A :class:`~statassist.core.SaSimulation` of five slots.

        * ``args`` - ``data``, ``feats``, ``factors``, ``factor_lv`` and
          ``input_scale``, and under a within design also ``within`` and ``id``.
          ``factors`` is a mapping holding one list per factor, each as long as
          ``data`` has rows, and ``factor_lv`` is the level order of each, exactly
          as it was passed in. These are the names the factorial comparison will
          take. When the design is wholly between subjects the two within keys are
          **absent** rather than present and empty.
        * ``truth`` - one row per feature, aligned with ``feats``, holding
          ``features``, ``pattern``, ``spread``, ``direction``, ``partner``,
          ``extreme_cell``, ``extreme_tied``, ``log2fc``, ``baseline`` and
          ``sd_subject``.
        * ``truth_term`` - one row per feature and model term, every main effect
          and every interaction of every order, holding ``features``, ``terms``,
          ``term_order``, ``is_within``, ``max_abs_delta`` and ``is_effect``. This
          is the table that scores an ANOVA table row by row, and the one that has
          no counterpart in :func:`simulate_multiple_groups`.
        * ``truth_cell`` - one row per feature and cell, holding ``features``, one
          column per factor, then ``is_ref``, ``delta``, ``center``, ``sd`` and
          ``n``. A feature the analysis missed can be looked up here rather than
          guessed at: a large ``sd`` explains a miss that the effect size alone
          does not.
        * ``truth_contrast`` - one row per feature and pair of levels, in the row
          order and direction a post-hoc table uses, holding ``features``,
          ``factor``, ``stratum``, ``contrast``, ``group1``, ``group2``, ``delta``
          and ``is_diff``. A ``stratum`` of ``None`` is the marginal contrast,
          averaged over the other factors; anything else names the combination of
          the other factors the contrast was taken inside, which is the simple
          effect.

    The five shapes:
        Each planted feature is given a magnitude ``d`` drawn from ``deg_log2fc``,
        positive for an up feature and negative for a down one, a shape from
        ``term_mix``, and for every shape but the first a partner factor drawn at
        random from the factors after the primary one. The shape decides which
        terms of the model end up carrying ``d``.

        * ``"main_only"`` - the primary factor moves and nothing else does. Every
          other main effect and every interaction is exactly zero. This is
          :func:`simulate_multiple_groups` inside a factorial frame, and the case
          a two-way analysis should answer with one row.
        * ``"additive"`` - the primary factor and the partner each move, and their
          interaction is exactly zero. The cell means are the sum of the two, so
          the profiles are parallel, and an interaction reported as significant
          here is a false positive by construction.
        * ``"interaction"`` - the primary factor moves and the size of its effect
          depends on the level of the partner. The primary main effect and the
          interaction are both real; the partner's own main effect is left at
          exactly zero, so the two terms that should be called are the only two
          there are.
        * ``"crossover"`` - pure interaction. The primary factor rises at the
          partner's reference level and falls at the others, by amounts arranged
          so that **both main effects are exactly zero** while the cells plainly
          differ. This is the shape a main-effect test has to miss and an
          interaction test has to catch, which is the reason a factorial design is
          analysed as one.
        * ``"nuisance_only"`` - the partner moves and the primary factor is exactly
          zero. Read as one factor, the primary factor looks null with inflated
          within-group spread; read as a factorial design, the spread is accounted
          for and belongs to a term of its own.

        A feature that was not planted has a delta of exactly zero in every cell
        and a component of exactly zero in every term. Both kinds of mistake are
        therefore defined for every row of ``truth_term``: a term called
        significant on a zero component is a false positive, and a non-zero
        component that was not called is a miss.

    How the effect is planted, and how it is reported:
        The effect is built in the space an ANOVA decomposes into: a main effect is
        a profile along its own factor that sums to zero, and an interaction sums
        to zero along each of its factors. The components are added up and the
        value at the reference cell is then subtracted from the whole array, which
        leaves the reference cell at exactly zero delta without touching any term,
        since a constant belongs to the grand mean alone.

        That is why the two tables read differently and both are right.
        ``truth_cell["delta"]`` is the shift of a cell from the reference cell, and
        for a ``"main_only"`` feature the cell at level ``j`` of the primary factor
        carries exactly what the ``pattern_mix`` profile put there.
        ``truth_term["max_abs_delta"]`` is the largest component of the ANOVA
        effect itself, which is the quantity that is exactly zero for a term that
        was not planted. Components smaller than
        :data:`~statassist.core.FACT_TOL` in absolute value are recorded as
        exactly zero: they are the rounding left over from averaging, and a term
        left holding ``3e-17`` would score every row of an ANOVA table against the
        wrong answer.

    Directions:
        ``direction`` is the sign of ``d``, and it is the sign of the primary
        factor's effect at the reference level of the partner. ``truth["log2fc"]``
        is the delta of whichever cell sits furthest from the reference cell, so
        for every shape but ``"crossover"`` an up feature is positive there. Under
        ``"crossover"`` the primary factor moves in opposite directions at
        different levels of the partner, so which cell is furthest, and its sign,
        follow from the shape rather than from ``direction``.

        ``truth_contrast["delta"]`` is ``group1 - group2`` with ``group1`` the
        later level of the factor, which is the direction and the row order a
        post-hoc table uses. It comes from
        :func:`~statassist.core.level_pairs`, the same helper the post-hoc tables
        are built from, so the two cannot drift apart.

        ``extreme_cell`` is the levels of that cell joined by a dot, so it reads
        back against ``truth_cell`` without a lookup. When more than one cell is
        equally far from the reference, which is what the ``"all"`` profile does on
        purpose, it records the first of them and ``extreme_tied`` is ``True``: the
        flag that says to score the magnitude rather than the name of the cell. It
        is ``True`` with ``extreme_cell`` missing for an unplanted feature, whose
        cells are all zero and none of which is furthest.

    Within and between:
        A subject belongs to one combination of the between factors and is
        measured under every combination of the within ones, so no subject is
        dropped and the within-subject rectangle is complete. Each subject gets an
        offset per feature, drawn once and added to all of its rows, which is the
        between-subject variation a within-subject test removes. The residual
        standard deviation still differs from cell to cell, so sphericity does not
        hold and the corrections a repeated measures analysis reports have
        something to report.

    Reproducibility:
        The seed reproduces this function within Python and does not reproduce R's
        numbers, for the reason :class:`~statassist.core.SaRandom` gives. What is
        promised instead of R's stream is the **order** draws are taken in, since
        that is what decides whether two calls differing in one argument share a
        prefix:

        1. ``baseline``, one per feature.
        2. The reference cell's standard deviation, one per feature.
        3. Each other cell's standard deviation, one per feature, cells in
           ascending order.
        4. ``sd_subject``, one per feature, only under a within design.
        5. Which features are planted, as one permutation.
        6. The up magnitudes, then the down magnitudes.
        7. The level profiles: one shuffle for the up set and one for the down set,
           and neither draws when the set holds fewer than two features.
        8. Per planted feature, in the order they were picked: the partner factor,
           then whatever the profile shape draws - which is one level for
           ``"single"`` and nothing for ``"all"`` or ``"gradient"``. A
           ``"main_only"`` feature draws no partner, so the stream depends on which
           shapes were handed out.
        9. The subject offsets, feature by feature, only under a within design.
        10. The observations, feature by feature.

        The two mixes never draw. How many features take each shape is a function
        of the weights alone, so it can be read off the arguments rather than
        looked up in the result.

    Raises:
        SaValueError: If ``factor_lv`` does not name at least two usable factors,
            if it names a column ``truth_cell`` uses, if ``within`` names a factor
            the design does not hold, if ``n_per_cell`` counts differently from the
            between-subject combinations, if ``n_up`` plus ``n_down`` asks for more
            features than there are, or if a range is unusable.

    Examples:
        A 4 x 2 design, both factors between subjects. The shapes are handed out in
        counts the weights fix rather than the seed.

        >>> sim = simulate_factorial_groups(n_feats=20, n_up=5, n_down=5, seed=1)
        >>> sim.args["factor_lv"]["treatment"]
        ['control', 'treat_A', 'treat_B', 'treat_C']
        >>> int((sim.truth["pattern"] == "crossover").sum())
        2

        A ``"crossover"`` feature has an interaction and no main effect at all,
        which is what makes the shape worth planting.

        >>> cross = sim.truth.loc[sim.truth["pattern"] == "crossover", "features"].iloc[0]
        >>> rows = sim.truth_term[sim.truth_term["features"] == cross]
        >>> list(rows["terms"])
        ['treatment', 'sex', 'treatment:sex']
        >>> list(rows.loc[rows["is_effect"], "terms"])
        ['treatment:sex']

        Time measured on the same subjects, treatment and sex between them: a mixed
        design, where ``args`` carries the two extra keys.

        >>> mixed = simulate_factorial_groups(
        ...     n_feats=10, n_up=2, n_down=2, n_per_cell=4,
        ...     factor_lv={"treatment": ["control", "treat_A"], "sex": ["male", "female"],
        ...                "time": ["T0", "T1", "T2"]},
        ...     within="time", seed=1,
        ... )
        >>> [key for key in mixed.args if key in ("within", "id")]
        ['within', 'id']
        >>> [key for key in sim.args if key in ("within", "id")]
        []

        Every subject appears under every time point, so the rectangle is complete.

        >>> sorted({mixed.args["id"].count(name) for name in set(mixed.args["id"])})
        [3]
    """
    design = _design(_DEFAULT_FACTOR_LV if factor_lv is None else factor_lv, within, n_per_cell)
    factor_lv_out = design.factor_lv
    fac_names = list(factor_lv_out)
    n_cells = design.n_cells
    has_within = len(design.within) > 0

    # `n_up` and `n_down` default to a fraction of `n_feats`, which R evaluates
    # lazily and so sees the checked value. Here they arrive as `None` and are
    # filled in below for the same reason: the fraction of an unchecked count is
    # not a count.
    n_feats = check_count(n_feats, "n_feats", 1)
    n_up = round(_PLANT_SHARE * n_feats) if n_up is None else n_up
    n_down = round(_PLANT_SHARE * n_feats) if n_down is None else n_down
    n_up = check_count(n_up, "n_up")
    n_down = check_count(n_down, "n_down")
    if n_up + n_down > n_feats:
        raise SaValueError(
            f"`n_up` + `n_down` is {n_up + n_down}, which is more features than the "
            f"{n_feats} that `n_feats` asks for."
        )
    shapes = check_pattern_mix(
        dict.fromkeys(FACT_SHAPES, 1.0) if term_mix is None else term_mix,
        FACT_SHAPES,
        "term_mix",
    )
    mix = check_pattern_mix(
        {"all": 1.0, "gradient": 1.0, "single": 1.0} if pattern_mix is None else pattern_mix
    )
    expr_lo, expr_hi = check_range(expr_range, "expr_range")
    ref_lo, ref_hi = check_range(ref_sd, "ref_sd", 0)
    cell_lo, cell_hi = check_range(cell_sd, "cell_sd", 0)
    deg_lo, deg_hi = check_range(deg_log2fc, "deg_log2fc", 0)
    interaction_scale = check_scalar_num(interaction_scale, "interaction_scale", 0, lower_open=True)
    subject_lo, subject_hi = check_range(subject_sd, "subject_sd", 0)
    if not isinstance(feat_prefix, str) or not feat_prefix:
        raise SaValueError("`feat_prefix` must be a single non-empty string.")

    rng = SaRandom(seed).rng

    feats = [f"{feat_prefix}_{i + 1}" for i in range(n_feats)]
    baseline = rng.uniform(expr_lo, expr_hi, n_feats)

    sd_mat = np.zeros((n_feats, n_cells))
    sd_mat[:, design.ref_cell] = rng.uniform(ref_lo, ref_hi, n_feats)
    for cell in range(n_cells):
        if cell != design.ref_cell:
            sd_mat[:, cell] = rng.uniform(cell_lo, cell_hi, n_feats)
    sd_subject = (
        rng.uniform(subject_lo, subject_hi, n_feats) if has_within else np.full(n_feats, np.nan)
    )

    delta = np.zeros((n_feats, n_cells))
    direction = np.full(n_feats, "none", dtype=object)
    pattern = np.full(n_feats, "none", dtype=object)
    spread = np.full(n_feats, "none", dtype=object)
    partner: list[str | None] = [None] * n_feats

    if n_up + n_down > 0:
        up_idx, down_idx = pick_up_down(n_feats, n_up, n_down, rng)
        direction[up_idx] = "up"
        direction[down_idx] = "down"

        plant_idx = np.concatenate([up_idx, down_idx])
        plant_mag = np.concatenate(
            [rng.uniform(deg_lo, deg_hi, n_up), -rng.uniform(deg_lo, deg_hi, n_down)]
        )
        # Each direction is split between the shapes on its own, so a mix holds
        # within the up set and within the down set rather than only in total.
        plant_shape = [
            name
            for count in (allocate(n_up, shapes), allocate(n_down, shapes))
            for name, times in count.items()
            for _ in range(times)
        ]
        # The two mixes are handed out in blocks over the same features, so without
        # a shuffle the term shape and the level profile would arrive in lockstep
        # and their crossing would never be covered. Shuffling the profiles leaves
        # both sets of counts exact and makes only the pairing random.
        plant_spread = [
            name
            for count in (allocate(n_up, mix), allocate(n_down, mix))
            for name in _shuffle([name for name, times in count.items() for _ in range(times)], rng)
        ]

        for k, i in enumerate(plant_idx):
            mate = _partner(plant_shape[k], fac_names, rng)
            effect = _plant(
                float(plant_mag[k]),
                plant_shape[k],
                plant_spread[k],
                mate,
                factor_lv_out,
                design.cells,
                interaction_scale,
                rng,
            )
            # The reference cell is put at exactly zero. A constant shift belongs
            # to the grand mean, so no term of the decomposition moves with it.
            delta[i] = effect - effect[design.ref_cell]
            pattern[i] = plant_shape[k]
            spread[i] = plant_spread[k]
            partner[int(i)] = mate

    center = baseline[:, None] + delta
    # One offset per subject and feature, drawn before the rows and added to all
    # of the ones that subject owns. Drawing it per row would make it noise rather
    # than a subject effect, and the within-subject tests would have nothing to
    # gain over the independent ones.
    # Drawn feature by feature, which is the order the docstring publishes and the
    # order R's `vapply()` over features takes. Filling a (units, features) array
    # instead would draw unit by unit and give a different stream for the same
    # seed.
    offsets = (
        rng.normal(0.0, sd_subject[:, None], size=(n_feats, design.n_units)).T
        if has_within
        else None
    )

    cell_idx = design.cell_idx
    values = rng.normal(center[:, cell_idx], sd_mat[:, cell_idx]).T
    if offsets is not None:
        values = values + offsets[design.subject_idx]

    args: dict[str, Any] = {
        "data": pd.DataFrame(values, columns=feats),
        "feats": feats,
        "factors": design.factors,
        "factor_lv": factor_lv_out,
    }
    if has_within:
        args["within"] = design.within
        args["id"] = design.subject
    args["input_scale"] = "log2"

    return SaSimulation(
        {
            "args": args,
            "truth": _truth(
                feats,
                delta,
                design,
                pattern,
                spread,
                direction,
                partner,
                baseline,
                sd_subject,
            ),
            "truth_term": _truth_term(feats, delta, design, pattern != "none"),
            "truth_cell": _truth_cell(feats, delta, center, sd_mat, design),
            "truth_contrast": _truth_contrast(feats, delta, design),
        }
    )
