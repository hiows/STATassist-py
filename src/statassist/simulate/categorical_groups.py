"""A contingency table whose association is known.

The port of ``R/simulate_categorical_groups.R``.

Two things this port settles differently from R, both of them the decision the
rest of the port already made.

The seed reproduces this package and not R
    ``seed=`` promises the same result from the same seed within Python. R's
    generators and NumPy's differ, so no draw here matches a draw there. What is
    promised instead is the **order** of consumption, written out under
    :func:`simulate_categorical_groups`.

An argument the caller did not pass is :data:`~statassist.core.validate.UNSET`
    R reads that off ``missing()``, which cannot survive a default being filled
    in. The sentinel says the same thing and survives it, and it is read before
    anything is validated for the same reason R records ``given`` before
    ``match.arg()``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.contingency import finite_or_na
from ..core.errors import SaValueError, warn
from ..core.random import SaRandom
from ..core.result import SaSimulation
from ..core.validate import UNSET, check_count, check_flag, check_num_vector, check_scalar_num

__all__ = ["CAT_PATTERNS", "simulate_categorical_groups"]

#: Which cells the planted association moves mass between.
#:
#: All three keep the margins fixed, so all three are the same kind of departure
#: at different places. Named in one place so the argument, its validation and
#: the dispatch in :func:`_perturb` cannot come to disagree about what exists.
CAT_PATTERNS: tuple[str, ...] = ("corner", "single", "gradient")

#: The two variables used when the caller names none.
_DEFAULT_CATEGORY_LV: dict[str, tuple[str, ...]] = {
    "cat_1": ("y", "n"),
    "cat_2": ("high", "mid", "low"),
}

#: The repeated conditions used when the caller names none and ``paired`` is set.
_DEFAULT_PAIRED_LV: dict[str, tuple[str, ...]] = {
    "before": ("fail", "pass"),
    "after": ("fail", "pass"),
}

#: How much association is planted when the caller names none.
_DEFAULT_ASSOC = 0.3

#: The transition probabilities used when the caller names none.
_DEFAULT_DISCORDANCE: tuple[float, float] = (0.25, 0.10)

#: What a matched simulation records under ``pattern``: not one of
#: :data:`CAT_PATTERNS`, because a transition is not a shape over cells.
_MATCHED_PATTERN = "transition"

#: How many conditions a matched design has before the paired measures stop
#: being defined and the sequence of response rates takes over.
_PAIRED_CONDITIONS = 2

#: The levels a matched design needs. McNemar's test and Cochran's Q are both
#: about a binary response.
_MATCHED_LEVELS = 2


def _check_given(given: Mapping[str, bool], paired: bool) -> None:
    """Warn about an argument belonging to the other design.

    Port of ``sa_sim_cat_check_given()``. Both lists are named rather than only
    the offending one. An argument passed to the wrong design is usually a call
    that meant the other design, so knowing what this one does read is the part
    that resolves it.
    """
    reads = ["discordance"] if paired else ["margins", "assoc", "pattern"]
    ignored = [name for name, was_given in given.items() if was_given and name not in reads]
    if not ignored:
        return

    warn(
        "a "
        + ("matched" if paired else "cross-classified")
        + " design reads "
        + ", ".join(reads)
        + ", so the value(s) given for "
        + ", ".join(ignored)
        + " were ignored. Set `paired = "
        + ("FALSE" if paired else "TRUE")
        + "` for the design those belong to."
    )


def _check_levels(category_lv: Any) -> dict[str, list[str]]:
    """Check the level sets a simulated table is built on.

    Port of ``sa_sim_cat_levels()``.
    """
    if (
        not isinstance(category_lv, Mapping)
        or len(category_lv) < 2
        or not all(isinstance(name, str) and name for name in category_lv)
    ):
        raise SaValueError(
            "`category_lv` must be a named mapping of at least two variables, each "
            "entry holding that variable's levels."
        )

    out: dict[str, list[str]] = {}
    for name, levels in category_lv.items():
        usable = isinstance(levels, Sequence) and not isinstance(levels, str)
        items = [str(level) for level in levels] if usable else []
        if not usable or len(items) < 2 or len(set(items)) != len(items):
            raise SaValueError(
                f"`category_lv['{name}']` must hold at least two distinct non-missing levels."
            )
        out[str(name)] = items
    return out


def _margins(margins: Any, category_lv: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    """Normalise the marginal probabilities of each variable.

    Port of ``sa_sim_cat_margins()``.
    """
    if margins is None:
        return {name: np.full(len(levels), 1 / len(levels)) for name, levels in category_lv.items()}
    if not isinstance(margins, Mapping) or set(margins) != set(category_lv):
        raise SaValueError(
            "`margins` must be a named mapping holding one entry per variable of "
            "`category_lv`: " + ", ".join(category_lv) + "."
        )

    out: dict[str, np.ndarray] = {}
    for name, levels in category_lv.items():
        weights = check_num_vector(margins[name], f"margins['{name}']", 0)
        if weights.size != len(levels):
            raise SaValueError(
                f"`margins['{name}']` must hold one weight per level of "
                f"`category_lv['{name}']`: got {weights.size} for {len(levels)} level(s)."
            )
        if weights.sum() <= 0 or bool(np.any(weights == 0)):
            raise SaValueError(
                f"`margins['{name}']` must give every level a positive weight; a level "
                "with none is a level to leave out of `category_lv`."
            )
        out[name] = weights / weights.sum()
    return out


def _zero_sum(k: int, pattern: str) -> np.ndarray:
    """One factor of the perturbation: a vector over ``k`` levels summing to zero."""
    if pattern == "corner":
        return np.array([1.0, -1.0] + [0.0] * (k - 2))
    if pattern == "single":
        return np.array([1.0] + [-1 / (k - 1)] * (k - 1))
    ramp = np.linspace(1.0, -1.0, k)
    return np.asarray(ramp - ramp.mean(), dtype=float)


def _perturb(independent: np.ndarray, assoc: float, pattern: str) -> np.ndarray:
    """Move mass between the cells without moving the margins.

    Port of ``sa_sim_cat_perturb()``. The perturbation is the outer product of two
    vectors that each sum to zero, which is what makes every row and column sum of
    it zero as well: the row sums are one vector scaled by the total of the other,
    and that total is zero. One rank is enough for every pattern here, and it is
    what keeps the step a single number to scale.
    """
    if assoc == 0:
        return np.array(independent, dtype=float, copy=True)

    step = np.outer(
        _zero_sum(independent.shape[0], pattern), _zero_sum(independent.shape[1], pattern)
    )

    # The largest step this shape can take is the one that first empties a cell.
    # Scaling by it is what gives `assoc` the same meaning whatever the margins
    # are, rather than a meaning that has to be found by trial for each table.
    losing = step < 0
    max_step = float((independent[losing] / -step[losing]).min())

    out = independent + assoc * max_step * step
    # A cell that reaches exactly zero at `assoc = 1` can land a hair below it
    # through floating point, and a negative probability is not something to draw
    # from.
    out[out < 0] = 0
    return np.asarray(out, dtype=float)


def _truth(
    planted: np.ndarray,
    independent: np.ndarray,
    n_samples: int,
    assoc: float,
    pattern: str,
) -> pd.DataFrame:
    """The population value of every measure a cross-classification defines.

    Port of ``sa_sim_cat_truth()``. Computed from the planted distribution rather
    than from the drawn table, so it carries no sampling error and is the number
    an estimate should approach.

    The mean square contingency ``sum((p - p_ind)^2 / p_ind)`` is what a
    chi-square statistic divided by ``n`` estimates, so Cramer's V follows from it
    without ``n`` entering anywhere. The contingency coefficient does not, since
    its denominator holds ``n`` itself, so it is not reported here.
    """
    phi_sq = float(((planted - independent) ** 2 / independent).sum())
    min_df = min(planted.shape) - 1

    out = {
        "n_samples": n_samples,
        "pattern": pattern,
        "assoc": assoc,
        "cramers_v": float(np.sqrt(phi_sq / min_df)),
    }

    # A 2 x 2 table is the only one on which a signed association is defined, so
    # the two signed measures are columns that are absent elsewhere rather than
    # columns of missing values.
    if planted.shape == (2, 2):
        cross = planted[0, 0] * planted[1, 1] - planted[0, 1] * planted[1, 0]
        margins = np.concatenate([planted.sum(axis=1), planted.sum(axis=0)])
        out["phi_coefficient"] = float(cross / np.sqrt(margins.prod()))
        # A structural zero at `assoc = 1` makes the ratio infinite, which is a
        # ratio that does not exist rather than one that is very large.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = planted[0, 0] * planted[1, 1] / (planted[0, 1] * planted[1, 0])
        out["odds_ratio"] = float(finite_or_na([ratio])[0])

    return pd.DataFrame([out])


def _truth_cell(
    planted: np.ndarray,
    independent: np.ndarray,
    total: float,
    row_lv: Sequence[str],
    col_lv: Sequence[str],
) -> pd.DataFrame:
    """The planted distribution, cell by cell.

    Port of ``sa_sim_cat_truth_cell()``. Keyed on ``row_level`` and ``col_level``,
    the columns the ``cells`` table of a comparison carries, so the two merge with
    neither side renamed - and laid out down the columns, which is the row order
    :func:`~statassist.core.categorical_cells` produces.

    Args:
        planted: The joint distribution the data was drawn from.
        independent: The product of its margins, which is what a test of
            independence holds it against.
        total: How many observations the table counts, which is the number of rows
            for a cross-classification and the number of subjects times the number
            of conditions for a repeated one. ``p_planted`` is a share of the whole
            table either way, so it is the only thing that turns it back into a
            count.
        row_lv: Row labels.
        col_lv: Column labels.
    """

    def down(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1, order="F")

    with np.errstate(divide="ignore", invalid="ignore"):
        lift = finite_or_na(planted / independent)

    return pd.DataFrame(
        {
            "row_level": list(row_lv) * len(col_lv),
            "col_level": [level for level in col_lv for _ in row_lv],
            "p_independent": down(independent),
            "p_planted": down(planted),
            "lift": down(lift),
            "expected_n": total * down(planted),
        }
    )


def _crossed(
    n_samples: int,
    category_lv: Mapping[str, list[str]],
    margins: Any,
    assoc: float,
    pattern: str,
    rng: np.random.Generator,
) -> SaSimulation:
    """Draw a cross-classification with a planted, margin-preserving association.

    Port of ``sa_sim_cat_crossed()``.
    """
    if len(category_lv) != 2:
        raise SaValueError(
            "a cross-classified simulation plants an association between exactly two "
            f"variables, and `category_lv` names {len(category_lv)}. Set "
            "`paired = TRUE` for repeated measurements of one thing."
        )
    variables = list(category_lv)
    row_lv = category_lv[variables[0]]
    col_lv = category_lv[variables[1]]

    probs = _margins(margins, category_lv)
    independent = np.outer(probs[variables[0]], probs[variables[1]])
    planted = _perturb(independent, assoc, pattern)

    # Drawing from the flattened joint keeps the two variables in step, which
    # drawing each in turn could not: an association is a statement about the
    # pair, so the pair is what is drawn. R normalises the weights itself and
    # NumPy requires them to sum to one, and at `assoc = 1` a clipped cell can
    # leave them a hair short, so they are normalised here.
    flat = planted.reshape(-1, order="F")
    drawn = rng.choice(flat.size, size=n_samples, replace=True, p=flat / flat.sum())
    row_at = drawn % len(row_lv)
    col_at = drawn // len(row_lv)

    data = pd.DataFrame(
        {
            variables[0]: [row_lv[at] for at in row_at],
            variables[1]: [col_lv[at] for at in col_at],
        }
    )
    _check_drawn(data, category_lv)

    return SaSimulation(
        {
            "args": {
                "data": data,
                "category_lv": {name: list(levels) for name, levels in category_lv.items()},
                "paired": False,
            },
            "truth": _truth(planted, independent, n_samples, assoc, pattern),
            "truth_cell": _truth_cell(planted, independent, n_samples, row_lv, col_lv),
        }
    )


def _check_drawn(data: pd.DataFrame, category_lv: Mapping[str, list[str]]) -> None:
    """Whether every named level was actually drawn.

    Port of ``sa_sim_cat_check_drawn()``. A level the draw happened to miss leaves
    an empty row or column, which a categorical comparison refuses rather than
    test, so the simulator says which one it was here instead of the analysis
    failing on data it was handed.
    """
    absent = []
    for name, levels in category_lv.items():
        seen = set(data[name])
        missed = [level for level in levels if level not in seen]
        if missed:
            absent.append(f"{name}: " + ", ".join(missed))
    if not absent:
        return

    warn(
        "no row was drawn at level(s) "
        + "; ".join(absent)
        + ", which leaves an empty row or column that no test of independence can "
        "be run on. Raise `n_samples`, or give the level more weight in `margins`."
    )


def _matched(
    n_samples: int,
    category_lv: Mapping[str, list[str]],
    discordance: Any,
    rng: np.random.Generator,
) -> SaSimulation:
    """Draw repeated binary measurements with a planted transition.

    Port of ``sa_sim_cat_matched()``. Every subject starts at either level with
    equal probability, which is what makes the two discordant cells half of their
    transition probabilities and the planted paired odds ratio their ratio. The
    chain then runs across the conditions, so the response rate of condition
    ``j + 1`` follows from that of condition ``j`` and nothing else, and the whole
    sequence of rates is known before a single subject is drawn.
    """
    moves = np.asarray(discordance, dtype=object).reshape(-1)
    if moves.size != _MATCHED_LEVELS:
        raise SaValueError(
            "`discordance` must be two transition probabilities, from the first "
            "level to the second and back."
        )
    checked = check_num_vector(discordance, "discordance", 0, 1)

    variables = list(category_lv)
    levels = category_lv[variables[0]]
    if any(category_lv[name] != levels for name in variables):
        raise SaValueError(
            "a matched simulation measures one thing repeatedly, so every entry of "
            "`category_lv` holds the same levels in the same order. They differ."
        )
    if len(levels) != _MATCHED_LEVELS:
        raise SaValueError(
            f"a matched simulation needs binary conditions, and `category_lv` holds "
            f"{len(levels)} level(s). McNemar's test and Cochran's Q are both about "
            "a binary response."
        )

    k = len(variables)
    move_up, move_down = float(checked[0]), float(checked[1])

    # The response, here and in the comparison, is the second level: the first is
    # the reference every other scenario reads against.
    state = np.zeros((n_samples, k), dtype=bool)
    state[:, 0] = rng.random(n_samples) < 0.5
    for j in range(k - 1):
        threshold = np.where(state[:, j], move_down, move_up)
        state[:, j + 1] = state[:, j] ^ (rng.random(n_samples) < threshold)

    rates = np.empty(k)
    rates[0] = 0.5
    for j in range(k - 1):
        rates[j + 1] = rates[j] * (1 - move_down) + (1 - rates[j]) * move_up

    data = pd.DataFrame(
        {name: [levels[int(at)] for at in state[:, j]] for j, name in enumerate(variables)}
    )

    return SaSimulation(
        {
            "args": {
                "data": data,
                "category_lv": {name: list(levels) for name in variables},
                "paired": True,
            },
            "truth": _truth_matched(n_samples, k, move_up, move_down, rates),
            "truth_cell": _truth_cell_matched(
                n_samples, k, variables, levels, move_up, move_down, rates
            ),
        }
    )


def _truth_matched(
    n_samples: int, k: int, move_up: float, move_down: float, rates: np.ndarray
) -> pd.DataFrame:
    """What a matched simulation planted.

    Port of ``sa_sim_cat_truth_matched()``.
    """
    b = 0.5 * move_up
    c = 0.5 * move_down

    out: dict[str, Any] = {
        "n_samples": n_samples,
        "pattern": _MATCHED_PATTERN,
        "n_conditions": k,
        "move_up": move_up,
        "move_down": move_down,
    }

    if k == _PAIRED_CONDITIONS:
        # A transition of zero leaves the ratio undefined rather than infinite,
        # which is what R's `sa_finite_or_na()` says about the same division.
        out["odds_ratio_paired"] = b / c if c > 0 else float("nan")
        out["risk_difference_paired"] = b - c
        out["cohens_g"] = b / (b + c) - 0.5 if b + c > 0 else float("nan")
    else:
        # Three or more conditions have no single pair to be about, so what was
        # planted is the climb in the response rate, which is what Q is testing.
        out["rate_first"] = float(rates[0])
        out["rate_last"] = float(rates[-1])
        out["rate_range"] = float(rates.max() - rates.min())

    return pd.DataFrame([out])


def _truth_cell_matched(
    n_samples: int,
    k: int,
    variables: Sequence[str],
    levels: Sequence[str],
    move_up: float,
    move_down: float,
    rates: np.ndarray,
) -> pd.DataFrame:
    """The planted distribution of a matched design, cell by cell.

    Port of ``sa_sim_cat_truth_cell_matched()``. Over two conditions this is the
    paired square table McNemar's test reads, and the null it is read against is
    symmetry, so ``p_symmetric`` is added: it is the average of each cell and its
    transpose, which is the share the ``expected`` column holds there. The
    diagonal is symmetric by construction, so its ``p_symmetric`` equals its
    ``p_planted`` and nothing was planted there to find.

    Over three or more conditions there is no such table, so it is the
    condition-by-response table Cochran's Q is about, which is the one the
    comparison carries as well. Marginal homogeneity and independence are the same
    arithmetic on that table, so ``p_independent`` already is the null and no
    column is added.
    """
    if k == _PAIRED_CONDITIONS:
        planted = np.array(
            [
                [0.5 * (1 - move_up), 0.5 * move_up],
                [0.5 * move_down, 0.5 * (1 - move_down)],
            ]
        )
        independent = np.outer(planted.sum(axis=1), planted.sum(axis=0))

        out = _truth_cell(planted, independent, n_samples, levels, levels)
        symmetric = (planted + planted.T) / 2
        out["p_symmetric"] = symmetric.reshape(-1, order="F")
        out["expected_symmetry_n"] = n_samples * symmetric.reshape(-1, order="F")
        return out

    planted = np.column_stack([1 - rates, rates]) / k
    independent = np.outer(planted.sum(axis=1), planted.sum(axis=0))
    # Every subject is measured under every condition, so the table this counts is
    # subjects times conditions rather than subjects.
    return _truth_cell(planted, independent, n_samples * k, variables, levels)


def simulate_categorical_groups(
    n_samples: int = 200,
    category_lv: Mapping[str, Sequence[str]] | None = None,
    margins: Mapping[str, Sequence[float]] | None = None,
    assoc: Any = UNSET,
    pattern: Any = UNSET,
    paired: bool = False,
    discordance: Any = UNSET,
    seed: int | None = None,
) -> SaSimulation:
    """Simulate a contingency table whose association is known.

    Generates two or more categorical variables with a chosen amount of
    association planted between them, and returns the planted answer alongside the
    data. Every quantity a categorical comparison estimates can then be checked
    against what was actually put there, which is what a real data set can never
    offer.

    The point of the exercise is the gap. A table drawn from a distribution is not
    that distribution: the observed Cramer's V carries a sampling error of its
    own, and it is biased upwards, because every departure from independence
    counts towards it whether it was planted or drawn. So a table simulated at
    ``assoc=0`` does not come back with an estimated association of zero, and
    reading that number as the answer rather than the p-value beside it is the
    mistake the simulator exists to make visible.

    Args:
        n_samples: Number of observations, meaning rows for a cross-classified
            design and subjects for a matched one.
        category_lv: Mapping giving the levels of each variable. Its names are the
            column names of the generated data and its order fixes the reference
            level of each variable. A matched design needs one binary level set
            shared by every condition, and the default changes to reflect that
            when ``paired`` is set.
        margins: Mapping of the marginal probabilities of each variable, in the
            order its levels are given, or ``None`` for a uniform margin. Each
            entry is normalised to sum to one, so relative weights are enough.
        assoc: How much association to plant, from 0 for exact independence to 1
            for the largest margin-preserving departure the margins allow.
            Defaults to ``0.3``.
        pattern: Which cells the association moves mass between, one of
            :data:`CAT_PATTERNS`. Defaults to ``"corner"``.
        paired: Whether the columns are repeated binary measurements on the same
            subject rather than different variables.
        discordance: The two transition probabilities of a matched design, from
            the first level to the second and back. Their ratio is the planted
            paired odds ratio, and their being equal is the strict null. Defaults
            to ``(0.25, 0.10)``.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy.

    Returns:
        A :class:`~statassist.core.SaSimulation` of three slots.

        * ``args`` - ``data``, ``category_lv`` and ``paired``, named after the
          arguments of the categorical comparison so that the analysis this data
          was made for runs on them unchanged.
        * ``truth`` - one row summarising what was planted: ``n_samples``,
          ``pattern``, and the population value of every association measure this
          design defines. These are the values the estimates should approach as
          ``n_samples`` grows, and they carry no sampling error.
        * ``truth_cell`` - one row per cell: ``row_level``, ``col_level``,
          ``p_independent``, ``p_planted``, ``lift`` and ``expected_n``, in the row
          order :func:`~statassist.core.categorical_cells` produces. It merges with
          the ``cells`` table of the comparison on ``row_level`` and ``col_level``
          with neither side renamed, so a residual can be read against the
          departure that produced it.

        A measure the design does not define is a **column that is not there**
        rather than a column of missing values. A cross-classification adds
        ``assoc`` and ``cramers_v``, and ``phi_coefficient`` and ``odds_ratio``
        only on a 2 x 2 table. A matched design adds the transition probabilities,
        and then either the three paired measures over two conditions or the
        sequence of response rates over three or more. ``truth_cell`` adds
        ``p_symmetric`` and ``expected_symmetry_n`` over two matched conditions,
        which is what the comparison's ``expected`` holds there, and adds nothing
        over three or more, where marginal homogeneity and independence are the
        same arithmetic.

    Raises:
        SaValueError: If the arguments do not describe a table this can be drawn
            from.

    Warns:
        SaWarning: If an argument belonging to the other design was passed, or if
            a level was named that no row was drawn at.

    How the association is planted:
        The margins are fixed first, and then mass is moved between the cells in a
        way that leaves them exactly where they were. That is what ``assoc``
        scales: a perturbation matrix whose every row and column sums to zero,
        added to the independent joint distribution.

        Keeping the margins fixed is what makes the null hypothesis the only thing
        that moves. A test of independence compares the table to the product of
        its own observed margins, so a construction that shifted the margins too
        would plant an association and change what it is being measured against at
        the same time.

        ``assoc`` is the fraction of the largest step of that shape the table can
        take before a cell reaches zero, so it runs from 0 to 1 whatever the
        margins are. ``assoc=0`` is the product of the margins **exactly**, so it
        is null in the strict sense and is what a type I error rate can be
        measured on. ``assoc=1`` puts a structural zero in the table, which is the
        one setting where the exact test and the approximation part company
        sharply.

    The shape of the association:
        ``"corner"`` moves mass into the two cells where both variables sit at
        their first level or both at their second, and out of the two where they
        disagree. On a 2 x 2 table this is the whole of the association, and the
        odds ratio it plants is above 1.

        ``"single"`` makes one level of each variable special and the rest alike:
        mass moves into the cell where both first levels meet and is taken evenly
        from the others. This is the shape a test of homogeneity is usually
        looking for, one category behaving differently from the pack.

        ``"gradient"`` is a monotone association, planted by a centred linear ramp
        over the levels of each variable in the order ``category_lv`` gives them.
        This is the shape an ordered variable such as ``high`` / ``mid`` / ``low``
        actually takes, and the one a chi-square test is least efficient at
        finding, since it spends degrees of freedom on departures that are not
        there.

    A matched design:
        ``paired`` generates repeated binary measurements on the same subject
        rather than a cross-classification, so ``assoc``, ``pattern`` and
        ``margins`` do not apply and ``discordance`` takes over. Every subject
        starts at either level with equal probability, and then moves between
        consecutive conditions: a subject at the first level takes the second with
        probability ``discordance[0]``, and one at the second takes the first with
        probability ``discordance[1]``.

        Two things follow, and both are what the matched tests are about. Over two
        conditions the planted paired odds ratio is exactly
        ``discordance[0] / discordance[1]``, because the equal start makes the two
        discordant cells half of each transition probability. Over three or more,
        the response rate climbs from condition to condition whenever the first
        probability exceeds the second, and that climb is what Cochran's Q is
        testing for. Equal probabilities are the strict null of both: the
        transitions cancel, the rate stays at one half, and nothing is planted.

    Which arguments each design reads:
        A cross-classification reads ``margins``, ``assoc`` and ``pattern``. A
        matched design reads ``discordance``. Passing one design an argument
        belonging to the other is a warning naming both lists rather than a value
        silently ignored, because the two designs plant different things and a
        call that names the wrong knob is a call that expected the other design.

    Reproducibility:
        The seed reproduces this function within Python and does not reproduce R's
        numbers, for the reason :class:`~statassist.core.SaRandom` gives. What is
        promised instead of R's stream is the **order** draws are taken in:

        1. A cross-classification draws **once**, ``n_samples`` cells from the
           flattened joint distribution.
        2. A matched design draws the starting state, ``n_samples`` uniforms, and
           then one further ``n_samples`` uniforms per transition, which is
           ``k - 1`` draws for ``k`` conditions.

        Nothing else consumes the generator, so the planted answer is a function
        of the arguments alone and does not move with the seed.

    Examples:
        >>> sim = simulate_categorical_groups(n_samples=300, assoc=0.4, seed=1)
        >>> list(sim["args"])
        ['data', 'category_lv', 'paired']
        >>> float(sim["truth"]["cramers_v"].iloc[0]) > 0
        True

        ``assoc=0`` is the product of the margins exactly, so it is null in the
        strict sense: the lift of every cell is 1.

        >>> null = simulate_categorical_groups(n_samples=300, assoc=0, seed=2)
        >>> bool((null["truth_cell"]["lift"] == 1).all())
        True

        A matched design plants the paired odds ratio as a ratio of transitions.

        >>> matched = simulate_categorical_groups(
        ...     n_samples=200, paired=True, discordance=(0.3, 0.1), seed=4
        ... )
        >>> round(float(matched["truth"]["odds_ratio_paired"].iloc[0]), 6)
        3.0
    """
    # Recorded before anything is filled in, since a default assigned here would
    # make the sentinel say the argument was supplied whether it was or not.
    given = {
        "margins": margins is not None,
        "assoc": assoc is not UNSET,
        "pattern": pattern is not UNSET,
        "discordance": discordance is not UNSET,
    }

    resolved_pattern = CAT_PATTERNS[0] if pattern is UNSET else pattern
    if resolved_pattern not in CAT_PATTERNS:
        raise SaValueError("`pattern` must be one of: " + ", ".join(CAT_PATTERNS) + ".")
    paired = check_flag(paired, "paired")
    n_samples = check_count(n_samples, "n_samples", 2)
    resolved_assoc = check_scalar_num(_DEFAULT_ASSOC if assoc is UNSET else assoc, "assoc", 0, 1)

    if category_lv is None:
        category_lv = _DEFAULT_PAIRED_LV if paired else _DEFAULT_CATEGORY_LV
    levels = _check_levels(category_lv)

    _check_given(given, paired)

    rng = SaRandom(seed).rng
    if paired:
        return _matched(
            n_samples,
            levels,
            _DEFAULT_DISCORDANCE if discordance is UNSET else discordance,
            rng,
        )
    return _crossed(n_samples, levels, margins, resolved_assoc, resolved_pattern, rng)
