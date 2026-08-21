"""Fisher's exact test on a table larger than 2 x 2, by enumeration with bounds.

R reaches this through FEXACT, Mehta and Patel's network algorithm, which has no
counterpart in ``scipy``: :func:`scipy.stats.fisher_exact` is 2 x 2 only. Rather
than fall back on a Monte Carlo p-value where R gives an exact one, the network
algorithm's idea is implemented here, because the idea is short even though the
Fortran is not.

The p-value is the total probability of every table with the observed margins
whose own probability is no greater than the observed table's. Enumerating them
one by one is out of the question - a three-by-four table of a hundred
observations has tens of millions of them - so whole subtrees are resolved without
being walked.

Filling the table column by column, the state after ``j`` columns is nothing but
the vector of row sums still to be placed: two different partial tables with the
same remainder have exactly the same set of completions. So three quantities can
be tabulated per state and reused:

* the total probability weight of all completions,
* the largest weight any single completion reaches,
* the smallest.

At a partial table, the largest bound says whether *every* completion already
qualifies, in which case the whole subtree contributes its total weight at once;
the smallest says whether none of them can, in which case the subtree contributes
nothing. Only a state that straddles the threshold is walked further, and there
are few of those.

The three tables are built backwards from the last column, one dense array per
column, so that a state is an index rather than a dictionary key and a whole
column's worth of states is filled by array arithmetic. That is the difference
between this being usable and not: walking the same recursion state by state in
Python takes seconds on a table this handles in a tenth of one.

Everything is in log space. The weight of a table is ``prod(1 / x_ij!)``, which
underflows on a table of any size, and its logarithm is a sum of log-gammas that
does not.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np

__all__ = [
    "FISHER_CELL_LIMIT",
    "FISHER_REL_ERR",
    "FISHER_TABLE_LIMIT",
    "count_tables",
    "fisher_exact_rxc",
]

#: How far above the observed probability a table still counts as "no greater".
#:
#: R's ``fisher.test()`` compares probabilities with a relative slack of ``1e-7``,
#: because two tables that are equally probable by construction can come out of
#: the arithmetic a rounding apart, and dropping one of a symmetric pair would put
#: a visible step in the p-value.
FISHER_REL_ERR = 1e-7

#: How many states one column's table may hold before the exact test is given up.
#:
#: The state is the vector of margins still to be placed, so the count is the
#: product of the shorter margin plus one apiece: it grows with the *shape* of the
#: table rather than with its total, and a table this does not fit is a wide one
#: rather than a large one. Checked before anything is allocated, so a table past
#: the limit costs nothing to refuse.
FISHER_CELL_LIMIT = 4_000_000

#: How many tables with the observed margins there may be before giving up.
#:
#: The bounds resolve most of the tree without walking it, but "most" is a
#: fraction rather than a bound: the walk still visits on the order of a tenth of
#: the tables, and a tenth of forty million is not something to do in Python. The
#: number of tables is counted exactly first, by the same convolution that builds
#: the summary arrays and at a quarter of its cost, so a table past the limit is
#: refused in milliseconds rather than after minutes.
#:
#: **This is where the port stops short of R.** R's FEXACT is Fortran with tighter
#: bounds, and it returns an exact p-value for tables this refuses - a
#: three-by-four table of a hundred observations, for instance. Reaching the limit
#: is reported the same way R reports an exhausted workspace: the p-value comes
#: back missing, ``enumerated`` is 0, and the caller points at
#: ``simulate_p_value = True``. A table that large is one where the chi-square
#: approximation standing beside it in the same result is the one to read anyway,
#: which is what makes the limit a cost rather than a gap.
FISHER_TABLE_LIMIT = 1_000_000


@cache
def _allocations(total: int, capacity: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Every way to split ``total`` among rows, each within its remaining sum.

    Memoised on the pair, because the same split is asked for once per column of
    every table of a given shape and the answer never depends on anything else.
    """
    if len(capacity) == 1:
        return ((total,),) if total <= capacity[0] else ()

    out: list[tuple[int, ...]] = []
    rest = capacity[1:]
    room = sum(rest)
    for take in range(max(0, total - room), min(total, capacity[0]) + 1):
        out.extend((take, *tail) for tail in _allocations(total - take, rest))
    return tuple(out)


def _logsumexp(values: list[float]) -> float:
    """Sum in log space, which is where all the weights live."""
    if not values:
        return -math.inf
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(value - top) for value in values))


def _state_shape(row_sums: tuple[int, ...]) -> tuple[int, ...]:
    """Dimensions of the dense array a boundary's states are indexed by."""
    return tuple(value + 1 for value in row_sums)


def count_tables(row_sums: tuple[int, ...], col_sums: tuple[int, ...]) -> float:
    """How many tables hold these margins.

    The same convolution :func:`_summary_tables` runs, with every weight at one,
    which is what turns a sum of probabilities into a count. In floating point,
    because the count of a wide table overflows an integer long before it matters
    whether it is exact.
    """
    shape = _state_shape(row_sums)
    counts = np.zeros(shape)
    counts[(0,) * len(row_sums)] = 1.0
    for column in range(len(col_sums) - 1, -1, -1):
        onward = counts
        counts = np.zeros(shape)
        for split in _allocations(col_sums[column], row_sums):
            source = tuple(slice(0, size - take) for size, take in zip(shape, split, strict=True))
            target = tuple(slice(take, None) for take in split)
            counts[target] += onward[source]
    return float(counts[row_sums])


def _summary_tables(
    row_sums: tuple[int, ...],
    col_sums: tuple[int, ...],
    log_weight: list[float],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The three per-state summaries, one dense array per column boundary.

    Returns:
        ``len(col_sums) + 1`` triples of ``(log_total, log_best, log_worst)``,
        each indexed by the remaining row sums. The last is the boundary past the
        final column, where the only state with any completion is the empty one.
    """
    shape = _state_shape(row_sums)
    n_columns = len(col_sums)

    empty_total = np.full(shape, -math.inf)
    empty_best = np.full(shape, -math.inf)
    empty_worst = np.full(shape, math.inf)
    origin = (0,) * len(row_sums)
    empty_total[origin] = 0.0
    empty_best[origin] = 0.0
    empty_worst[origin] = 0.0

    tables = [(empty_total, empty_best, empty_worst)]
    for column in range(n_columns - 1, -1, -1):
        onward_total, onward_best, onward_worst = tables[0]
        splits = _allocations(col_sums[column], row_sums)

        best = np.full(shape, -math.inf)
        worst = np.full(shape, math.inf)
        # Two passes, because a sum in log space needs its own maximum first.
        moves = []
        for split in splits:
            source = tuple(slice(0, size - take) for size, take in zip(shape, split, strict=True))
            target = tuple(slice(take, None) for take in split)
            here = math.fsum([log_weight[take] for take in split])
            moves.append((source, target, here))
            np.maximum(best[target], onward_best[source] + here, out=best[target])
            np.minimum(worst[target], onward_worst[source] + here, out=worst[target])

        top = np.full(shape, -math.inf)
        for source, target, here in moves:
            np.maximum(top[target], onward_total[source] + here, out=top[target])
        accumulated = np.zeros(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            for source, target, here in moves:
                accumulated[target] += np.exp(onward_total[source] + here - top[target])
            total = np.where(np.isneginf(top), -math.inf, top + np.log(accumulated))

        tables.insert(0, (total, best, worst))
    return tables


def fisher_exact_rxc(counts: Any) -> float:
    """The exact p-value of Fisher's test on a table of any shape.

    Args:
        counts: Two-dimensional counts, at least 2 x 2.

    Returns:
        The total probability of every table with these margins whose probability
        is no greater than this one's. ``NaN`` if the table is past
        :data:`FISHER_CELL_LIMIT` or :data:`FISHER_TABLE_LIMIT`, which the caller
        reports as ``enumerated = 0``.
    """
    table = np.asarray(counts, dtype=np.int64)
    by_row = tuple(int(value) for value in table.sum(axis=1))
    by_col = tuple(int(value) for value in table.sum(axis=0))

    # Fill along whichever margin leaves fewer states to hold, which is not always
    # the longer one: a margin's states cost the product of its entries plus one,
    # so three large sums can be dearer than four small ones.
    if math.prod(size + 1 for size in by_col) < math.prod(size + 1 for size in by_row):
        table = table.T
        by_row, by_col = by_col, by_row
    if math.prod(size + 1 for size in by_row) > FISHER_CELL_LIMIT:
        return math.nan
    if count_tables(by_row, by_col) > FISHER_TABLE_LIMIT:
        return math.nan

    total = int(table.sum())
    log_weight = [-math.lgamma(k + 1) for k in range(total + 1)]
    # A table's own log weight, and the ceiling every other table is judged
    # against. `log1p` rather than `log`, since the slack is a rounding.
    observed = math.fsum([log_weight[int(value)] for value in table.reshape(-1)])
    ceiling = observed + math.log1p(FISHER_REL_ERR)

    tables = _summary_tables(by_row, by_col, log_weight)

    def walk(remaining: tuple[int, ...], column: int, accumulated: float) -> list[float]:
        log_total, log_best, log_worst = (float(part[remaining]) for part in tables[column])
        if log_total == -math.inf:
            return []
        if accumulated + log_best <= ceiling:
            # Even the most probable completion is no more probable than the
            # observed table, so the whole subtree is in and needs no walking.
            return [accumulated + log_total]
        if accumulated + log_worst > ceiling:
            return []

        out: list[float] = []
        for split in _allocations(by_col[column], remaining):
            here = math.fsum([log_weight[take] for take in split])
            out.extend(
                walk(
                    tuple(left - take for left, take in zip(remaining, split, strict=True)),
                    column + 1,
                    accumulated + here,
                )
            )
        return out

    qualifying = _logsumexp(walk(by_row, 0, 0.0))

    # The constant that turns a weight into a probability: the same for every
    # table with these margins, so it is applied once at the end.
    log_scale = (
        math.fsum([math.lgamma(value + 1) for value in by_row])
        + math.fsum([math.lgamma(value + 1) for value in by_col])
        - math.lgamma(total + 1)
    )
    return float(min(math.exp(log_scale + qualifying), 1.0))
