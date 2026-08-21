"""The contingency table kernels, against R's numbers."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from math import lgamma
from typing import Any

import numpy as np
import pandas as pd
import pytest
from golden import assert_close, assert_frame_close, load_case

from statassist.core.contingency import (
    expected_independence,
    expected_symmetry,
    finite_or_na,
)
from statassist.core.errors import SaInternalError, SaValueError
from statassist.core.random import SaRandom
from statassist.kernel._fisher import (
    FISHER_REL_ERR,
    FISHER_TABLE_LIMIT,
    count_tables,
    fisher_exact_rxc,
)
from statassist.kernel.categorical import (
    ASSOC_COLUMNS,
    MCNEMAR_EXACT_MAX_DISCORDANT,
    _resample_fixed_margins,
    assoc_measures,
    assoc_measures_paired,
    assoc_measures_repeated,
    assoc_row,
    chisq,
    cochran_q,
    fisher,
    has_zero_cell,
    mcnemar,
    odds_ratio,
    phi,
)

#: Where R's own root-finding stops, which is nearer than ``1e-8``.
#:
#: Every number in this file is graded at the tolerance of the phase except the
#: three ``fisher.test()`` reads off a 2 x 2 table: the interval bounds, which R
#: solves with ``uniroot(tol = .Machine$double.eps^0.25)``, and the conditional
#: odds ratio, which it maximises with ``optimize()`` at the same tolerance. That
#: is ``1.2e-4`` on the odds ratio scale, so R's answer is the approximate one of
#: the two and SciPy's brentq is the sharper. The largest divergence across the
#: frozen cases is ``5.6e-5``, and every other column of the same rows still has
#: to reach ``1e-8``.
FISHER_ROOT_RTOL = 1e-4

#: The three columns that tolerance is for.
_ROOTED = ("lower_conf", "upper_conf", "odds_ratio_cond")


def assert_fisher_close(actual: dict[str, float], expected: dict[str, Any]) -> None:
    """Grade the root-found columns of a Fisher row apart from the rest."""
    assert_close(
        {name: value for name, value in actual.items() if name not in _ROOTED},
        {name: value for name, value in expected.items() if name not in _ROOTED},
    )
    assert_close(
        {name: actual[name] for name in _ROOTED},
        {name: expected[name] for name in _ROOTED},
        rtol=FISHER_ROOT_RTOL,
    )


def fixed_margin_tables(row_sums: np.ndarray, col_sums: np.ndarray) -> Iterator[np.ndarray]:
    """Every table with these margins, by brute force.

    The definition the pruning kernel is graded against, and the enumeration any
    statistic's exact conditional p-value can be read off. Affordable only on a
    small table, which is the whole reason the kernel does not work this way.
    """
    span = [range(int(value) + 1) for value in np.tile(col_sums, (len(row_sums) - 1, 1)).ravel()]
    for head in product(*span):
        candidate = np.array(head, dtype=np.int64).reshape(len(row_sums) - 1, len(col_sums))
        last = col_sums - candidate.sum(axis=0)
        if np.any(last < 0) or not np.array_equal(candidate.sum(axis=1), row_sums[:-1]):
            continue
        yield np.vstack([candidate, last])


def log_table_weight(table: Any) -> float:
    """The log probability of a table, up to the constant its margins fix."""
    return -sum(lgamma(int(cell) + 1) for cell in np.asarray(table).reshape(-1))


def tables_from_long(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rebuild each named table from the one long input frame.

    The row and column order is the order of first appearance, which is the order
    ``expand.grid`` laid the frame out in and therefore the order R's own
    ``dimnames`` were in.
    """
    out: dict[str, np.ndarray] = {}
    for name, block in frame.groupby("table", sort=False):
        rows = list(dict.fromkeys(block["row_level"]))
        cols = list(dict.fromkeys(block["col_level"]))
        wide = block.pivot(index="row_level", columns="col_level", values="count")
        out[str(name)] = wide.loc[rows, cols].to_numpy(dtype=float)
    return out


@pytest.fixture(scope="module")
def cat_tables() -> dict[str, np.ndarray]:
    frame, _ = load_case("cat_chisq")
    return tables_from_long(frame)


# --------------------------------------------------------------------------- #
# core/contingency.py
# --------------------------------------------------------------------------- #


def test_the_expectation_builders_reproduce_r(cat_tables: dict[str, np.ndarray]) -> None:
    _, expected = load_case("cat_expected")

    # `jsonlite` writes a matrix as a list of its rows, so the nested shape is
    # part of what is compared: a transposed expectation would fail here.
    assert_close(expected_independence(cat_tables["t2x2"]).tolist(), expected["independence_2x2"])
    assert_close(expected_independence(cat_tables["t3x4"]).tolist(), expected["independence_3x4"])
    assert_close(expected_symmetry(cat_tables["pair_small"]).tolist(), expected["symmetry_small"])
    assert_close(
        expected_symmetry(cat_tables["pair_one_way"]).tolist(), expected["symmetry_one_way"]
    )
    assert_close(
        finite_or_na([1.5, np.inf, -np.inf, np.nan, np.nan, 0.0]).tolist(),
        expected["finite_or_na"],
    )


def test_symmetry_is_a_claim_about_a_square_table() -> None:
    with pytest.raises(SaInternalError, match="square table"):
        expected_symmetry(np.arange(6.0).reshape(2, 3))


def test_an_expectation_under_independence_keeps_the_margins(
    cat_tables: dict[str, np.ndarray],
) -> None:
    table = cat_tables["t3x4"]
    expected = expected_independence(table)
    assert_close(expected.sum(axis=0).tolist(), table.sum(axis=0).tolist())
    assert_close(expected.sum(axis=1).tolist(), table.sum(axis=1).tolist())


# --------------------------------------------------------------------------- #
# chisq
# --------------------------------------------------------------------------- #


def test_chisq_reproduces_r(cat_tables: dict[str, np.ndarray]) -> None:
    _, expected = load_case("cat_chisq")

    assert_close(chisq(cat_tables["t2x2"]), expected["t2x2_corrected"])
    assert_close(chisq(cat_tables["t2x2"], correct=False), expected["t2x2_plain"])
    assert_close(chisq(cat_tables["t2x4"]), expected["t2x4_corrected"])
    assert_close(chisq(cat_tables["t2x4"], correct=False), expected["t2x4_plain"])
    assert_close(chisq(cat_tables["t3x4"]), expected["t3x4"])
    assert_close(chisq(cat_tables["t3x3_small"]), expected["t3x3_small"])


def test_yates_is_a_two_by_two_rule_and_not_a_one_degree_rule(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """The one place ``scipy.stats.chi2_contingency`` would have disagreed.

    A 2 x 4 table has three degrees of freedom, so scipy leaves it alone and so
    does R. A 2 x 2 table is where they agree. The rule they key on differs, and a
    2 x N table is where that shows.
    """
    wide = cat_tables["t2x4"]
    assert chisq(wide)["statistic"] == chisq(wide, correct=False)["statistic"]

    small = cat_tables["t2x2"]
    assert chisq(small)["statistic"] < chisq(small, correct=False)["statistic"]


def test_a_table_with_an_empty_margin_has_no_chi_square() -> None:
    with pytest.raises(SaValueError, match="no observation"):
        chisq(np.array([[4.0, 0.0], [7.0, 0.0]]))
    with pytest.raises(SaValueError, match="no observation"):
        chisq(np.array([[4.0, 6.0], [0.0, 0.0]]))


def test_a_simulated_chisq_reports_no_degrees_of_freedom(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """The draws cannot match R's, so the contract is checked instead.

    The statistic is the uncorrected one whatever ``correct`` says, because a
    replicate is not corrected either and comparing the two would be comparing
    different quantities. And there are no degrees of freedom to report, since
    nothing was referred to a chi-square distribution.
    """
    table = cat_tables["t3x3_small"]
    row = chisq(table, simulate_p_value=True, n_resamples=19999, seed=11)

    assert np.isnan(row["df"])
    assert list(row) == list(chisq(table))
    assert row["statistic"] == pytest.approx(chisq(table, correct=False)["statistic"])
    assert 0 < row["pval"] <= 1

    # Same seed, same answer; a different one moves it only a little.
    assert row["pval"] == chisq(table, simulate_p_value=True, n_resamples=19999, seed=11)["pval"]
    other = chisq(table, simulate_p_value=True, n_resamples=19999, seed=12)["pval"]
    assert abs(other - row["pval"]) < 0.02


def test_a_simulated_chisq_converges_on_the_exact_conditional_p_value(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """What the Monte Carlo branch estimates, computed by brute force.

    Not Fisher's p-value: this one orders the tables by their chi-square statistic
    and Fisher's orders them by their probability, so the two disagree on the same
    table and only one of them is what these draws are about.
    """
    table = cat_tables["t3x3_small"]
    integer = np.rint(table).astype(np.int64)
    by_row = integer.sum(axis=1)
    by_col = integer.sum(axis=0)
    expectation = expected_independence(table)

    def statistic(candidate: np.ndarray) -> float:
        return float(np.sum(np.sort(((candidate - expectation) ** 2 / expectation).ravel())[::-1]))

    observed = statistic(integer)
    weights = [
        (statistic(candidate), np.exp(log_table_weight(candidate)))
        for candidate in fixed_margin_tables(by_row, by_col)
    ]
    total = sum(weight for _, weight in weights)
    exact = sum(weight for value, weight in weights if value >= observed) / total

    simulated = chisq(table, simulate_p_value=True, n_resamples=39999, seed=5)["pval"]
    assert abs(simulated - exact) < 0.01


def test_a_resampled_table_keeps_the_margins_it_was_drawn_from(
    cat_tables: dict[str, np.ndarray],
) -> None:
    table = cat_tables["t3x4"]
    draws = _resample_fixed_margins(table, 200, SaRandom(3))
    reshaped = draws.reshape(-1, *table.shape)

    assert np.array_equal(reshaped.sum(axis=2), np.tile(table.sum(axis=1), (len(reshaped), 1)))
    assert np.array_equal(reshaped.sum(axis=1), np.tile(table.sum(axis=0), (len(reshaped), 1)))
    # The draw is a draw and not the observed table over and over.
    assert not np.all(reshaped == table)


# --------------------------------------------------------------------------- #
# fisher
# --------------------------------------------------------------------------- #


def test_fisher_reproduces_r(cat_tables: dict[str, np.ndarray]) -> None:
    _, expected = load_case("cat_fisher")

    assert_fisher_close(fisher(cat_tables["t2x2"]), expected["t2x2"])
    assert_fisher_close(fisher(cat_tables["t2x2"], conf_level=0.90), expected["t2x2_conf_90"])
    assert_fisher_close(fisher(cat_tables["t2x2_zero"]), expected["t2x2_zero"])

    # The r x c branch has nothing root-found in it, so these are graded whole.
    assert_close(fisher(cat_tables["t3x3_small"]), expected["t3x3_small"])
    assert_close(fisher(cat_tables["t3x3_mid"]), expected["t3x3_mid"])
    assert_close(fisher(cat_tables["t2x4"]), expected["t2x4"])


def test_a_table_past_the_enumeration_limit_is_reported_and_not_raised(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """Where this port stops short of R, stated as a test rather than as a comment.

    ``t3x4`` is inside R's workspace and outside this port's limit; ``t_big`` is
    outside both. Either way the p-value is absent, ``enumerated`` is 0, and
    nothing is raised - the chi-square test standing beside it in the same result
    is what the caller reads instead.
    """
    _, expected = load_case("cat_fisher")

    for name in ("t3x4", "t_big"):
        row = fisher(cat_tables[name])
        assert row["enumerated"] == 0
        assert np.isnan(row["pval"])
        assert row["n_used"] == expected[name]["n_used"]

    # R answered `t3x4` exactly, so the fixture records a p-value there and the
    # divergence is this and only this.
    assert expected["t3x4"]["enumerated"] == 1
    assert expected["t_big"]["enumerated"] == 0


def test_the_enumeration_limit_is_what_decides_which_tables_are_refused(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """The limit is on the table count, so the count is what the cases straddle."""
    counted = {}
    for name in ("t3x3_small", "t3x3_mid", "t2x4", "t3x4"):
        table = np.rint(cat_tables[name]).astype(np.int64)
        by_row = tuple(int(value) for value in table.sum(axis=1))
        by_col = tuple(int(value) for value in table.sum(axis=0))
        counted[name] = min(count_tables(by_row, by_col), count_tables(by_col, by_row))

    assert counted["t3x3_small"] < counted["t2x4"] < counted["t3x3_mid"]
    assert counted["t3x3_mid"] < FISHER_TABLE_LIMIT < counted["t3x4"]


def test_an_exact_p_value_sums_the_probability_of_every_table_no_likelier() -> None:
    """A table small enough to enumerate by brute force, checked against one.

    The kernel prunes whole subtrees rather than walking them, so the pruning is
    graded against the definition on a table where the definition is affordable.
    """
    table = np.array([[3, 1, 2], [2, 4, 1], [1, 2, 3]])
    by_row = table.sum(axis=1)
    by_col = table.sum(axis=0)

    ceiling = log_table_weight(table) + np.log1p(FISHER_REL_ERR)
    total = sum(
        np.exp(log_table_weight(candidate))
        for candidate in fixed_margin_tables(by_row, by_col)
        if log_table_weight(candidate) <= ceiling
    )

    scale = (
        sum(lgamma(int(value) + 1) for value in by_row)
        + sum(lgamma(int(value) + 1) for value in by_col)
        - lgamma(int(table.sum()) + 1)
    )
    assert fisher_exact_rxc(table) == pytest.approx(float(np.exp(scale) * total), rel=1e-12)


def test_the_exact_p_value_does_not_depend_on_how_the_table_is_laid_out(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """Transposing a table renames its variables and changes no probability.

    Worth a test because the kernel picks which margin to fill along by which one
    is cheaper to hold, so a transposed table takes the other branch.
    """
    for name in ("t3x3_small", "t3x3_mid", "t2x4"):
        table = cat_tables[name]
        assert fisher(table)["pval"] == pytest.approx(fisher(table.T)["pval"], rel=1e-12)


def test_a_simulated_fisher_converges_on_the_exact_one(
    cat_tables: dict[str, np.ndarray],
) -> None:
    table = cat_tables["t3x3_mid"]
    row = fisher(table, simulate_p_value=True, n_resamples=19999, seed=7)

    assert row["enumerated"] == 1
    assert np.isnan(row["odds_ratio_cond"])
    assert abs(row["pval"] - fisher(table)["pval"]) < 0.02


def test_the_conditional_odds_ratio_is_not_the_sample_one(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """Both are exported, and the docstring's claim is that they differ.

    The conditional estimate is pulled towards 1, and most so where the counts
    are smallest, which is the table the exact test exists for.
    """
    table = cat_tables["t2x2"]
    conditional = fisher(table)["odds_ratio_cond"]
    sample = float(odds_ratio(table)["estimate"].iloc[0])
    assert conditional != pytest.approx(sample)
    assert 1 < conditional < sample

    # A zero cell leaves the conditional estimate unbounded, where the sample one
    # is finite only because the Haldane-Anscombe correction made it so. The two
    # answer the zero differently and both answers are reported.
    zero = cat_tables["t2x2_zero"]
    assert np.isinf(fisher(zero)["odds_ratio_cond"])
    assert np.isinf(fisher(zero)["upper_conf"])
    assert np.isfinite(float(odds_ratio(zero)["estimate"].iloc[0]))


def test_only_a_two_by_two_table_has_an_interval(cat_tables: dict[str, np.ndarray]) -> None:
    for name in ("t3x3_small", "t2x4"):
        row = fisher(cat_tables[name])
        assert np.isnan(row["lower_conf"])
        assert np.isnan(row["upper_conf"])
        assert np.isnan(row["odds_ratio_cond"])
        assert np.isnan(row["statistic"])
        assert np.isnan(row["df"])


# --------------------------------------------------------------------------- #
# mcnemar
# --------------------------------------------------------------------------- #


def test_mcnemar_reproduces_r(cat_tables: dict[str, np.ndarray]) -> None:
    _, expected = load_case("cat_mcnemar")
    small = cat_tables["pair_small"]
    large = cat_tables["pair_large"]

    assert_close(mcnemar(small), expected["small_default"])
    assert_close(mcnemar(small, exact=False), expected["small_forced_chisq"])
    assert_close(mcnemar(small, correct=False, exact=False), expected["small_plain_chisq"])
    assert_close(mcnemar(large), expected["large_default"])
    assert_close(mcnemar(large, exact=True), expected["large_forced_exact"])
    assert_close(mcnemar(large, correct=False), expected["large_plain_chisq"])
    assert_close(mcnemar(cat_tables["pair_one_way"]), expected["one_way"])


def test_the_default_branch_turns_on_the_discordant_pair_count(
    cat_tables: dict[str, np.ndarray],
) -> None:
    small = mcnemar(cat_tables["pair_small"])
    large = mcnemar(cat_tables["pair_large"])

    assert small["n_discordant"] < MCNEMAR_EXACT_MAX_DISCORDANT
    assert small["exact_used"] == 1
    assert large["n_discordant"] >= MCNEMAR_EXACT_MAX_DISCORDANT
    assert large["exact_used"] == 0


def test_mcnemar_reads_only_the_discordant_cells(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """Changing a concordant cell moves ``n_used`` and nothing else."""
    table = cat_tables["pair_small"].copy()
    before = mcnemar(table, exact=False)
    table[0, 0] += 40
    after = mcnemar(table, exact=False)

    assert after["n_used"] == before["n_used"] + 40
    assert after["statistic"] == before["statistic"]
    assert after["pval"] == before["pval"]


def test_the_uncorrected_statistic_is_the_symmetry_residual_sum(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """The docstring's claim that the cell table and the p-value are one thing."""
    table = cat_tables["pair_large"]
    expected = expected_symmetry(table)
    residual_sum = float(np.sum((table - expected) ** 2 / np.where(expected > 0, expected, np.nan)))
    assert mcnemar(table, correct=False)["statistic"] == pytest.approx(residual_sum)


def test_a_table_with_no_discordance_has_no_mcnemar_test() -> None:
    with pytest.raises(SaValueError, match="no discordance"):
        mcnemar(np.array([[12.0, 0.0], [0.0, 9.0]]))


# --------------------------------------------------------------------------- #
# cochran_q
# --------------------------------------------------------------------------- #


def test_cochran_q_and_kendalls_w_reproduce_r() -> None:
    frame, expected = load_case("cat_cochran_q")
    fit = cochran_q(frame)

    assert_close(fit, expected["q"])
    assert_frame_close(
        assoc_measures_repeated(fit["statistic"], len(frame.index), len(frame.columns)),
        expected["kendalls_w"],
    )


def test_a_constant_subject_moves_neither_side_of_cochran_q() -> None:
    """It cancels out of both, which is more than the docstring claims.

    A subject who answered the same way throughout shifts every column count and
    the total by the same amount, so it drops out of the numerator; and it
    contributes ``k * row_n - row_n^2``, which is zero at both extremes, so it
    drops out of the denominator too. It is kept in ``n_used`` all the same, and
    that is where it shows: Kendall's W divides by the subject count, so dropping
    such subjects does move the effect size even though the test is untouched.
    """
    frame, _ = load_case("cat_cochran_q")
    matrix = frame.to_numpy(dtype=float)
    k = matrix.shape[1]
    constant = np.isin(matrix.sum(axis=1), (0, k))
    varying = matrix[~constant]

    assert constant.any()
    full = cochran_q(matrix)
    trimmed = cochran_q(varying)
    assert full["statistic"] == pytest.approx(trimmed["statistic"])
    assert full["n_used"] == len(matrix)
    assert trimmed["n_used"] == len(varying)

    w_full = float(assoc_measures_repeated(full["statistic"], len(matrix), k)["estimate"].iloc[0])
    w_trimmed = float(
        assoc_measures_repeated(trimmed["statistic"], len(varying), k)["estimate"].iloc[0]
    )
    assert w_trimmed > w_full


def test_no_within_subject_variation_leaves_no_cochran_q() -> None:
    with pytest.raises(SaValueError, match="no within-subject variation"):
        cochran_q(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


def test_cochran_q_on_two_conditions_is_mcnemar_uncorrected() -> None:
    """Cochran's Q extends McNemar's test, so on two conditions it is that test."""
    responses = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    both = responses.sum(axis=0)
    table = np.array(
        [
            [float(np.sum((responses[:, 0] == 0) & (responses[:, 1] == 0))), 0.0],
            [0.0, 0.0],
        ]
    )
    table[0, 1] = float(np.sum((responses[:, 0] == 0) & (responses[:, 1] == 1)))
    table[1, 0] = float(np.sum((responses[:, 0] == 1) & (responses[:, 1] == 0)))
    table[1, 1] = float(np.sum((responses[:, 0] == 1) & (responses[:, 1] == 1)))
    assert both.size == 2

    assert cochran_q(responses)["statistic"] == pytest.approx(
        mcnemar(table, correct=False, exact=False)["statistic"]
    )


# --------------------------------------------------------------------------- #
# association measures
# --------------------------------------------------------------------------- #


def test_the_association_measures_reproduce_r(cat_tables: dict[str, np.ndarray]) -> None:
    _, expected = load_case("cat_association")

    assert_frame_close(assoc_measures(cat_tables["t2x2"]), expected["t2x2"])
    assert_frame_close(
        assoc_measures(cat_tables["t2x2"], conf_level=0.90), expected["t2x2_conf_90"]
    )
    assert_frame_close(assoc_measures(cat_tables["t2x2_zero"]), expected["t2x2_zero"])
    assert_frame_close(assoc_measures(cat_tables["t3x4"]), expected["t3x4"])
    assert_frame_close(assoc_measures_paired(cat_tables["pair_small"]), expected["paired_small"])
    assert_frame_close(
        assoc_measures_paired(cat_tables["pair_small"], conf_level=0.90),
        expected["paired_conf_90"],
    )
    assert_frame_close(
        assoc_measures_paired(cat_tables["pair_one_way"]), expected["paired_one_way"]
    )
    assert_frame_close(odds_ratio(cat_tables["t2x2"]), expected["odds_ratio"])
    assert_frame_close(odds_ratio(cat_tables["t2x2_zero"]), expected["odds_ratio_zero"])

    assert_close(phi(cat_tables["t2x2"]), expected["phi_2x2"])
    assert_close(phi(cat_tables["t2x2_zero"]), expected["phi_zero"])
    assert_close(has_zero_cell(cat_tables["t2x2"]), expected["has_zero_2x2"])
    assert_close(has_zero_cell(cat_tables["t2x2_zero"]), expected["has_zero_zero"])
    assert_close(has_zero_cell(cat_tables["t3x4"]), expected["has_zero_3x4"])


def test_an_association_table_is_built_from_assoc_rows() -> None:
    row = assoc_row("cramers_v", 0.25)

    assert list(row.columns) == list(ASSOC_COLUMNS)
    assert row["measure"].iloc[0] == "cramers_v"
    assert row["estimate"].iloc[0] == 0.25
    assert np.isnan(row["lower_conf"].iloc[0])
    assert np.isnan(row["upper_conf"].iloc[0])


def test_an_unbounded_estimate_is_reported_as_absent() -> None:
    """``finite_or_na`` is what a plot depends on, so an infinity must not pass."""
    row = assoc_row("odds_ratio_paired", np.inf, 3.0, np.inf)

    assert np.isnan(row["estimate"].iloc[0])
    assert row["lower_conf"].iloc[0] == 3.0
    assert np.isnan(row["upper_conf"].iloc[0])


def test_which_measures_a_table_gets_depends_on_its_shape(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """Phi and the odds ratio only exist on a 2 x 2 table."""
    assert list(assoc_measures(cat_tables["t2x2"])["measure"]) == [
        "cramers_v",
        "contingency_coefficient",
        "phi_coefficient",
        "odds_ratio",
    ]
    assert list(assoc_measures(cat_tables["t3x4"])["measure"]) == [
        "cramers_v",
        "contingency_coefficient",
    ]


def test_the_effect_size_does_not_move_when_the_p_value_correction_does(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """Cramer's V is built on the uncorrected statistic, whatever the test used."""
    table = cat_tables["t2x2"]
    n = table.sum()
    plain = chisq(table, correct=False)["statistic"]
    v = float(assoc_measures(table)["estimate"].iloc[0])

    assert v == pytest.approx(np.sqrt(plain / n))
    assert v == pytest.approx(abs(phi(table)))


def test_phi_carries_the_sign_cramers_v_drops(cat_tables: dict[str, np.ndarray]) -> None:
    table = cat_tables["t2x2"]
    flipped = table[:, ::-1]

    assert phi(table) > 0
    assert phi(flipped) == pytest.approx(-phi(table))
    # Which is the same direction the odds ratio reads.
    assert float(odds_ratio(table)["estimate"].iloc[0]) > 1
    assert float(odds_ratio(flipped)["estimate"].iloc[0]) < 1


def test_phi_has_no_value_where_a_margin_is_empty() -> None:
    assert np.isnan(phi(np.array([[5.0, 3.0], [0.0, 0.0]])))


def test_only_a_two_by_two_table_can_hold_a_zero_cell_worth_reporting() -> None:
    """A zero in a larger table is not what the correction is about."""
    assert has_zero_cell(np.array([[4.0, 0.0], [3.0, 6.0]]))
    assert not has_zero_cell(np.array([[4.0, 0.0, 2.0], [3.0, 6.0, 1.0]]))


def test_the_haldane_anscombe_correction_keeps_the_odds_ratio_finite() -> None:
    table = np.array([[14.0, 0.0], [5.0, 11.0]])
    row = odds_ratio(table)

    assert np.isfinite(row["estimate"].iloc[0])
    assert np.isfinite(row["lower_conf"].iloc[0])
    assert np.isfinite(row["upper_conf"].iloc[0])
    # Half an observation everywhere, which is what the uncorrected ratio of the
    # shifted table is.
    shifted = table + 0.5
    assert row["estimate"].iloc[0] == pytest.approx(
        shifted[0, 0] * shifted[1, 1] / (shifted[0, 1] * shifted[1, 0])
    )


def test_the_odds_ratio_interval_is_built_on_the_log_scale(
    cat_tables: dict[str, np.ndarray],
) -> None:
    row = odds_ratio(cat_tables["t2x2"])
    estimate = float(row["estimate"].iloc[0])
    lower = float(row["lower_conf"].iloc[0])
    upper = float(row["upper_conf"].iloc[0])

    assert 0 < lower < estimate < upper
    # Symmetric in the log, so asymmetric in the ratio.
    assert np.log(estimate) - np.log(lower) == pytest.approx(np.log(upper) - np.log(estimate))
    assert estimate - lower != pytest.approx(upper - estimate)


def test_a_narrower_level_gives_a_narrower_interval(
    cat_tables: dict[str, np.ndarray],
) -> None:
    table = cat_tables["t2x2"]
    wide = odds_ratio(table, conf_level=0.99)
    narrow = odds_ratio(table, conf_level=0.90)

    assert wide["lower_conf"].iloc[0] < narrow["lower_conf"].iloc[0]
    assert wide["upper_conf"].iloc[0] > narrow["upper_conf"].iloc[0]


def test_a_one_way_paired_table_keeps_the_two_bounded_measures(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """The odds ratio runs off the scale there and the other two carry the finding.

    ``pair_one_way`` has every discordant pair moving the same way, so the odds
    ratio hits an end of its range: 0 as the table stands, and unbounded - hence
    absent - once transposed so that the movement is the other way. ``cohens_g``
    reaches its own extreme of a half in both, with the sign saying which way.
    """
    table = cat_tables["pair_one_way"]
    forward = assoc_measures_paired(table).set_index("measure")
    reverse = assoc_measures_paired(table.T).set_index("measure")

    assert forward.loc["odds_ratio_paired", "estimate"] == 0.0
    assert np.isnan(reverse.loc["odds_ratio_paired", "estimate"])

    assert forward.loc["cohens_g", "estimate"] == -0.5
    assert reverse.loc["cohens_g", "estimate"] == 0.5

    assert forward.loc["risk_difference_paired", "estimate"] == pytest.approx(
        -table[1, 0] / table.sum()
    )
    assert reverse.loc["risk_difference_paired", "estimate"] == pytest.approx(
        table[1, 0] / table.sum()
    )


def test_the_paired_interval_and_the_exact_test_agree_about_a_half(
    cat_tables: dict[str, np.ndarray],
) -> None:
    """The docstring's claim, checked on both sides of significance."""
    for name in ("pair_small", "pair_large"):
        table = cat_tables[name]
        row = assoc_measures_paired(table).set_index("measure").loc["cohens_g"]
        # `cohens_g` is the discordant share less a half, so it excludes zero
        # exactly when the interval for the share excludes a half.
        excludes_half = not (row["lower_conf"] <= 0 <= row["upper_conf"])
        assert excludes_half == (mcnemar(table, exact=True)["pval"] < 0.05)


def test_no_discordance_leaves_every_paired_measure_but_the_difference_absent() -> None:
    table = np.array([[12.0, 0.0], [0.0, 9.0]])
    out = assoc_measures_paired(table).set_index("measure")

    assert np.isnan(out.loc["odds_ratio_paired", "estimate"])
    assert np.isnan(out.loc["cohens_g", "estimate"])
    assert out.loc["risk_difference_paired", "estimate"] == 0.0


def test_kendalls_w_rescales_cochran_q_by_its_largest_possible_value() -> None:
    """Which is what makes it a measure rather than a statistic that grows with n."""
    frame, _ = load_case("cat_cochran_q")
    matrix = frame.to_numpy(dtype=float)
    k = matrix.shape[1]

    once = cochran_q(matrix)
    twice = cochran_q(np.vstack([matrix, matrix]))
    w_once = float(assoc_measures_repeated(once["statistic"], len(matrix), k)["estimate"].iloc[0])
    w_twice = float(
        assoc_measures_repeated(twice["statistic"], 2 * len(matrix), k)["estimate"].iloc[0]
    )

    assert twice["statistic"] == pytest.approx(2 * once["statistic"])
    assert w_twice == pytest.approx(w_once)
    assert 0 <= w_once <= 1
