"""Crossed-factor data with a known answer in four tables.

The draws are not R's, so this is graded on the contract and on the statistics
rather than against a frozen fixture. The contract is the larger half here: four
truth tables and an ``args`` the factorial comparison will unpack, with a row
order and a direction a post-hoc table is scored against, and a sign error in any
of them would score every later table backwards while every number stayed
plausible.

What replaces the fixture is the fact that a planted effect can be recomputed.
The simulator says which terms it moved; decomposing the cell deltas says which
terms moved. Those are two independent statements about the same design, and the
five shapes exist precisely so that they can disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statassist import (
    diagnose_distribution,
    screen_outliers,
    simulate_factorial_groups,
    summarize_descriptive_stats,
)
from statassist.core.errors import SaValueError
from statassist.core.factorial import fact_cell_labels, fact_component, fact_grid, fact_terms
from statassist.kernel.factorial import factorial_anova, factorial_plan
from statassist.simulate.factorial_groups import (
    FACT_SHAPES,
    RESERVED_FACTOR_NAMES,
    _flip,
    _profile,
)

#: A three-factor design, so that an interaction of order three exists to test.
THREE_WAY = {
    "treatment": ["control", "treat_A", "treat_B"],
    "sex": ["male", "female"],
    "time": ["T0", "T1"],
}


def cell_deltas(sim, feature: str) -> np.ndarray:
    """The planted shift of every cell of one feature, in grid order.

    Read back off ``truth_cell`` rather than out of the simulator's internals, so
    that a test which recomputes the decomposition is reading the table a user
    would read.
    """
    rows = sim.truth_cell[sim.truth_cell["features"] == feature]
    return rows["delta"].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_the_slots_are_the_five_the_result_promises() -> None:
    sim = simulate_factorial_groups(n_feats=6, n_per_cell=3, seed=1)
    assert list(sim) == ["args", "truth", "truth_term", "truth_cell", "truth_contrast"]


def test_args_is_named_after_the_comparison_that_consumes_it() -> None:
    sim = simulate_factorial_groups(n_feats=6, n_per_cell=3, seed=1)
    assert list(sim.args) == ["data", "feats", "factors", "factor_lv", "input_scale"]

    mixed = simulate_factorial_groups(
        n_feats=6, n_per_cell=3, factor_lv=THREE_WAY, within="time", seed=1
    )
    # The two within keys arrive between `factor_lv` and `input_scale`, and are
    # absent rather than empty when the design is wholly between subjects.
    assert list(mixed.args) == [
        "data",
        "feats",
        "factors",
        "factor_lv",
        "within",
        "id",
        "input_scale",
    ]


def test_the_four_tables_hold_their_columns_in_order() -> None:
    sim = simulate_factorial_groups(n_feats=4, n_per_cell=3, factor_lv=THREE_WAY, seed=1)

    assert sim.truth.columns.tolist() == [
        "features",
        "pattern",
        "spread",
        "direction",
        "partner",
        "extreme_cell",
        "extreme_tied",
        "log2fc",
        "baseline",
        "sd_subject",
    ]
    assert sim.truth_term.columns.tolist() == [
        "features",
        "terms",
        "term_order",
        "is_within",
        "max_abs_delta",
        "is_effect",
    ]
    # One column per factor, in declaration order, between `features` and the
    # rest. That is what makes a factor named after one of the others an error.
    assert sim.truth_cell.columns.tolist() == [
        "features",
        "treatment",
        "sex",
        "time",
        "is_ref",
        "delta",
        "center",
        "sd",
        "n",
    ]
    assert sim.truth_contrast.columns.tolist() == [
        "features",
        "factor",
        "stratum",
        "contrast",
        "group1",
        "group2",
        "delta",
        "is_diff",
    ]


def test_every_table_is_as_long_as_its_axes_say() -> None:
    sim = simulate_factorial_groups(n_feats=5, n_per_cell=3, factor_lv=THREE_WAY, seed=1)
    n_feats = 5
    n_cells = 3 * 2 * 2
    n_terms = len(fact_terms(list(THREE_WAY)))

    assert len(sim.truth.index) == n_feats
    assert len(sim.truth_term.index) == n_feats * n_terms
    assert len(sim.truth_cell.index) == n_feats * n_cells
    assert len(sim.truth_contrast.index) % n_feats == 0
    assert sim.args["data"].shape == (n_cells * 3, n_feats)


def test_every_table_is_feature_major_and_aligned_with_feats() -> None:
    """A truth table read in blocks has to hand back the features in ``feats`` order."""
    sim = simulate_factorial_groups(n_feats=4, n_per_cell=3, factor_lv=THREE_WAY, seed=1)
    feats = sim.args["feats"]

    assert list(sim.truth["features"]) == feats
    for table in (sim.truth_term, sim.truth_cell, sim.truth_contrast):
        blocks = len(table.index) // len(feats)
        assert list(table["features"]) == list(np.repeat(feats, blocks))


def test_the_cells_are_counted_the_way_the_layout_counts_them() -> None:
    """``truth_cell`` is in ``fact_grid`` order, first factor varying fastest."""
    sim = simulate_factorial_groups(n_feats=2, n_per_cell=3, factor_lv=THREE_WAY, seed=1)
    grid = fact_grid(THREE_WAY)
    block = sim.truth_cell[sim.truth_cell["features"] == "prot_1"]

    for name, levels in THREE_WAY.items():
        wanted = [levels[at] for at in grid[name]]
        assert list(block[name]) == wanted
    # The reference cell is the first, and it is the only one flagged.
    assert list(block["is_ref"]) == [True] + [False] * (len(block.index) - 1)
    assert block["delta"].iloc[0] == 0.0


def test_the_terms_are_in_the_order_an_anova_table_lists_them() -> None:
    sim = simulate_factorial_groups(n_feats=2, n_per_cell=3, factor_lv=THREE_WAY, seed=1)
    block = sim.truth_term[sim.truth_term["features"] == "prot_1"]

    assert list(block["terms"]) == [
        "treatment",
        "sex",
        "time",
        "treatment:sex",
        "treatment:time",
        "sex:time",
        "treatment:sex:time",
    ]
    assert list(block["term_order"]) == [1, 1, 1, 2, 2, 2, 3]


def test_which_terms_sit_in_the_within_subject_stratum_is_recorded() -> None:
    """A mixed ANOVA has two error strata, so a term row has to say which is its."""
    mixed = simulate_factorial_groups(
        n_feats=2, n_per_cell=3, factor_lv=THREE_WAY, within="time", seed=1
    )
    block = mixed.truth_term[mixed.truth_term["features"] == "prot_1"]
    by_term = dict(zip(block["terms"], block["is_within"], strict=True))

    assert by_term == {
        "treatment": False,
        "sex": False,
        "time": True,
        "treatment:sex": False,
        "treatment:time": True,
        "sex:time": True,
        "treatment:sex:time": True,
    }

    between = simulate_factorial_groups(n_feats=2, n_per_cell=3, factor_lv=THREE_WAY, seed=1)
    assert not between.truth_term["is_within"].any()


def test_the_marginal_contrast_comes_before_the_simple_effects() -> None:
    """A table read top to bottom moves from the main effect to what may hide it."""
    sim = simulate_factorial_groups(n_feats=1, n_per_cell=3, seed=1)
    rows = sim.truth_contrast

    assert rows["stratum"].iloc[0] is None
    treatment = rows[rows["factor"] == "treatment"]
    assert list(dict.fromkeys(treatment["stratum"])) == [None, "male", "female"]
    # And each factor's block is contiguous, in declaration order.
    assert list(dict.fromkeys(rows["factor"])) == ["treatment", "sex"]


def test_a_stratum_is_missing_on_the_marginal_row_and_named_otherwise() -> None:
    sim = simulate_factorial_groups(n_feats=1, n_per_cell=3, factor_lv=THREE_WAY, seed=1)
    rows = sim.truth_contrast[sim.truth_contrast["factor"] == "treatment"]
    named = [value for value in dict.fromkeys(rows["stratum"]) if value is not None]

    assert None in list(rows["stratum"])
    # The stratum names the combination of the *other* factors, in their order.
    assert named == ["male.T0", "female.T0", "male.T1", "female.T1"]


def test_a_labelled_column_holds_strings_and_not_categories() -> None:
    """R builds these with ``stringsAsFactors = FALSE``; the port keeps plain text."""
    sim = simulate_factorial_groups(n_feats=3, n_per_cell=3, seed=1)

    for table, columns in (
        (sim.truth, ("features", "pattern", "spread", "direction")),
        (sim.truth_term, ("features", "terms")),
        (sim.truth_cell, ("features", "treatment", "sex")),
        (sim.truth_contrast, ("features", "factor", "contrast", "group1", "group2")),
    ):
        for column in columns:
            assert not isinstance(table[column].dtype, pd.CategoricalDtype)
            assert all(isinstance(value, str) for value in table[column])

    assert sim.args["factor_lv"]["treatment"] == ["control", "treat_A", "treat_B", "treat_C"]
    assert all(isinstance(level, str) for level in sim.args["factors"]["treatment"])


def test_a_within_design_gives_every_subject_every_within_cell() -> None:
    """No subject is dropped, so the within-subject rectangle is complete."""
    mixed = simulate_factorial_groups(
        n_feats=3, n_per_cell=4, factor_lv=THREE_WAY, within=["time"], seed=1
    )
    frame = pd.DataFrame({"id": mixed.args["id"], **mixed.args["factors"]})

    per_subject = frame.groupby("id").size()
    assert set(per_subject) == {len(THREE_WAY["time"])}
    # Each subject sits at one combination of the between factors and moves
    # through every level of the within one.
    for name in ("treatment", "sex"):
        assert (frame.groupby("id")[name].nunique() == 1).all()
    assert (frame.groupby("id")["time"].nunique() == len(THREE_WAY["time"])).all()


def test_naming_every_factor_within_gives_a_fully_repeated_design() -> None:
    sim = simulate_factorial_groups(
        n_feats=3, n_per_cell=4, factor_lv=THREE_WAY, within=list(THREE_WAY), seed=1
    )
    n_cells = 3 * 2 * 2

    assert sim.args["within"] == list(THREE_WAY)
    assert len(set(sim.args["id"])) == 4
    assert sim.args["data"].shape[0] == 4 * n_cells
    assert sim.truth_term["is_within"].all()


def test_within_is_reordered_into_the_order_the_factors_were_declared() -> None:
    """So that every table built from either list reads in one order."""
    sim = simulate_factorial_groups(
        n_feats=2, n_per_cell=3, factor_lv=THREE_WAY, within=["time", "sex"], seed=1
    )
    assert sim.args["within"] == ["sex", "time"]


def test_one_size_spreads_and_a_vector_carries_one_per_between_combination() -> None:
    sizes = [4, 5, 6, 7]
    sim = simulate_factorial_groups(
        n_feats=2,
        n_per_cell=sizes,
        factor_lv={"treatment": ["control", "treat_A"], "sex": ["male", "female"]},
        seed=1,
    )
    block = sim.truth_cell[sim.truth_cell["features"] == "prot_1"]

    assert list(block["n"]) == sizes
    assert sim.args["data"].shape[0] == sum(sizes)


def test_a_within_factor_holds_no_size_of_its_own() -> None:
    """The sizes count the between combinations, so a within factor is not among them."""
    sizes = [3, 4, 5, 6, 7, 8]
    sim = simulate_factorial_groups(
        n_feats=2, n_per_cell=sizes, factor_lv=THREE_WAY, within="time", seed=1
    )
    block = sim.truth_cell[sim.truth_cell["features"] == "prot_1"]

    # Twelve sizes would be needed if `time` counted among the combinations; it
    # does not, so six cover treatment x sex and each repeats across `time`.
    assert len(sizes) == 3 * 2
    assert list(block["n"]) == sizes * len(THREE_WAY["time"])
    assert len(set(sim.args["id"])) == sum(sizes)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_data() -> None:
    first = simulate_factorial_groups(n_feats=5, n_per_cell=3, seed=7)
    again = simulate_factorial_groups(n_feats=5, n_per_cell=3, seed=7)
    other = simulate_factorial_groups(n_feats=5, n_per_cell=3, seed=8)

    pd.testing.assert_frame_equal(first.args["data"], again.args["data"])
    pd.testing.assert_frame_equal(first.truth_cell, again.truth_cell)
    assert not first.args["data"].equals(other.args["data"])


def test_how_many_features_take_each_shape_does_not_move_with_the_seed() -> None:
    """Both mixes are split by largest remainder, so the counts are the arguments'."""
    counts = []
    for seed in (1, 2, 3):
        sim = simulate_factorial_groups(n_feats=40, n_up=10, n_down=10, n_per_cell=3, seed=seed)
        counts.append(
            (
                sim.truth["pattern"].value_counts().to_dict(),
                sim.truth["spread"].value_counts().to_dict(),
            )
        )
    assert counts[0] == counts[1] == counts[2]
    # Ten of each direction over five shapes is two apiece, in each direction.
    assert counts[0][0] == dict.fromkeys(FACT_SHAPES, 4) | {"none": 20}


def test_each_direction_is_split_between_the_shapes_on_its_own() -> None:
    """A mix that held only in total would leave a shape one-directional."""
    sim = simulate_factorial_groups(n_feats=50, n_up=15, n_down=15, n_per_cell=3, seed=4)
    planted = sim.truth[sim.truth["pattern"] != "none"]
    crossed = pd.crosstab(planted["pattern"], planted["direction"])

    assert set(crossed.columns) == {"up", "down"}
    assert (crossed == 3).all().all()


def test_a_zero_weight_leaves_a_shape_out_entirely() -> None:
    sim = simulate_factorial_groups(
        n_feats=20,
        n_up=5,
        n_down=5,
        n_per_cell=3,
        term_mix={"crossover": 1.0, "main_only": 0.0},
        pattern_mix={"all": 1.0, "gradient": 0.0, "single": 0.0},
        seed=1,
    )
    assert set(sim.truth["pattern"]) == {"none", "crossover"}
    assert set(sim.truth["spread"]) == {"none", "all"}


# --------------------------------------------------------------------------- #
# What was planted, recomputed
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def planted():
    """One large draw, so that every shape has several features in it."""
    return simulate_factorial_groups(
        n_feats=60, n_up=20, n_down=20, n_per_cell=3, factor_lv=THREE_WAY, seed=11
    )


def test_the_terms_a_shape_moves_are_the_terms_the_shape_promises(planted) -> None:
    """The claim the five shapes exist to make, read off ``truth_term``.

    A shape is defined by which terms it leaves at exactly zero, so this is the
    table that says whether an interaction call is a finding or a false positive.
    The partner factor is drawn, so a term is checked against the feature's own
    partner rather than against a fixed name.
    """
    for row in planted.truth.itertuples(index=False):
        if row.pattern == "none":
            continue
        terms = planted.truth_term[planted.truth_term["features"] == row.features]
        effect = dict(zip(terms["terms"], terms["is_effect"], strict=True))
        primary = next(iter(THREE_WAY))

        if row.pattern == "main_only":
            assert effect[primary]
            assert sum(effect.values()) == 1
            continue

        mate = row.partner
        # Terms are labelled in declaration order, and the primary factor is
        # declared first, so the interaction reads primary-first whichever
        # partner was drawn.
        interaction = f"{primary}:{mate}"
        wanted = {
            "additive": {primary, mate},
            "interaction": {primary, interaction},
            "crossover": {interaction},
            "nuisance_only": {mate},
        }[row.pattern]
        assert {name for name, hit in effect.items() if hit} == wanted


def test_the_recorded_component_is_the_one_a_decomposition_finds(planted) -> None:
    """``max_abs_delta`` recomputed from ``truth_cell``, which is the whole claim.

    The simulator builds the effect out of centred profiles and reports the
    largest component of each term. Decomposing the cell deltas is the other
    direction of the same statement, and the two are computed by different code:
    one plants, the other takes inclusion-exclusion over the grid.
    """
    grid = fact_grid(THREE_WAY)
    terms = fact_terms(list(THREE_WAY))

    for feature in planted.args["feats"]:
        deltas = cell_deltas(planted, feature)
        rows = planted.truth_term[planted.truth_term["features"] == feature]
        found = [float(np.abs(fact_component(deltas, grid, term)).max()) for term in terms]
        np.testing.assert_allclose(rows["max_abs_delta"].to_numpy(), found, rtol=1e-12)


def test_an_unplanted_feature_is_exactly_zero_everywhere(planted) -> None:
    """Exactly, not nearly: ``is_effect`` and ``is_diff`` are comparisons to zero."""
    null_feats = set(planted.truth.loc[planted.truth["pattern"] == "none", "features"])
    assert null_feats

    for table, column in (
        (planted.truth_cell, "delta"),
        (planted.truth_term, "max_abs_delta"),
        (planted.truth_contrast, "delta"),
    ):
        rows = table[table["features"].isin(null_feats)]
        assert (rows[column] == 0.0).all()

    truth = planted.truth[planted.truth["features"].isin(null_feats)]
    assert (truth["log2fc"] == 0.0).all()
    assert truth["extreme_cell"].isna().all()
    # Every cell is equally far from the reference, which is to say none is
    # furthest, so the tie flag says to score the magnitude and not the name.
    assert truth["extreme_tied"].all()
    assert truth["partner"].isna().all()
    assert set(truth["spread"]) == {"none"}


def test_a_crossover_has_an_interaction_and_no_main_effect_at_all(planted) -> None:
    """The shape a main-effect test has to miss, which is the reason for the design."""
    rows = planted.truth[planted.truth["pattern"] == "crossover"]
    assert len(rows.index) > 0

    for row in rows.itertuples(index=False):
        terms = planted.truth_term[planted.truth_term["features"] == row.features]
        by_order = terms.groupby("term_order")["is_effect"].any()
        assert not by_order[1]
        assert by_order[2]
        # And the cells plainly differ all the same, which is what makes it a
        # trap rather than a null feature.
        assert np.abs(cell_deltas(planted, row.features)).max() > 0.5


def test_the_reference_cell_carries_no_shift_and_no_term_moved_with_it(planted) -> None:
    """Shifting the array is a constant, and a constant is the grand mean's."""
    grid = fact_grid(THREE_WAY)
    for feature in planted.args["feats"]:
        deltas = cell_deltas(planted, feature)
        assert deltas[0] == 0.0
        shifted = deltas + 3.7
        for term in fact_terms(list(THREE_WAY)):
            np.testing.assert_allclose(
                fact_component(shifted, grid, term), fact_component(deltas, grid, term), atol=1e-12
            )


def test_the_centre_of_a_cell_is_its_baseline_plus_its_shift(planted) -> None:
    for row in planted.truth.itertuples(index=False):
        block = planted.truth_cell[planted.truth_cell["features"] == row.features]
        np.testing.assert_allclose(block["center"], row.baseline + block["delta"], rtol=1e-12)


def test_a_main_only_feature_carries_the_profile_the_pattern_mix_put_there(planted) -> None:
    """``truth_cell`` reads as ``simulate_multiple_groups`` does, uncentred.

    The effect is built from centred profiles, so this is a claim about the shift
    the array takes afterwards: the reference cell goes to zero, and what the
    primary factor then holds is the raw profile rather than the centred one.
    """
    grid = fact_grid(THREE_WAY)
    rows = planted.truth[planted.truth["pattern"] == "main_only"]
    assert len(rows.index) > 0

    for row in rows.itertuples(index=False):
        deltas = cell_deltas(planted, row.features)
        by_level = {
            level: deltas[np.asarray(grid["treatment"]) == level]
            for level in range(len(THREE_WAY["treatment"]))
        }
        # Constant within a level of the primary factor, since nothing else moved.
        for values in by_level.values():
            np.testing.assert_allclose(values, values[0], atol=1e-12)
        assert by_level[0][0] == 0.0
        if row.spread == "all":
            # Every non-reference level carries the whole magnitude.
            assert by_level[1][0] == pytest.approx(by_level[2][0])
        if row.spread == "gradient":
            assert abs(by_level[1][0]) < abs(by_level[2][0])
        if row.spread == "single":
            assert sum(values[0] != 0.0 for values in by_level.values()) == 1


def test_an_up_feature_reads_positive_wherever_the_shape_lets_it(planted) -> None:
    """``log2fc`` and ``direction`` agree for every shape that has one direction."""
    rows = planted.truth[planted.truth["pattern"].isin(["main_only", "additive", "interaction"])]
    assert len(rows.index) > 0

    for row in rows.itertuples(index=False):
        assert (row.log2fc > 0) == (row.direction == "up")


def test_a_crossover_reverses_the_primary_factor_along_the_partner(planted) -> None:
    """Which is why ``log2fc`` there follows the shape rather than ``direction``.

    The defining property of the shape, read off ``truth_cell``: the primary
    factor moves one way at the partner's reference level and the other way at the
    next one, so no single sign describes the feature and the extreme cell can sit
    on either side.
    """
    grid = fact_grid(THREE_WAY)
    primary = np.asarray(grid[next(iter(THREE_WAY))], dtype=int)
    rows = planted.truth[planted.truth["pattern"] == "crossover"]
    assert len(rows.index) > 0

    for row in rows.itertuples(index=False):
        deltas = cell_deltas(planted, row.features)
        mate = np.asarray(grid[row.partner], dtype=int)

        def shift(level: int, at: int, deltas=deltas, mate=mate) -> float:
            """How far the primary factor's level ``at`` sits from its reference."""
            held = mate == level
            return float(deltas[held & (primary == at)][0] - deltas[held & (primary == 0)][0])

        # The `"single"` profile leaves most levels of the primary factor at zero,
        # so the level it did move is the one to read the reversal off.
        levels = range(len(THREE_WAY[next(iter(THREE_WAY))]))
        moved_at = max(levels, key=lambda at: abs(shift(0, at)))
        assert shift(0, moved_at) != 0.0
        assert shift(0, moved_at) * shift(1, moved_at) < 0
        assert (shift(0, moved_at) > 0) == (row.direction == "up")


def test_the_extreme_cell_is_the_furthest_one_and_names_a_row_of_truth_cell(planted) -> None:
    for row in planted.truth.itertuples(index=False):
        block = planted.truth_cell[planted.truth_cell["features"] == row.features]
        deltas = block["delta"].to_numpy(dtype=float)
        largest = np.abs(deltas).max()

        assert row.log2fc == pytest.approx(deltas[int(np.abs(deltas).argmax())])
        assert row.extreme_tied == bool((np.abs(deltas) == largest).sum() > 1)
        if largest == 0:
            assert row.extreme_cell is None
            continue
        # The label reads back against `truth_cell` without a lookup table.
        labels = [
            ".".join(str(block[name].iloc[at]) for name in THREE_WAY)
            for at in range(len(block.index))
        ]
        assert row.extreme_cell == labels[int(np.abs(deltas).argmax())]


def test_a_contrast_is_group1_minus_group2_over_the_cells_it_averages(planted) -> None:
    """The direction a post-hoc table reads, recomputed from ``truth_cell``.

    A marginal contrast averages the other factors away and a simple effect holds
    them at one combination, so the two are different averages of the same cells
    and both are checked here.
    """
    grid = fact_grid(THREE_WAY)
    labels = {
        name: [THREE_WAY[name][at] for at in np.asarray(grid[name], dtype=int)]
        for name in THREE_WAY
    }

    for feature in planted.args["feats"][:6]:
        deltas = cell_deltas(planted, feature)
        rows = planted.truth_contrast[planted.truth_contrast["features"] == feature]
        for row in rows.itertuples(index=False):
            others = [name for name in THREE_WAY if name != row.factor]
            held = np.ones(len(deltas), dtype=bool)
            if row.stratum is not None:
                for name, level in zip(others, row.stratum.split("."), strict=True):
                    held &= np.array([value == level for value in labels[name]])
            own = np.array(labels[row.factor])
            wanted = (
                deltas[held & (own == row.group1)].mean()
                - deltas[held & (own == row.group2)].mean()
            )
            assert row.delta == pytest.approx(wanted, abs=1e-10)
            assert row.is_diff == (abs(wanted) >= 1e-8)


def test_a_contrast_that_is_zero_by_construction_reads_as_exactly_zero(planted) -> None:
    """Averaging divides, so the rounding has to be cleared rather than tolerated."""
    rows = planted.truth_contrast[~planted.truth_contrast["is_diff"]]
    assert len(rows.index) > 0
    assert (rows["delta"] == 0.0).all()

    # A `"nuisance_only"` feature's primary factor moved nothing, so every one of
    # its contrasts along that factor is exactly zero - marginal and simple alike.
    nuisance = set(planted.truth.loc[planted.truth["pattern"] == "nuisance_only", "features"])
    assert nuisance
    block = planted.truth_contrast[
        planted.truth_contrast["features"].isin(nuisance)
        & (planted.truth_contrast["factor"] == "treatment")
    ]
    assert (block["delta"] == 0.0).all()
    assert not block["is_diff"].any()


def test_a_marginal_contrast_can_be_zero_where_a_simple_effect_is_not(planted) -> None:
    """The reason both are in the table, on the shape that makes the point."""
    crossover = set(planted.truth.loc[planted.truth["pattern"] == "crossover", "features"])
    assert crossover

    found = False
    for feature in crossover:
        rows = planted.truth_contrast[
            (planted.truth_contrast["features"] == feature)
            & (planted.truth_contrast["factor"] == "treatment")
        ]
        marginal = rows[rows["stratum"].isna()]
        simple = rows[rows["stratum"].notna()]
        if not marginal["is_diff"].any() and simple["is_diff"].any():
            found = True
    assert found


# --------------------------------------------------------------------------- #
# The two helpers whose arithmetic the shapes rest on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_a_profile_is_centred_so_that_it_is_a_main_effect(k: int) -> None:
    rng = np.random.default_rng(0)
    for spread in ("all", "gradient", "single"):
        profile = _profile(1.5, spread, k, rng)
        assert len(profile) == k
        assert profile.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_the_flip_signs_average_to_zero_and_reach_one(k: int) -> None:
    """Which is what keeps a crossover pure and its size the size asked for."""
    signs = _flip(k)

    assert len(signs) == k
    assert signs.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(signs).max() == pytest.approx(1.0)
    # Alternating, so the effect reverses rather than merely differing.
    assert signs[0] > 0 > signs[1]


def test_a_crossover_of_any_partner_size_has_both_main_effects_at_zero() -> None:
    """The claim ``_flip`` exists for, checked past the two-level case."""
    for size in (2, 3, 4, 5):
        factor_lv = {"treatment": ["control", "treat_A", "treat_B"], "sex": list("abcde"[:size])}
        sim = simulate_factorial_groups(
            n_feats=4,
            n_up=2,
            n_down=2,
            n_per_cell=3,
            factor_lv=factor_lv,
            term_mix={"crossover": 1.0},
            seed=1,
        )
        by_order = sim.truth_term.groupby("term_order")["is_effect"]
        assert not by_order.any()[1]
        assert by_order.any()[2]


def test_interaction_scale_sets_how_much_of_the_effect_the_interaction_carries() -> None:
    small, large = (
        simulate_factorial_groups(
            n_feats=6,
            n_up=3,
            n_down=3,
            n_per_cell=3,
            term_mix={"interaction": 1.0},
            interaction_scale=scale,
            seed=5,
        )
        for scale in (0.2, 1.0)
    )

    def interaction(sim):
        rows = sim.truth_term[sim.truth_term["terms"] == "treatment:sex"]
        return rows["max_abs_delta"].to_numpy()

    # The same seed gives the same magnitudes, so the interaction scales with the
    # argument and the main effect does not move at all.
    np.testing.assert_allclose(interaction(large), interaction(small) * (1.0 / 0.2), rtol=1e-12)
    main = [sim.truth_term[sim.truth_term["terms"] == "treatment"] for sim in (small, large)]
    np.testing.assert_allclose(main[0]["max_abs_delta"], main[1]["max_abs_delta"], rtol=1e-12)


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_one_factor_is_the_other_simulator() -> None:
    with pytest.raises(SaValueError, match="at least two crossed factors"):
        simulate_factorial_groups(factor_lv={"treatment": ["control", "treat_A"]})
    with pytest.raises(SaValueError, match="at least two crossed factors"):
        simulate_factorial_groups(factor_lv=["control", "treat_A"])


@pytest.mark.parametrize("reserved", RESERVED_FACTOR_NAMES)
def test_a_factor_cannot_be_named_after_a_column_of_truth_cell(reserved: str) -> None:
    with pytest.raises(SaValueError, match="already use as columns"):
        simulate_factorial_groups(
            factor_lv={reserved: ["x", "y"], "sex": ["male", "female"]},
        )


def test_a_factor_needs_two_distinct_non_empty_levels() -> None:
    for levels in (["only"], ["a", "a"], ["a", ""], "ab", [1, 2]):
        with pytest.raises(SaValueError, match="two distinct non-empty level"):
            simulate_factorial_groups(
                factor_lv={"treatment": levels, "sex": ["male", "female"]},
            )


def test_within_has_to_name_factors_the_design_holds() -> None:
    with pytest.raises(SaValueError, match="does not hold"):
        simulate_factorial_groups(within="nope")
    with pytest.raises(SaValueError, match="distinct names"):
        simulate_factorial_groups(factor_lv=THREE_WAY, within=["time", "time"])


def test_n_per_cell_counts_the_between_subject_combinations() -> None:
    with pytest.raises(SaValueError, match="this design has 8"):
        simulate_factorial_groups(n_per_cell=[3, 4, 5])
    with pytest.raises(SaValueError, match="this design has 4"):
        simulate_factorial_groups(factor_lv=THREE_WAY, within="treatment", n_per_cell=[3, 4, 5])
    with pytest.raises(SaValueError, match=r"`n_per_cell` must be"):
        simulate_factorial_groups(n_per_cell="six")


def test_a_cell_needs_at_least_two_observations() -> None:
    with pytest.raises(SaValueError, match="`n_per_cell`"):
        simulate_factorial_groups(n_per_cell=1)
    with pytest.raises(SaValueError, match=r"`n_per_cell\[2\]`"):
        simulate_factorial_groups(
            n_per_cell=[3, 3, 1, 3],
            factor_lv={"treatment": ["control", "treat_A"], "sex": ["male", "female"]},
        )


def test_more_planted_features_than_features_is_refused() -> None:
    with pytest.raises(SaValueError, match="more features than"):
        simulate_factorial_groups(n_feats=4, n_up=3, n_down=3)


def test_a_scale_of_zero_would_leave_an_interaction_shape_with_no_interaction() -> None:
    with pytest.raises(SaValueError, match="interaction_scale"):
        simulate_factorial_groups(interaction_scale=0)


def test_an_unknown_shape_name_is_refused_rather_than_ignored() -> None:
    with pytest.raises(SaValueError, match="unknown shape"):
        simulate_factorial_groups(term_mix={"main_only": 1.0, "nonesuch": 1.0})
    with pytest.raises(SaValueError, match="unknown shape"):
        simulate_factorial_groups(pattern_mix={"all": 1.0, "crossover": 1.0})


def test_an_unusable_range_is_refused() -> None:
    for name in ("expr_range", "ref_sd", "cell_sd", "deg_log2fc", "subject_sd"):
        with pytest.raises(SaValueError, match=name):
            simulate_factorial_groups(**{name: (3.0, 1.0)})


def test_the_feature_prefix_has_to_be_a_name() -> None:
    for prefix in ("", None, 3, ["a"]):
        with pytest.raises(SaValueError, match="feat_prefix"):
            simulate_factorial_groups(feat_prefix=prefix)


# --------------------------------------------------------------------------- #
# The described family takes `args` as it stands
# --------------------------------------------------------------------------- #


def test_the_data_reaches_the_description_family_without_being_reshaped() -> None:
    """``args`` is named for the factorial comparison, which does not exist yet.

    What can be checked now is that the pieces it holds are the ones the described
    family already takes: a wide frame, a feature list, and one grouping vector
    per factor.
    """
    sim = simulate_factorial_groups(n_feats=8, n_up=2, n_down=2, n_per_cell=6, seed=1)
    data = sim.args["data"]
    feats = sim.args["feats"]

    ungrouped = summarize_descriptive_stats(data=data, feats=feats)
    assert len(ungrouped.index) == len(feats)

    grouped = summarize_descriptive_stats(
        data=data,
        feats=feats,
        group=sim.args["factors"]["treatment"],
        group_lv=sim.args["factor_lv"]["treatment"],
    )
    assert len(grouped.index) == len(feats) * len(sim.args["factor_lv"]["treatment"])

    diagnosed = diagnose_distribution(
        data=data,
        feats=feats,
        group=sim.args["factors"]["sex"],
        group_lv=sim.args["factor_lv"]["sex"],
    )
    assert len(diagnosed.normality.index) > 0
    assert set(diagnosed.normality["group"]) == set(sim.args["factor_lv"]["sex"])

    screened = screen_outliers(data=data, feats=feats)
    assert set(screened["features"]) <= set(feats)


def test_the_terms_the_answer_names_are_the_terms_an_anova_reports() -> None:
    """The simulator and the kernel enumerate the model separately, and must agree.

    A truth table that named its terms differently from the ANOVA table it is meant
    to score would merge on nothing, and the failure would read as a recall figure
    rather than as an error.
    """
    sim = simulate_factorial_groups(n_feats=3, n_per_cell=5, factor_lv=THREE_WAY, seed=1)
    plan = factorial_plan(sim.args["factor_lv"], fact_grid(sim.args["factor_lv"]))
    block = sim.truth_term[sim.truth_term["features"] == "prot_1"]

    assert plan.labels == list(block["terms"])
    assert plan.orders == list(block["term_order"])


def test_a_planted_effect_is_recovered_by_the_kernel_it_was_planted_for() -> None:
    """The simulation is scored against an ANOVA, which is what makes it useful.

    Not a claim about a rate, which would need far more features than a test should
    draw. The claim is the weaker and more useful one: a term that was planted is
    found more often than a term that was not, on the same features and the same
    fit. A sign error or a mislabelled term would put these the other way round.
    """
    sim = simulate_factorial_groups(
        n_feats=40, n_up=20, n_down=20, n_per_cell=12, deg_log2fc=(2.0, 3.0), seed=17
    )
    factor_lv = sim.args["factor_lv"]
    cells = fact_grid(factor_lv)
    labels = fact_cell_labels(factor_lv, cells)
    plan = factorial_plan(factor_lv, cells)

    cell_of = pd.Series(
        [
            ".".join(sim.args["factors"][name][row] for name in factor_lv)
            for row in range(len(sim.args["data"].index))
        ]
    )
    called: dict[tuple[str, str], list[bool]] = {}
    for feature in sim.args["feats"]:
        values = sim.args["data"][feature].to_numpy(dtype=float)
        fit = factorial_anova(
            {label: values[(cell_of == label).to_numpy()] for label in labels}, plan
        )
        planted = dict(
            zip(
                sim.truth_term.loc[sim.truth_term["features"] == feature, "terms"],
                sim.truth_term.loc[sim.truth_term["features"] == feature, "is_effect"],
                strict=True,
            )
        )
        for term, pval in zip(fit.terms.index, fit.terms["pval"], strict=True):
            called.setdefault((term, "planted" if planted[term] else "null"), []).append(
                bool(pval < 0.05)
            )

    for term in plan.labels:
        hit = np.mean(called[(term, "planted")])
        miss = np.mean(called[(term, "null")])
        assert hit > 0.5, f"{term}: planted effects found only {hit:.0%} of the time"
        assert miss < 0.2, f"{term}: null terms called {miss:.0%} of the time"
        assert hit > miss


def test_a_cell_of_the_design_is_one_group_the_description_family_can_take() -> None:
    """The cell labels are a grouping vector, so a per-cell description is a call."""
    sim = simulate_factorial_groups(n_feats=6, n_per_cell=4, seed=1)
    factors = sim.args["factors"]
    cell = [".".join(values) for values in zip(factors["treatment"], factors["sex"], strict=True)]
    cell_lv = list(dict.fromkeys(sim.truth_cell["treatment"] + "." + sim.truth_cell["sex"]))

    described = summarize_descriptive_stats(
        data=sim.args["data"], feats=sim.args["feats"], group=cell, group_lv=cell_lv
    )
    # Every cell holds what `truth_cell` says it holds.
    counts = described.groupby("group")["n"].first()
    block = sim.truth_cell[sim.truth_cell["features"] == "prot_1"]
    wanted = dict(zip(cell_lv, block["n"], strict=True))
    assert {name: int(value) for name, value in counts.items()} == wanted
