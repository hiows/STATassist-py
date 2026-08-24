"""``compare_factorial_groups`` against its result contract and planted truths."""

from __future__ import annotations

import numpy as np
import pytest

from statassist import compare_factorial_groups, simulate_factorial_groups
from statassist.core.errors import SaValueError
from statassist.core.result import SaFactorial


@pytest.fixture(scope="module")
def planted():
    sim = simulate_factorial_groups(n_feats=12, n_per_cell=8, seed=1)
    res = compare_factorial_groups(**sim.args, diagnose=False)
    return sim, res


class TestContract:
    def test_returns_a_factorial_result(self, planted):
        _, res = planted
        assert isinstance(res, SaFactorial)
        assert res.analysis == "factorial_comparison"

    def test_slots_match_the_factorial_contract(self, planted):
        _, res = planted
        assert list(res.tests) == ["anova_test"]
        assert "terms" in res and "cells" in res
        assert "pairwise" not in res
        assert "posthoc" in res

    def test_terms_carry_the_contract_columns(self, planted):
        _, res = planted
        for name in (
            "features",
            "terms",
            "term_order",
            "log2_effect",
            "pval",
            "pval_adj",
        ):
            assert name in res.terms.columns

    def test_cells_carry_factor_columns_and_means(self, planted):
        _, res = planted
        for name in ("features", "cell", "n", "mean", "sd", "se"):
            assert name in res.cells.columns
        for name in res.design["factor_lv"]:
            assert name in res.cells.columns

    def test_effect_names_cells_not_levels(self, planted):
        _, res = planted
        assert "extreme_cell" in res.effect.columns
        assert "extreme_level" not in res.effect.columns

    def test_sim_args_unpack_without_translation(self, planted):
        sim, res = planted
        assert list(res.features) == list(sim.args["feats"])

    def test_whole_model_df1_is_n_cells_minus_one(self, planted):
        _, res = planted
        row = res.tests["anova_test"].iloc[0]
        assert row["df1"] == len(res.design["group_lv"]) - 1
        assert np.isfinite(row["pval"])


class TestPlantedTruth:
    def test_extreme_cell_matches_truth(self, planted):
        sim, res = planted
        effect = res.effect.set_index("features")
        truth = sim.truth.set_index("features")
        planted_feats = truth.index[truth["direction"] != "none"]
        for feature in planted_feats:
            if truth.loc[feature, "extreme_tied"]:
                continue
            assert effect.loc[feature, "extreme_cell"] == truth.loc[feature, "extreme_cell"]

    def test_log2fc_sign_matches_truth_for_non_crossover(self, planted):
        sim, res = planted
        effect = res.effect.set_index("features")
        truth = sim.truth.set_index("features")
        for feature, row in truth.iterrows():
            if row["direction"] == "none" or row["pattern"] == "crossover":
                continue
            assert np.sign(effect.loc[feature, "log2fc"]) == np.sign(row["log2fc"])


class TestWithin:
    def test_within_is_refused(self):
        sim = simulate_factorial_groups(n_feats=4, n_per_cell=4, seed=2)
        with pytest.raises(SaValueError, match="within"):
            compare_factorial_groups(**sim.args, within=["treatment"], diagnose=False)


class TestPosthocOff:
    def test_posthoc_false_omits_the_slot(self):
        sim = simulate_factorial_groups(n_feats=4, n_per_cell=6, seed=3)
        res = compare_factorial_groups(**sim.args, posthoc=False, diagnose=False)
        assert "posthoc" not in res
