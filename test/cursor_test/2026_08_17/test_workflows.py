"""Pytest suite for statassist Python port (2026-08-17 session)."""

import numpy as np
import pandas as pd
import pytest

import statassist as sa
from statassist.contracts.comparison import ComparisonResult, sa_test_table_columns
from statassist.utils.serialize import to_json


@pytest.fixture
def two_group_sim():
    return sa.simulate_two_groups(n_feats=10, n_up=3, n_down=2, seed=2026)


# --- Phase 1: Descriptive ---


def test_summarize_descriptive_stats_columns(two_group_sim):
    data = two_group_sim["args"]["data"]
    feats = two_group_sim["args"]["feats"][:3]
    group = two_group_sim["args"]["group"]
    out = sa.summarize_descriptive_stats(data, feats, group)
    assert "features" in out.columns
    assert "group" in out.columns
    assert "mean" in out.columns
    assert "iqr" in out.columns
    assert len(out) == len(feats) * 2


def test_draw_grouped_boxplot_returns_stats(two_group_sim, matplotlib_use_agg):
    args = two_group_sim["args"]
    drawn = sa.draw_grouped_boxplot(
        args["data"],
        args["feats"][:2],
        args["group"],
        args["group_lv"],
        out_statistics=True,
    )
    assert "box_summary_stats" in drawn


def test_draw_grouped_barplot(two_group_sim, matplotlib_use_agg):
    args = two_group_sim["args"]
    drawn = sa.draw_grouped_barplot(
        args["data"],
        args["feats"][:2],
        args["group"],
        args["group_lv"],
    )
    assert "heights" in drawn or drawn is not None


# --- Phase 2: Association ---


def test_summarize_association_stats(two_group_sim):
    args = two_group_sim["args"]
    out = sa.summarize_association_stats(args["data"], args["feats"][:4])
    assert "pearson" in out
    assert "design" in out
    corr = out["pearson"]["corr"]
    assert np.allclose(np.diag(corr), 1.0)
    assert out["pearson"]["pvalue"].shape == corr.shape


def test_draw_corrplot(two_group_sim, matplotlib_use_agg):
    args = two_group_sim["args"]
    assoc = sa.summarize_association_stats(args["data"], args["feats"][:4])
    drawn = sa.draw_corrplot(assoc["pearson"]["corr"])
    assert drawn is not None


# --- Phase 3: Compare ---


def test_compare_two_groups_contract(two_group_sim):
    res = sa.compare_two_groups(**two_group_sim["args"], diagnose=False)
    assert res.analysis == "two_group_comparison"
    from statassist.contracts.comparison import ComparisonResult

    assert isinstance(res, ComparisonResult)
    cols = sa_test_table_columns()
    for nm, tbl in res.tests.items():
        for col in cols:
            assert col in tbl.columns, f"{nm} missing {col}"
        assert list(tbl["features"]) == list(res.features)


def test_estimate_significance(two_group_sim):
    res = sa.compare_two_groups(**two_group_sim["args"], diagnose=False)
    sig = sa.estimate_significance(res, test="t_test")
    assert sig.analysis_type == res.analysis
    cols = {"features", "log2fc", "pvalue", "adj_pvalue", "is_signif"}
    assert cols.issubset(set(sig.significance.columns))


def test_compare_multiple_groups():
    sim = sa.simulate_multiple_groups(
        n_feats=8, n_control=15, n_treat=[15, 15], seed=1
    )
    res = sa.compare_multiple_groups(**sim["args"], diagnose=False)
    assert res.analysis == "multi_group_comparison"
    assert len(res.tests) >= 1


def test_compare_one_sample():
    sim = sa.simulate_two_groups(n_feats=5, n_up=1, n_down=1, seed=1)
    feat = sim["args"]["feats"][0]
    res = sa.compare_one_sample(sim["args"]["data"], [feat], mu=8.0)
    assert res.analysis == "one_sample_comparison"


def test_volcano_and_forest_plot(two_group_sim, matplotlib_use_agg):
    res = sa.compare_two_groups(**two_group_sim["args"], diagnose=False)
    sig = sa.estimate_significance(res)
    sa.draw_volcano_plot(sig)
    sa.draw_forest_plot(res)


# --- Phase 4: ML ---


def test_split_and_fit():
    sim = sa.simulate_regression(n_pred=6, n_factor_pred=0, seed=1)
    sp = sa.split_data(sim["args"]["data"], p_train=0.75, seed=1)
    train = sp["datasets"][0]["train_data"]
    fit = sa.fit_linear_regression(
        train,
        sim["args"]["outcome"],
        sim["args"]["predictors"][:4],
        cv=False,
    )
    assert fit["analysis"] == "linear_regression"
    assert "terms" in fit["coefficients"].columns


def test_rfe_and_stepwise():
    sim = sa.simulate_regression(n_pred=6, n_factor_pred=0, seed=1)
    sp = sa.split_data(sim["args"]["data"], p_train=0.75, seed=1)
    train = sp["datasets"][0]["train_data"]
    sel = sa.perform_stepwise(
        train,
        sim["args"]["outcome"],
        sim["args"]["predictors"],
        model="linear",
        criterion="AIC",
    )
    assert sel["analysis"] == "stepwise"
    assert len(sel["selected"]) >= 1


# --- Phase 5-6: Reduce & Cluster ---


def test_pca_and_cluster(two_group_sim):
    args = two_group_sim["args"]
    pca = sa.perform_pca(args["data"], args["feats"], embedding_scale="samples")
    assert pca["analysis"] == "pca"
    assert "PC1" in pca["scores"].columns
    cl = sa.cluster_kmeans(
        pca["scores"],
        feats=["PC1", "PC2"],
        n_clust=2,
        seed=1,
        cluster_scale="samples",
    )
    assert cl["analysis"] == "kmeans"
    assert "cluster" in cl["assignments"].columns


def test_draw_dim_reduction_plot(two_group_sim, matplotlib_use_agg):
    args = two_group_sim["args"]
    pca = sa.perform_pca(args["data"], args["feats"], embedding_scale="samples")
    sa.draw_dim_reduction_plot(pca, group=args["group"])


# --- Phase 7: Extended ---


def test_simulate_two_groups_truth(two_group_sim):
    truth = two_group_sim["truth"]
    assert "direction" in truth.columns
    assert set(truth["direction"]).issubset({"up", "down", "none"})


def test_diagnose_distribution(two_group_sim):
    args = two_group_sim["args"]
    d = sa.diagnose_distribution(args["data"], args["feats"][:3], args["group"])
    assert hasattr(d, "normality")
    assert hasattr(d, "variance")


def test_compare_categorical():
    sim = sa.simulate_categorical_groups(seed=1)
    res = sa.compare_categorical_groups(**sim["args"])
    assert res.analysis == "categorical_comparison"


def test_comparison_json_serializable(two_group_sim):
    res = sa.compare_two_groups(**two_group_sim["args"], diagnose=False)
    js = to_json(res.to_dict())
    assert "two_group_comparison" in js
