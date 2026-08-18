"""Smoke tests for R-style print.sa_* parity via __repr__."""

from __future__ import annotations

import re

import statassist as sa


def test_repr_comparison_two_group():
    sim = sa.simulate_two_groups(n_feats=30, n_up=8, n_down=8, seed=2026)
    comp = sa.compare_two_groups(**sim["args"], diagnose=False)
    text = repr(comp)
    assert "<sa_two_group> two_group_comparison" in text
    assert "groups   : control vs case" in text
    assert "features : 30" in text
    assert "$t_test" in text
    assert re.search(r"\d+ of 30", text)


def test_repr_significance():
    sim = sa.simulate_two_groups(n_feats=30, n_up=8, n_down=8, seed=2026)
    comp = sa.compare_two_groups(**sim["args"], diagnose=False)
    sig = sa.estimate_significance(comp, test="t_test", log2fc_cutoff=1, pval_cutoff=0.05)
    text = repr(sig)
    assert "<sa_significance>" in text
    assert "two_group_comparison" in text


def test_repr_categorical_and_significance():
    sim = sa.simulate_categorical_groups(seed=2026)
    cat = sa.compare_categorical_groups(**sim["args"])
    text = repr(cat)
    assert "<sa_categorical>" in text

    sig_cell = sa.estimate_categorical_significance(cat, by="cell")
    assert "reading  : cell" in repr(sig_cell)

    sig_table = sa.estimate_categorical_significance(cat, by="table", test="chisq_test")
    assert "reading  : table" in repr(sig_table)
    assert "chisq_test" in repr(sig_table)


def test_repr_diagnosis():
    sim = sa.simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=2026)
    diag = sa.diagnose_distribution(
        sim["args"]["data"],
        sim["args"]["feats"][:5],
        sim["args"]["group"],
        sim["args"]["group_lv"],
    )
    assert "<sa_diagnosis>" in repr(diag)


def test_repr_model_selection_performance():
    sim = sa.simulate_regression(n_pred=6, n_factor_pred=0, seed=2026)
    fit = sa.fit_linear_regression(
        sim["args"]["data"],
        sim["args"]["outcome"],
        sim["args"]["predictors"][:4],
        cv=False,
    )
    assert "<sa_model>" in repr(fit)

    rfe = sa.perform_rfe(
        sim["args"]["data"],
        sim["args"]["outcome"],
        sim["args"]["predictors"],
        model="linear",
        seed=2026,
    )
    assert "<sa_selection>" in repr(rfe)

    sp = sa.split_data(sim["args"]["data"], p_train=0.7, seed=2026)
    test_data = sp["datasets"][0]["test_data"]
    eval_reg = sa.evaluate_regression_models(
        baseline_model=fit,
        newdata=test_data,
        answer=test_data[sim["args"]["outcome"]],
    )
    assert "<sa_performance>" in repr(eval_reg)


def test_repr_reduction_cluster_split():
    sim = sa.simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=2026)
    pca = sa.perform_pca(data=sim["args"]["data"], feats=sim["args"]["feats"])
    assert "<sa_reduction>" in repr(pca)

    clust = sa.cluster_kmeans(
        pca["scores"],
        feats=["PC1", "PC2"],
        n_clust=2,
        seed=2026,
        cluster_scale="samples",
    )
    assert "<sa_cluster>" in repr(clust)

    split = sa.split_data(sim["args"]["data"], p_train=0.7, seed=2026)
    assert "<sa_split>" in repr(split)
