#!/usr/bin/env python3
"""PNGs for README.md. Run from the statassist-py directory:

    py tools/render_readme_figures.py

Every figure the README links to is drawn here, from the same calls the README
quotes, so a figure cannot disagree with the text beside it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
sys.path.insert(0, str(ROOT / "src"))

import statassist as sa  # noqa: E402

plt.show = lambda *args, **kwargs: None  # headless save

DPI = 110
RED_COLS = ["#1B9E77", "#1B9E77", "#7570B3", "#7570B3", "#7570B3", "#E7298A", "#66A61E", "#66A61E"]


def figure(name: str, width: int, height: int, fn) -> None:
    """Run *fn*, save the current (or returned) figure, close all."""
    plt.close("all")
    result = fn()
    if isinstance(result, dict) and "fig" in result:
        fig = result["fig"]
    elif isinstance(result, plt.Figure):
        fig = result
    elif isinstance(result, plt.Axes):
        fig = result.figure
    else:
        fig = plt.gcf()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"README-{name}.png"
    fig.set_size_inches(width / DPI, height / DPI)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  {path}")


def as_predictors(terms: list[str]) -> list[str]:
    out = []
    for t in terms:
        m = re.match(r"^Q\('(.+)'\)$", str(t))
        if m:
            t = m.group(1)
        t = re.sub(r"high$", "", t)
        t = re.sub(r"mid$", "", t)
        if t and t not in out:
            out.append(t)
    return out


def draw_fit_scatter(y, y_hat, main: str) -> None:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    lim = (min(y.min(), y_hat.min()), max(y.max(), y_hat.max()))
    corr = float(np.corrcoef(y, y_hat)[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(lim, lim, color="gray", lw=2, ls=":")
    coef = np.polyfit(y, y_hat, 1)
    xs = np.array(lim)
    ys = coef[0] * xs + coef[1]
    ax.plot(xs, ys, color="red", lw=2)
    ax.scatter(y, y_hat, s=20)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("y (test data)")
    ax.set_ylabel("y_hat")
    ax.set_title(f"{main}\nCorr = {corr:.3f}")
    intercept = coef[1]
    slope = coef[0]
    sign = "- " if intercept < 0 else "+ "
    ax.legend(
        [f"y = {slope:.3f}x {sign}{abs(intercept):.3f}"],
        loc="lower right",
        frameon=False,
    )


def draw_roc_multi(response, probs: dict, main: str) -> None:
    levels = ("control", "case")
    y_bin = (np.asarray(response) == levels[1]).astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [1, 0], color="gray", lw=2, ls=":")
    cols = ["black", "red", "gray"]
    ltys = ["--", "-", "-"]
    labels = []
    for i, (name, prob) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y_bin, np.asarray(prob), pos_label=1)
        auc_val = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            ls=ltys[i % len(ltys)],
            color=cols[i % len(cols)],
        )
        labels.append(f"{name} ({auc_val:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(main)
    ax.legend(labels, loc="lower right", frameon=False)


def label_plot(x, y, labels, xlab, ylab, main) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, c=RED_COLS[: len(x)], s=80)
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi, yi, lab, fontsize=9, fontweight="bold", ha="center", va="bottom")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main)


def draw_factorial_boxplot(
    data: pd.DataFrame,
    feat: str,
    factor_lv: dict[str, list[str]],
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Ad-hoc crossed boxplot until utils.factorial is ported."""
    fig, ax = plt.subplots(figsize=(8, 6))
    primary, secondary = list(factor_lv.keys())[:2]
    p_lv, s_lv = factor_lv[primary], factor_lv[secondary]
    width = 0.8 / len(s_lv)
    offsets = np.linspace(-(len(s_lv) - 1) / 2, (len(s_lv) - 1) / 2, len(s_lv)) * width
    colors = ["#4E79A7", "#F28E2B"]
    for j, (sec, off) in enumerate(zip(s_lv, offsets)):
        vals = [
            data.loc[(data[primary] == p) & (data[secondary] == sec), feat].to_numpy()
            for p in p_lv
        ]
        pos = np.arange(len(p_lv)) + off
        bp = ax.boxplot(vals, positions=pos, widths=width * 0.9, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[j % len(colors)])
        ax.plot([], [], color=colors[j % len(colors)], label=sec)
    ax.set_xticks(np.arange(len(p_lv)))
    ax.set_xticklabels(p_lv)
    ax.set_xlabel(primary)
    ax.set_ylabel(feat)
    ax.legend(title=secondary, loc="upper right", frameon=False)
    if ylim:
        ax.set_ylim(ylim)


CV = dict(cv=False, seed=2026)
LASSO = dict(penalty="lasso", lambda_=[0.1])
SVM_GRID = dict(C=[0.5], sigma=[0.05])


def top_five(model: dict) -> list[str]:
    imp = sa.coef(model)
    terms = imp.sort_values("estimate", ascending=False)["terms"].head(5).tolist()
    return as_predictors(terms)


def main() -> None:
    print("=== 1. two groups ===")
    sim2 = sa.simulate_two_groups(n_feats=30, n_up=8, n_down=8, seed=2026)
    args2 = sim2["args"]
    comp2 = sa.compare_two_groups(
        data=args2["data"],
        feats=args2["feats"],
        group=args2["group"],
        group_lv=args2["group_lv"],
        input_scale=args2["input_scale"],
        diagnose=False,
    )
    sig2 = sa.estimate_significance(
        comp2,
        test="t_test",
        log2fc_cutoff=1,
        pval_cutoff=0.05,
        adj_type="BH",
    )
    first_ten = [f"gene_{i}" for i in range(1, 11)]

    figure("volcano", 700, 650, lambda: sa.draw_volcano_plot(sig2, xlim=(-3, 3)))
    figure(
        "boxplot",
        1200,
        620,
        lambda: sa.draw_grouped_boxplot(
            data=args2["data"],
            feats=first_ten,
            group=args2["group"],
            group_lv=args2["group_lv"],
            ylim=(-5, 20),
        ),
    )
    figure(
        "butterfly",
        800,
        650,
        lambda: sa.draw_butterfly_hist(
            data=args2["data"],
            feat="gene_8",
            group=args2["group"],
            group_lv=args2["group_lv"],
            bins=25,
        ),
    )
    figure(
        "forest-pvalue",
        900,
        620,
        lambda: sa.draw_forest_plot(
            comp2,
            test="t_test",
            type="pvalue",
            feats=first_ten,
            sort_by="pvalue",
        ),
    )
    figure(
        "forest-estimate",
        900,
        620,
        lambda: sa.draw_forest_plot(
            comp2,
            test="t_test",
            type="estimate",
            feats=first_ten,
            sort_by="pvalue",
            xlim=(-6, 6),
        ),
    )
    figure(
        "heatmap",
        900,
        770,
        lambda: sa.draw_heatmap(
            data=args2["data"],
            group=args2["group"],
            group_lv=args2["group_lv"],
            hclust_method="ward.D2",
            show_sample_names=False,
        ),
    )

    print("=== 2. three or more groups ===")
    sim_n = sa.simulate_multiple_groups(
        n_feats=10,
        n_control=50,
        n_treat=[50, 50, 50],
        group_lv=["control", "treat_1", "treat_2", "treat_3"],
        seed=2026,
    )
    args_n = sim_n["args"]
    comp_n = sa.compare_multiple_groups(
        data=args_n["data"],
        feats=args_n["feats"],
        group=args_n["group"],
        group_lv=args_n["group_lv"],
        id=args_n.get("id"),
        input_scale=args_n["input_scale"],
        paired=False,
        diagnose=False,
    )
    sig_n = sa.estimate_significance(
        comp_n, test="anova_test", pval_cutoff=0.05, adj_type="BH"
    )
    figure("multi-volcano", 700, 650, lambda: sa.draw_volcano_plot(sig_n, xlim=(-4, 4)))
    figure(
        "multi-boxplot",
        1200,
        620,
        lambda: sa.draw_grouped_boxplot(
            data=args_n["data"],
            feats=args_n["feats"],
            group=args_n["group"],
            group_lv=args_n["group_lv"],
            ylim=(-10, 20),
        ),
    )
    figure(
        "multi-posthoc",
        900,
        520,
        lambda: sa.draw_forest_plot(
            comp_n,
            test="anova_test",
            type="posthoc",
            feats=["gene_1"],
            sort_by="pvalue",
        ),
    )

    print("=== 3. supervised learning: the shared data ===")
    cor_mat = sa.make_block_cor(
        n_features=8,
        blocks=[
            {"features": [1, 2], "cor": 0.8},
            {"features": [3, 4, 5], "cor": 0.5},
        ],
    )
    beta = [0.0, 1.2, 0.0, 0.0, 0.54, 0.0, -1.79, -0.88]
    sim_reg = sa.simulate_regression(
        n_samples=200, n_pred=8, beta=beta, cor_mat=cor_mat, seed=2026
    )
    reg_args = sim_reg["args"]
    reg_split = sa.split_data(
        data=reg_args["data"],
        p_train=0.75,
        times=1,
        seed=2026,
    )
    reg_train = reg_split["datasets"][0]["train_data"]
    reg_test = reg_split["datasets"][0]["test_data"]

    sim_cls = sa.simulate_classification(n_samples=200, n_pred=8, n_pos=2, n_neg=2, seed=2026)
    cls_args = sim_cls["args"]
    cls_split = sa.split_data(
        data=cls_args["data"],
        stratified=cls_args["data"]["y"],
        p_train=0.75,
        times=1,
        seed=2026,
    )
    cls_train = cls_split["datasets"][0]["train_data"]
    cls_test = cls_split["datasets"][0]["test_data"]

    print("=== 4. linear and logistic regression ===")
    lin_all = sa.fit_linear_regression(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=reg_args["predictors"],
        **CV,
    )
    lin_keep = as_predictors(
        sa.coef(lin_all)["terms"].iloc[1:][sa.coef(lin_all)["pval"].iloc[1:] < 0.01].tolist()
    )
    lin = sa.fit_linear_regression(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=lin_keep,
        **CV,
    )
    figure(
        "linear-regression",
        700,
        650,
        lambda: draw_fit_scatter(
            reg_test["y"], sa.predict(lin, newdata=reg_test), "Linear regression"
        ),
    )

    log_all = sa.fit_logistic_regression(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=cls_args["predictors"],
        outcome_lv=cls_args["outcome_lv"],
        **CV,
    )
    log_coef = sa.coef(log_all)
    log_terms = log_coef["terms"].iloc[1:]
    log_signif = as_predictors(log_terms[log_coef["pval"].iloc[1:] < 0.05].tolist())
    log_null = as_predictors(log_terms[log_coef["pval"].iloc[1:] >= 0.05].tolist())
    if not log_signif:
        log_signif = [p for p in cls_args["predictors"] if p in sim_cls["truth"].loc[sim_cls["truth"]["role"] == "signal", "predictors"].tolist()][:3]
    if not log_null:
        log_null = [p for p in cls_args["predictors"] if p not in log_signif][:2]
    log_signif_fit = sa.fit_logistic_regression(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=log_signif,
        outcome_lv=cls_args["outcome_lv"],
        **CV,
    )
    log_null_fit = sa.fit_logistic_regression(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=log_null,
        outcome_lv=cls_args["outcome_lv"],
        **CV,
    )
    figure(
        "logistic-regression",
        700,
        650,
        lambda: draw_roc_multi(
            cls_test["y"],
            {
                "All features": sa.predict(log_all, newdata=cls_test, type="response"),
                "Selected features": sa.predict(log_signif_fit, newdata=cls_test, type="response"),
                "Not significant features": sa.predict(
                    log_null_fit, newdata=cls_test, type="response"
                ),
            },
            "Logistic regression",
        ),
    )

    print("=== 5. elastic net ===")
    enet_reg_all = sa.fit_elastic_net(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=reg_args["predictors"],
        **LASSO,
        **CV,
    )
    enet_reg_keep = as_predictors(
        [
            t
            for t, sel in zip(
                sa.coef(enet_reg_all)["terms"].iloc[1:],
                sa.coef(enet_reg_all)["selected"].iloc[1:],
            )
            if sel
        ]
    )
    enet_reg = sa.fit_elastic_net(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=enet_reg_keep,
        **LASSO,
        **CV,
    )
    figure(
        "elastic-net-regression",
        700,
        650,
        lambda: draw_fit_scatter(
            reg_test["y"], sa.predict(enet_reg, newdata=reg_test), "Elastic net (LASSO)"
        ),
    )

    enet_cls_all = sa.fit_elastic_net(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=cls_args["predictors"],
        outcome_lv=cls_args["outcome_lv"],
        **LASSO,
        **CV,
    )
    enet_cls_keep = as_predictors(
        [
            t
            for t, sel in zip(
                sa.coef(enet_cls_all)["terms"].iloc[1:],
                sa.coef(enet_cls_all)["selected"].iloc[1:],
            )
            if sel
        ]
    )
    enet_cls = sa.fit_elastic_net(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=enet_cls_keep,
        outcome_lv=cls_args["outcome_lv"],
        **LASSO,
        **CV,
    )
    enet_cls_dropped = [p for p in cls_args["predictors"] if p not in enet_cls_keep]
    enet_cls_null = sa.fit_logistic_regression(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=enet_cls_dropped,
        outcome_lv=cls_args["outcome_lv"],
        **CV,
    )
    figure(
        "elastic-net-roc",
        700,
        650,
        lambda: draw_roc_multi(
            cls_test["y"],
            {
                "All features": sa.predict(enet_cls_all, newdata=cls_test, type="response"),
                "Kept features": sa.predict(enet_cls, newdata=cls_test, type="response"),
                "Dropped features": sa.predict(enet_cls_null, newdata=cls_test, type="response"),
            },
            "Elastic net (LASSO)",
        ),
    )

    print("=== 6. random forest ===")
    rf_reg_all = sa.fit_rf(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=reg_args["predictors"],
        mtry=[5],
        ntree=500,
        **CV,
    )
    rf_reg = sa.fit_rf(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=top_five(rf_reg_all),
        ntree=500,
        **CV,
    )
    figure(
        "random-forest-regression",
        700,
        650,
        lambda: draw_fit_scatter(
            reg_test["y"], sa.predict(rf_reg, newdata=reg_test), "Random forest"
        ),
    )

    rf_cls_all = sa.fit_rf(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=cls_args["predictors"],
        outcome_lv=cls_args["outcome_lv"],
        mtry=[5],
        ntree=500,
        **CV,
    )
    rf_cls_keep = top_five(rf_cls_all)
    rf_cls = sa.fit_rf(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=rf_cls_keep,
        outcome_lv=cls_args["outcome_lv"],
        ntree=500,
        **CV,
    )
    rf_cls_rest = sa.fit_rf(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=[p for p in cls_args["predictors"] if p not in rf_cls_keep],
        outcome_lv=cls_args["outcome_lv"],
        ntree=500,
        **CV,
    )
    figure(
        "random-forest-roc",
        700,
        650,
        lambda: draw_roc_multi(
            cls_test["y"],
            {
                "All features": sa.predict(rf_cls_all, newdata=cls_test, type="response"),
                "Top importance": sa.predict(rf_cls, newdata=cls_test, type="response"),
                "Low importance features": sa.predict(rf_cls_rest, newdata=cls_test, type="response"),
            },
            "Random forest",
        ),
    )

    print("=== 7. support vector machine ===")
    svm_reg_all = sa.fit_svm(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=reg_args["predictors"],
        **SVM_GRID,
        **CV,
    )
    svm_reg = sa.fit_svm(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=top_five(svm_reg_all),
        **SVM_GRID,
        **CV,
    )
    figure(
        "svm-regression",
        700,
        650,
        lambda: draw_fit_scatter(
            reg_test["y"], sa.predict(svm_reg, newdata=reg_test), "Support vector machine"
        ),
    )

    svm_cls_all = sa.fit_svm(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=cls_args["predictors"],
        outcome_lv=cls_args["outcome_lv"],
        **SVM_GRID,
        **CV,
    )
    svm_cls_keep = top_five(svm_cls_all)
    svm_cls = sa.fit_svm(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=svm_cls_keep,
        outcome_lv=cls_args["outcome_lv"],
        **SVM_GRID,
        **CV,
    )
    svm_cls_rest = sa.fit_svm(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=[p for p in cls_args["predictors"] if p not in svm_cls_keep],
        outcome_lv=cls_args["outcome_lv"],
        **SVM_GRID,
        **CV,
    )
    figure(
        "svm-roc",
        700,
        650,
        lambda: draw_roc_multi(
            cls_test["y"],
            {
                "All features": sa.predict(svm_cls_all, newdata=cls_test, type="response"),
                "Top importance": sa.predict(svm_cls, newdata=cls_test, type="response"),
                "Low importance features": sa.predict(
                    svm_cls_rest, newdata=cls_test, type="response"
                ),
            },
            "Support vector machine",
        ),
    )

    print("=== 8. dimension reduction ===")
    red_cor = sa.make_block_cor(
        n_features=8,
        blocks=[
            {"features": [1, 2], "cor": 0.8},
            {"features": [3, 4, 5], "cor": 0.5},
            {"features": [7, 8], "cor": 0.9},
        ],
    )
    red_data = sa.simulate_regression(
        n_samples=200, n_pred=8, cor_mat=red_cor, beta=[0.0] * 8, intercept=0, seed=2026
    )["args"]["data"]
    red_feats = [f"x_{i}" for i in range(1, 9)]

    pca = sa.perform_pca(
        data=red_data,
        feats=red_feats,
        embedding_scale="features",
        center=True,
        scale=True,
    )
    figure(
        "pca",
        700,
        650,
        lambda: label_plot(
            pca["scores"]["PC1"],
            pca["scores"]["PC2"],
            pca["scores"]["points"],
            f"PC1 ({pca['variance']['prop_var'].iloc[0]:.2f}%)",
            f"PC2 ({pca['variance']['prop_var'].iloc[1]:.2f}%)",
            "PCA of the features",
        ),
    )

    tsne = sa.perform_tsne(
        data=red_data,
        feats=red_feats,
        embedding_scale="features",
        center=True,
        scale=True,
        seed=2026,
    )
    tsne_cols = [c for c in tsne["scores"].columns if c != "points"]
    figure(
        "tsne",
        700,
        650,
        lambda: label_plot(
            tsne["scores"][tsne_cols[0]],
            tsne["scores"][tsne_cols[1]],
            tsne["scores"]["points"],
            tsne_cols[0],
            tsne_cols[1],
            "t-SNE of the features",
        ),
    )

    umap_res = sa.perform_umap(
        data=red_data,
        feats=red_feats,
        embedding_scale="features",
        n_neighbors=3,
        center=False,
        scale=False,
        seed=2026,
    )
    umap_cols = [c for c in umap_res["scores"].columns if c != "points"]
    figure(
        "umap",
        700,
        650,
        lambda: label_plot(
            umap_res["scores"][umap_cols[0]],
            umap_res["scores"][umap_cols[1]],
            umap_res["scores"]["points"],
            umap_cols[0],
            umap_cols[1],
            "UMAP of the features",
        ),
    )

    print("=== 9. factorial crossed design ===")
    fact_feats = [f"prot_{i}" for i in range(1, 101)]
    sim_fact = sa.simulate_factorial_groups(
        seed=2026,
        n_per_cell=20,
        n_feats=100,
        feat_prefix="prot",
        factor_lv={"treatment": ["control", "A", "B", "C"], "sex": ["male", "female"]},
    )
    fact_args = sim_fact["args"]
    fact_comp = sa.compare_factorial_groups(
        data=fact_args["data"],
        feats=fact_args["feats"],
        factors=fact_args["factors"],
        factor_lv=fact_args["factor_lv"],
        control_label={"treatment": "control", "sex": "male"},
        input_scale="raw",
        diagnose=False,
    )
    sig_fact_term = sa.estimate_significance(fact_comp, by="term")
    interest_feats = [f"prot_{i}" for i in range(1, 21)]
    interest_feat = "prot_14"

    figure(
        "factorial-forest-pvalue",
        900,
        520,
        lambda: sa.draw_forest_plot(
            fact_comp,
            type="pvalue",
            feats=interest_feats,
            sort_by="pvalue",
        ),
    )
    figure("factorial-volcano", 900, 650, lambda: sa.draw_volcano_plot(sig_fact_term))
    figure(
        "factorial-forest-estimate",
        900,
        520,
        lambda: sa.draw_forest_plot(
            fact_comp,
            feats=[interest_feat],
            sort_by="pvalue",
            xlim=(-6, 6),
        ),
    )
    figure(
        "factorial-interaction",
        800,
        620,
        lambda: sa.draw_interaction_plot(fact_comp, feat=interest_feat, factor="treatment"),
    )
    figure(
        "factorial-boxplot",
        800,
        620,
        lambda: draw_factorial_boxplot(
            data=fact_args["data"],
            feat=interest_feat,
            factor_lv=fact_args["factor_lv"],
            ylim=(5, 25),
        ),
    )

    print("=== 10. categorical contingency table ===")
    sim_cat = sa.simulate_categorical_groups(seed=2026)
    cat_args = sim_cat["args"]
    cat_comp = sa.compare_categorical_groups(
        data=cat_args["data"],
        category_lv=cat_args["category_lv"],
        control_label={"smoker": "n", "grade": "mid"},
        paired=False,
        diagnose=False,
    )
    figure("mosaic", 800, 620, lambda: sa.draw_mosaic_plot(cat_comp))

    print("=== 11. grouped barplot ===")
    sim_bar = sa.simulate_two_groups(n_feats=10, n_up=3, n_down=3, seed=2026)
    bar_args = sim_bar["args"]
    figure(
        "grouped-barplot",
        900,
        620,
        lambda: sa.draw_grouped_barplot(
            data=bar_args["data"],
            feats=bar_args["feats"],
            group=bar_args["group"],
            group_lv=bar_args["group_lv"],
            control_label="control",
            errorbar="se",
        ),
    )

    print("=== 12. feature-pair association ===")
    assoc_cor = sa.make_block_cor(
        n_features=10,
        blocks=[
            {"features": list(range(1, 4)), "cor": 0.9},
            {"features": [4, 5], "cor": 0.5, "against": [6, 7]},
        ],
    )
    assoc_sim = sa.simulate_regression(n_pred=10, cor_mat=assoc_cor, seed=2026)
    assoc_data = assoc_sim["args"]["data"].iloc[:, 1:]
    assoc_feats = list(assoc_data.columns)
    assoc = sa.summarize_association_stats(data=assoc_data, feats=assoc_feats)
    figure("corrplot", 700, 700, lambda: sa.draw_corrplot(assoc["pearson"]["corr"]))
    figure(
        "corrplot-masked",
        700,
        700,
        lambda: sa.draw_corrplot(
            assoc["pearson"]["corr"],
            pvalue=assoc["pearson"]["adj_pvalue"],
        ),
    )

    print("=== 13. evaluate regression models ===")
    eval_rfe = sa.perform_rfe(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=reg_args["predictors"],
        seed=2026,
    )
    eval_sel = eval_rfe["selected"]
    eval_lin = sa.fit_linear_regression(
        data=reg_train, outcome=reg_args["outcome"], predictors=eval_sel, **CV
    )
    eval_lasso = sa.fit_elastic_net(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=eval_sel,
        **LASSO,
        **CV,
    )
    eval_rf = sa.fit_rf(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=eval_sel,
        mtry=[min(5, len(eval_sel))],
        **CV,
    )
    eval_svm = sa.fit_svm(
        data=reg_train,
        outcome=reg_args["outcome"],
        predictors=eval_sel,
        **SVM_GRID,
        **CV,
    )
    eval_reg = sa.evaluate_regression_models(
        baseline_model=eval_lin,
        new_models={"lasso": eval_lasso, "rf": eval_rf, "svm": eval_svm},
        newdata=reg_test,
        answer=reg_test["y"],
        baseline_label="linear",
    )
    figure(
        "eval-regression",
        800,
        650,
        lambda: sa.draw_prediction_plot(eval_reg, type="overlay", points=False),
    )

    print("=== 14. evaluate classification models ===")
    eval_rfe_cls = sa.perform_rfe(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=cls_args["predictors"],
        outcome_lv=cls_args["outcome_lv"],
        control_label="control",
        seed=2026,
        model="logistic",
    )
    eval_sel_cls = eval_rfe_cls["selected"]
    eval_log = sa.fit_logistic_regression(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=eval_sel_cls,
        outcome_lv=cls_args["outcome_lv"],
        control_label="control",
        **CV,
    )
    eval_lasso_cls = sa.fit_elastic_net(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=eval_sel_cls,
        outcome_lv=cls_args["outcome_lv"],
        **LASSO,
        **CV,
    )
    eval_rf_cls = sa.fit_rf(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=eval_sel_cls,
        outcome_lv=cls_args["outcome_lv"],
        mtry=[min(5, len(eval_sel_cls))],
        **CV,
    )
    eval_svm_cls = sa.fit_svm(
        data=cls_train,
        outcome=cls_args["outcome"],
        predictors=eval_sel_cls,
        outcome_lv=cls_args["outcome_lv"],
        **SVM_GRID,
        **CV,
    )
    eval_cls = sa.evaluate_classification_models(
        baseline_model=eval_log,
        new_models={"lasso": eval_lasso_cls, "rf": eval_rf_cls, "svm": eval_svm_cls},
        newdata=cls_test,
        answer=cls_test["y"],
        outcome_lv=cls_args["outcome_lv"],
        control_label="control",
        baseline_label="logistic",
    )
    figure(
        "eval-classification",
        650,
        650,
        lambda: sa.draw_roc_curve(eval_cls, anno_auc=True),
    )

    print("=== 15. cluster on an embedding ===")
    clust_sim = sa.simulate_two_groups(n_feats=50, deg_log2fc=(5, 10), seed=2026)
    clust_args = clust_sim["args"]
    clust_pca = sa.perform_pca(
        data=clust_args["data"],
        feats=clust_args["feats"],
        embedding_scale="samples",
    )
    clust_km_pca = sa.cluster_kmeans(
        data=clust_pca["scores"],
        feats=["PC1", "PC2"],
        cluster_scale="samples",
        n_clust=2,
        seed=2026,
    )
    figure(
        "cluster-pca-group",
        700,
        650,
        lambda: sa.draw_dim_reduction_plot(
            clust_pca,
            group=clust_args["group"],
        ),
    )
    figure(
        "cluster-pca-cluster",
        700,
        650,
        lambda: sa.draw_dim_reduction_plot(
            clust_pca,
            cluster_result=clust_km_pca,
        ),
    )

    clust_umap = sa.perform_umap(
        data=clust_args["data"],
        feats=clust_args["feats"],
        embedding_scale="samples",
        seed=2026,
    )
    umap_score_cols = [c for c in clust_umap["scores"].columns if c != "points"][:2]
    clust_km_umap = sa.cluster_kmeans(
        data=clust_umap["scores"],
        feats=umap_score_cols,
        cluster_scale="samples",
        n_clust=2,
        seed=2026,
    )
    figure(
        "cluster-umap-group",
        700,
        650,
        lambda: sa.draw_dim_reduction_plot(
            clust_umap,
            group=clust_args["group"],
        ),
    )
    figure(
        "cluster-umap-cluster",
        700,
        650,
        lambda: sa.draw_dim_reduction_plot(
            clust_umap,
            cluster_result=clust_km_umap,
        ),
    )

    n_png = len(list(FIG_DIR.glob("README-*.png")))
    print(f"\nWrote {n_png} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
