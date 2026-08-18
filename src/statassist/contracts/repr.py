"""R ``print.sa_*`` summaries for Python result contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statassist.contracts.repr_fmt import (
    sa_cat_field,
    sa_class_tag,
    sa_fmt_est,
    sa_fmt_num,
    sa_fmt_pval,
    sa_left,
    sa_signif_count,
    sa_verdict_count,
)


def repr_sa_comparison(obj: Any, alpha: float = 0.05) -> str:
    design = obj.design if hasattr(obj, "design") else obj["design"]
    params = obj.parameters if hasattr(obj, "parameters") else obj["parameters"]
    features = obj.features if hasattr(obj, "features") else obj["features"]
    tests = obj.tests if hasattr(obj, "tests") else obj["tests"]
    test_info = obj.test_info if hasattr(obj, "test_info") else obj["test_info"]
    terms = getattr(obj, "terms", None) if hasattr(obj, "terms") else obj.get("terms")
    posthoc = getattr(obj, "posthoc", None) if hasattr(obj, "posthoc") else obj.get("posthoc")
    diagnostics = getattr(obj, "diagnostics", None) if hasattr(obj, "diagnostics") else obj.get(
        "diagnostics"
    )
    analysis = obj.analysis if hasattr(obj, "analysis") else obj["analysis"]

    lines = [f"<{sa_class_tag(obj)}> {analysis}"]
    if design.get("factor_lv"):
        parts = [f"{nm} ({len(lv)})" for nm, lv in design["factor_lv"].items()]
        lines.append(
            f"  factors  : {' x '.join(parts)}  ({len(design['group_lv'])} cells, independent)"
        )
        anova_type = str(design.get("anova_type", "factorial")).replace("_", "-")
        lines.append(f"  anova    : {anova_type}, Type {params.get('ss_type', 'III')} sums of squares")
    elif design.get("group_lv") is None:
        lines.append(f"  mu       : {design.get('mu')}")
    else:
        paired = " (paired)" if design.get("paired") else " (independent)"
        if design.get("paired") and design.get("pairing"):
            paired = f" (paired by {design['pairing']})"
        lines.append(f"  groups   : {' vs '.join(design['group_lv'])}{paired}")
    lines.append(f"  features : {len(features)}")
    lines.append(
        "  settings : "
        f"alternative = {params.get('alternative')}, "
        f"conf_level = {params.get('conf_level')}, "
        f"p_adjust = {params.get('p_adjust')}"
    )

    lines.append("\n  tests")
    width = max((len(nm) for nm in tests), default=0)
    for nm, tbl in tests.items():
        n_signif, n_total, n_failed = sa_signif_count(tbl, alpha)
        failed = f"  ({n_failed} not computed)" if n_failed else ""
        lines.append(
            f"    ${sa_left(nm, width)}  {n_signif} of {n_total} at pval_adj <= {alpha}{failed}"
        )
        lines.append(f"    {' ' * (width + 2)}{test_info[nm]['label']}")
        if posthoc and nm in posthoc and len(posthoc[nm]):
            ph = posthoc[nm]
            n_pairs = int((ph["pval_adj"].notna() & (ph["pval_adj"] <= alpha)).sum())
            n_feats = ph["features"].nunique()
            label = test_info[nm].get("posthoc_label", "")
            lines.append(
                f"    {' ' * (width + 2)}post-hoc: {n_pairs} of {len(ph)} contrast(s) "
                f"over {n_feats} feature(s), {label}"
            )

    if terms is not None and len(terms):
        lines.append("\n  terms")
        labels = terms["terms"].drop_duplicates().tolist()
        tw = max(len(str(x)) for x in labels)
        for nm in labels:
            rows = terms.loc[terms["terms"] == nm]
            n_signif = int((rows["pval_adj"].notna() & (rows["pval_adj"] <= alpha)).sum())
            lines.append(f"    {sa_left(str(nm), tw)}  {n_signif} of {len(rows)} at pval_adj <= {alpha}")

    if diagnostics is not None:
        lines.append("\n  $diagnostics attached")
    unmatched = design.get("unmatched_ids") or []
    if unmatched:
        lines.append(f"\n  dropped  : {len(unmatched)} unpaired id(s)")
    if design.get("n_dropped", 0):
        lines.append(f"  dropped  : {design['n_dropped']} row(s) outside `group_lv`")
    return "\n".join(lines)


def repr_sa_significance(obj: Any) -> str:
    sig = obj.significance
    if isinstance(sig, pd.DataFrame):
        tables = {"": sig}
        head = sig
    else:
        tables = sig
        head = next(iter(tables.values()))

    attrs = head.attrs
    lines = [f"<{sa_class_tag(obj)}> {obj.analysis_type}"]
    lines.append(f"  test     : {attrs.get('test')}  ({attrs.get('test_label')})")
    adj = attrs.get("adj_type", "none")
    lines.append(
        "  cutoffs  : "
        f"abs(log2fc) >= {attrs.get('log2fc_cutoff')}, "
        f"adj_pvalue <= {attrs.get('pval_cutoff')}  ({adj})"
    )
    if isinstance(sig, pd.DataFrame):
        lines.append(f"  verdict  : {sa_verdict_count(sig)}")
    else:
        axis = "term" if attrs.get("term") is not None else "contrast"
        lines.append(f"\n  $significance, one table per {axis}")
        width = max(len(nm) for nm in tables)
        for nm, tbl in tables.items():
            lines.append(f"    {sa_left(nm, width)}  {sa_verdict_count(tbl)}")
    return "\n".join(lines)


def repr_sa_categorical(obj: Any, alpha: float = 0.05) -> str:
    from statassist.contracts.categorical import sa_null_label

    design = obj.design
    params = obj.parameters
    lines = [f"<{sa_class_tag(obj)}> {obj.analysis}"]
    paired = (
        f"matched by {design['pairing']}"
        if design.get("paired")
        else "independent"
    )
    lines.append(
        f"  table    : {design['row_var']} ({design['dim'][0]}) x "
        f"{design['col_var']} ({design['dim'][1]})  "
        f"({design['dim'][0] * design['dim'][1]} cells, {paired})"
    )
    lines.append(f"  null     : {sa_null_label(design['null'])}")
    lines.append(f"  observed : {design['n_used']} row(s)")
    sim = ""
    if params.get("simulate_p_value"):
        sim = f", simulated on {params.get('n_resamples')} resample(s)"
    lines.append(
        f"  settings : conf_level = {params.get('conf_level')}, "
        f"correct = {params.get('correct')}{sim}"
    )

    lines.append("\n  tests")
    width = max(len(nm) for nm in obj.tests)
    for nm, tbl in obj.tests.items():
        row = tbl.iloc[0]
        pval = row["pval"]
        if pd.isna(pval):
            verdict = "not computed"
        elif pval <= alpha:
            verdict = f"null rejected at {alpha}"
        else:
            verdict = f"null retained at {alpha}"
        lines.append(
            f"    ${sa_left(nm, width)}  pval = {sa_fmt_pval(pval)}  ({verdict})"
        )
        lines.append(f"    {' ' * (width + 2)}{obj.test_info[nm]['label']}")

    lines.append("\n  association")
    aw = max(len(str(x)) for x in obj.association["measure"])
    for _, row in obj.association.iterrows():
        interval = ""
        if pd.notna(row.get("lower_conf")):
            interval = f"  [{sa_fmt_est(row['lower_conf'])}, {sa_fmt_est(row['upper_conf'])}]"
        lines.append(
            f"    {sa_left(str(row['measure']), aw)}  {sa_fmt_est(row['estimate'])}{interval}"
        )
    return "\n".join(lines)


def repr_sa_categorical_significance(tbl: pd.DataFrame) -> str:
    from statassist.contracts.categorical import sa_null_label

    attrs = tbl.attrs
    by = attrs.get("by", "cell")
    dims = attrs.get("table_dim", [])
    analysis = attrs.get("analysis", "categorical_comparison")
    lines = [f"<sa_categorical_significance> {analysis}"]
    lines.append(f"  reading  : {by}  ({' x '.join(str(d) for d in dims)} table)")
    lines.append(f"  null     : {sa_null_label(attrs.get('null', ''))}")
    if by == "cell":
        lines.append(
            "  cutoffs  : "
            f"abs(log2_lift) >= {attrs.get('log2_lift_cutoff', attrs.get('ratio_cutoff'))}, "
            f"adj_pvalue <= {attrs.get('pval_cutoff')}  ({attrs.get('adj_type')})"
        )
        verdict = sa_verdict_count(tbl).replace("significant", "cell(s) significant")
        lines.append(f"  verdict  : {verdict}")
        hits = tbl.index[tbl["is_signif"] == True].tolist()  # noqa: E712
        if hits:
            lines.append("\n  cells")
            labels = [
                f"{tbl.loc[i, 'row_level']} : {tbl.loc[i, 'col_level']}" for i in hits
            ]
            width = max(len(x) for x in labels)
            for i, label in zip(hits, labels):
                lift_col = "lift" if "lift" in tbl.columns else "ratio"
                lines.append(
                    f"    {sa_left(label, width)}  lift = {sa_fmt_est(tbl.loc[i, lift_col])}, "
                    f"adj_pvalue = {sa_fmt_pval(tbl.loc[i, 'adj_pvalue'])}"
                )
        return "\n".join(lines)

    lines.append(f"  test     : {attrs.get('test')}  ({attrs.get('test_label')})")
    cutoff = attrs.get("effect_cutoff")
    measure = tbl["measure"].iloc[0]
    cutoff_text = ""
    if cutoff is not None:
        if _sa_assoc_scale(measure) == "ratio":
            cutoff_text = f"{measure} >= {cutoff} or <= {1 / cutoff}, "
        else:
            cutoff_text = f"abs({measure}) >= {cutoff}, "
    lines.append(f"  cutoffs  : {cutoff_text}pvalue <= {attrs.get('pval_cutoff')}")
    row = tbl.iloc[0]
    interval = ""
    if pd.notna(row.get("lower_conf")):
        interval = f"  [{sa_fmt_est(row['lower_conf'])}, {sa_fmt_est(row['upper_conf'])}]"
    flag = row["is_signif"]
    if flag is True or flag == True:  # noqa: E712
        verdict_word = "significant"
    elif flag is False or flag == False:  # noqa: E712
        verdict_word = "not significant"
    else:
        verdict_word = "undecided"
    lines.append(
        f"  verdict  : {measure} = {sa_fmt_est(row['estimate'])}{interval}  ({verdict_word})"
    )
    return "\n".join(lines)


def _sa_assoc_scale(measure: str) -> str:
    if measure in ("odds_ratio", "odds_ratio_paired"):
        return "ratio"
    return "magnitude"


def _cv_settings(params: dict[str, Any]) -> str:
    if not params.get("cv"):
        return "no resampling"
    text = str(params.get("cv_method", "cv"))
    if params.get("n_fold") is not None and not (isinstance(params["n_fold"], float) and np.isnan(params["n_fold"])):
        text += f", {params['n_fold']} fold(s)"
    if params.get("n_repeat") is not None and not (
        isinstance(params["n_repeat"], float) and np.isnan(params["n_repeat"])
    ):
        text += f" x {params['n_repeat']} repeat(s)"
    if params.get("conf_level") is not None:
        text += f", conf_level = {params['conf_level']}"
    return text


def repr_sa_model(obj: Any, n: int = 10) -> str:
    design = obj["design"]
    params = obj["parameters"]
    coefs = obj["coefficients"]
    lines = [f"<sa_model> {obj['analysis']}"]
    lines.append(f"  outcome  : {design['outcome']}  ({design['outcome_type']})")
    if design.get("outcome_lv"):
        odds = "the odds of " if "odds_ratio" in coefs.columns else ""
        lines.append(
            f"             modelling {odds}{design['outcome_lv'][1]} against "
            f"{design['outcome_lv'][0]}, {design.get('n_events')} of {design['n_used']} row(s)"
        )
    dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)" if design.get("n_dropped") else ""
    lines.append(f"  rows     : {design['n_used']} used{dropped}")
    lines.append(
        f"  terms    : {len(obj['terms'])} over {len(design['predictors'])} predictor(s)"
    )
    lines.append(f"  settings : {_cv_settings(params)}")
    if params.get("penalty") is not None:
        chosen = (
            f"  (chosen from {params['n_candidates']} candidate(s))"
            if params.get("cv")
            else ""
        )
        lines.append(
            f"  penalty  : {params['penalty']}, alpha = {sa_fmt_num(params.get('alpha'))}, "
            f"lambda = {sa_fmt_num(params.get('lambda'))}{chosen}"
        )
    if params.get("ntree") is not None:
        chosen = (
            f"  (mtry chosen from {params['n_candidates']} candidate(s))"
            if params.get("cv") and params.get("n_candidates", 0) > 1
            else ""
        )
        lines.append(
            f"  forest   : {params['ntree']} tree(s), mtry = {params.get('mtry')}, "
            f"nodesize = {params.get('nodesize')}{chosen}"
        )
    if params.get("kernel") is not None:
        chosen = (
            f"  (chosen from {params['n_candidates']} candidate(s))"
            if params.get("cv") and params.get("n_candidates", 0) > 1
            else ""
        )
        lines.append(
            f"  kernel   : {params['kernel']}, C = {sa_fmt_num(params.get('C'))}, "
            f"sigma = {sa_fmt_num(params.get('sigma'))}{chosen}"
        )

    inference = "pval" in coefs.columns
    importance = not inference and "selected" not in coefs.columns
    heading = "importance  (permutation)" if importance else "coefficients"
    lines.append(f"\n  {heading}")
    shown = coefs.head(n)
    width = max((len(str(t)) for t in shown["terms"]), default=0)
    for _, row in shown.iterrows():
        extra = ""
        if inference:
            extra = (
                f"  [{sa_fmt_num(row['lower_conf'])}, {sa_fmt_num(row['upper_conf'])}]  "
                f"p = {sa_fmt_num(row['pval'])}"
            )
        elif "selected" in coefs.columns:
            extra = f"  {'selected' if row['selected'] else 'dropped'}"
        lines.append(
            f"    {sa_left(str(row['terms']), width)}  {sa_fmt_num(row['estimate']):>10}{extra}"
        )
    if len(coefs) > len(shown):
        lines.append(f"    ... and {len(coefs) - len(shown)} more term(s) in $coefficients")

    stats = ", ".join(f"{k} = {sa_fmt_num(v)}" for k, v in obj["fit_stats"].items())
    lines.append("")
    lines.append(sa_cat_field("fit", stats).rstrip())
    perf = obj.get("performance")
    if perf is not None and len(perf):
        row = perf.iloc[0]
        metrics = obj["engine"]["metrics"]
        scored = []
        for m in metrics:
            sd = f"{m}SD"
            piece = f"{m} = {sa_fmt_num(row[m])}"
            if sd in perf.columns and pd.notna(row.get(sd)):
                piece += f" (SD {sa_fmt_num(row[sd])})"
            scored.append(piece)
        resamp = obj.get("resampling")
        suffix = f" over {len(resamp)} resample(s)" if resamp is not None and len(resamp) else ""
        lines.append(sa_cat_field("resample", ", ".join(scored) + suffix).rstrip())
    if design.get("dropped_predictors"):
        lines.append(
            f"  dropped  : {', '.join(design['dropped_predictors'])} (single valued)"
        )
    return "\n".join(lines)


def repr_sa_selection(obj: Any, n: int = 10) -> str:
    design = obj["design"]
    params = obj["parameters"]
    chose_by = params.get("metric") or params.get("criterion", "metric")
    lines = [f"<sa_selection> {obj['analysis']}"]
    lines.append(f"  outcome  : {design['outcome']}  ({design['outcome_type']})")
    if design.get("outcome_lv"):
        lines.append(
            f"             modelling {design['outcome_lv'][1]} against "
            f"{design['outcome_lv'][0]}, {design.get('n_events')} of {design['n_used']} row(s)"
        )
    dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)" if design.get("n_dropped") else ""
    lines.append(f"  rows     : {design['n_used']} used{dropped}")

    profile = obj["profile"]
    if obj["analysis"] == "rfe":
        sizes = ", ".join(str(int(x)) for x in profile["n_vars"])
        search = f"{obj['engine']['label']} over {len(obj['candidates'])} candidate(s), size(s) {sizes}"
    elif obj["analysis"] == "stepwise":
        search = f"{obj['engine']['label']} over {len(obj['candidates'])} candidate(s), {len(profile) - 1} step(s)"
    else:
        search = f"{obj['engine']['label']} over {len(obj['candidates'])} candidate(s), {len(profile)} model(s) compared"
    lines.append(sa_cat_field("search", search).rstrip())

    if obj["analysis"] == "stepwise":
        direction = "maximised" if params.get("maximize") else "minimised"
        lines.append(
            f"  settings : {params.get('direction')} search, {chose_by} {direction} "
            f"at {sa_fmt_num(params.get('k'))} per parameter"
        )
    else:
        direction = "maximised" if params.get("maximize") else "minimised"
        fold = ""
        if params.get("n_fold") is not None and not (
            isinstance(params["n_fold"], float) and np.isnan(params["n_fold"])
        ):
            fold = f", {params['n_fold']} fold(s)"
        rep = ""
        if params.get("n_repeat") is not None and not (
            isinstance(params["n_repeat"], float) and np.isnan(params["n_repeat"])
        ):
            rep = f" x {params['n_repeat']} repeat(s)"
        lines.append(
            f"  settings : {params.get('cv_method')}{fold}{rep}, {chose_by} {direction}"
        )

    best = profile.loc[profile["chosen"]].iloc[0]
    digits = 6 if obj["analysis"] == "stepwise" else 3
    sd_col = f"{chose_by}SD"
    sd_text = ""
    if sd_col in profile.columns and pd.notna(best.get(sd_col)):
        sd_text = f" (SD {sa_fmt_num(best[sd_col], 2)})"
    lines.append(
        f"  selected : {len(obj['selected'])} of {len(obj['candidates'])}  "
        f"({chose_by} = {sa_fmt_num(best[chose_by], digits)}{sd_text})"
    )

    ranking = obj["ranking"].head(n)
    lines.append("\n  ranking")
    width = max(len(str(c)) for c in ranking["candidates"]) if len(ranking) else 0
    for _, row in ranking.iterrows():
        mark = " *" if row["selected"] else ""
        lines.append(
            f"    {sa_left(str(row['candidates']), width)}  {sa_fmt_num(row['estimate'])}{mark}"
        )
    if len(obj["ranking"]) > len(ranking):
        lines.append(f"    ... and {len(obj['ranking']) - len(ranking)} more in $ranking")
    return "\n".join(lines)


def repr_sa_performance(obj: Any, n: int = 10) -> str:
    design = obj["design"]
    classification = obj["analysis"] == "classification_performance"
    lines = [f"<sa_performance> {obj['analysis']}"]
    lines.append(f"  outcome  : {design['outcome']}  ({design['outcome_type']})")
    if design.get("outcome_lv"):
        lines.append(
            f"             scoring the probability of {design['outcome_lv'][1]} against "
            f"{design['outcome_lv'][0]}, {design.get('n_events')} of {design['n_used']} row(s)"
        )
    dropped = f"  ({design['n_dropped']} row(s) dropped)" if design.get("n_dropped") else ""
    lines.append(f"  rows     : {design['n_used']} scored{dropped}")
    lines.append(f"  models   : {len(obj['models'])}, baseline = {design['baseline']}")
    if classification:
        lines.append(
            f"  threshold: {obj['parameters'].get('threshold')}  "
            "(accuracy, sensitivity and specificity only)"
        )

    metrics = obj["metrics"].head(n)
    lines.append("\n  metrics")
    width = max(len(str(m)) for m in metrics["model"]) if len(metrics) else 0
    for _, row in metrics.iterrows():
        if classification:
            text = (
                f"auc = {sa_fmt_num(row['auc'])}, "
                f"[{sa_fmt_num(row['auc_lower_conf'])}, {sa_fmt_num(row['auc_upper_conf'])}], "
                f"brier = {sa_fmt_num(row['brier'])}, accuracy = {sa_fmt_num(row['accuracy'])}"
            )
        else:
            text = (
                f"cor = {sa_fmt_num(row['cor'])}, r_squared = {sa_fmt_num(row['r_squared'])}, "
                f"rmse = {sa_fmt_num(row['rmse'])}, mae = {sa_fmt_num(row['mae'])}"
            )
        lines.append(f"    {sa_left(str(row['model']), width)}  {text}")
    if len(obj["metrics"]) > len(metrics):
        lines.append(f"    ... and {len(obj['metrics']) - len(metrics)} more model(s) in $metrics")
    return "\n".join(lines)


def _scaling_label(params: dict[str, Any]) -> str:
    if params.get("center") and params.get("scale"):
        return "centred and scaled"
    if params.get("center"):
        return "centred"
    if params.get("scale"):
        return "scaled"
    return "none, values as they arrived"


def repr_sa_reduction(obj: Any, n: int = 3) -> str:
    design = obj["design"]
    params = obj["parameters"]
    dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)" if design.get("n_dropped") else ""
    lines = [f"<sa_reduction> {obj['analysis']}"]
    lines.append(
        f"  data     : {design['n_used']} sample(s) x {design['n_feats']} feature(s){dropped}"
    )
    lines.append(f"  points   : {len(obj['points'])} {design['point_type']}(s)")
    lines.append(f"  scaling  : {_scaling_label(params)}")

    variance = obj.get("variance")
    if variance is not None and len(variance):
        shown = variance.head(n)
        parts = [
            f"{row.component} {sa_fmt_num(row.prop_var, 4)}%"
            for _, row in shown.iterrows()
        ]
        cum = shown["cum_var"].iloc[-1]
        lines.append(
            sa_cat_field(
                "variance",
                f"{', '.join(parts)}  ({len(shown)} of {len(variance)} component(s), "
                f"{sa_fmt_num(cum, 4)}% cumulative)",
            ).rstrip()
        )
    if obj["analysis"] == "tsne":
        seed = f"  (seed = {params['seed']})" if params.get("seed") is not None else ""
        lines.append(
            f"  tsne     : {params.get('n_dim')} dimension(s), perplexity = "
            f"{sa_fmt_num(params.get('perplexity'))}, theta = {sa_fmt_num(params.get('theta'))}{seed}"
        )
    if obj["analysis"] == "umap":
        seed = f"  (seed = {params['seed']})" if params.get("seed") is not None else ""
        lines.append(
            f"  umap     : {params.get('n_dim')} dimension(s), method = {params.get('method')}, "
            f"n_neighbors = {params.get('n_neighbors')}, min_dist = {sa_fmt_num(params.get('min_dist'))}, "
            f"{params.get('metric')}{seed}"
        )
    if design.get("dropped_feats"):
        lines.append(
            sa_cat_field("dropped", f"{', '.join(design['dropped_feats'])} (no variance)").rstrip()
        )
    return "\n".join(lines)


def repr_sa_cluster(obj: Any, n: int = 10) -> str:
    design = obj["design"]
    params = obj["parameters"]
    dropped = f"  ({design['n_dropped']} incomplete row(s) dropped)" if design.get("n_dropped") else ""
    lines = [f"<sa_cluster> {obj['analysis']}"]
    lines.append(
        f"  data     : {design['n_used']} sample(s) x {design['n_feats']} feature(s){dropped}"
    )
    lines.append(f"  points   : {len(obj['points'])} {design['point_type']}(s)")
    lines.append(f"  scaling  : {_scaling_label(params)}")
    noise = f"  ({design['n_noise']} point(s) left as noise)" if design.get("n_noise") else ""
    lines.append(f"  clusters : {design['n_clusters']}{noise}")

    if design.get("n_clusters", 0) > 0:
        shown = obj["clusters"].head(n)
        parts = [
            f"#{int(row.cluster)} n = {int(row.size)}, s = {sa_fmt_num(row.silhouette)}"
            for _, row in shown.iterrows()
        ]
        suffix = (
            f"  ({len(shown)} of {design['n_clusters']} shown)"
            if len(shown) < design["n_clusters"]
            else ""
        )
        lines.append(sa_cat_field("sizes", "; ".join(parts) + suffix).rstrip())
        sil = obj["assignments"]["silhouette"]
        assigned = sil.notna().sum()
        mean_s = sil.mean(skipna=True)
        lines.append(
            sa_cat_field(
                "silhouette",
                f"mean {sa_fmt_num(mean_s)} over the {assigned} assigned "
                f"{design['point_type']}(s), on the {params.get('dist_method')} distance",
            ).rstrip()
        )
    if obj["analysis"] == "hclust":
        lines.append(
            f"  linkage  : {params.get('hclust_method')}, cut at k = {params.get('n_clust')}"
        )
    if obj["analysis"] == "kmeans":
        seed = f"  (seed = {params['seed']})" if params.get("seed") is not None else ""
        lines.append(
            f"  kmeans   : k = {params.get('n_clust')}, {params.get('n_start')} start(s), "
            f"{sa_fmt_num(params.get('tot_withinss'), 5)} within-cluster ss{seed}"
        )
    return "\n".join(lines)


def repr_sa_split(obj: Any) -> str:
    design = obj["design"]
    params = obj["parameters"]
    lines = ["<sa_split> train/test partition"]
    unit = ""
    if design.get("id") is not None and not (
        isinstance(design["id"], float) and np.isnan(design["id"])
    ):
        unit = f"  ({design['n_units']} unit(s) of `{design['id']}`)"
    lines.append(f"  rows     : {design['n_rows']}{unit}")
    strat = design.get("stratified")
    if strat is None or (isinstance(strat, float) and np.isnan(strat)):
        strat_text = "none"
    else:
        strat_text = str(strat)
    lines.append(f"  stratify : {strat_text}")
    if design.get("strata_n"):
        parts = [f"{k} {v}" for k, v in design["strata_n"].items()]
        lines.append(f"             {', '.join(parts)}")

    seed = f", seed = {params['seed']}" if params.get("seed") is not None else ""
    lines.append(
        f"  settings : p_train = {params['p_train']}, times = {params['times']}{seed}"
    )

    lines.append("\n  splits")
    achieved = params.get("achieved_p", [])
    if isinstance(achieved, dict):
        achieved_map = achieved
    else:
        achieved_map = {
            f"Resample{i + 1}": achieved[i] for i in range(len(achieved))
        }
    datasets = obj["datasets"]
    names = list(achieved_map.keys()) if achieved_map else [f"Resample{i + 1}" for i in range(len(datasets))]
    width = max(len(nm) for nm in names) if names else 8
    for i, ds in enumerate(datasets):
        nm = names[i] if i < len(names) else f"Resample{i + 1}"
        p = achieved_map.get(nm, params["p_train"])
        lines.append(
            f"    ${sa_left(nm, width)}  train {len(ds['train_data'])} / test {len(ds['test_data'])}  "
            f"(p = {sa_fmt_num(p, 3)})"
        )
    return "\n".join(lines)


def repr_sa_diagnosis(obj: Any) -> str:
    alpha = obj.parameters["alpha"]
    s = obj.summary
    lines = [f"<sa_diagnosis> {obj.analysis}"]
    lines.append(f"  features : {len(obj.features)}")
    if obj.design.get("grouped"):
        groups = ", ".join(obj.design.get("group_lv", []))
    else:
        groups = "none, so no variance test"
    lines.append(f"  groups   : {groups}")
    lines.append(
        f"  settings : alpha = {alpha}, outlier criterion = {obj.parameters.get('criterion')}"
    )

    def count_ok(flag_col: str) -> int:
        flag = s[flag_col]
        return int((flag.notna() & ~flag.astype(bool)).sum())

    lines.append("\n  checks")
    lines.append(
        f"    normality  {count_ok('normal_ok')} of {len(s)} feature(s) have a group "
        f"failing Shapiro-Wilk at {alpha}"
    )
    if len(obj.variance) > 0:
        lines.append(
            f"    variance   {count_ok('variance_ok')} of {len(s)} feature(s) fail Levene at {alpha}"
        )
    n_feat_flagged = int((s["n_outliers"] > 0).sum())
    lines.append(
        f"    outliers   {len(obj.outliers)} observation(s) flagged across {n_feat_flagged} feature(s)"
    )
    lines.append("\n  A failed check never changes which tests run. It changes which of")
    lines.append("  them deserves the most weight, and that judgement stays with you.")
    return "\n".join(lines)
