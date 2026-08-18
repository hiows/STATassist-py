#!/usr/bin/env python3
"""Fix R syntax leftovers in README.md python blocks."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _r_colon_range(match: re.Match[str]) -> str:
    start, end = int(match.group(1)), int(match.group(2))
    return f"list(range({start}, {end + 1}))"


def fix_python_block(block: str) -> str:
    block = block.replace("<-", "=")

    draw_fixes = [
        (r"draw_forest_sa\.draw_forest_sa\.draw_forest_plot", "sa.draw_forest_plot"),
        (r"sa\.draw_forest_sa\.draw_forest_plot", "sa.draw_forest_plot"),
        (r"draw_volcano_sa\.draw_forest_plot", "sa.draw_volcano_plot"),
        (r"draw_mosaic_sa\.draw_mosaic_sa\.draw_forest_plot", "sa.draw_mosaic_plot"),
        (r"draw_mosaic_sa\.draw_forest_plot", "sa.draw_mosaic_plot"),
    ]
    for pattern, repl in draw_fixes:
        block = re.sub(pattern, repl, block)

    block = re.sub(r"sa\.sa\.", "sa.", block)
    block = re.sub(r"(\w+)\[([^,\]]+),\s*\]", r"\1[\2]", block)
    block = re.sub(
        r"seq\((-?\d+),\s*(\d+),\s*by\s*=\s*1\)",
        lambda m: f"list(range({m.group(1)}, {int(m.group(2)) + 1}))",
        block,
    )
    block = re.sub(r"C\s*=\s*2\^seq\(-5,\s*10,\s*by\s*=\s*2\)", "C=[2**i for i in range(-5, 11, 2)]", block)
    block = block.replace("as.numeri", ".astype(int)")
    block = block.replace("data.frame", "pd.DataFrame")

    block = re.sub(
        r"control_label\s*=\s*list\(([^)]+)\)",
        lambda m: "control_label = {" + re.sub(r"(\w+)\s*=", r'"\1":', m.group(1)) + "}",
        block,
    )
    block = re.sub(
        r"category_lv\s*=\s*list\(([^)]+)\)",
        lambda m: "category_lv = {" + re.sub(r"(\w+)\s*=\s*c\(([^)]+)\)", r'"\1": [\2]', m.group(1)) + "}",
        block,
    )

    block = re.sub(r"assoc\.pearson\.(\w+)", r'assoc["pearson"]["\1"]', block)
    block = re.sub(
        r"fact_comp\.terms\.features",
        'fact_comp.terms["features"]',
        block,
    )
    block = re.sub(
        r"fact_comp\.terms\[fact_comp\.terms\.term ==",
        'fact_comp.terms[fact_comp.terms["term"] ==',
        block,
    )

    block = re.sub(
        r"list\(lasso\s*=\s*eval_lasso,\s*rf\s*=\s*eval_rf,\s*svm\s*=\s*eval_svm\)",
        '{"lasso": eval_lasso, "rf": eval_rf, "svm": eval_svm}',
        block,
    )

    def _fix_new_models(match: re.Match[str]) -> str:
        body = match.group(1)
        body = re.sub(r"(\w+)\s*=\s*(eval_\w+)", r'"\1": \2', body)
        return f"new_models = {{{body}}}"

    block = re.sub(
        r"new_models\s*=\s*list\(\s*([^)]+)\s*\)",
        _fix_new_models,
        block,
        flags=re.DOTALL,
    )

    def _fix_block_entry(match: re.Match[str]) -> str:
        text = match.group(0)
        text = re.sub(r"list\(features\s*=\s*", '{"features": ', text)
        text = re.sub(r"(\d+):(\d+)", _r_colon_range, text)
        text = re.sub(r",\s*cor\s*=\s*([\d.]+)", r', "cor": \1', text)
        text = re.sub(r",\s*against\s*=\s*", ', "against": ', text)
        text = text.rstrip().rstrip(")")
        if not text.endswith("}"):
            text += "}"
        return text

    block = re.sub(
        r"list\(\s*features\s*=\s*\d+:\d+[^)]*\)",
        _fix_block_entry,
        block,
    )

    block = re.sub(
        r"blocks\s*=\s*list\(\s*((?:\{[^}]+\}\s*,?\s*)+)\)",
        lambda m: "blocks = [\n    "
        + ",\n    ".join(part.strip().rstrip(",") for part in re.findall(r"\{[^}]+\}", m.group(1)))
        + ",\n  ]",
        block,
        flags=re.DOTALL,
    )

    block = re.sub(r"blocks\s*=\s*list\(\s*\n", "blocks = [\n", block)
    block = re.sub(r"\n\s*\)\s*\n\)", "\n  ]\n)", block)

    block = re.sub(
        r'assoc\["pearson"\]\["corr"\]\[(\d+):(\d+),\s*(\d+):(\d+)\]',
        lambda m: f'assoc["pearson"]["corr"].iloc[{int(m.group(1))}:{int(m.group(2))}, {int(m.group(3))}:{int(m.group(4))}]',
        block,
    )
    block = re.sub(
        r'multi\.tests\["anova_test"\]\[,\s*\[',
        'multi.tests["anova_test"].loc[:, [',
        block,
    )
    block = re.sub(
        r'ph\[ph\.features == "prot_1", \[',
        'ph.loc[ph["features"] == "prot_1", [',
        block,
    )
    block = re.sub(
        r'multi\.pairwise\["anova_test"\]\[\["treat_3 - control"\]\]\[\s*,\s*\[',
        'multi.pairwise["anova_test"]["treat_3 - control"].loc[:, [',
        block,
    )
    block = re.sub(
        r'rm_res\.tests\["anova_test"\]\[1:3,\s*c\(',
        'rm_res.tests["anova_test"].iloc[0:3][[',
        block,
    )
    block = re.sub(r'c\("features", "f_stat"', '"features", "f_stat"', block)
    block = re.sub(
        r'sim_fact\["truth"\]_term',
        'sim_fact["truth_term"]',
        block,
    )
    block = re.sub(
        r'subset\(sim_fact\["truth_term"\], features == "prot_14"\)',
        'sim_fact["truth_term"].loc[sim_fact["truth_term"]["features"] == "prot_14"]',
        block,
    )
    block = re.sub(
        r'subset\(sim_reg\["truth"\], role == "signal"\)',
        'sim_reg["truth"].loc[sim_reg["truth"]["role"] == "signal"]',
        block,
    )
    block = re.sub(
        r'subset\(sim_cls\["truth"\], role != "null"\)\$predictors',
        'sim_cls["truth"].loc[sim_cls["truth"]["role"] != "null", "predictors"]',
        block,
    )
    block = re.sub(
        r'data = assoc_sim\["args"\]\["data"\]\[, -1, drop = False\]',
        'data = assoc_sim["args"]["data"].iloc[:, 1:]',
        block,
    )
    block = re.sub(
        r'feats = colnames\(assoc_sim\["args"\]\["data"\]\)\[-1\]',
        'feats = list(assoc_sim["args"]["data"].columns[1:])',
        block,
    )
    block = re.sub(
        r'(\w+) = (\w+)\.datasets\[\[(\d+)\]\]\$(\w+)',
        lambda m: f'{m.group(1)} = {m.group(2)}["datasets"][{int(m.group(3)) - 1}]["{m.group(4)}"]',
        block,
    )
    block = re.sub(
        r'sa\.coef\((\w+)\)\[,\s*\[',
        r'sa.coef(\1).loc[:, [',
        block,
    )
    block = re.sub(
        r'terms_kept = sa\.coef\(lin\)\$terms\[-1\]\[sa\.coef\(lin\)\$pval\[-1\] < 0\.01\]',
        'terms_kept = sa.coef(lin)["terms"].iloc[1:][sa.coef(lin)["pval"].iloc[1:] < 0.01]',
        block,
    )
    block = re.sub(
        r'unique\(sub\("high\$", "", sub\("mid\$", "", terms_kept\)\)\)',
        'list(dict.fromkeys(re.sub(r"high$", "", re.sub(r"mid$", "", t)) for t in terms_kept))',
        block,
    )
    block = re.sub(
        r'round\(cor\(test_data\.y, y_hat\), 3\)',
        'round(float(np.corrcoef(test_data["y"], y_hat)[0, 1]), 3)',
        block,
    )
    block = re.sub(
        r'stratified = sim_cls\["args"\]\["data"\]\.y,',
        'stratified = sim_cls["args"]["data"]["y"],',
        block,
    )
    block = re.sub(
        r'sa\.compare_one_sample\(flag, "is_case", mu = 0\.5, p = 0\.5\)\$tests\.prop_test',
        'sa.compare_one_sample(flag, "is_case", mu = 0.5, p = 0.5).tests["prop_test"]',
        block,
    )
    block = re.sub(
        r'pca\.scores\[,\s*1:3\]',
        'pca.scores.iloc[:, 0:3]',
        block,
    )
    block = re.sub(
        r'table\(clust_sim\["args"\]\["group"\], clust_km\.assignments\.cluster\)',
        'pd.crosstab(clust_sim["args"]["group"], clust_km.assignments["cluster"])',
        block,
    )
    block = re.sub(
        r'table\(planted = planted, called = verdict\["is_signif"\] %in% True\)',
        'pd.crosstab(planted, verdict["is_signif"].astype(bool), rownames=["planted"], colnames=["called"])',
        block,
    )
    block = re.sub(
        r'rownames\(drawn\.matrix\)',
        'drawn.matrix.index.tolist()',
        block,
    )
    block = re.sub(
        r'rle\(as\.character\(sim\["args"\]\["group"\]\)\[drawn\.sample_order\]\)',
        'pd.Series(sim["args"]["group"][drawn.sample_order]).value_counts()',
        block,
    )
    block = re.sub(
        r'names\(multi\.pairwise\["anova_test"\]\)',
        'list(multi.pairwise["anova_test"].keys())',
        block,
    )
    block = re.sub(
        r'names\(sa\.coef\(svm\)\)',
        'list(sa.coef(svm).columns)',
        block,
    )
    block = re.sub(
        r'answer\s*=\s*test_data\.y,',
        'answer = test_data["y"],',
        block,
    )
    block = re.sub(
        r'answer\s*=\s*cls_test\.y,',
        'answer = cls_test["y"],',
        block,
    )
    block = re.sub(
        r'red_data = sa\.simulate_classification\(cor_mat = red_cor, seed = 2026\)\$args\.data',
        'red_data = sa.simulate_classification(cor_mat = red_cor, seed = 2026)["args"]["data"]',
        block,
    )

    block = re.sub(
        r'\[\s*,\s*(["\'][^"\']+["\'](?:,\s*["\'][^"\']+["\'])*)\s*\]',
        r"[\1]",
        block,
    )

    block = re.sub(
        r'\)\$selected',
        ')["selected"]',
        block,
    )
    block = re.sub(
        r'\blambda\s*=\s*\[',
        'lambda_ = [',
        block,
    )
    block = re.sub(
        r', cex\.anno = [^,)]+',
        '',
        block,
    )
    block = block.replace(
        "# keep the order `feats` names",
        "# keep the order named by feats",
    )

    return block


SEE_RESULT_REPLACEMENTS: list[tuple[str, str, str]] = [
    ("The verdict comes back", "sig"),
    ("![Term-wise volcano plot", "sig_fact_term"),
    ("At the default cutoffs every cell misses", "sig_cell"),
    ("Half of these features fail the variance check", "d"),
    ("`p_train` is a proportion of rows", "dataset"),
    ("`coef()` on the result is the whole table", "lin"),
    ("All four planted predictors clear 0.05 here", "log_fit"),
    ("This is `sa_selection`, the fourth result contract", "rfe"),
    ("Same contract, same `candidates` axis", "step_sel"),
    ("The three predictors at the bottom are **negative**", "rf"),
    ("Unlike the forest, this importance is measured", "svm"),
        ("All four planted predictors clear 0.05 here", "log_fit"),
        ("There is no p-value beside the deltas", "eval_reg"),
        ("The logistic baseline's AUC of 0.911", "eval_cls"),
    ("The three blocks come out as three groups", "pca"),
    ("`perform_umap()` is the one that standardises nothing", "tsne"),
    ("| t-SNE | UMAP |", "umap_res"),
    ("Fifty features were moved up or down", "clust_km"),
]


def fix_see_result_tables(text: str) -> str:
    placeholder = "```python\n# See result tables: .effect, .tests, .scores, .metrics, ...\n```"
    while placeholder in text:
        idx = text.index(placeholder)
        after = text[idx + len(placeholder) :]
        var = "comp_res"
        for marker, name in SEE_RESULT_REPLACEMENTS:
            if marker in after[:500]:
                var = name
                break
        # factorial section: first placeholder before cramers_v block
        if var == "comp_res" and "n_cells ref_center" in after[:800]:
            var = "fact_comp"
        elif var == "comp_res" and "cramers_v" in after[:800]:
            var = "cat_comp"
        text = text.replace(placeholder, f"```python\n{var}\n```", 1)
    return text


def fix_readme(text: str) -> str:
    text = text.replace(
        'remotes::install_github("hiows/STATassist@v1.0.0")',
        '# pip install git+https://github.com/hiows/STATassist@v1.0.0  # R cross-check only',
    )
    text = text.replace("Wide `data.frame`", "Wide `DataFrame`")
    text = text.replace(
        '`group_lv` is `c("control", "case")`',
        '`group_lv` is `["control", "case"]`',
    )
    text = fix_see_result_tables(text)

    parts: list[str] = []
    in_python = False
    python_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```python"):
            in_python = True
            python_lines = []
            parts.append(line)
            continue
        if in_python and line.strip() == "```":
            in_python = False
            block = "".join(python_lines)
            parts.append(fix_python_block(block))
            parts.append(line)
            continue
        if in_python:
            python_lines.append(line)
        else:
            parts.append(line)

    return "".join(parts)


def main() -> None:
    text = README.read_text(encoding="utf-8")
    README.write_text(fix_readme(text), encoding="utf-8")
    print("README python blocks normalized.")


if __name__ == "__main__":
    main()
