#!/usr/bin/env python3
"""Transcribe R STATassist v1.0.0 README to statassist-py/README.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "tools" / "r_readme_v1.0.0.md"

EXPORTS = [
    "summarize_descriptive_stats", "summarize_association_stats", "sa_describe_vector",
    "sa_skewness", "sa_kurtosis", "draw_grouped_boxplot", "draw_grouped_barplot",
    "draw_heatmap", "draw_corrplot", "draw_butterfly_hist", "draw_volcano_plot",
    "draw_forest_plot", "draw_interaction_plot", "draw_mosaic_plot", "compare_one_sample",
    "compare_two_groups", "compare_multiple_groups", "compare_factorial_groups",
    "compare_categorical_groups", "center_by_control", "diagnose_distribution",
    "screen_outliers", "estimate_significance", "estimate_categorical_significance",
    "split_data", "fit_linear_regression", "fit_logistic_regression", "fit_elastic_net",
    "fit_rf", "fit_svm", "perform_rfe", "perform_stepwise", "predict", "coef",
    "evaluate_regression_models", "evaluate_classification_models", "draw_prediction_plot",
    "draw_roc_curve", "perform_pca", "perform_tsne", "perform_umap", "draw_dim_reduction_plot",
    "cluster_kmeans", "cluster_hclust", "cluster_dbscan", "cluster_snn",
    "simulate_two_groups", "simulate_multiple_groups", "simulate_factorial_groups",
    "simulate_categorical_groups", "simulate_regression", "simulate_classification",
    "make_block_cor",
]

PLOT_MAP = {
    r"plot\s*\(\s*comp_res\s*\)": "sa.draw_forest_plot(comp_res)",
    r"plot\s*\(\s*cat_comp\s*\)": "sa.draw_mosaic_plot(cat_comp)",
    r"plot\s*\(\s*eval_reg\s*\)": "sa.draw_prediction_plot(eval_reg)",
    r"plot\s*\(\s*eval_cls\s*\)": "sa.draw_prediction_plot(eval_cls)",
    r"plot\s*\(\s*(\w+)\s*\)": r"sa.draw_forest_plot(\1)",
}


def load_source() -> str:
    if BUNDLED.is_file():
        return BUNDLED.read_text(encoding="utf-8")
    raise FileNotFoundError(f"R README source not found: {BUNDLED}")


def transform_header(text: str) -> str:
    text = text.replace("# STATassist\n", "# statassist-py\n\nPython port of [R STATassist](https://github.com/hiows/STATassist).\n\n", 1)
    old_deps = (
        "The comparison, diagnostic and visualisation functions use base R only (`stats`, `graphics`, "
        "`grDevices`, `utils`). The modelling and dimension-reduction functions are built on `caret` "
        "and call `glmnet`, `randomForest`, `kernlab`, `Rtsne` and `umap` through it. The two "
        "density-based clustering functions call `dbscan` directly."
    )
    new_deps = (
        "Core dependencies are **SciPy**, **statsmodels**, **scikit-learn**, and **matplotlib**. "
        "Optional **`umap-learn`** and **`openTSNE`** power `perform_umap()` and `perform_tsne()` "
        "(install with `pip install \"statassist-py[reduce]\"`). Density-based clustering uses "
        "**scikit-learn** DBSCAN."
    )
    text = text.replace(old_deps, new_deps)
    return text


def transform_installation(text: str) -> str:
    pattern = re.compile(
        r"## Installation\n\nInstall from GitHub:\n\n```r\n.*?```\n\nEach release is tagged.*?\n\n",
        re.DOTALL,
    )
    replacement = """## Installation

```bash
pip install statassist-py
pip install "statassist-py[reduce]"   # t-SNE / UMAP
pip install "statassist-py[all]"      # reduce + dev extras
```

For cross-validation against R, install the pinned release:

```r
remotes::install_github("hiows/STATassist@v1.0.0")
```

"""
    return pattern.sub(replacement, text, count=1)


def r_dollar_to_dot(line: str) -> str:
    """obj$slot -> obj.slot; obj$tests$t -> obj.tests['t'] style."""
    def repl(m: re.Match) -> str:
        parts = m.group(0).split("$")
        base = parts[0]
        chain = parts[1:]
        out = base
        for i, p in enumerate(chain):
            if re.match(r"^\w+$", p):
                out += f".{p}" if i == 0 else f".{p}"
            else:
                out += f'["{p}"]'
        return out

    return re.sub(r"[A-Za-z_][\w]*(\$[\w]+)+", repl, line)


def transform_code_line(line: str) -> str:
    for pat, rep in PLOT_MAP.items():
        line = re.sub(pat, rep, line)
    line = line.replace("library(STATassist)", "import statassist as sa")
    line = line.replace("NULL", "None")
    line = re.sub(r"\bTRUE\b", "True", line)
    line = re.sub(r"\bFALSE\b", "False", line)
    line = line.replace("man/figures/", "docs/figures/")
    # sim$args -> sim["args"]
    line = re.sub(r"(\w+)\$args", r'\1["args"]', line)
    line = re.sub(r"(\w+)\$truth", r'\1["truth"]', line)
    line = re.sub(r"(\w+)\$split_args", r'\1["split_args"]', line)
    line = r_dollar_to_dot(line)
    # do.call(fn, sim$args) already handled; do.call(x, list(...)) -> x(**{...})
    line = re.sub(
        r"do\.call\((\w+),\s*(\w+)\[\"args\"\]\)",
        r"sa.\1(**\2['args'])",
        line,
    )
    for fn in sorted(EXPORTS, key=len, reverse=True):
        if fn in ("predict", "coef"):
            continue
        line = re.sub(rf"(?<![\w.]){fn}\(", f"sa.{fn}(", line)
        line = re.sub(rf"sa\.sa\.{fn}\(", f"sa.{fn}(", line)
    line = re.sub(r"(?<![\w.])predict\(", "sa.predict(", line)
    line = re.sub(r"(?<![\w.])coef\(", "sa.coef(", line)
    # c("a", "b") -> ["a", "b"] simple
    line = re.sub(r'c\(([^)]+)\)', lambda m: "[" + m.group(1) + "]", line)
    # R-only
    line = line.replace("all.equal(", "np.allclose(")
    line = line.replace("# install.packages", "# pip install")
    return line


def transform_paired_sleep_block(text: str) -> str:
    old = """Paired example (`sleep`, same subjects under two drugs):

```r
paired_res <- compare_two_groups(
  data     = sleep["extra"],
  feats    = "extra",
  group    = sleep$group,
  group_lv = c("1", "2"),
  id       = sleep$ID,
  paired   = TRUE,
  alternative = "less"
)
paired_res$tests$t_test
```"""
    new = """Paired example (simulated repeated measures, same subjects under two conditions):

```python
paired_sim = sa.simulate_two_groups(n_feats=1, n_up=1, paired=True, seed=2026)
paired_res = sa.compare_two_groups(
    **paired_sim["args"],
    alternative="less",
    diagnose=False,
)
paired_res.tests["t_test"]
```"""
    return text.replace(old, new)


def transform_s3_blocks(text: str) -> str:
    """Replace R S3 print blocks with table-oriented Python hints."""
    replacements = [
        (
            "```\n<sa_two_group> two_group_comparison\n  groups   : control vs case  (independent)\n  features : 30\n  settings : alternative = two.sided, conf_level = 0.95, p_adjust = BH\n\n  tests\n    $t_test       13 of 30 at pval_adj <= 0.05\n                 Welch's t-test\n    $wilcox_test  11 of 30 at pval_adj <= 0.05\n                 Wilcoxon rank sum test (Mann-Whitney U test)\n    $robust_test  12 of 30 at pval_adj <= 0.05\n                 Brunner-Munzel test\n\n  $diagnostics attached\n```",
            "```python\n# comp_res.analysis, comp_res.design, comp_res.tests keys\ncomp_res.tests[\"t_test\"].head()\n```",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Generic: strip remaining <sa_* print blocks to a short note
    text = re.sub(
        r"```\n<sa_[^>]+>[\s\S]*?```",
        "```python\n# See result tables: .effect, .tests, .scores, .metrics, ...\n```",
        text,
    )
    return text


def post_process_code(text: str) -> str:
    """Fix patterns the line transformer missed."""
    text = re.sub(r'(\[["\']\w+["\']\])\$(\w+)', r'\1["\2"]', text)
    text = text.replace("<-", "=")
    text = re.sub(r"sa\.sa\.", "sa.", text)
    text = re.sub(r"\)\$significance", ").significance", text)
    text = re.sub(r"(\w+)\$significance", r"\1.significance", text)
    text = re.sub(r"(\w+)\.(\w+)\.(\w+)", lambda m: f'{m.group(1)}.{m.group(2)}["{m.group(3)}"]' if m.group(2) in ("tests", "posthoc", "pairwise", "metrics", "datasets") else m.group(0), text)
    # R output blocks: leave as illustrative tables
    text = text.replace("do.call(compare_two_groups, sim$args)", 'sa.compare_two_groups(**sim["args"])')
    text = text.replace("do.call(compare_two_groups, sim$args)", 'sa.compare_two_groups(**sim["args"])')
    return text


def post_process_readme(text: str) -> str:
    lines = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            in_code = True
            lines.append(line)
            continue
        if in_code and line.strip() == "```":
            in_code = False
            lines.append(line)
            continue
        if in_code:
            line = post_process_code(line)
            line = re.sub(r'paste0\("gene_", 1:10\)', '[f"gene_{i}" for i in range(1, 11)]', line)
            line = re.sub(r'paste0\("prot_", 1:(\d+)\)', r'[f"prot_{i}" for i in range(1, \1 + 1)]', line)
            line = re.sub(r'paste0\("x_", 1:(\d+)\)', r'[f"x_{i}" for i in range(1, \1 + 1)]', line)
            line = line.replace("1:10", "range(1, 11)")
            line = line.replace("sim_raw[] <- 2^sim_raw", "sim_raw = 2 ** sim_raw")
            line = re.sub(r"(\w+)\.posthoc\.(\w+)", r'\1.posthoc["\2"]', line)
            line = re.sub(r"(\w+)\.pairwise\.(\w+)", r'\1.pairwise["\2"]', line)
            line = re.sub(r"verdict\.is_signif", 'verdict["is_signif"]', line)
            line = re.sub(r"drawn\$group_densities\$case", 'drawn["group_densities"]["case"]', line)
        lines.append(line)
    return "\n".join(lines)


def process(text: str) -> str:
    text = transform_header(text)
    text = transform_installation(text)
    text = transform_paired_sleep_block(text)
    text = text.replace("```r\n", "```python\n")
    text = text.replace("man/figures/", "docs/figures/")

    out_lines = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            in_code = True
            out_lines.append(line)
            continue
        if in_code and line.strip() == "```":
            in_code = False
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(transform_code_line(line))
        else:
            out_lines.append(line)
    text = "\n".join(out_lines)
    text = transform_s3_blocks(text)
    text = post_process_readme(text)

    footer = """

---

## Python stack

| Role | Packages |
| --- | --- |
| Tests, distributions | SciPy, statsmodels |
| Models, CV, clustering | scikit-learn |
| Penalised regression | scikit-learn `ElasticNet` / `LogisticRegression` |
| Plots | matplotlib |
| t-SNE / UMAP (optional) | openTSNE, umap-learn |

## Testing

```bash
cd statassist-py
py -m pytest -v
```

Tests live in [`test/cursor_test/`](../test/cursor_test/).

## Regenerating figures

```bash
cd statassist-py
py tools/render_readme_figures.py
```

## Known Python differences

- **`perform_stepwise()`** — simplified backward search (not full R `stats::step()`).
- **`simulate_*()`** — same contract as R; internal generators are simplified in places (e.g. `simulate_classification()` has no `cor_mat`; use `simulate_regression(cor_mat=...)` for correlated features).
- **`compare_factorial_groups()`** — two-way Type III ANOVA centre; repeated-measures factorial not implemented.
- **`draw_grouped_boxplot()`** — crossed factorial layout not yet ported; README uses an ad-hoc render helper for that figure.
- **Plots** — matplotlib rather than base R graphics; no S3 `plot()` dispatch (call `draw_*()` explicitly).

"""
    # Insert footer before Author if not already extended
    if "## Python stack" not in text:
        text = re.sub(r"\n---\n\n## Author\n", footer + "\n---\n\n## Author\n", text, count=1)
    return text + "\n"


def main() -> None:
    src = load_source()
    out = process(src)
    (ROOT / "README.md").write_text(out, encoding="utf-8")
    print(f"Wrote {ROOT / 'README.md'} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
