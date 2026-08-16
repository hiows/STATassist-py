#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "README.md"
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
out = []
skip_next = 0
for i, line in enumerate(lines):
    if skip_next:
        skip_next -= 1
        continue
    if line.startswith("roc_") and "pROC::ro[" in line:
        out.append("# AUC on held-out rows is shown in the ROC figure.\n")
        # skip round(...) and optional #> line
        if i + 1 < len(lines) and "pROC::auc" in lines[i + 1]:
            skip_next = 1
            if i + 2 < len(lines) and lines[i + 2].startswith("#>"):
                skip_next = 2
        continue
    if "pROC::" in line or line.startswith("round(as.numeri"):
        continue
    out.append(line)

text = "".join(out)
replacements = [
    ("head(fact_comp.effect, 3)", "fact_comp.effect.head(3)"),
    ("head(sig_cell.significance, 4)", "sig_cell.significance.head(4)"),
    (
        'sa.summarize_descriptive_stats(sim["args"]["data"], paste0("gene_", 1:3))',
        'sa.summarize_descriptive_stats(sim["args"]["data"], [f"gene_{i}" for i in range(1, 4)])',
    ),
    (
        "do.call(compare_two_groups, sim$args)",
        'sa.compare_two_groups(**sim["args"])',
    ),
    ("C          = 2^seq(-5, 10, by = 2)", "C=[0.5], sigma=[0.05]"),
]
for old, new in replacements:
    text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("pROC remaining:", text.count("pROC"))
