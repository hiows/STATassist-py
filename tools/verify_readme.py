#!/usr/bin/env python3
"""Verify README figure links and spot-run key examples."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import statassist as sa  # noqa: E402


def check_figures() -> list[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for m in re.finditer(r"docs/figures/(README-[^)\]]+\.png)", readme):
        path = ROOT / "docs" / "figures" / m.group(1)
        if not path.is_file():
            missing.append(str(path))
    return missing


def spot_run() -> None:
    sim2 = sa.simulate_two_groups(n_feats=30, n_up=8, n_down=8, seed=2026)
    comp2 = sa.compare_two_groups(**sim2["args"], diagnose=False)
    sig2 = sa.estimate_significance(comp2, test="t_test", log2fc_cutoff=1, pval_cutoff=0.05)
    assert len(sig2.significance) == 30

    sim_multi = sa.simulate_multiple_groups(
        n_feats=100, n_control=50, n_treat=[50, 50, 50], seed=2026
    )
    multi = sa.compare_multiple_groups(**sim_multi["args"], diagnose=False)
    assert multi.analysis == "multi_group_comparison"

    clust_sim = sa.simulate_two_groups(n_feats=50, deg_log2fc=(5, 10), seed=2026)
    clust_pca = sa.perform_pca(
        data=clust_sim["args"]["data"],
        feats=clust_sim["args"]["feats"],
        embedding_scale="samples",
    )
    assert "PC1" in clust_pca["scores"].columns


def main() -> None:
    missing = check_figures()
    if missing:
        print("Missing figures:")
        for m in missing:
            print(" ", m)
        raise SystemExit(1)
    print(f"All {len(re.findall(r'docs/figures/README-', (ROOT / 'README.md').read_text(encoding='utf-8')))} figure links resolve.")
    spot_run()
    print("Spot-run checks passed.")


if __name__ == "__main__":
    main()
