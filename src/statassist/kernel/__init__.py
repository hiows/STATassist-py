"""The statistical kernels: plain numbers in, one named row out.

Port of the ``kernel_*.R`` files. Every function here takes arrays and returns a
``dict[str, float]`` whose keys are, in order, the columns of the table the
caller assembles from it. Nothing in this package looks at a DataFrame, a group
label or a feature name, which is what lets the same kernel serve
:func:`~statassist.diagnose_distribution` and a comparison scenario without
either of them knowing about the other.

The R files say why the formulas are written out rather than taken from a
package, and the reason survives the port: a wrapper around a different third
party in each language would make the two disagree over defaults rather than
over arithmetic, which is the hardest kind of difference to find. Written out
once, the formula is the specification, and ``testdata/golden/`` is what holds
this side to it.
"""

from __future__ import annotations

from .anova import (
    friedman,
    kruskal,
    oneway_anova,
    rm_anova,
    sphericity,
    split_groups,
    welch_anova,
    yuen_anova,
)
from .categorical import (
    ASSOC_COLUMNS,
    MCNEMAR_EXACT_MAX_DISCORDANT,
    assoc_measures,
    assoc_measures_paired,
    assoc_measures_repeated,
    assoc_row,
    chisq,
    cochran_q,
    fisher,
    has_zero_cell,
    mcnemar,
    odds_ratio,
    phi,
)
from .cluster import NOISE_LABEL, silhouette
from .diagnostic import (
    bartlett,
    flag_outliers,
    grubbs,
    ks_normal,
    levene,
    shapiro,
)
from .factorial import (
    QR_RANK_TOL,
    SS_TYPES,
    TERM_COLUMNS,
    CellMatrix,
    FactorialFit,
    FactorialPlan,
    SsPair,
    contr_sum,
    fact_cell_matrix,
    fact_ss_plan,
    factorial_anova,
    factorial_plan,
    factorial_tukey,
)
from .performance import (
    Placement,
    auc,
    auc_delong,
    brier,
    check_response,
    delong_test,
    idi,
    nri,
    placement_values,
    roc_points,
    threshold_scores,
)
from .posthoc import (
    conover,
    dunn,
    games_howell,
    pair_matrix,
    pairwise_paired_t,
    pairwise_yuen,
    posthoc_columns,
    tukey,
    yuen_independent,
)
from .robust import (
    brunner_munzel,
    t_ci,
    t_pval,
    trimmed_mean,
    winsorize,
    winsorized_normal_var,
    yuen_paired,
)

__all__ = [
    "ASSOC_COLUMNS",
    "MCNEMAR_EXACT_MAX_DISCORDANT",
    "NOISE_LABEL",
    "QR_RANK_TOL",
    "SS_TYPES",
    "TERM_COLUMNS",
    "CellMatrix",
    "FactorialFit",
    "FactorialPlan",
    "Placement",
    "SsPair",
    "assoc_measures",
    "assoc_measures_paired",
    "assoc_measures_repeated",
    "assoc_row",
    "auc",
    "auc_delong",
    "bartlett",
    "brier",
    "brunner_munzel",
    "check_response",
    "chisq",
    "cochran_q",
    "conover",
    "contr_sum",
    "delong_test",
    "dunn",
    "fact_cell_matrix",
    "fact_ss_plan",
    "factorial_anova",
    "factorial_plan",
    "factorial_tukey",
    "fisher",
    "flag_outliers",
    "friedman",
    "games_howell",
    "grubbs",
    "has_zero_cell",
    "idi",
    "kruskal",
    "ks_normal",
    "levene",
    "mcnemar",
    "nri",
    "odds_ratio",
    "oneway_anova",
    "pair_matrix",
    "pairwise_paired_t",
    "pairwise_yuen",
    "phi",
    "placement_values",
    "posthoc_columns",
    "rm_anova",
    "roc_points",
    "shapiro",
    "silhouette",
    "sphericity",
    "split_groups",
    "t_ci",
    "t_pval",
    "threshold_scores",
    "trimmed_mean",
    "tukey",
    "welch_anova",
    "winsorize",
    "winsorized_normal_var",
    "yuen_anova",
    "yuen_independent",
    "yuen_paired",
]
