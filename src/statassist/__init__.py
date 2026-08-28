"""Python port of the R package STATassist.

The shared helper layer is :mod:`statassist.core` and the numeric engines are in
:mod:`statassist.kernel`; the public ``verb_object`` API is re-exported here as
each phase lands.

Two families are in place. The simulation family is what makes every later phase
testable: a comparison run on real data can only be judged against another
comparison, while one run on :func:`simulate_two_groups` can be judged against
the answer that was planted. The description family is what comes before a test
is chosen - what the data looks like, whether the assumptions hold, and what a
feature reads as once the control group is taken out of it.

The learning families are the two that answer about a margin rather than about a
group. :func:`fit_linear_regression` and the rest predict an outcome,
:func:`evaluate_regression_models` scores what they predicted, and
:func:`perform_rfe` and :func:`perform_stepwise` ask the question one level up -
which of the predictors were worth having. On the other side nothing is predicted:
the ``cluster_*`` four group the points of a wide table without being told what the
groups are, :func:`perform_pca` and the other two place the same points in fewer
coordinates, and :func:`draw_dim_reduction_plot` puts those two together, since
whether a clustering recovered a grouping that was known all along is a question
about one picture.
"""

__version__ = "0.1.0.dev0"

from .cluster import (
    cluster_dbscan,
    cluster_hclust,
    cluster_kmeans,
    cluster_snn,
)
from .compare import (
    compare_categorical_groups,
    compare_factorial_groups,
    compare_multiple_groups,
    compare_one_sample,
    compare_two_groups,
)
from .diagnose import diagnose_distribution, screen_outliers
from .estimate import estimate_categorical_significance, estimate_significance
from .evaluate import evaluate_classification_models, evaluate_regression_models
from .fit import (
    fit_elastic_net,
    fit_linear_regression,
    fit_logistic_regression,
    fit_rf,
    fit_svm,
)
from .plot import (
    draw_butterfly_hist,
    draw_corrplot,
    draw_dim_reduction_plot,
    draw_forest_plot,
    draw_grouped_barplot,
    draw_grouped_boxplot,
    draw_heatmap,
    draw_interaction_plot,
    draw_mosaic_plot,
    draw_prediction_plot,
    draw_roc_curve,
    draw_volcano_plot,
)
from .reduce import perform_pca, perform_tsne, perform_umap
from .select import perform_rfe, perform_stepwise
from .simulate import (
    make_block_cor,
    simulate_categorical_groups,
    simulate_classification,
    simulate_factorial_groups,
    simulate_multiple_groups,
    simulate_regression,
    simulate_two_groups,
    split_data,
)
from .summarize import summarize_association_stats, summarize_descriptive_stats
from .transform import center_by_control

__all__ = [
    "__version__",
    "center_by_control",
    "cluster_dbscan",
    "cluster_hclust",
    "cluster_kmeans",
    "cluster_snn",
    "compare_categorical_groups",
    "compare_factorial_groups",
    "compare_multiple_groups",
    "compare_one_sample",
    "compare_two_groups",
    "diagnose_distribution",
    "draw_butterfly_hist",
    "draw_corrplot",
    "draw_dim_reduction_plot",
    "draw_forest_plot",
    "draw_grouped_barplot",
    "draw_grouped_boxplot",
    "draw_heatmap",
    "draw_interaction_plot",
    "draw_mosaic_plot",
    "draw_prediction_plot",
    "draw_roc_curve",
    "draw_volcano_plot",
    "estimate_categorical_significance",
    "estimate_significance",
    "evaluate_classification_models",
    "evaluate_regression_models",
    "fit_elastic_net",
    "fit_linear_regression",
    "fit_logistic_regression",
    "fit_rf",
    "fit_svm",
    "make_block_cor",
    "perform_pca",
    "perform_rfe",
    "perform_stepwise",
    "perform_tsne",
    "perform_umap",
    "screen_outliers",
    "simulate_categorical_groups",
    "simulate_classification",
    "simulate_factorial_groups",
    "simulate_multiple_groups",
    "simulate_regression",
    "simulate_two_groups",
    "split_data",
    "summarize_association_stats",
    "summarize_descriptive_stats",
]
