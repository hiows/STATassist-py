"""STATassist — Standardised statistical comparison workflows (Python port)."""

from statassist.cluster.cluster_dbscan import cluster_dbscan
from statassist.cluster.cluster_hclust import cluster_hclust
from statassist.cluster.cluster_kmeans import cluster_kmeans
from statassist.cluster.cluster_snn import cluster_snn
from statassist.compare.center_by_control import center_by_control
from statassist.compare.compare_categorical_groups import compare_categorical_groups
from statassist.compare.compare_factorial_groups import compare_factorial_groups
from statassist.compare.compare_multiple_groups import compare_multiple_groups
from statassist.compare.compare_one_sample import compare_one_sample
from statassist.compare.compare_two_groups import compare_two_groups
from statassist.compare.diagnose_distribution import diagnose_distribution
from statassist.compare.estimate_categorical_significance import (
    estimate_categorical_significance,
)
from statassist.compare.estimate_significance import estimate_significance
from statassist.compare.screen_outliers import screen_outliers
from statassist.evaluate.evaluate_classification_models import evaluate_classification_models
from statassist.evaluate.evaluate_regression_models import evaluate_regression_models
from statassist.model.fit_elastic_net import fit_elastic_net
from statassist.model.fit_linear_regression import fit_linear_regression
from statassist.model.fit_logistic_regression import fit_logistic_regression
from statassist.model.fit_rf import fit_rf
from statassist.model.fit_svm import fit_svm
from statassist.model.perform_rfe import perform_rfe
from statassist.model.perform_stepwise import perform_stepwise
from statassist.model.predict import coef, predict
from statassist.model.split_data import split_data
from statassist.associate.summarize_association_stats import summarize_association_stats
from statassist.describe.summarize_descriptive_stats import summarize_descriptive_stats
from statassist.plot.draw_butterfly_hist import draw_butterfly_hist
from statassist.plot.draw_corrplot import draw_corrplot
from statassist.plot.draw_dim_reduction_plot import draw_dim_reduction_plot
from statassist.plot.draw_forest_plot import draw_forest_plot
from statassist.plot.draw_grouped_barplot import draw_grouped_barplot
from statassist.plot.draw_grouped_boxplot import draw_grouped_boxplot
from statassist.plot.draw_heatmap import draw_heatmap
from statassist.plot.draw_interaction_plot import draw_interaction_plot
from statassist.plot.draw_mosaic_plot import draw_mosaic_plot
from statassist.plot.draw_prediction_plot import draw_prediction_plot
from statassist.plot.draw_roc_curve import draw_roc_curve
from statassist.plot.draw_volcano_plot import draw_volcano_plot
from statassist.simulate.make_block_cor import make_block_cor
from statassist.simulate.simulate_categorical_groups import simulate_categorical_groups
from statassist.simulate.simulate_classification import simulate_classification
from statassist.simulate.simulate_factorial_groups import simulate_factorial_groups
from statassist.simulate.simulate_multiple_groups import simulate_multiple_groups
from statassist.simulate.simulate_regression import simulate_regression
from statassist.simulate.simulate_two_groups import simulate_two_groups
from statassist.utils.describe import sa_describe_vector, sa_kurtosis, sa_skewness
from statassist.reduce.perform_pca import perform_pca
from statassist.reduce.perform_tsne import perform_tsne
from statassist.reduce.perform_umap import perform_umap

__all__ = [
    "summarize_descriptive_stats",
    "summarize_association_stats",
    "sa_describe_vector",
    "sa_skewness",
    "sa_kurtosis",
    "draw_grouped_boxplot",
    "draw_grouped_barplot",
    "draw_heatmap",
    "draw_corrplot",
    "draw_butterfly_hist",
    "draw_volcano_plot",
    "draw_forest_plot",
    "draw_interaction_plot",
    "draw_mosaic_plot",
    "compare_one_sample",
    "compare_two_groups",
    "compare_multiple_groups",
    "compare_factorial_groups",
    "compare_categorical_groups",
    "center_by_control",
    "diagnose_distribution",
    "screen_outliers",
    "estimate_significance",
    "estimate_categorical_significance",
    "split_data",
    "fit_linear_regression",
    "fit_logistic_regression",
    "fit_elastic_net",
    "fit_rf",
    "fit_svm",
    "perform_rfe",
    "perform_stepwise",
    "predict",
    "coef",
    "evaluate_regression_models",
    "evaluate_classification_models",
    "draw_prediction_plot",
    "draw_roc_curve",
    "perform_pca",
    "perform_tsne",
    "perform_umap",
    "draw_dim_reduction_plot",
    "cluster_kmeans",
    "cluster_hclust",
    "cluster_dbscan",
    "cluster_snn",
    "simulate_two_groups",
    "simulate_multiple_groups",
    "simulate_factorial_groups",
    "simulate_categorical_groups",
    "simulate_regression",
    "simulate_classification",
    "make_block_cor",
]

__version__ = "1.0.0"
