"""The column names each kind of result table must carry.

The port of the ``sa_*_columns()`` family in ``R/result.R``. The per-test
statistics differ from test to test; these are the columns a consumer is allowed
to rely on regardless of which test produced the table, so they are stated once
here and checked once in :func:`statassist.core.result.new_comparison`.

The names are kept exactly as R spells them. A Python caller reading
``tbl["pval_adj"]`` and an R caller reading ``tbl$pval_adj`` are then looking at
the same column, which is what makes a result object written out as JSON in one
language readable in the other.
"""

from __future__ import annotations

__all__ = [
    "RATIO_MEASURES",
    "assoc_scale",
    "association_columns",
    "categorical_cell_columns",
    "categorical_nulls",
    "categorical_test_columns",
    "cell_table_columns",
    "classification_comparison_columns",
    "classification_metric_columns",
    "cluster_assignment_columns",
    "cluster_table_columns",
    "curve_columns",
    "model_coef_columns",
    "model_inference_columns",
    "null_label",
    "pairwise_table_columns",
    "posthoc_stat_columns",
    "posthoc_table_columns",
    "prediction_columns",
    "reduction_variance_columns",
    "regression_comparison_columns",
    "regression_metric_columns",
    "selection_profile_columns",
    "selection_ranking_columns",
    "stepwise_profile_columns",
    "term_table_columns",
    "test_table_columns",
]


def test_table_columns() -> list[str]:
    """Columns every omnibus test table in a comparison result must carry."""
    return ["features", "n_used", "pval", "pval_adj", "lower_conf", "upper_conf"]


def posthoc_table_columns() -> list[str]:
    """Columns every post-hoc table must carry.

    A post-hoc table holds one row per feature *and pair of levels* rather than
    one row per feature, which is why it lives in its own slot instead of
    alongside the omnibus tables. ``contrast`` is the readable pair label and
    ``group1`` / ``group2`` are the two levels it is made of, in the direction
    ``estimate`` reads as ``group1 - group2``.
    """
    return [
        "features",
        "contrast",
        "group1",
        "group2",
        "n1",
        "n2",
        "estimate",
        "stderr",
        "statistic",
        "df",
        "pval",
        "pval_adj",
        "lower_conf",
        "upper_conf",
    ]


def posthoc_stat_columns() -> list[str]:
    """The post-hoc columns that carry a number rather than a label."""
    labels = {"features", "contrast", "group1", "group2"}
    return [name for name in posthoc_table_columns() if name not in labels]


def term_table_columns() -> list[str]:
    """Columns a term table must carry.

    A factorial analysis answers on two axes. Whether a feature responds to the
    design at all is one question per feature and lives in ``tests``; which part
    of the design it responds to is one question per feature and model term and
    lives here, because a table of that shape cannot satisfy the
    one-row-per-feature alignment every other table is held to.

    ``terms`` is the term label in ``terms()`` form, ``a`` for a main effect and
    ``a:b`` for an interaction, and ``term_order`` is how many factors it spans.
    The pair is what a simulator's ``truth_term`` is keyed on, so the two tables
    merge without either side being renamed.
    """
    return [
        "features",
        "terms",
        "term_order",
        "n_used",
        "df",
        "ss",
        "f_stat",
        "log2_effect",
        "pval",
        "pval_adj",
    ]


def cell_table_columns() -> list[str]:
    """Columns a cell table must carry.

    One row per feature and cell of the crossed grid. The table also carries one
    column per factor, named after the factor, so a subset can be taken on a
    factor without parsing the ``cell`` label. Those extra columns are why these
    six names are reserved: a factor may not be called any of them.
    """
    return ["features", "cell", "n", "mean", "sd", "se"]


def categorical_nulls() -> list[str]:
    """The null hypotheses a cell table can be built against.

    Port of ``sa_categorical_nulls()``. Three, one per design, and every one of
    them is a statement the result also carries a p-value for.

    ``"independence"`` is two variables cross-classified on one sample: a cell is
    expected at the product of its margins over the total, which is what the
    chi-square test of independence and Fisher's exact test are both about.

    ``"symmetry"`` is one thing measured twice, crossed against itself: a cell is
    expected at the average of it and its transpose, so the diagonal is expected
    to be exactly what it is and only the discordant cells carry a residual. That
    is McNemar's test.

    ``"marginal_homogeneity"`` is three or more repeated conditions summarised as
    a condition-by-response table: every condition is expected to show the pooled
    response rate. On that table it is arithmetically the same formula as
    independence and it is a different claim about the world. That is Cochran's Q,
    and it is why the name is carried beside the numbers rather than inferred from
    them.
    """
    return ["independence", "symmetry", "marginal_homogeneity"]


def null_label(null: str) -> str:
    """A one-line reading of what a null hypothesis says.

    Port of ``sa_null_label()``. Read by ``repr()`` of a categorical result and
    by the mosaic key, so the two cannot describe the same shading differently.
    A name this does not know is handed back as it arrived, which is what R's
    ``switch()`` default does.
    """
    readings = {
        "independence": "independence -- a cell is expected at the product of its margins",
        "symmetry": "symmetry -- a cell is expected at the average of it and its transpose",
        "marginal_homogeneity": (
            "marginal homogeneity -- every condition is expected at the pooled rate"
        ),
    }
    return readings.get(null, null)


def categorical_cell_columns() -> list[str]:
    """Columns the cell table of a contingency scenario must carry.

    Port of ``sa_categorical_cell_columns()``. The canonical form of the table. A
    matrix would say the same thing more compactly, but it is the one shape the
    result contract does not take: a labelled matrix survives a trip through JSON
    in some writers and loses its labels in others, and the labels are the whole
    content of a contingency table. One row per cell keeps them beside the number
    they belong to.

    ``expected``, ``residual`` and ``std_residual`` are all read under the null
    :func:`categorical_nulls` named and mean nothing without it. ``residual`` is
    the Pearson residual, the quantity that squares and sums to the statistic of
    the test that null belongs to. ``std_residual`` is the standardized residual,
    which is referred to a standard normal and so says which cells are
    individually surprising; its variance correction is derived under
    independence and has no counterpart under symmetry, so it is missing there
    rather than a number that looks comparable and is not.
    """
    return [
        "row_level",
        "col_level",
        "observed",
        "expected",
        "residual",
        "std_residual",
        "prop_total",
        "prop_row",
        "prop_col",
    ]


def categorical_test_columns() -> list[str]:
    """Columns every test table of a contingency scenario must carry.

    Port of ``sa_categorical_test_columns()``. One row rather than one row per
    feature, and no ``pval_adj``: there is a single table and therefore a single
    question, so there is no family to adjust across, and a column holding a copy
    of ``pval`` under another name would suggest otherwise.

    ``statistic`` and ``df`` are missing for a test that has neither, which
    Fisher's exact test does not: it conditions on the margins and reads a
    probability off the hypergeometric distribution rather than referring a
    statistic to a null one. The columns exist for every test, as in the
    comparison contract; being finite is not required.
    """
    return ["n_used", "statistic", "df", "pval", "lower_conf", "upper_conf"]


def association_columns() -> list[str]:
    """Columns the association table must carry.

    Port of ``sa_association_columns()``. One row per measure rather than one
    column per measure, because which measures exist depends on the design and on
    the size of the table, and a wide table would carry a column of missing values
    for every measure the design does not define.
    """
    return ["measure", "estimate", "lower_conf", "upper_conf"]


#: The association measures that are ratios centred at 1 rather than at zero.
#:
#: Two of the measures this scenario reports are odds ratios and the rest are
#: centred at zero, which is the whole of what a cutoff or a printed rule has to
#: know about a measure's scale.
RATIO_MEASURES = ("odds_ratio", "odds_ratio_paired")


def assoc_scale(measure: str) -> str:
    """Which scale an association measure lives on.

    Port of ``sa_assoc_scale()``. The zero-centred measures are read on their
    magnitude, which is a no-op for the ones that cannot be negative and the
    correct reading for the ones that can: ``cohens_g`` and
    ``risk_difference_paired`` fall below zero when the later condition lowers
    the response, and a departure downwards is as large as the same one upwards.

    Returns:
        ``"ratio"`` or ``"magnitude"``.
    """
    return "ratio" if measure in RATIO_MEASURES else "magnitude"


def model_coef_columns() -> list[str]:
    """Columns every coefficient table of a fitted model must carry.

    Port of ``sa_model_coef_columns()``. What every model answers is which terms
    it has and what it estimated for each, so those two are the contract and the
    rest is per-model. ``terms`` is also the row order every other table in a
    :class:`~statassist.core.result.SaModel` follows, the way ``features`` is in
    a comparison result.
    """
    return ["terms", "estimate"]


def model_inference_columns() -> list[str]:
    """The inference columns a model either fills or does not have.

    Port of ``sa_model_inference_columns()``. ``statistic`` is a t value for a
    linear model and a Wald z for a logistic one, which is why the column is not
    named after either.

    They come as a group, and a table that has none of them is a different kind
    of table rather than one that lost its values: a penalized fit's estimates
    are deliberately biased and a standard error assumes an unbiased one, so
    there is nothing to put here and ``"pval" in table.columns`` is what tells a
    consumer which kind of table it is holding.
    """
    return ["stderr", "statistic", "df", "pval", "lower_conf", "upper_conf"]


def prediction_columns() -> list[str]:
    """Columns the prediction table of an evaluation must carry.

    Port of ``sa_prediction_columns()``. Long rather than wide: one row per model
    and scored row, so that a set of models of any size fits the same table and a
    plot can be drawn from it by taking a subset rather than by choosing columns.

    ``row`` is the position in ``newdata`` the prediction was made for, which is
    what lets a caller put a prediction back beside the row it belongs to after
    the rows no model could predict have been left out.
    """
    return ["model", "row", "observed", "predicted"]


def regression_metric_columns() -> list[str]:
    """Columns the metric table of a regression evaluation must carry.

    Port of ``sa_regression_metric_columns()``. Six ways of being wrong, and they
    disagree usefully.

    ``cor`` is rank-free agreement, so a model that predicts the right ordering
    on the wrong scale scores well on it and badly on ``r_squared``. ``bias`` is
    the mean residual, which is the one that says a model is wrong in a
    direction rather than merely wrong. And the two calibration numbers are the
    line through predicted against observed: a slope below 1 is the signature of
    a model whose predictions are too spread out for the outcome they are
    predicting, which no summary of the errors reports.
    """
    return [
        "model",
        "n_used",
        "cor",
        "r_squared",
        "rmse",
        "mae",
        "bias",
        "calib_slope",
        "calib_intercept",
    ]


def classification_metric_columns() -> list[str]:
    """Columns the metric table of a classification evaluation must carry.

    Port of ``sa_classification_metric_columns()``. The first three are
    threshold-free and the last three are not, which is the division worth
    knowing: ``auc`` and ``brier`` describe the predicted probabilities
    themselves, while ``accuracy``, ``sensitivity`` and ``specificity`` describe
    what happens when they are turned into calls at
    ``parameters["threshold"]``.
    """
    return [
        "model",
        "n_used",
        "n_events",
        "auc",
        "auc_lower_conf",
        "auc_upper_conf",
        "brier",
        "accuracy",
        "sensitivity",
        "specificity",
    ]


def regression_comparison_columns() -> list[str]:
    """Columns the comparison table of a regression evaluation must carry.

    Port of ``sa_regression_comparison_columns()``. Differences and no p-values,
    which is the honest shape: there is no paired test of two correlations or two
    root mean squared errors on the same rows that this package implements, so a
    column of them would be a number nobody could stand behind.
    """
    return ["model", "delta_cor", "delta_r_squared", "delta_rmse", "delta_mae"]


def classification_comparison_columns() -> list[str]:
    """Columns the comparison table of a classification evaluation must carry.

    Port of ``sa_classification_comparison_columns()``. Three paired tests, and
    each answers a question the others do not.

    ``delta_auc`` is DeLong's test: whether the new model ranks the events above
    the non-events better. ``idi`` is whether it separates the two groups'
    predicted probabilities further apart, which a model can do without changing
    any ordering. ``nri`` is whether the rows it moved were moved the right way,
    reported as a total and split into the events and the non-events, since a
    model that raises everyone's probability improves the events and damages the
    others by the same rule.
    """
    return [
        "model",
        "delta_auc",
        "delta_auc_lower_conf",
        "delta_auc_upper_conf",
        "delta_auc_pval",
        "idi",
        "idi_lower_conf",
        "idi_upper_conf",
        "idi_pval",
        "nri",
        "nri_event",
        "nri_nonevent",
        "nri_lower_conf",
        "nri_upper_conf",
        "nri_pval",
    ]


def curve_columns() -> list[str]:
    """Columns the ROC curve table of a classification evaluation must carry.

    Port of ``sa_curve_columns()``. One row per operating point per model, with
    ``threshold`` the prediction at which that point is reached. The first point
    of every curve has an infinite threshold, which is the point where nothing is
    called an event; written out as JSON that becomes ``null``, and the
    coordinates beside it are what the curve is drawn from either way.
    """
    return ["model", "threshold", "sensitivity", "specificity"]


def reduction_variance_columns() -> list[str]:
    """Columns the component table of a principal component analysis must carry.

    One row per component, and every component is here: a share of the variance
    is only a share if the rest of it is there to be a share of. ``prop_var`` and
    ``cum_var`` are percentages rather than fractions, which is what an axis label
    reads as.
    """
    return ["component", "sdev", "prop_var", "cum_var"]


def cluster_assignment_columns() -> list[str]:
    """Columns the per-point table of a clustering must carry.

    One row per point, in the row order ``points`` is in, which is what lets an
    assignment be painted straight onto the scores of a reduction of the same
    rows. ``cluster`` is a whole number and ``0`` means noise, so a cluster number
    of zero is not a cluster.
    """
    return ["points", "cluster", "silhouette"]


def cluster_table_columns() -> list[str]:
    """Columns the per-cluster table of a clustering must carry.

    One row per cluster that was found and never a row for noise, so the length
    of the table is the number of groups there were and what did not join one is
    ``design["n_noise"]``.
    """
    return ["cluster", "size", "silhouette"]


def selection_ranking_columns() -> list[str]:
    """Columns the per-candidate table of a feature selection must carry.

    One row per predictor that was offered, in the order the search ranked them,
    which is the row order ``candidates`` is in. ``estimate`` is whatever the
    search ranked by and ``engine["importance"]`` names it, since a t statistic
    and a rise in a criterion are the same column under two readings.
    """
    return ["candidates", "estimate", "rank", "selected"]


def selection_profile_columns() -> list[str]:
    """Columns every profile table must carry, whichever search filled it.

    One row per model the search compared, and only these two are shared: an
    elimination's rows are subset sizes and each carries its resampled metrics,
    while a stepwise path's rows are steps and carry both criteria. ``n_vars``
    counts predictors rather than coefficients, so a factor counts once.
    """
    return ["n_vars", "chosen"]


def stepwise_profile_columns() -> list[str]:
    """Columns the profile table of a stepwise search carries, in order.

    Both criteria are reported whichever one searched, since they are two charges
    against the same likelihood and the fit that has one has the other. ``step``
    is the move that reached this model, empty on the row the search started at.
    """
    return ["n_vars", "AIC", "BIC", "step", "chosen"]


def pairwise_table_columns() -> list[str]:
    """Columns every pairwise table must carry.

    The same numbers as a post-hoc table, rearranged into one rectangular table
    per contrast so a single contrast can be read on its own, plus
    ``fold_change`` and ``log2fc``: the ratio of the two group centres, which no
    post-hoc procedure reports because it does not depend on which test was run.
    """
    return ["features", "contrast", "group1", "group2", "fold_change", "log2fc"] + (
        posthoc_stat_columns()
    )
