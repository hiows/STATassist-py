"""Run every applicable multi-group test at once.

Port of ``R/compare_multiple_groups.R``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify, warn
from ..core.result import SaComparison, new_comparison
from ..core.tables import feature_table, posthoc_table
from ..core.validate import (
    UNSET,
    align_by_subject,
    check_flag,
    check_p_adjust,
    check_scalar_num,
    control_first,
    validate_wide_input,
)
from ..diagnose.distribution import diagnose_samples
from ..kernel.anova import friedman, kruskal, oneway_anova, rm_anova, welch_anova, yuen_anova
from ..kernel.posthoc import (
    conover,
    dunn,
    games_howell,
    pairwise_paired_t,
    pairwise_yuen,
    posthoc_columns,
    tukey,
)
from ..transform._foldchange import INPUT_SCALES, resolve_fc_mean
from ._multi import (
    feature_samples,
    group_centers,
    multi_fold_change,
    pairwise_tables,
    require_groups,
)

__all__ = ["compare_multiple_groups"]

#: Per feature, either the samples by level or a subjects-by-conditions frame.
PerFeature = dict[str, Any]


class Spec(NamedTuple):
    """One omnibus test and the post-hoc procedure that shares its assumptions.

    Keeping the two in one object is what makes it impossible to follow a
    rank-based omnibus test with a parametric comparison by accident.
    """

    id: str
    label: str
    columns: list[str]
    omnibus: Callable[[str], dict[str, float]]
    posthoc_id: str
    posthoc_label: str
    posthoc_familywise: bool
    posthoc: Callable[[str], pd.DataFrame]


def compare_multiple_groups(
    data: Any,
    feats: Any,
    group: Any,
    group_lv: Any,
    control_label: Any = None,
    id: Any = None,  # noqa: A002 - matches the R argument name
    paired: bool = False,
    conf_level: float = 0.95,
    tr: float = 0.2,
    posthoc: bool = True,
    posthoc_alpha: float = 0.05,
    fc_mean: Any = UNSET,
    input_scale: str = INPUT_SCALES[0],
    p_adjust: str = "BH",
    posthoc_p_adjust: str = "holm",
    diagnose: bool = True,
) -> SaComparison:
    """Run every applicable multi-group test at once.

    Compares three or more group levels across any number of numeric features and
    returns the omnibus tests side by side, each followed by the post-hoc
    procedure that shares its assumptions. As with
    :func:`~statassist.compare_two_groups`, nothing is chosen on the caller's
    behalf: reporting the parametric, the rank-based and the robust result
    together makes disagreement between them visible.

    Which family runs depends on ``paired``:

    ==============  ==========================  =========================
    slot            ``paired=False``            ``paired=True``
    ==============  ==========================  =========================
    ``anova_test``  One-way ANOVA               Repeated measures ANOVA
    ``welch_test``  Welch's ANOVA               not applicable
    ``robust_test`` Yuen's trimmed mean ANOVA   not applicable
    ``kruskal_test``Kruskal-Wallis              Friedman
    ==============  ==========================  =========================

    Each one is followed by its own post-hoc procedure, never by a borrowed one:
    one-way ANOVA by Tukey's HSD, Welch's ANOVA by Games-Howell, the trimmed mean
    ANOVA by pairwise Yuen tests, Kruskal-Wallis by Dunn's test, repeated
    measures ANOVA by pairwise paired t-tests and Friedman by Conover's test.

    Args:
        data: Wide frame (or 2-D array), one row per observation and one column
            per feature.
        feats: Names of the numeric columns to test. One output row per entry.
        group: Grouping vector with one entry per row of ``data``.
        group_lv: At least three group levels. The first is the reference the
            ``effect`` table is expressed against and the level every post-hoc
            contrast subtracts, so a contrast reads ``treat_1 - control``. Rows
            belonging to any other level are dropped.
        control_label: The level to hold as the reference. Naming it moves that
            level to the front of ``group_lv`` and leaves the rest in the order
            given, so the fold change and every post-hoc contrast are expressed
            against it without the levels having to be retyped.
        id: Subject identifier with one entry per row of ``data``. Required when
            ``paired=True`` and ignored otherwise. Row order pairing, which
            :func:`~statassist.compare_two_groups` allows, is deliberately not
            offered here: with three or more conditions it would also have to
            assume every condition is stored in the same subject order, and
            there is no way to notice when it is not.
        paired: Whether the levels of ``group`` are repeated conditions measured
            on the subjects named by ``id``.
        conf_level: Confidence level for the post-hoc intervals.
        tr: Trimming proportion for the trimmed mean ANOVA, in ``[0, 0.5)``.
            Ignored when ``paired=True``.
        posthoc: Whether to run the pairwise stage. When ``False`` the result
            carries no ``posthoc`` or ``pairwise`` slot at all.
        posthoc_alpha: A feature enters the post-hoc stage when its omnibus
            ``pval_adj`` is at or below this value. Set it to 1 to compare every
            feature regardless of its omnibus result.
        fc_mean: Which centre the fold change divides, ``"arith"`` or
            ``"geom"``. Left unset it is ``"geom"`` when ``input_scale="log2"``
            and ``"arith"`` otherwise.
        input_scale: The scale ``data`` arrives on, ``"raw"`` or ``"log2"``. This
            changes the ``effect`` table only, never the tests.
        p_adjust: Multiplicity adjustment applied across ``feats`` within each
            omnibus table. ``"none"`` disables it.
        posthoc_p_adjust: Multiplicity adjustment applied across the contrasts
            within each feature of a post-hoc table. Ignored for Tukey's HSD and
            Games-Howell, whose p-values are already family-wise.
        diagnose: Whether to attach the assumption checks the tests rest on.

    Returns:
        A :class:`~statassist.core.result.SaComparison` laid out as
        :func:`~statassist.compare_two_groups` returns, plus the ``posthoc`` and
        ``pairwise`` slots that only a comparison of three or more levels has.
        ``effect`` holds ``ref_center``, ``extreme_level``, ``extreme_center``,
        ``fold_change`` and ``log2fc``, the ratio putting the level furthest
        from the reference over the reference.

    Raises:
        SaValueError: If an argument is unusable, or if the conditions cannot be
            lined up by subject.

    Warns:
        SaWarning: If a test could not be run for at least one feature. Those
            rows come back missing and are named together.

    Examples:
        A simulator's ``args`` is named after this function's arguments, so the
        analysis is one call away.

        >>> from statassist import simulate_multiple_groups
        >>> sim = simulate_multiple_groups(
        ...     n_feats=4, n_control=10, n_treat=(10, 10), n_up=1, n_down=1, seed=1
        ... )
        >>> res = compare_multiple_groups(**sim.args)
        >>> list(res.tests)
        ['anova_test', 'welch_test', 'robust_test', 'kruskal_test']

        The pairwise stage is the same numbers in two shapes: stacked in
        ``posthoc``, and one rectangular table per contrast in ``pairwise``.

        >>> list(res.posthoc["anova_test"].columns)[:4]
        ['features', 'contrast', 'group1', 'group2']
        >>> list(res.pairwise["anova_test"])
        ['treat_1 - control', 'treat_2 - control', 'treat_2 - treat_1']

        Every pairwise table holds every feature, whether or not it qualified.

        >>> list(res.pairwise["anova_test"]["treat_1 - control"]["features"])
        ['prot_1', 'prot_2', 'prot_3', 'prot_4']

        The planted direction comes back in the sign of ``log2fc``.

        >>> up = sim.truth.loc[sim.truth["direction"] == "up", "features"].iloc[0]
        >>> bool(res.effect.set_index("features").loc[up, "log2fc"] > 0)
        True
    """
    if input_scale not in INPUT_SCALES:
        raise SaValueError("`input_scale` must be one of: " + ", ".join(INPUT_SCALES) + ".")
    mean_type = resolve_fc_mean(fc_mean, input_scale)
    paired = check_flag(paired, "paired")
    posthoc = check_flag(posthoc, "posthoc")
    diagnose = check_flag(diagnose, "diagnose")
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    tr = check_scalar_num(tr, "tr", 0, 0.5, upper_open=True)
    posthoc_alpha = check_scalar_num(posthoc_alpha, "posthoc_alpha", 0, 1, lower_open=True)
    p_adjust = check_p_adjust(p_adjust, "p_adjust")
    posthoc_p_adjust = check_p_adjust(posthoc_p_adjust, "posthoc_p_adjust")

    if paired and id is None:
        raise SaValueError(
            "`paired = True` needs `id` to say which rows belong to the same "
            "subject. Three or more conditions cannot be matched by row order."
        )
    if id is not None and not paired:
        warn("`id` is only used to match repeated conditions and is ignored when `paired = False`.")

    validated = validate_wide_input(data, feats, group, group_lv, id=id, min_levels=3)
    frame = validated.data
    names = validated.feats
    if validated.group is None:  # pragma: no cover - min_levels forces a grouping
        raise SaValueError("`group` must be supplied.")
    levels = control_first([str(level) for level in validated.group.categories], control_label)
    membership = np.asarray(validated.group.astype(str), dtype=object)

    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")

    if paired:
        aligned = align_by_subject(validated.id, membership, levels)
        unmatched = aligned.unmatched
        if unmatched:
            notify(f"Dropped {len(unmatched)} subject(s) missing at least one condition.")
        per_feature = _repeated_samples(frame, names, aligned.idx)
    else:
        unmatched = []
        per_feature = _independent_samples(frame, names, membership, levels)

    centers = group_centers(per_feature, names, levels, mean_type, paired, input_scale)
    effect = multi_fold_change(centers, names, levels, mean_type)

    specs = (
        _repeated_specs(per_feature, conf_level)
        if paired
        else _independent_specs(per_feature, levels, conf_level, tr)
    )

    # The specs are keyed by feature name and `feature_table` calls by position,
    # so the index is resolved here rather than inside every spec.
    def by_index(spec: Spec) -> Callable[[int], dict[str, float]]:
        return lambda index: spec.omnibus(names[index])

    tests = {
        name: feature_table(
            names,
            spec.columns,
            spec.label,
            fun=by_index(spec),
            p_adjust_method=p_adjust,
        )
        for name, spec in specs.items()
    }

    posthoc_tables: dict[str, pd.DataFrame] = {}
    pairwise = {}
    n_posthoc = dict.fromkeys(specs, 0)
    if posthoc:
        for name, spec in specs.items():
            adjusted = tests[name]["pval_adj"].to_numpy(dtype=float)
            qualified = [
                feature
                for feature, value in zip(names, adjusted, strict=True)
                if not np.isnan(value) and value <= posthoc_alpha
            ]
            n_posthoc[name] = len(qualified)
            posthoc_tables[name] = posthoc_table(
                qualified,
                levels,
                posthoc_columns(),
                spec.posthoc_label,
                fun=spec.posthoc,
                # A studentised range p-value is already family-wise over the
                # same set of contrasts an adjustment would be applied to, so
                # adjusting it again would correct twice for one comparison.
                p_adjust_method="none" if spec.posthoc_familywise else posthoc_p_adjust,
            )
            pairwise[name] = pairwise_tables(posthoc_tables[name], centers.centers, names, levels)

    return new_comparison(
        analysis="multi_group_comparison",
        features=names,
        design={
            "group_lv": levels,
            "paired": paired,
            "pairing": "id" if paired else None,
            "n_dropped": validated.n_dropped,
            "unmatched_ids": unmatched,
        },
        parameters={
            "alternative": "two.sided",
            "conf_level": conf_level,
            "tr": float("nan") if paired else tr,
            "fc_mean": mean_type,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
            "posthoc": posthoc,
            "posthoc_alpha": posthoc_alpha,
            "posthoc_p_adjust": posthoc_p_adjust,
            "n_posthoc": n_posthoc,
        },
        effect=effect,
        tests=tests,
        posthoc=posthoc_tables,
        pairwise=pairwise,
        test_info={
            name: {
                "id": spec.id,
                "label": spec.label,
                "paired": paired,
                "posthoc_id": spec.posthoc_id,
                "posthoc_label": spec.posthoc_label,
            }
            for name, spec in specs.items()
        },
        diagnostics=(diagnose_samples(per_feature, names, levels, paired) if diagnose else None),
        subclass="sa_multi_group",
    )


def _independent_samples(
    frame: pd.DataFrame,
    feats: Sequence[str],
    membership: np.ndarray,
    group_lv: Sequence[str],
) -> PerFeature:
    """The observations of each feature, by level, missing values dropped.

    Handled per feature rather than once for the whole frame: a feature with a
    missing value somewhere costs only itself an observation, which is why
    ``n_used`` can differ between features.
    """
    positions = {level: np.flatnonzero(membership == level) for level in group_lv}
    out: PerFeature = {}
    for name in feats:
        column = frame[name].to_numpy(dtype=float)
        held = {}
        for level in group_lv:
            values = column[positions[level]]
            held[level] = values[~np.isnan(values)]
        out[str(name)] = held
    return out


def _repeated_samples(
    frame: pd.DataFrame,
    feats: Sequence[str],
    idx: pd.DataFrame,
) -> PerFeature:
    """One complete subjects-by-conditions rectangle per feature.

    A subject with a missing value anywhere across the conditions is dropped
    from that feature, since a within-subject test has nothing to compare a
    partial subject against.
    """
    out: PerFeature = {}
    for name in feats:
        column = frame[name].to_numpy(dtype=float)
        matrix = pd.DataFrame(
            column[idx.to_numpy(dtype=int)],
            index=idx.index,
            columns=idx.columns,
        )
        out[str(name)] = matrix.dropna()
    return out


def _independent_specs(
    per_feature: PerFeature,
    group_lv: Sequence[str],
    conf_level: float,
    tr: float,
) -> dict[str, Spec]:
    """Omnibus and post-hoc pairs for independent levels.

    Port of ``sa_multi_specs_independent()``.
    """
    levels = [str(level) for level in group_lv]

    def samples(feature: str, n_min: int) -> Mapping[str, np.ndarray]:
        return require_groups(feature_samples(per_feature[feature], levels, False), n_min)

    return {
        "anova_test": Spec(
            id="oneway_anova",
            label="One-way ANOVA",
            columns=[
                "n_used",
                "n_groups",
                "f_stat",
                "df1",
                "df2",
                "eta_sq",
                "omega_sq",
                "pval",
                "lower_conf",
                "upper_conf",
            ],
            omnibus=lambda feature: oneway_anova(samples(feature, 2)),
            posthoc_id="tukey_hsd",
            posthoc_label="Tukey HSD",
            posthoc_familywise=True,
            posthoc=lambda feature: tukey(samples(feature, 2), conf_level),
        ),
        "welch_test": Spec(
            id="welch_anova",
            label="Welch's one-way ANOVA",
            columns=[
                "n_used",
                "n_groups",
                "f_stat",
                "df1",
                "df2",
                "eta_sq",
                "omega_sq",
                "pval",
                "lower_conf",
                "upper_conf",
            ],
            omnibus=lambda feature: welch_anova(samples(feature, 2)),
            posthoc_id="games_howell",
            posthoc_label="Games-Howell post-hoc test",
            posthoc_familywise=True,
            posthoc=lambda feature: games_howell(samples(feature, 2), conf_level),
        ),
        "robust_test": Spec(
            id="yuen_anova",
            label="Yuen's trimmed mean one-way ANOVA",
            columns=[
                "n_used",
                "n_groups",
                "f_stat",
                "df1",
                "df2",
                "robust_eta_sq",
                "pval",
                "lower_conf",
                "upper_conf",
            ],
            omnibus=lambda feature: yuen_anova(samples(feature, 2), tr),
            posthoc_id="pairwise_yuen",
            posthoc_label="Pairwise Yuen tests",
            posthoc_familywise=False,
            posthoc=lambda feature: pairwise_yuen(samples(feature, 2), tr, conf_level),
        ),
        "kruskal_test": Spec(
            id="kruskal_wallis",
            label="Kruskal-Wallis test",
            columns=[
                "n_used",
                "n_groups",
                "h_stat",
                "df",
                "epsilon_sq",
                "eta_sq_rank",
                "pval",
                "lower_conf",
                "upper_conf",
            ],
            omnibus=lambda feature: kruskal(samples(feature, 1)),
            posthoc_id="dunn_test",
            posthoc_label="Dunn's post-hoc test",
            posthoc_familywise=False,
            posthoc=lambda feature: dunn(samples(feature, 1), conf_level),
        ),
    }


def _repeated_specs(per_feature: PerFeature, conf_level: float) -> dict[str, Spec]:
    """Omnibus and post-hoc pairs for repeated conditions.

    Port of ``sa_multi_specs_repeated()``.
    """
    return {
        "anova_test": Spec(
            id="repeated_measures_anova",
            label="Repeated measures ANOVA",
            columns=[
                "n_used",
                "n_groups",
                "f_stat",
                "df1",
                "df2",
                "partial_eta_sq",
                "gen_eta_sq",
                "mauchly_w",
                "mauchly_pval",
                "gg_eps",
                "pval_gg",
                "hf_eps",
                "pval_hf",
                "pval",
                "lower_conf",
                "upper_conf",
            ],
            omnibus=lambda feature: rm_anova(per_feature[feature]),
            posthoc_id="pairwise_paired_t",
            posthoc_label="Pairwise paired t-tests",
            posthoc_familywise=False,
            posthoc=lambda feature: pairwise_paired_t(per_feature[feature], conf_level),
        ),
        "kruskal_test": Spec(
            id="friedman_test",
            label="Friedman test",
            columns=[
                "n_used",
                "n_groups",
                "chi_sq",
                "df",
                "kendalls_w",
                "pval",
                "lower_conf",
                "upper_conf",
            ],
            omnibus=lambda feature: friedman(per_feature[feature]),
            posthoc_id="conover_posthoc",
            posthoc_label="Conover post-hoc test",
            posthoc_familywise=False,
            posthoc=lambda feature: conover(per_feature[feature], conf_level),
        ),
    }
