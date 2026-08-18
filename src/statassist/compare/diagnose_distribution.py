"""Check the assumptions a comparison rests on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from statassist.compare.screen_outliers import screen_outliers
from statassist.contracts.repr import repr_sa_diagnosis
from statassist.kernels.diagnostic import sa_bartlett, sa_ks_normal, sa_levene, sa_shapiro
from statassist.utils.describe import sa_kurtosis, sa_skewness
from statassist.utils.metadata import sa_metadata
from statassist.utils.validate import sa_check_scalar_num, sa_split_for_screening


class sa_result:
    pass


class sa_diagnosis(sa_result):
    pass


@dataclass(repr=False)
class DiagnosisResult:
    analysis: str
    features: list[str]
    design: dict[str, Any]
    parameters: dict[str, Any]
    normality: pd.DataFrame
    variance: pd.DataFrame
    outliers: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, str] = field(default_factory=sa_metadata)

    def __repr__(self) -> str:
        return repr_sa_diagnosis(self)


def sa_normality_table(
    per_feature: dict[str, dict[str, np.ndarray]],
    feats: list[str],
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for f in feats:
        samples = per_feature[f]
        for j, (group, v) in enumerate(samples.items()):
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            try:
                sh = sa_shapiro(v) if v.size >= 3 else {
                    "shapiro_stat": np.nan,
                    "shapiro_pval": np.nan,
                }
            except Exception:
                sh = {"shapiro_stat": np.nan, "shapiro_pval": np.nan}
            try:
                ks = sa_ks_normal(v) if v.size >= 2 else {
                    "ks_stat": np.nan,
                    "ks_pval": np.nan,
                }
            except Exception:
                ks = {"ks_stat": np.nan, "ks_pval": np.nan}
            blocks.append(
                pd.DataFrame(
                    {
                        "features": [f],
                        "group": [group],
                        "n_used": [v.size],
                        "shapiro_stat": [sh["shapiro_stat"]],
                        "shapiro_pval": [sh["shapiro_pval"]],
                        "ks_stat": [ks["ks_stat"]],
                        "ks_pval": [ks["ks_pval"]],
                        "skewness": [sa_skewness(v) if v.size else np.nan],
                        "excess_kurtosis": [sa_kurtosis(v) if v.size else np.nan],
                    }
                )
            )
    return pd.concat(blocks, ignore_index=True)


def sa_variance_table(
    per_feature: dict[str, dict[str, np.ndarray]],
    feats: list[str],
    grouped: bool,
    center: str,
    trim: float,
) -> pd.DataFrame:
    columns = [
        "features", "n_used", "n_groups", "levene_stat", "levene_df1",
        "levene_df2", "levene_pval", "bartlett_stat", "bartlett_df", "bartlett_pval",
    ]
    if not grouped:
        return pd.DataFrame(columns=columns)

    rows = []
    for f in feats:
        samples = per_feature[f]
        try:
            lev = sa_levene(samples, center, trim)
        except Exception:
            lev = {
                "levene_stat": np.nan,
                "levene_df1": np.nan,
                "levene_df2": np.nan,
                "levene_pval": np.nan,
            }
        try:
            bart = sa_bartlett(samples)
        except Exception:
            bart = {
                "bartlett_stat": np.nan,
                "bartlett_df": np.nan,
                "bartlett_pval": np.nan,
            }
        rows.append(
            {
                "features": f,
                "n_used": sum(v.size for v in samples.values()),
                "n_groups": len(samples),
                **lev,
                **bart,
            }
        )
    return pd.DataFrame(rows)[columns]


def sa_new_diagnosis(
    features: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    normality: pd.DataFrame,
    variance: pd.DataFrame,
    outliers: pd.DataFrame,
    alpha: float,
) -> DiagnosisResult:
    summary = pd.DataFrame({"features": features})
    summary["n_levels"] = [
        int((normality["features"] == f).sum()) for f in features
    ]
    summary["n_outliers"] = [
        int((outliers["features"] == f).sum()) if len(outliers) else 0
        for f in features
    ]
    summary["min_shapiro_pval"] = [
        float(normality.loc[normality["features"] == f, "shapiro_pval"].min(skipna=True))
        if normality.loc[normality["features"] == f, "shapiro_pval"].notna().any()
        else np.nan
        for f in features
    ]
    summary["normal_ok"] = summary["min_shapiro_pval"] > alpha
    if len(variance) == 0:
        summary["variance_ok"] = np.nan
    else:
        summary["variance_ok"] = [
            variance.loc[variance["features"] == f, "levene_pval"].iloc[0] > alpha
            if f in variance["features"].values
            else np.nan
            for f in features
        ]

    obj = DiagnosisResult(
        analysis="distribution_diagnosis",
        features=features,
        design=design,
        parameters=parameters,
        normality=normality,
        variance=variance,
        outliers=outliers,
        summary=summary,
    )
    obj.__class__ = type("DiagnosisResult", (sa_diagnosis, DiagnosisResult), {})
    return obj


def sa_diagnose_samples(
    per_feature: list[Any],
    feats: list[str],
    group_lv: list[str],
    paired: bool,
    alpha: float = 0.05,
) -> dict[str, Any]:
    from statassist.utils.foldchange import sa_feature_samples

    samples_by_feature: dict[str, dict[str, np.ndarray]] = {}
    for i, f in enumerate(feats):
        if paired:
            mat = per_feature[i]
            samples_by_feature[f] = {
                group_lv[j]: mat.iloc[:, j].to_numpy() for j in range(len(group_lv))
            }
        else:
            samples_by_feature[f] = per_feature[i]

    normality = sa_normality_table(samples_by_feature, feats)
    variance = sa_variance_table(samples_by_feature, feats, not paired, "median", 0.1)
    diag = sa_new_diagnosis(
        feats,
        design={"group_lv": group_lv, "grouped": True},
        parameters={"alpha": alpha},
        normality=normality,
        variance=variance,
        outliers=pd.DataFrame(columns=["features"]),
        alpha=alpha,
    )
    return {
        "normality": normality,
        "variance": variance,
        "summary": diag.summary,
    }


def diagnose_distribution(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: Any | None = None,
    group_lv: list[str] | None = None,
    *,
    alpha: float = 0.05,
    criterion: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.5,
    center: str = "median",
    trim: float = 0.1,
) -> DiagnosisResult:
    if criterion not in ("iqr", "robust_z", "grubbs"):
        raise ValueError('`criterion` must be one of "iqr", "robust_z", or "grubbs".')
    if center not in ("median", "mean", "trimmed"):
        raise ValueError('`center` must be one of "median", "mean", or "trimmed".')
    sa_check_scalar_num(alpha, "alpha", 0, 1, lower_open=True)
    sa_check_scalar_num(trim, "trim", 0, 0.5, upper_open=True)

    split = sa_split_for_screening(data, feats, group, group_lv)
    groups_used = list(split["rows"].keys()) if split["grouped"] else [np.nan]

    per_feature: dict[str, dict[str, np.ndarray]] = {}
    for f in feats:
        per_feature[f] = {
            lv: split["data"][f].to_numpy()[rows]
            for lv, rows in split["rows"].items()
        }

    outliers = screen_outliers(
        data, feats, group, group_lv,
        criterion=criterion,
        iqr_multiplier=iqr_multiplier,
        z_threshold=z_threshold,
        alpha=alpha,
    )

    return sa_new_diagnosis(
        features=feats,
        design={
            "group_lv": groups_used if split["grouped"] else None,
            "grouped": split["grouped"],
        },
        parameters={
            "alpha": alpha,
            "criterion": criterion,
            "iqr_multiplier": iqr_multiplier,
            "z_threshold": z_threshold,
            "center": center,
            "trim": trim,
        },
        normality=sa_normality_table(per_feature, feats),
        variance=sa_variance_table(
            per_feature, feats, split["grouped"], center, trim
        ),
        outliers=outliers,
        alpha=alpha,
    )
