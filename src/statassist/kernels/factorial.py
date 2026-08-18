"""Kernels of a fully crossed between-subject analysis (R kernel_factorial.R).

Two things set this apart from the one-way kernels. A factorial analysis answers
on two axes at once, the whole model and the individual terms, and both come out
of one call so that the two ends of a result cannot come to disagree about the
mean square error they share.

And the arithmetic runs on the cell means rather than on the observations. A
fully crossed model gives every row of a cell the same predictor values, so the
residual sum of squares of any sub-model is the within-cell sum of squares plus
the weighted residual sum of squares of the cell means, with the cell counts as
weights. Every sum of squares here is a difference of two such residuals, so the
within-cell part cancels. It is exact rather than approximate: the two
formulations have the same normal equations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.kernels.anova import sa_oneway_anova
from statassist.kernels.posthoc import sa_posthoc_columns
from statassist.utils.factorial_utils import sa_fact_term_labels, sa_fact_terms
from statassist.utils.validate import sa_row


def sa_contr_sum(n: int) -> np.ndarray:
    """R ``stats::contr.sum(n)``: identity above, a row of -1 below."""
    out = np.zeros((n, n - 1), dtype=float)
    out[: n - 1, :] = np.eye(n - 1)
    out[n - 1, :] = -1.0
    return out


def sa_fact_cell_matrix(
    factor_lv: dict[str, list[str]],
    cells: pd.DataFrame,
) -> dict[str, Any]:
    """Sum-to-zero model matrix of the cells, intercept first then terms in order.

    Sum-to-zero coding is what makes a term's columns orthogonal to the terms it
    does not contain in a balanced design, which is why the three sum-of-squares
    types agree there and why Type III means what the unweighted marginal means
    say it does. The columns of an interaction are the elementwise products of
    the columns of the factors it is over.
    """
    terms = sa_fact_terms(list(factor_lv.keys()))
    codes = {f: sa_contr_sum(len(lv)) for f, lv in factor_lv.items()}
    n_cells = len(cells)

    x = np.ones((n_cells, 1), dtype=float)
    assign = [0]
    for k, term in enumerate(terms, start=1):
        block = np.ones((n_cells, 1), dtype=float)
        for f in term:
            cf = codes[f][cells[f].to_numpy(dtype=int) - 1, :]
            b, c = block.shape[1], cf.shape[1]
            block = np.tile(block, (1, c)) * np.repeat(cf, b, axis=1)
        x = np.hstack([x, block])
        assign.extend([k] * block.shape[1])

    return {"x": x, "assign": np.asarray(assign, dtype=int), "terms": terms}


def sa_fact_ss_plan(
    terms: list[list[str]],
    assign: np.ndarray,
    ss_type: str,
) -> list[dict[str, np.ndarray]]:
    """The two models whose difference is each term's sum of squares.

    Every sum of squares in an ANOVA table is a model comparison, and the three
    types differ only in which pair of models is compared:

    - ``III`` — everything else stays in and the term comes out.
    - ``II``  — the term is added to the model holding every term that does not
      contain it.
    - ``I``   — sequential, so the answer depends on declaration order.
    """
    n_terms = len(terms)
    cols_of = [np.flatnonzero(assign == k + 1) for k in range(n_terms)]
    intercept = np.flatnonzero(assign == 0)

    plan: list[dict[str, np.ndarray]] = []
    for k in range(n_terms):
        if ss_type == "III":
            keep = [u for u in range(n_terms) if u != k]
        elif ss_type == "II":
            # A term contains this one when its factors include all of them,
            # which is also true of the term itself, so it drops out without a
            # second condition.
            keep = [
                u for u in range(n_terms) if not set(terms[k]).issubset(terms[u])
            ]
        elif ss_type == "I":
            keep = list(range(k))
        else:
            raise ValueError(f"internal error: unknown `ss_type` `{ss_type}`.")

        base = np.sort(
            np.concatenate([intercept] + [cols_of[u] for u in keep])
            if keep
            else intercept
        )
        full = np.sort(np.concatenate([base, cols_of[k]]))
        plan.append({"base": base, "full": full})
    return plan


def sa_factorial_plan(
    factor_lv: dict[str, list[str]],
    cells: pd.DataFrame,
    ss_type: str,
) -> dict[str, Any]:
    """Everything about the model that does not depend on the data.

    Settled once and handed to the kernel rather than rebuilt per feature.
    """
    mat = sa_fact_cell_matrix(factor_lv, cells)
    return {
        "x": mat["x"],
        "assign": mat["assign"],
        "terms": mat["terms"],
        "labels": sa_fact_term_labels(mat["terms"]),
        "orders": [len(t) for t in mat["terms"]],
        "ss": sa_fact_ss_plan(mat["terms"], mat["assign"], ss_type),
    }


_TERM_COLUMNS = [
    "n_used",
    "df",
    "ss",
    "ms",
    "f_stat",
    "df_error",
    "eta_sq",
    "partial_eta_sq",
    "pval",
]


def sa_factorial_anova(
    samples: dict[str, np.ndarray],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Factorial analysis of variance, whole model and term by term.

    The whole-model test is the one-way ANOVA that treats the cells as groups,
    which is the same test as the F test of a fully crossed model, so it is
    delegated rather than rewritten.
    """
    labels = list(samples.keys())
    values = list(samples.values())
    n = np.array([v.size for v in values], dtype=float)
    if np.any(n == 0):
        empty = ", ".join(labels[i] for i in np.flatnonzero(n == 0))
        raise ValueError(
            "cell(s) with no usable observation, which leaves a crossed model "
            f"with nothing to estimate there: {empty}."
        )

    model = sa_oneway_anova(values)
    model["n_cells"] = model.pop("n_groups")

    means = np.array([float(np.mean(v)) for v in values])
    ss_within = float(sum(float(np.sum((v - m) ** 2)) for v, m in zip(values, means)))
    df_error = float(model["df2"])
    ms_error = ss_within / df_error
    grand = float(np.sum(n * means) / np.sum(n))
    ss_total = ss_within + float(np.sum(n * (means - grand) ** 2))

    # One weighted least squares problem per distinct sub-model. Several terms
    # ask for the same one under Type I and II, so the answers are kept.
    sw = np.sqrt(n)
    xw = plan["x"] * sw[:, None]
    yw = means * sw
    seen: dict[str, dict[str, float]] = {}

    def fit(cols: np.ndarray) -> dict[str, float]:
        key = ",".join(str(c) for c in cols)
        got = seen.get(key)
        if got is None:
            xs = xw[:, cols]
            coef, _, rank, _ = np.linalg.lstsq(xs, yw, rcond=None)
            resid = yw - xs @ coef
            got = {"rss": float(resid @ resid), "rank": int(rank)}
            seen[key] = got
        return got

    rows = []
    for pair in plan["ss"]:
        base = fit(pair["base"])
        full = fit(pair["full"])
        df = full["rank"] - base["rank"]
        # Subtracting two residual sums of squares of nearly equal size can land
        # a hair below zero on a term that explains nothing at all.
        ss = max(base["rss"] - full["rss"], 0.0)
        if df < 1:
            rows.append(
                sa_row(
                    n_used=float(np.sum(n)),
                    df=0.0,
                    ss=ss,
                    ms=np.nan,
                    f_stat=np.nan,
                    df_error=df_error,
                    eta_sq=np.nan,
                    partial_eta_sq=np.nan,
                    pval=np.nan,
                )[_TERM_COLUMNS].to_numpy(dtype=float)
            )
            continue
        ms = ss / df
        f_stat = ms / ms_error
        rows.append(
            sa_row(
                n_used=float(np.sum(n)),
                df=float(df),
                ss=ss,
                ms=ms,
                f_stat=f_stat,
                df_error=df_error,
                eta_sq=ss / ss_total,
                partial_eta_sq=ss / (ss + ss_within),
                pval=float(stats.f.sf(f_stat, df, df_error)),
            )[_TERM_COLUMNS].to_numpy(dtype=float)
        )

    return {
        "model": model,
        "terms": pd.DataFrame(rows, index=plan["labels"], columns=_TERM_COLUMNS),
        "means": means,
        "n": n,
        "ms_error": ms_error,
        "df_error": df_error,
    }


def sa_factorial_tukey(
    fit: dict[str, Any],
    skeleton: dict[str, Any],
    nmeans: np.ndarray,
    rows: np.ndarray,
    conf_level: float = 0.95,
) -> np.ndarray:
    """Tukey-Kramer comparisons of marginal means and of simple effects.

    A marginal mean is the *unweighted* mean of the cell means, so a level's
    mean is not pulled towards whichever combination of the other factors
    happened to be sampled most. The weights being ``1/m``, the variance of a
    difference is the Kramer form, which reduces to Tukey's own when the cells
    are equal in size.

    The family is one block of contrasts rather than the whole table: the
    studentised range is over the number of levels of the factor being compared,
    so the p-values are family-wise within each block without further
    adjustment.
    """
    means = fit["means"]
    n = fit["n"]
    mse = fit["ms_error"]
    df = fit["df_error"]
    if mse <= 0:
        raise ValueError(
            "the mean square error of the model is zero, so no contrast can be "
            "scaled."
        )

    out = []
    for k in rows:
        s1 = skeleton["sel1"][k]
        s2 = skeleton["sel2"][k]
        estimate = float(np.mean(means[s1]) - np.mean(means[s2]))
        variance = mse * (
            float(np.sum((1 / len(s1)) ** 2 / n[s1]))
            + float(np.sum((1 / len(s2)) ** 2 / n[s2]))
        )
        # The studentised range is the range of the means over the standard
        # error of one of them, so the divisor carries a 1/2 that a t statistic
        # does not.
        stderr = float(np.sqrt(variance / 2))
        q_stat = estimate / stderr
        q_crit = float(stats.studentized_range.ppf(conf_level, int(nmeans[k]), df))
        out.append(
            sa_row(
                n1=float(np.sum(n[s1])),
                n2=float(np.sum(n[s2])),
                estimate=estimate,
                stderr=stderr,
                statistic=q_stat,
                df=df,
                pval=float(
                    stats.studentized_range.sf(abs(q_stat), int(nmeans[k]), df)
                ),
                lower_conf=estimate - q_crit * stderr,
                upper_conf=estimate + q_crit * stderr,
            )[sa_posthoc_columns()].to_numpy(dtype=float)
        )

    return np.array(out, dtype=float).reshape(len(rows), len(sa_posthoc_columns()))
