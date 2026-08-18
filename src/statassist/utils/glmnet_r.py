"""glmnet's penalised path and caret's performance summaries, R conventions kept.

Two conventions have to be carried over for a Python penalised fit to mean the
same thing an R one does.

glmnet standardises both sides before it optimises and puts the coefficients
back on the original scale afterwards, and it divides the requested lambda by
the standard deviation of the outcome on the way in. Handing scikit-learn the
raw columns instead solves a different problem that happens to have the same
argument names.

And caret's ``Rsquared`` is a squared correlation between prediction and
observation, not one minus a residual sum of squares over a total one. The two
agree on a training fit and disagree on a held-out fold, which is the only place
this is ever used.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import enet_path


def sa_r_sd(x: np.ndarray) -> float:
    """glmnet's scale factor: the population standard deviation, ``1/n``."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))


def sa_glmnet_path(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lambdas: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200000,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of a Gaussian glmnet fit over a lambda path.

    Both sides are standardised, the path is solved in that space, and the
    coefficients are put back on the original scale, which is what
    ``glmnet(standardize = TRUE)`` does. The lambda handed to the solver is
    divided by the outcome's standard deviation for the same reason glmnet's
    Fortran does it: the objective it scales is written for a unit-variance
    outcome.

    Returns the intercepts, one per lambda, and the coefficient matrix shaped
    ``(n_features, n_lambda)``, both in the order `lambdas` arrived in.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    xm = x.mean(axis=0)
    xs = np.sqrt(((x - xm) ** 2).mean(axis=0))
    # A constant column has nothing to standardise by and nothing to contribute.
    xs = np.where(xs > 0, xs, 1.0)
    z = (x - xm) / xs

    ym = float(y.mean())
    ys = sa_r_sd(y)
    if ys == 0:
        ys = 1.0
    w = (y - ym) / ys

    lambdas = np.asarray(lambdas, dtype=float)
    scaled = lambdas / ys

    if alpha <= 0:
        # scikit-learn's coordinate descent refuses a pure L2 penalty, and the
        # ridge solution is a linear system anyway.
        gram = z.T @ z / n
        rhs = z.T @ w / n
        raw = np.column_stack(
            [np.linalg.solve(gram + lam * np.eye(z.shape[1]), rhs) for lam in scaled]
        )
    else:
        # enet_path sorts the path descending and returns it that way.
        order = np.argsort(scaled)[::-1]
        _, coefs, _ = enet_path(
            z,
            w,
            l1_ratio=alpha,
            alphas=scaled[order],
            tol=tol,
            max_iter=max_iter,
            check_input=False,
        )
        raw = np.empty_like(coefs)
        raw[:, order] = coefs

    beta = ys * raw / xs[:, None]
    intercept = ym - beta.T @ xm
    return intercept, beta


def sa_post_resample(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """``caret::postResample()`` for a numeric outcome."""
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    if pred.size == 0:
        return {"RMSE": np.nan, "Rsquared": np.nan, "MAE": np.nan}

    resid = pred - obs
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    # A fold where every prediction is the same leaves the correlation
    # undefined, which is where caret's "missing values in resampled
    # performance measures" note comes from. Tested on the range rather than on
    # the standard deviation: a penalty strong enough to zero every coefficient
    # leaves predictions that are equal to each other but whose NumPy standard
    # deviation is a rounding artefact rather than zero.
    if pred.size < 2 or np.ptp(pred) == 0 or np.ptp(obs) == 0:
        r2 = np.nan
    else:
        r2 = float(np.corrcoef(pred, obs)[0, 1] ** 2)
    return {"RMSE": rmse, "Rsquared": r2, "MAE": mae}


def sa_post_resample_class(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """``caret::postResample()`` for a factor outcome: accuracy and Cohen's kappa."""
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    if pred.size == 0:
        return {"Accuracy": np.nan, "Kappa": np.nan}

    accuracy = float(np.mean(pred == obs))
    levels = np.unique(np.concatenate([pred, obs]))
    counts = np.zeros((len(levels), len(levels)), dtype=float)
    index = {lv: i for i, lv in enumerate(levels)}
    for p, o in zip(pred, obs):
        counts[index[p], index[o]] += 1
    total = counts.sum()
    expected = float(
        np.sum(counts.sum(axis=1) * counts.sum(axis=0)) / (total * total)
    )
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else np.nan
    return {"Accuracy": accuracy, "Kappa": float(kappa)}
