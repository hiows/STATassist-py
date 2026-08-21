"""Two independent groups with a known answer planted in them.

The port of ``R/simulate_two_groups.R``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.random import SaRandom
from ..core.result import SaSimulation
from ..core.validate import check_count, check_range
from ._patterns import pick_up_down

__all__ = ["simulate_two_groups"]


def _check_two_labels(group_lv: object, arg: str = "group_lv") -> list[str]:
    """Two distinct labels, the first being the reference."""
    items = (
        list(group_lv) if isinstance(group_lv, Sequence) and not isinstance(group_lv, str) else []
    )
    ok = len(items) == 2 and all(isinstance(item, str) for item in items) and items[0] != items[1]
    if not ok:
        raise SaValueError(f"`{arg}` must be two distinct non-missing group labels.")
    return [str(item) for item in items]


def simulate_two_groups(
    n_feats: int = 100,
    n_case: int = 50,
    n_control: int = 50,
    n_up: int = 15,
    n_down: int = 15,
    expr_range: tuple[float, float] = (2, 12),
    case_sd: tuple[float, float] = (1.8, 3.2),
    control_sd: tuple[float, float] = (1.2, 2.4),
    deg_log2fc: tuple[float, float] = (1, 2.5),
    group_lv: Sequence[str] = ("control", "case"),
    seed: int | None = None,
) -> SaSimulation:
    """Simulate a two-group experiment whose answer is known.

    Generates log2-scale expression data for two independent groups with a fixed
    number of features moved up and down on purpose, and returns the planted
    answer alongside the data. Every quantity a comparison estimates can then be
    checked against what was actually put there, which is what a real data set
    can never offer.

    The point of the exercise is the gap. A comparison does not recover every
    feature that was planted, and the reasons it misses them are the three things
    worth understanding about a volcano plot: the p-value may not clear its
    cutoff, the multiplicity adjustment may take it back, and the estimated
    ``log2fc`` carries a sampling error of its own, so a feature planted just
    above the magnitude cutoff lands below it about half the time. The defaults
    are set so that a run recovers most but not all of what was planted, because
    a simulation that recovers everything teaches none of this.

    Args:
        n_feats: Number of features to generate. Columns are named ``gene_1``
            upwards.
        n_case: Observations in the case group.
        n_control: Observations in the control group. It does not have to match
            ``n_case``.
        n_up: How many features are moved up in the case group.
        n_down: How many features are moved down in the case group. ``n_up`` plus
            ``n_down`` cannot exceed ``n_feats``, and every other feature is left
            with a true fold change of exactly zero.
        expr_range: Range the baseline log2 expression of each feature is drawn
            from. The default spans what log2 CPM or RMA values usually cover.
            Both groups share the baseline, which is what makes an unplanted
            feature null.
        case_sd: Range the per-feature standard deviation of the case group is
            drawn from.
        control_sd: The same for the control group. The two are drawn
            independently, so the groups end up with unequal variances, which is
            the situation Welch's t-test and the Brunner-Munzel test exist for.
            Pass the same range twice for a homoscedastic data set. The defaults
            leave roughly four planted features in five recoverable at the
            default cutoffs; narrowing them recovers nearly everything and
            widening them costs recall quickly.
        deg_log2fc: Range the magnitude of the planted effect is drawn from, on
            the log2 scale. The default of ``(1, 2.5)`` is a two-fold to roughly
            six-fold change, which straddles the ``log2fc_cutoff = 1`` that
            significance estimation applies by default.
        group_lv: The two group labels, the first being the control and the
            second the one the effect is applied to. Passed straight through to
            the returned arguments, so it also fixes the direction a comparison
            reads: a planted increase comes back as a positive ``log2fc`` because
            the control is the reference.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy. Unlike R, seeding is local to the call, so nothing needs
            putting back afterwards - and the numbers a seed gives are this
            package's, not R's.

    Returns:
        A :class:`~statassist.core.SaSimulation` of two slots.

        * ``args`` - ``data``, ``feats``, ``group``, ``group_lv`` and
          ``input_scale``, named after the arguments of the two-group comparison
          so that ``compare_two_groups(**sim.args)`` runs it. ``input_scale`` is
          ``"log2"``, since that is the scale the data is on.
        * ``truth`` - one row per feature, aligned with ``feats``, holding
          ``features``, ``direction`` (``"up"``, ``"down"`` or ``"none"``),
          ``log2fc`` (the effect that was planted, exactly ``0`` for ``"none"``),
          ``baseline`` and the two group standard deviations. The last three are
          there so that a feature the comparison missed can be looked up rather
          than guessed at: a large ``sd_case`` explains a miss that the effect
          size alone does not.

    How the data is built:
        Each feature gets a baseline ``b`` drawn from ``expr_range``, two
        standard deviations drawn from ``case_sd`` and ``control_sd``, and a
        planted effect ``d`` that is a positive draw from ``deg_log2fc`` when the
        feature is one of the ``n_up``, a negative draw when it is one of the
        ``n_down``, and ``0`` otherwise. Case observations are then normal around
        ``b + d`` and control observations normal around ``b``.

        Because the baseline is shared, the true log2 fold change of a feature is
        ``d`` and nothing else. An unplanted feature is null in the strict sense,
        so a feature called significant is a false positive by definition and the
        multiplicity adjustment can be judged on it. A model that gave each group
        its own random offset would look more lifelike and would make both the
        recall and the false positive rate impossible to compute, since an
        unplanted feature would then differ between the groups too.

        The data is on the log2 scale throughout, so ``input_scale = "log2"``
        comes back with it. The effect is added rather than multiplied, which is
        what makes ``deg_log2fc`` a difference of log2 means rather than a ratio.

    Raises:
        SaValueError: If a count or a range is unusable, or if ``n_up`` plus
            ``n_down`` asks for more features than there are.

    Examples:
        How many features were planted is a function of the arguments, not of the
        seed.

        >>> sim = simulate_two_groups(seed=1)
        >>> {d: int((sim.truth["direction"] == d).sum()) for d in ("up", "down", "none")}
        {'up': 15, 'down': 15, 'none': 70}

        The names in ``args`` are the comparison's own, so the analysis is one
        call away as ``compare_two_groups(**sim.args)``.

        >>> list(sim.args)
        ['data', 'feats', 'group', 'group_lv', 'input_scale']
        >>> sim.args["data"].shape
        (100, 100)

        A planted feature carries the effect that was put there, so a comparison
        can be scored on it rather than against another comparison.

        >>> planted = sim.truth["direction"] != "none"
        >>> bool((sim.truth.loc[planted, "log2fc"].abs() >= 1).all())
        True
    """
    n_feats = check_count(n_feats, "n_feats", 1)
    n_case = check_count(n_case, "n_case", 2)
    n_control = check_count(n_control, "n_control", 2)
    n_up = check_count(n_up, "n_up")
    n_down = check_count(n_down, "n_down")
    if n_up + n_down > n_feats:
        raise SaValueError(
            f"`n_up` + `n_down` is {n_up + n_down}, which is more features than "
            f"the {n_feats} that `n_feats` asks for."
        )
    expr_lo, expr_hi = check_range(expr_range, "expr_range")
    case_lo, case_hi = check_range(case_sd, "case_sd", 0)
    control_lo, control_hi = check_range(control_sd, "control_sd", 0)
    deg_lo, deg_hi = check_range(deg_log2fc, "deg_log2fc", 0)
    levels = _check_two_labels(group_lv)

    rng = SaRandom(seed).rng

    feats = [f"gene_{i + 1}" for i in range(n_feats)]
    baseline = rng.uniform(expr_lo, expr_hi, n_feats)
    sd_case = rng.uniform(case_lo, case_hi, n_feats)
    sd_control = rng.uniform(control_lo, control_hi, n_feats)

    direction = np.full(n_feats, "none", dtype=object)
    delta = np.zeros(n_feats)
    if n_up + n_down > 0:
        up_idx, down_idx = pick_up_down(n_feats, n_up, n_down, rng)
        direction[up_idx] = "up"
        direction[down_idx] = "down"
        delta[up_idx] = rng.uniform(deg_lo, deg_hi, n_up)
        delta[down_idx] = -rng.uniform(deg_lo, deg_hi, n_down)

    def draw(n: int, center: np.ndarray, spread: np.ndarray) -> np.ndarray:
        # One draw for the whole block, where R loops over features. The stream
        # differs from R's either way, and a feature-by-feature loop would only
        # make the same data set slower to produce.
        return rng.normal(loc=center, scale=spread, size=(n, n_feats))

    # Drawn case first and stacked control first. `group_lv` names the control
    # ahead of the case group, so the rows have to follow, while the order the
    # draws consume the random stream is left alone: reversing that instead would
    # hand a seed that used to give one data set a different one.
    case_values = draw(n_case, baseline + delta, sd_case)
    control_values = draw(n_control, baseline, sd_control)

    data = pd.DataFrame(np.vstack([control_values, case_values]), columns=feats)

    return SaSimulation(
        {
            "args": {
                "data": data,
                "feats": feats,
                "group": [levels[0]] * n_control + [levels[1]] * n_case,
                "group_lv": levels,
                "input_scale": "log2",
            },
            "truth": pd.DataFrame(
                {
                    "features": feats,
                    "direction": direction.astype(str),
                    "log2fc": delta,
                    "baseline": baseline,
                    "sd_case": sd_case,
                    "sd_control": sd_control,
                }
            ),
        }
    )
