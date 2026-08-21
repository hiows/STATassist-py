"""One control against any number of treatment groups, with a known answer.

The port of ``R/simulate_multiple_groups.R``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.random import SaRandom
from ..core.result import SaSimulation
from ..core.tables import level_pairs
from ..core.validate import UNSET, check_count, check_flag, check_range, fmt_num
from ._patterns import allocate, pattern_delta, pick_up_down
from ._patterns import pattern_mix as check_pattern_mix

__all__ = ["simulate_multiple_groups"]

#: Treatment group sizes used when the caller names none.
_DEFAULT_N_TREAT = (50, 50, 50)
#: Share of the features moved in each direction when the caller names no count.
_PLANT_SHARE = 0.15


class Design(NamedTuple):
    """The levels of the design and how many observations each one holds.

    Attributes:
        group_lv: Level labels, the control first.
        sizes: One size per level, in the same order.
    """

    group_lv: list[str]
    sizes: list[int]


def _design(
    n_control: Any,
    n_treat: Any,
    group_lv: Any,
    use_default: bool,
    paired: bool,
) -> Design:
    """Work out the group labels and the size of each one.

    Port of ``sa_sim_design()``. The count, the labels and the sizes all come
    from the same pair of arguments, so they are settled together rather than in
    two passes that could disagree. ``n_treat`` carries one size per treatment
    group, which makes its length the number of them; ``group_lv`` carries the
    labels, which makes its length say the same thing. When both are given they
    have to agree, and the one case where they need not is a single size, which
    has an obvious number of groups to be spread over as soon as the labels say
    how many there are.

    Args:
        n_control: The argument as received.
        n_treat: The argument as received.
        group_lv: The argument as received, possibly ``None``.
        use_default: Whether ``n_treat`` was left at its default, which is what
            R reads off ``missing(n_treat)``.
        paired: Whether the levels are repeated conditions.
    """
    array = np.atleast_1d(np.asarray(n_treat))
    numeric = array.ndim == 1 and array.dtype.kind in "iuf" and array.dtype != bool
    if not numeric or array.size == 0:
        raise SaValueError("`n_treat` must be one or more group sizes, one per treatment group.")
    sizes_in: list[Any] = list(array)

    if group_lv is None:
        if len(sizes_in) < 2:
            raise SaValueError(
                "`n_treat` holds one size per treatment group, and there must be "
                "at least two of them for a comparison of three or more levels. "
                f"Pass a size per group, such as `n_treat = [{fmt_num(sizes_in[0])}] * 3`, "
                "or use simulate_two_groups() for two groups in all."
            )
        levels = ["control"] + [f"treat_{k + 1}" for k in range(len(sizes_in))]
    else:
        items = (
            list(group_lv)
            if isinstance(group_lv, Sequence) and not isinstance(group_lv, str)
            else []
        )
        distinct = len(set(items)) == len(items)
        if len(items) < 3 or not all(isinstance(item, str) for item in items) or not distinct:
            raise SaValueError(
                "`group_lv` must be at least three distinct non-missing group "
                "labels, the first being the control."
            )
        levels = [str(item) for item in items]
        n_wanted = len(levels) - 1
        # Labels say how many groups there are, so one size has somewhere to go.
        # The default is treated as one size for the same reason: it says how big
        # a group should be, not how many of them the caller wanted.
        if use_default or len(sizes_in) == 1:
            sizes_in = [sizes_in[0]] * n_wanted
        if len(sizes_in) != n_wanted:
            raise SaValueError(
                f"`group_lv` names {n_wanted} treatment group(s) after the control, "
                f"but `n_treat` gives {len(sizes_in)} size(s)."
            )

    sizes = [check_count(n_control, "n_control", 2)]
    sizes += [check_count(size, f"n_treat[{k}]", 2) for k, size in enumerate(sizes_in)]

    if paired and len(set(sizes)) > 1:
        raise SaValueError(
            "`paired = True` measures every condition on the same subjects, so "
            "every group holds the same number of them, but the sizes given are "
            + ", ".join(str(size) for size in sizes)
            + "."
        )
    return Design(group_lv=levels, sizes=sizes)


def _truth(
    feats: list[str],
    delta: np.ndarray,
    group_lv: list[str],
    pattern: np.ndarray,
    direction: np.ndarray,
    baseline: np.ndarray,
    sd_subject: np.ndarray,
) -> pd.DataFrame:
    """Feature-level answer, aligned with the ``effect`` table.

    Port of ``sa_sim_truth()``. ``delta`` is features by levels, the control
    column being zero throughout.
    """
    treat_delta = delta[:, 1:]
    abs_delta = np.abs(treat_delta)
    largest = abs_delta.max(axis=1)
    # A shape that moves every treatment group by the same amount leaves no
    # single level furthest from the control, so the tie is reported rather than
    # broken silently and scored on.
    tied = (abs_delta == largest[:, None]).sum(axis=1) > 1
    # R's `max.col(ties.method = "first")` and `argmax` both take the first
    # maximum, so the level a tie names is the same in both.
    which_max = abs_delta.argmax(axis=1)

    extreme_level: list[str | None] = [
        None if size == 0 else group_lv[1 + int(column)]
        for size, column in zip(largest, which_max, strict=True)
    ]

    return pd.DataFrame(
        {
            "features": feats,
            "pattern": pattern.astype(str),
            "direction": direction.astype(str),
            "extreme_level": extreme_level,
            "extreme_tied": tied,
            "log2fc": treat_delta[np.arange(len(feats)), which_max],
            "baseline": baseline,
            "sd_subject": sd_subject,
        }
    )


def _truth_group(
    feats: list[str],
    delta: np.ndarray,
    center: np.ndarray,
    sd_mat: np.ndarray,
    group_lv: list[str],
    sizes: list[int],
) -> pd.DataFrame:
    """Per feature and level answer.

    Port of ``sa_sim_truth_group()``. R flattens with ``as.vector(t(delta))``,
    which is feature-major; ``ravel()`` is the same order.
    """
    n_lv = len(group_lv)
    return pd.DataFrame(
        {
            "features": np.repeat(feats, n_lv),
            "group": np.tile(group_lv, len(feats)),
            "is_ref": np.tile(np.arange(n_lv) == 0, len(feats)),
            "delta": delta.ravel(),
            "center": center.ravel(),
            "sd": sd_mat.ravel(),
            "n": np.tile(sizes, len(feats)),
        }
    )


def _truth_contrast(feats: list[str], delta: np.ndarray, group_lv: list[str]) -> pd.DataFrame:
    """Per feature and pair answer, in the post-hoc table's own direction.

    Port of ``sa_sim_truth_contrast()``. The pairs come from
    :func:`~statassist.core.level_pairs` rather than from a second listing of
    them here, so the row order and the ``group1 - group2`` direction cannot
    drift apart from the tables this is meant to score.
    """
    pairs = level_pairs(group_lv)
    diffs = delta[:, pairs["i"].to_numpy()] - delta[:, pairs["j"].to_numpy()]
    flat = diffs.ravel()

    return pd.DataFrame(
        {
            "features": np.repeat(feats, len(pairs.index)),
            "contrast": np.tile(pairs["contrast"].to_numpy(), len(feats)),
            "group1": np.tile(pairs["group1"].to_numpy(), len(feats)),
            "group2": np.tile(pairs["group2"].to_numpy(), len(feats)),
            "delta": flat,
            "is_diff": flat != 0,
        }
    )


def simulate_multiple_groups(
    n_feats: int = 100,
    n_control: int = 50,
    n_treat: Any = UNSET,
    n_up: int | None = None,
    n_down: int | None = None,
    pattern_mix: dict[str, float] | None = None,
    expr_range: tuple[float, float] = (2, 12),
    control_sd: tuple[float, float] = (1.2, 2.4),
    treat_sd: tuple[float, float] = (1.8, 3.2),
    deg_log2fc: tuple[float, float] = (1, 2.5),
    paired: bool = False,
    subject_sd: tuple[float, float] = (2, 4),
    group_lv: Sequence[str] | None = None,
    feat_prefix: str = "prot",
    seed: int | None = None,
) -> SaSimulation:
    """Simulate a control-versus-treatments experiment whose answer is known.

    The multi-group counterpart of :func:`simulate_two_groups`. Generates
    log2-scale abundance data for one control group and any number of treatment
    groups, and returns the planted answer alongside the data so that a
    comparison can be scored against what was actually put there.

    With two groups there is one thing to get right: whether a feature moved.
    With three or more there are two, and they fail separately. The omnibus test
    asks whether the levels are all alike, and the post-hoc stage asks which of
    them differ. A feature can clear the first and be misread by the second, and
    a shape of effect that the omnibus finds easy can be the one the post-hoc
    stage finds hard. That is why the effect is planted in three shapes rather
    than one, and why the answer comes back in three tables rather than one.

    Args:
        n_feats: Number of features to generate. Columns are named ``prot_1``
            upwards, or whatever ``feat_prefix`` asks for.
        n_control: Number of observations in the control group.
        n_treat: Observations in each treatment group, one entry per group, so
            that its length is how many treatment groups there are. Pass
            ``[50, 40, 30]`` for three treatment groups of different sizes and
            ``[30] * 5`` for five of the same size. There is no separate argument
            for the number of groups: the sizes already say it, and two arguments
            that could disagree would have to be settled somewhere.

            A multi-group comparison needs at least three levels in all, so this
            needs at least two entries. A single number is allowed only when
            ``group_lv`` says how many groups to spread it over.
        n_up: How many features are moved up in the treatment groups.
        n_down: How many are moved down. Their sum cannot exceed ``n_feats``, and
            every other feature is left with a true effect of exactly zero in
            every group. ``None`` takes a fraction of ``n_feats`` rather than a
            fixed count, so that asking for fewer features plants fewer effects
            instead of failing; at the default ``n_feats = 100`` that is the 15
            and 15 the defaults were tuned at.
        pattern_mix: Relative weights over the three shapes an effect can take,
            keyed by shape name and described under "The three shapes" below.
            ``None`` weighs them equally. Set a weight to zero to leave that
            shape out. The planted features are split between the shapes in these
            proportions by the largest remainder method rather than drawn at
            random, so the counts are exactly what the weights ask for and do not
            move with the seed.
        expr_range: Range the baseline log2 abundance of each feature is drawn
            from. Every level shares the baseline, which is what makes an
            unplanted feature null.
        control_sd: Range the per-feature standard deviation of the control group
            is drawn from.
        treat_sd: The same for each treatment group. Every group draws its own,
            so the design is heteroscedastic, which is the situation Welch's
            ANOVA and Games-Howell exist for. Pass the same range twice for equal
            variances.
        deg_log2fc: Range the magnitude of the planted effect is drawn from, on
            the log2 scale. This is the magnitude at the level that carries the
            full effect; the ``"gradient"`` shape places the intermediate levels
            below it.
        paired: If ``True``, the levels are treated as repeated conditions
            measured on the same subjects, and ``args`` gains ``id`` and
            ``paired`` so that the comparison runs the within-subject tests.
        subject_sd: Range the per-feature subject standard deviation is drawn
            from. A subject's offset is drawn once per feature and reused across
            every condition, which is what a within-subject test exists to
            remove. The default is deliberately of the same order as the residual
            spread, so that analysing the same table without ``id`` costs most of
            the recall. Ignored when ``paired`` is ``False``.
        group_lv: Group labels, the first being the control that every effect is
            planted against. ``None`` gives ``control``, ``treat_1`` and so on,
            one label per entry of ``n_treat`` plus the control. When supplied it
            says how many groups there are, so an ``n_treat`` of length one is
            spread over them and an ``n_treat`` left at its default is replaced
            by that many groups of the default size. Supplying labels and sizes
            that count differently is an error rather than a guess.
        feat_prefix: Prefix for the generated feature names. ``"prot"`` gives
            ``prot_1``, ``prot_2`` and so on.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy.

    Returns:
        A :class:`~statassist.core.SaSimulation` of four slots.

        * ``args`` - ``data``, ``feats``, ``group``, ``group_lv`` and
          ``input_scale``, named after the arguments of the multi-group
          comparison so that ``compare_multiple_groups(**sim.args)`` runs it.
          Under ``paired=True`` it also carries ``id`` and ``paired``; when the
          design is not paired those two keys are **absent** rather than present
          and empty.
        * ``truth`` - one row per feature, aligned with ``feats``, holding
          ``features``, ``pattern``, ``direction``, ``extreme_level``,
          ``extreme_tied``, ``log2fc``, ``baseline`` and ``sd_subject``. This is
          the table that scores the effect and omnibus tables.
        * ``truth_group`` - one row per feature and level, holding ``features``,
          ``group``, ``is_ref``, ``delta``, ``center``, ``sd`` and ``n``. A
          feature the comparison missed can be looked up here rather than guessed
          at: a large ``sd`` explains a miss that the effect size alone does not.
        * ``truth_contrast`` - one row per feature and pair of levels, in the row
          order and direction the post-hoc tables use, holding ``features``,
          ``contrast``, ``group1``, ``group2``, ``delta`` and ``is_diff``.

    The three shapes:
        Each planted feature is given a magnitude ``d`` drawn from
        ``deg_log2fc``, positive for an up feature and negative for a down one,
        and one of three shapes that decides what each treatment group does with
        it.

        * ``"all"`` - every treatment group is shifted by ``d``. Only the control
          stands apart, so the omnibus test has the whole effect to work with and
          every contrast against the control should be found.
        * ``"gradient"`` - treatment group ``g`` of ``k`` is shifted by
          ``d * g / k``, so the last one carries the full effect and the ones
          before it carry a fraction. This is the dose-response shape, and its
          early contrasts are the ones a post-hoc stage loses first.
        * ``"single"`` - one treatment group, chosen at random, is shifted by
          ``d`` and the rest are left at exactly zero. The omnibus test is
          diluted here, since most of the levels it compares are alike, so this
          is the shape it misses most often. When it does clear the cutoff,
          exactly the contrasts involving that one level should come back.

        A feature that was not planted has a delta of exactly zero in every
        group. Both kinds of mistake are therefore defined: a contrast called
        significant on a zero delta is a false positive, and a non-zero delta
        that was not called is a miss.

    Directions:
        ``truth["log2fc"]`` is the delta of whichever level sits furthest from
        the control, which is the quantity the effect table estimates. A
        treatment group that went up gives a positive value in both.

        ``truth_contrast["delta"]`` reads the same way, because the post-hoc
        tables do: a post-hoc estimate is ``group1 - group2`` with ``group1`` the
        later level of ``group_lv``, and the control is the first level.

        Under the ``"all"`` shape every treatment group carries the same delta, so
        no single level is furthest from the control. ``extreme_level`` then
        records the first of the tied levels and ``extreme_tied`` is ``True``,
        which is the flag that says to score the magnitude rather than the name of
        the level. It is also ``True``, with ``extreme_level`` missing, for an
        unplanted feature.

    Repeated conditions:
        Under ``paired=True`` each subject is measured under every condition, so
        no subject is dropped. Each subject gets an offset per feature, drawn once
        and added to all of its conditions, which is the between-subject variation
        the within-subject tests remove. The residual standard deviation still
        differs between conditions, so sphericity does not hold and the Mauchly,
        Greenhouse-Geisser and Huynh-Feldt columns of the repeated measures ANOVA
        have something to report.

        The same subjects appearing under every condition also means every group
        holds the same number of them, so ``n_control`` and every entry of
        ``n_treat`` have to agree. Unequal sizes are rejected rather than quietly
        levelled, since the sizes are the clearest statement of which design was
        meant.

    Raises:
        SaValueError: If the labels and sizes count differently, if ``n_up`` plus
            ``n_down`` asks for more features than there are, if a range is
            unusable, or if ``paired`` is asked for with unequal group sizes.

    Examples:
        The default is three treatment groups against a control, and the shapes
        are handed out in counts the weights fix rather than the seed.

        >>> sim = simulate_multiple_groups(n_feats=30, n_up=6, n_down=6, seed=1)
        >>> sim.args["group_lv"]
        ['control', 'treat_1', 'treat_2', 'treat_3']
        >>> {p: int((sim.truth["pattern"] == p).sum()) for p in ("all", "gradient", "single")}
        {'all': 4, 'gradient': 4, 'single': 4}

        Labels alone say how many groups there are, so one size is spread over
        them.

        >>> dose = simulate_multiple_groups(
        ...     n_feats=20, n_treat=25, group_lv=["dmso", "low", "mid", "high"], seed=1
        ... )
        >>> [dose.args["group"].count(level) for level in dose.args["group_lv"]]
        [50, 25, 25, 25]

        A paired design adds the two keys the within-subject tests need, and
        leaves them out otherwise.

        >>> rep_sim = simulate_multiple_groups(
        ...     n_feats=10, n_control=12, n_treat=[12, 12], paired=True, seed=1
        ... )
        >>> [key for key in rep_sim.args if key in ("id", "paired")]
        ['id', 'paired']
        >>> [key for key in sim.args if key in ("id", "paired")]
        []
    """
    paired = check_flag(paired, "paired")
    use_default = n_treat is UNSET
    design = _design(
        n_control,
        _DEFAULT_N_TREAT if use_default else n_treat,
        group_lv,
        use_default,
        paired,
    )
    levels = design.group_lv
    sizes = design.sizes
    n_lv = len(levels)
    n_treat_groups = n_lv - 1

    # `n_up` and `n_down` default to a fraction of `n_feats`, which R evaluates
    # lazily and so sees the checked value. Here they arrive as `None` and are
    # filled in below for the same reason: the fraction of an unchecked count is
    # not a count.
    n_feats = check_count(n_feats, "n_feats", 1)
    n_up = round(_PLANT_SHARE * n_feats) if n_up is None else n_up
    n_down = round(_PLANT_SHARE * n_feats) if n_down is None else n_down
    n_up = check_count(n_up, "n_up")
    n_down = check_count(n_down, "n_down")
    if n_up + n_down > n_feats:
        raise SaValueError(
            f"`n_up` + `n_down` is {n_up + n_down}, which is more features than "
            f"the {n_feats} that `n_feats` asks for."
        )
    mix = check_pattern_mix(
        {"all": 1.0, "gradient": 1.0, "single": 1.0} if pattern_mix is None else pattern_mix
    )
    expr_lo, expr_hi = check_range(expr_range, "expr_range")
    control_lo, control_hi = check_range(control_sd, "control_sd", 0)
    treat_lo, treat_hi = check_range(treat_sd, "treat_sd", 0)
    deg_lo, deg_hi = check_range(deg_log2fc, "deg_log2fc", 0)
    subject_lo, subject_hi = check_range(subject_sd, "subject_sd", 0)
    if not isinstance(feat_prefix, str) or not feat_prefix:
        raise SaValueError("`feat_prefix` must be a single non-empty string.")

    rng = SaRandom(seed).rng

    feats = [f"{feat_prefix}_{i + 1}" for i in range(n_feats)]
    baseline = rng.uniform(expr_lo, expr_hi, n_feats)

    sd_mat = np.zeros((n_feats, n_lv))
    sd_mat[:, 0] = rng.uniform(control_lo, control_hi, n_feats)
    for g in range(n_treat_groups):
        sd_mat[:, 1 + g] = rng.uniform(treat_lo, treat_hi, n_feats)
    sd_subject = (
        rng.uniform(subject_lo, subject_hi, n_feats) if paired else np.full(n_feats, np.nan)
    )

    delta = np.zeros((n_feats, n_lv))
    direction = np.full(n_feats, "none", dtype=object)
    pattern = np.full(n_feats, "none", dtype=object)

    if n_up + n_down > 0:
        up_idx, down_idx = pick_up_down(n_feats, n_up, n_down, rng)
        direction[up_idx] = "up"
        direction[down_idx] = "down"

        plant_idx = np.concatenate([up_idx, down_idx])
        plant_mag = np.concatenate(
            [rng.uniform(deg_lo, deg_hi, n_up), -rng.uniform(deg_lo, deg_hi, n_down)]
        )
        # Each direction is split between the shapes on its own, so the mix holds
        # within the up set and within the down set rather than only in total.
        # The indices were drawn at random already, so handing the shapes out in
        # blocks still lands them on random features.
        plant_pat = [
            name
            for count in (allocate(n_up, mix), allocate(n_down, mix))
            for name, times in count.items()
            for _ in range(times)
        ]

        for k, i in enumerate(plant_idx):
            delta[i, 1:] = pattern_delta(plant_mag[k], plant_pat[k], n_treat_groups, rng)
            pattern[i] = plant_pat[k]

    center = baseline[:, None] + delta
    # One offset per subject and feature, drawn before the conditions and added
    # to all of them. Drawing it inside the condition loop would make it noise
    # rather than a subject effect, and the within-subject tests would have
    # nothing to gain over the independent ones.
    offsets = rng.normal(0, sd_subject, size=(sizes[0], n_feats)) if paired else None

    blocks = []
    for g in range(n_lv):
        values = rng.normal(center[:, g], sd_mat[:, g], size=(sizes[g], n_feats))
        blocks.append(values + offsets if offsets is not None else values)

    data = pd.DataFrame(np.vstack(blocks), columns=feats)

    args: dict[str, Any] = {
        "data": data,
        "feats": feats,
        "group": [level for level, size in zip(levels, sizes, strict=True) for _ in range(size)],
        "group_lv": levels,
    }
    if paired:
        # Every condition holds the same subjects, in the same order, so the
        # labels repeat once per level rather than per row.
        args["id"] = [f"subject_{i + 1}" for i in range(sizes[0])] * n_lv
        args["paired"] = True
    args["input_scale"] = "log2"

    return SaSimulation(
        {
            "args": args,
            "truth": _truth(feats, delta, levels, pattern, direction, baseline, sd_subject),
            "truth_group": _truth_group(feats, delta, center, sd_mat, levels, sizes),
            "truth_contrast": _truth_contrast(feats, delta, levels),
        }
    )
