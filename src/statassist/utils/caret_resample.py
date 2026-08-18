"""caret's resampling indices, transcribed onto the R-compatible RNG.

caret draws its folds with R's own ``sample()``, so the only way a Python fold
can be the fold R used is to consume the same RNG stream in the same order.
That is what ``rng_r.py`` is for, and it is why this is a transcription of
``caret::createFolds`` rather than a call to a scikit-learn splitter.

The stratification is the part that is easy to get wrong. A numeric outcome is
cut into between two and five quantile bins and the *string* labels of those
bins are what the classes are sorted by, so the order the folds are filled in
depends on how ``cut()`` chose to print its break points. Both the printing and
the sorting are reproduced here.
"""

from __future__ import annotations

import numpy as np

from statassist.utils.rng_r import get_rng, sa_r_seed


def _format_c(x: float, digits: int) -> str:
    """R ``formatC(x, digits = digits, width = 1L)`` for a double."""
    return "%.*g" % (digits, x + 0.0)


def sa_r_cut_labels(
    breaks: np.ndarray,
    include_lowest: bool = True,
    right: bool = True,
    dig_lab: int = 3,
) -> list[str]:
    """Interval labels exactly as ``base::cut.default()`` writes them."""
    nb = len(breaks)
    chars: list[str] = []
    ok = False
    for dig in range(dig_lab, max(12, dig_lab) + 1):
        chars = [_format_c(float(b), dig) for b in breaks]
        ok = all(chars[i] != chars[i + 1] for i in range(nb - 1))
        if ok:
            break
    if not ok:
        return [f"Range_{i + 1}" for i in range(nb - 1)]

    open_ch, close_ch = ("(", "]") if right else ("[", ")")
    labels = [
        f"{open_ch}{chars[i]},{chars[i + 1]}{close_ch}" for i in range(nb - 1)
    ]
    if include_lowest:
        if right:
            labels[0] = "[" + labels[0][1:]
        else:
            labels[-1] = labels[-1][:-1] + "]"
    return labels


def sa_r_bincode(
    x: np.ndarray,
    breaks: np.ndarray,
    right: bool = True,
    include_lowest: bool = True,
) -> np.ndarray:
    """R ``.bincode()``: 1-based interval number, 0 where the value falls outside."""
    x = np.asarray(x, dtype=float)
    if right:
        code = np.searchsorted(breaks, x, side="left")
        if include_lowest:
            code = np.where(x == breaks[0], 1, code)
    else:
        code = np.searchsorted(breaks, x, side="right")
        if include_lowest:
            code = np.where(x == breaks[-1], len(breaks) - 1, code)
    code = np.where((code < 1) | (code > len(breaks) - 1), 0, code)
    return code.astype(int)


def _stratum_labels(y: np.ndarray, k: int) -> np.ndarray:
    """The class each observation belongs to, as caret decides it."""
    if np.issubdtype(y.dtype, np.number):
        cuts = len(y) // k
        cuts = min(max(cuts, 2), 5)
        breaks = np.unique(np.quantile(y, np.linspace(0, 1, cuts)))
        labels = sa_r_cut_labels(breaks)
        codes = sa_r_bincode(y, breaks)
        return np.array(
            [labels[c - 1] if c > 0 else None for c in codes], dtype=object
        )
    return y.astype(str).astype(object)


def sa_create_folds(
    y: np.ndarray,
    k: int = 10,
    return_train: bool = False,
) -> list[np.ndarray]:
    """``caret::createFolds(y, k, list = TRUE)``, 0-based indices.

    Consumes the ambient R RNG, so wrap the call in ``sa_r_seed(seed)`` to line
    the folds up with an R session that called ``set.seed(seed)``.
    """
    rng = get_rng()
    y = np.asarray(y)
    n = len(y)

    fold_vector = np.zeros(n, dtype=int)
    if k < n:
        strata = _stratum_labels(y, k)
        # `factor(as.character(y))` orders its levels by sorting the strings.
        for cls in sorted({s for s in strata if s is not None}):
            idx = np.flatnonzero(strata == cls)
            num = len(idx)
            min_reps = num // k
            if min_reps > 0:
                spares = num % k
                seq_vector = np.tile(np.arange(1, k + 1), min_reps)
                if spares > 0:
                    seq_vector = np.concatenate(
                        [seq_vector, rng.sample_int(k, spares)]
                    )
                perm = rng.sample_int(len(seq_vector), len(seq_vector))
                fold_vector[idx] = seq_vector[perm - 1]
            else:
                fold_vector[idx] = rng.sample_int(k, num)
    else:
        fold_vector = np.arange(1, n + 1)

    all_rows = np.arange(n)
    out = [all_rows[fold_vector == f] for f in np.unique(fold_vector)]
    if return_train:
        out = [np.setdiff1d(all_rows, hold) for hold in out]
    return out


def sa_create_multi_folds(
    y: np.ndarray,
    k: int = 10,
    times: int = 5,
) -> list[np.ndarray]:
    """``caret::createMultiFolds()``: ``times`` repeats of ``k`` train indices."""
    out: list[np.ndarray] = []
    for _ in range(times):
        out.extend(sa_create_folds(y, k=k, return_train=True))
    return out


def sa_fold_names(k: int, times: int) -> list[str]:
    """The ``Fold01.Rep1`` style names caret gives the same list."""
    fold_width = len(str(k))
    rep_width = len(str(times))
    return [
        f"Fold{str(f).zfill(fold_width)}.Rep{str(r).zfill(rep_width)}"
        for r in range(1, times + 1)
        for f in range(1, k + 1)
    ]


R_INT_MAX = 2147483647


def sa_caret_resample_index(
    y: np.ndarray,
    method: str,
    n_fold: int,
    n_repeat: int,
    seed: int,
    chain: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """The fold index caret builds for a given seed.

    With ``chain``, the folds are two steps away from the seed the caller set
    rather than one: ``train()`` draws
    ``rs_seed <- sample.int(.Machine$integer.max, 1L)`` from the ambient stream
    and only then reseeds with it, inside
    ``withr::with_seed(rs_seed, make_resamples(...))``. Skipping that draw gives
    perfectly valid folds that are not the folds R used.

    ``rfe()`` builds its index straight from the ambient stream instead, before
    it draws anything else, so a search passes ``chain = False``.
    """
    y = np.asarray(y)
    rs_seed = seed
    if chain:
        with sa_r_seed(seed):
            rs_seed = int(get_rng().sample_int(R_INT_MAX, 1)[0])

    with sa_r_seed(rs_seed):
        if method == "repeatedcv":
            train = sa_create_multi_folds(y, k=n_fold, times=n_repeat)
            names = sa_fold_names(n_fold, n_repeat)
        elif method == "cv":
            train = sa_create_folds(y, k=n_fold, return_train=True)
            width = len(str(n_fold))
            names = [f"Fold{str(i).zfill(width)}" for i in range(1, n_fold + 1)]
        elif method == "LOOCV":
            train = sa_create_folds(y, k=len(y), return_train=True)
            width = len(str(len(y)))
            names = [f"Fold{str(i).zfill(width)}" for i in range(1, len(y) + 1)]
        else:
            raise ValueError(f"internal error: unhandled resampling method {method}.")
    return train, names


class RFoldSplitter:
    """A scikit-learn style splitter that replays caret's index list."""

    def __init__(self, train_indices: list[np.ndarray], n_samples: int) -> None:
        self._train = [np.asarray(t, dtype=int) for t in train_indices]
        self._n = n_samples

    def get_n_splits(self, x=None, y=None, groups=None) -> int:
        return len(self._train)

    def split(self, x=None, y=None, groups=None):
        all_rows = np.arange(self._n)
        for train in self._train:
            yield train, np.setdiff1d(all_rows, train)
