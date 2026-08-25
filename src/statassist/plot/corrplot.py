"""The correlation matrix drawn as a heatmap, which is what a corrplot is.

Port of ``R/draw_corrplot.R``. There is no second drawing engine here:
:func:`~statassist.draw_heatmap` already owns the cells, the diverging ramp, the
colour key and the dendrograms, and this function is the three decisions a
correlation matrix needs that a feature-by-sample matrix does not.

The first is that nothing may be standardised. A coefficient is already on a
common scale, and z-scoring it would replace the number the reader came for. The
second is that both axes hold the same features, so they need one order rather
than two: a symmetric matrix clustered twice can come back with its rows and
columns in different orders, and the diagonal then wanders off the diagonal. The
clustering is therefore done once here and the heatmap is handed a matrix that is
already in that order. The third is that a cell may be blanked for having no
evidence behind it, which has to happen after the clustering, so that what is
drawn and what the tree was built from stay the same matrix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ..core.errors import SaValueError, notify
from ..core.validate import check_flag, check_range, check_scalar_num
from .heatmap import HCLUST_METHODS, LINKAGE_NAMES, Clustering, draw_heatmap

__all__ = ["CORR_LIMITS", "draw_corrplot"]

#: The range the colours span by default.
#:
#: The range a correlation can take, rather than the range this matrix happens to
#: hold, so that the same colour means the same strength from one plot to the next.
CORR_LIMITS = (-1.0, 1.0)

#: Fewest features there is a correlation matrix to draw between.
_MIN_FEATS = 2

#: What :func:`draw_corrplot` decides for :func:`~statassist.draw_heatmap`, and so
#: will not pass on a second time.
#:
#: R lists the character expansions here too, since its ``...`` would catch them.
#: They are named arguments of this function instead, which forwards them, so
#: there is no second way to reach them and nothing to refuse.
_DECIDED = ("data", "group", "group_lv", "scale", "cluster_feats", "cluster_samples")


class CorrInput(NamedTuple):
    """The coefficient matrix and its p-values, whichever input they came from."""

    corr: pd.DataFrame
    pvalue: pd.DataFrame | None


def draw_corrplot(
    cor_matrix: Any,
    method: str | None = None,
    pvalue: Any = None,
    use_adjusted: bool = True,
    sig_level: float = 0.05,
    cluster: bool = True,
    hclust_method: str = "average",
    zlim: Any = CORR_LIMITS,
    anno: bool = True,
    main: str | None = None,
    cex_anno: float = 1.0,
    cex_axis: float = 0.9,
    cex_main: float = 1.5,
    cex_legend: float = 1.2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Draw a correlation matrix, with the cells that failed the test left blank.

    One cell per pair of features, coloured by the coefficient on a fixed -1 to 1
    scale, with both axes in one order so that the diagonal runs corner to corner
    and blocks of features that move together sit next to each other. Given the
    p-values as well, the pairs that did not clear ``sig_level`` are drawn as
    blank cells, so that what is left coloured is what there is evidence for.

    :func:`~statassist.draw_heatmap` draws it. This function decides what it is
    handed: the matrix unscaled, the colour range fixed at the range a
    correlation can take, one clustering shared by the two axes, and the blanking
    applied afterwards.

    Args:
        cor_matrix: The result of
            :func:`~statassist.summarize_association_stats`, or a correlation
            matrix on its own. A matrix must be square and symmetric, with the
            features as its labels, and hold at least two of them.
        method: Which coefficient to draw when ``cor_matrix`` is a result, naming
            one of the slots it holds. ``None`` takes the first method it was
            computed with. It must be ``None`` when ``cor_matrix`` is a matrix,
            there being only one matrix to draw.
        pvalue: Matrix of p-values laid out like ``cor_matrix``, used only when
            ``cor_matrix`` is a matrix. When ``cor_matrix`` is a result the
            p-values come from the same slot as the coefficients, so this must be
            left ``None``.
        use_adjusted: Whether to read ``adj_pvalue`` rather than ``pvalue`` out of
            the result. Ignored when ``pvalue`` is supplied directly.
        sig_level: Largest p-value a cell may have and still be drawn. Cells above
            it are blanked. The diagonal is never blanked, a feature not being
            tested against itself.
        cluster: Whether to reorder the features by clustering them. ``False``
            keeps the order they arrive in.
        hclust_method: Linkage, one of
            :data:`~statassist.plot.heatmap.HCLUST_METHODS`.
        zlim: Length-2 range the colours span, :data:`CORR_LIMITS` by default.
        anno: Whether to write each coefficient on its cell, rounded to two
            decimal places by the heatmap. Blanked cells stay blank.
        main: Plot title.
        cex_anno: Character expansion for those cell labels.
        cex_axis, cex_main, cex_legend: Character expansion for the axis labels,
            the title and the colour key.
        **kwargs: Passed to :func:`~statassist.draw_heatmap`. The arguments this
            function decides - the data, the grouping, the scaling and the
            clustering of either axis - cannot be given again.

    Returns:
        Everything :func:`~statassist.draw_heatmap` returns, and beside it

        ``corr``
            The matrix as it was drawn, in the drawn order and with the blanked
            cells missing.
        ``pvalue``
            The p-values in that same order, or ``None``.
        ``order``
            The permutation of the input the clustering chose.
        ``hclust``
            The :class:`~statassist.plot.Clustering` behind it, or ``None`` when
            the features were not clustered.
        ``n_masked``
            How many cells were blanked.
        ``feats``
            The features ``order`` indexes into, which is not recoverable from the
            drawn matrix once the permutation has been applied to it.

    Raises:
        SaValueError: If ``cor_matrix`` is not a square symmetric numeric matrix
            of correlations or a result holding one, if the two ways of naming the
            p-values are both used, or if an argument this function decides is
            passed again.

    Notes:
        The distance the clustering runs on is ``1 - corr``, the same one
        :func:`~statassist.draw_heatmap` means by ``dist_method="correlation"``,
        so a corrplot and a heatmap of the same features group them the same way.
        It is computed once and both axes are permuted by it. A matrix holding a
        missing value, which is what a feature with no variance leaves behind, has
        no distance for that feature, and rather than fail the features are left
        in their input order with a message saying so.

        Blanking happens after the clustering rather than before it. A cell removed
        for its p-value would otherwise change the tree, and the picture would no
        longer be the matrix the reader is being shown.

        A cell whose p-value is missing, a pair that could not be tested, is left
        as it arrived rather than blanked: there is no evidence against it either,
        and the coefficient beside it is usually already missing.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import draw_corrplot, simulate_two_groups
        >>> from statassist import summarize_association_stats
        >>> sim = simulate_two_groups(n_feats=6, n_up=2, n_down=2, seed=3)
        >>> res = summarize_association_stats(
        ...     sim.args["data"], sim.args["feats"], methods="pearson"
        ... )
        >>> drawn = draw_corrplot(res, main="six features")
        >>> drawn["corr"].shape
        (6, 6)
    """
    if hclust_method not in HCLUST_METHODS:
        raise SaValueError("`hclust_method` must be one of: " + ", ".join(HCLUST_METHODS) + ".")
    use_adjusted = check_flag(use_adjusted, "use_adjusted")
    cluster = check_flag(cluster, "cluster")
    anno = check_flag(anno, "anno")
    sig_level = check_scalar_num(sig_level, "sig_level", 0, 1, lower_open=True)
    cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    limits = check_range(zlim, "zlim")
    passed_twice = [name for name in _DECIDED if name in kwargs]
    if passed_twice:
        raise SaValueError(
            "`draw_corrplot()` decides these arguments of `draw_heatmap()`, so they "
            "cannot be given again: " + ", ".join(passed_twice) + "."
        )

    read = _corrplot_input(cor_matrix, method, pvalue, use_adjusted)
    feats = list(read.corr.columns)

    order, tree = _corrplot_order(read.corr, cluster, hclust_method, feats)
    drawn = read.corr.iloc[order, order]
    held = None if read.pvalue is None else read.pvalue.iloc[order, order]

    n_masked = 0
    if held is not None:
        blank = (~held.isna()).to_numpy() & (held.to_numpy(dtype=float) > sig_level)
        np.fill_diagonal(blank, False)
        n_masked = int(blank.sum())
        drawn = drawn.mask(pd.DataFrame(blank, index=drawn.index, columns=drawn.columns))

    # The matrix is symmetric, so the transpose the heatmap takes on the way in
    # leaves it unchanged and the features come out on both axes.
    out = draw_heatmap(
        data=drawn,
        group=None,
        group_lv=None,
        scale="none",
        zlim=limits,
        cluster_feats=False,
        cluster_samples=False,
        anno=anno,
        cex_anno=cex_anno,
        main=main,
        cex_axis=cex_axis,
        cex_main=cex_main,
        cex_legend=cex_legend,
        **kwargs,
    )

    out["corr"] = drawn
    out["pvalue"] = held
    out["order"] = order
    out["hclust"] = tree
    out["n_masked"] = n_masked
    out["feats"] = feats
    return out


def _corrplot_input(
    cor_matrix: Any, method: str | None, pvalue: Any, use_adjusted: bool
) -> CorrInput:
    """Read the coefficient matrix and its p-values out of either kind of input.

    Port of ``sa_corrplot_input()``. :func:`draw_corrplot` takes the result of
    :func:`~statassist.summarize_association_stats` so that the two functions meet
    in one line, and a bare matrix so that a correlation computed some other way
    can still be drawn. Which one arrived decides where the p-values come from,
    and the two ways of naming them cannot both be used.
    """
    is_result = (
        isinstance(cor_matrix, Mapping)
        and isinstance(cor_matrix.get("design"), Mapping)
        and cor_matrix["design"].get("methods") is not None
    )

    if is_result:
        if pvalue is not None:
            raise SaValueError(
                "`pvalue` cannot be given for a `summarize_association_stats()` "
                "result: the p-values come from the same slot as the coefficients. "
                "Use `use_adjusted` to choose between them."
            )
        methods = [str(name) for name in cor_matrix["design"]["methods"]]
        if method is None:
            method = methods[0]
        elif not isinstance(method, str) or method not in methods:
            raise SaValueError(
                "`method` must name one of the methods `cor_matrix` holds: "
                + ", ".join(methods)
                + "."
            )
        slot = cor_matrix[method]
        corr = slot["corr"]
        pvalue = slot["adj_pvalue"] if use_adjusted else slot["pvalue"]
    else:
        if method is not None:
            raise SaValueError(
                "`method` names a slot of a `summarize_association_stats()` result, "
                "and `cor_matrix` is a matrix. Leave it None."
            )
        corr = cor_matrix

    frame = _as_square(corr)
    return CorrInput(corr=frame, pvalue=_as_pvalue(pvalue, frame))


def _as_square(corr: Any) -> pd.DataFrame:
    """The coefficient matrix, checked for being one, labelled on both axes."""
    if isinstance(corr, pd.DataFrame):
        values = corr.to_numpy()
        labels = [str(name) for name in corr.columns]
    else:
        values = np.asarray(corr)
        labels = None
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise SaValueError(
            "`cor_matrix` must be a numeric correlation matrix or the result of "
            "`summarize_association_stats()`."
        )
    values = values.astype(float)
    if values.shape[0] != values.shape[1]:
        raise SaValueError(
            f"`cor_matrix` must be square, but is {values.shape[0]} by {values.shape[1]}."
        )
    if values.shape[1] < _MIN_FEATS:
        raise SaValueError(
            f"`draw_corrplot()` needs at least {_MIN_FEATS} features to draw, but got "
            f"{values.shape[1]}."
        )
    if not np.allclose(values, values.T, equal_nan=True):
        raise SaValueError(
            "`cor_matrix` must be symmetric: a correlation between two features is "
            "one number, so the two cells that hold it must agree."
        )
    finite = values[np.isfinite(values)]
    if finite.size > 0 and (finite.min() < CORR_LIMITS[0] or finite.max() > CORR_LIMITS[1]):
        raise SaValueError(
            f"`cor_matrix` holds value(s) outside [{CORR_LIMITS[0]:g}, "
            f"{CORR_LIMITS[1]:g}], so it is not a matrix of correlations."
        )

    # A matrix from a correlation always carries its labels; one assembled by hand
    # may not, and the heatmap would then invent a name for one axis only.
    if labels is None:
        labels = [f"V{index + 1}" for index in range(values.shape[1])]
    return pd.DataFrame(values, index=labels, columns=labels)


def _as_pvalue(pvalue: Any, corr: pd.DataFrame) -> pd.DataFrame | None:
    """The p-values, checked against the matrix they belong to."""
    if pvalue is None:
        return None
    if isinstance(pvalue, pd.DataFrame):
        values = pvalue.to_numpy()
        named = [str(name) for name in pvalue.columns]
    else:
        values = np.asarray(pvalue)
        named = None
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or values.shape != corr.shape:
        raise SaValueError(
            "`pvalue` must be a numeric matrix laid out like `cor_matrix`: "
            f"{corr.shape[0]} by {corr.shape[1]}."
        )
    if named is not None and named != list(corr.columns):
        raise SaValueError(
            "`pvalue` must name the same features as `cor_matrix`, in the same order."
        )
    return pd.DataFrame(values.astype(float), index=corr.index, columns=corr.columns)


def _corrplot_order(
    corr: pd.DataFrame, cluster: bool, hclust_method: str, feats: list[str]
) -> tuple[np.ndarray, Clustering | None]:
    """One order for both axes, from the ``1 - corr`` distance.

    Clustered once rather than per axis: a symmetric matrix clustered twice can
    come back with its rows and columns in different orders, and the diagonal then
    wanders off the diagonal.
    """
    identity = np.arange(len(feats))
    if not cluster:
        return identity, None

    distances = 1 - corr.to_numpy(dtype=float)
    # `squareform` reads the off-diagonal entries, which is what `as.dist()` takes,
    # and refuses a diagonal that is not zero to floating point.
    np.fill_diagonal(distances, 0.0)
    condensed = squareform(distances, checks=False)
    if not np.isfinite(condensed).all():
        notify(
            "Some pair of features has no correlation to measure a distance between, "
            "so the features are drawn in the order they arrived."
        )
        return identity, None

    tree = linkage(condensed, method=LINKAGE_NAMES[hclust_method])
    order = np.asarray(dendrogram(tree, no_plot=True)["leaves"], dtype=int)
    return order, Clustering(
        linkage=tree,
        order=order,
        method=hclust_method,
        dist_method="correlation",
        labels=feats,
    )
