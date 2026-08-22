"""``kernel/cluster.py`` against the numbers R produced."""

from __future__ import annotations

import numpy as np
import pytest
from golden import assert_close, load_case
from scipy.spatial.distance import pdist, squareform

from statassist.core.errors import SaValueError
from statassist.kernel.cluster import NOISE_LABEL, cluster_dist, silhouette


@pytest.fixture(scope="module")
def sil_case() -> tuple[np.ndarray, np.ndarray, dict]:
    """The frozen points, their labels and R's widths.

    Two real clusters, a singleton, two noise points and two points sitting on
    top of each other, which is every convention the kernel has in one input.
    """
    frame, expected = load_case("cluster_silhouette")
    points = frame[["x", "y"]].to_numpy(dtype=float)
    return squareform(pdist(points)), frame["cluster"].to_numpy(dtype=int), expected


def test_silhouette_reproduces_r(sil_case: tuple[np.ndarray, np.ndarray, dict]) -> None:
    distances, labels, expected = sil_case

    assert_close(silhouette(distances, labels).tolist(), expected["mixed"])
    assert_close(
        silhouette(distances, np.where(labels > NOISE_LABEL, 1, NOISE_LABEL)).tolist(),
        expected["one_cluster"],
    )
    assert_close(
        silhouette(distances, np.where(labels == NOISE_LABEL, 1, labels)).tolist(),
        expected["no_noise"],
    )
    assert_close(
        silhouette(distances, np.full(labels.size, NOISE_LABEL)).tolist(),
        expected["all_noise"],
    )


def test_a_condensed_distance_vector_is_read_the_same_way(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    """R takes a ``dist`` object; there is no such thing here, so both are taken."""
    distances, labels, _ = sil_case
    condensed = squareform(distances)

    assert condensed.ndim == 1
    np.testing.assert_array_equal(silhouette(condensed, labels), silhouette(distances, labels))


def test_the_distances_have_to_cover_the_points_they_are_labelled_with(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    distances, labels, _ = sil_case

    with pytest.raises(SaValueError, match="labels"):
        silhouette(distances[:-1, :-1], labels)
    with pytest.raises(SaValueError, match="square"):
        silhouette(distances[:, :-1], labels)


def test_noise_scores_nothing_and_counts_for_nothing(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    """The convention that costs the most to get wrong, so it gets its own test.

    Noise is not a cluster, so a point cannot be near to it in the sense ``b``
    measures, and it is not in a cluster either, so it enters no ``a``. Dropping
    the noise points entirely must therefore leave every other point's width
    where it was.
    """
    distances, labels, _ = sil_case
    widths = silhouette(distances, labels)
    noise = labels == NOISE_LABEL

    assert noise.any()
    assert np.all(np.isnan(widths[noise]))

    kept = ~noise
    without = silhouette(distances[np.ix_(kept, kept)], labels[kept])
    assert_close(without.tolist(), widths[kept].tolist())


def test_a_singleton_scores_zero_rather_than_one(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    """It has no ``a``, and calling that a perfect fit would rank it best in the data."""
    distances, labels, _ = sil_case
    widths = silhouette(distances, labels)
    sizes = np.bincount(labels)
    alone = np.flatnonzero((labels > NOISE_LABEL) & (sizes[labels] == 1))

    assert alone.size == 1
    assert widths[alone[0]] == 0.0
    # And it is far from everything, so an `a` of zero would have scored it 1.
    assert distances[alone[0]].max() == distances.max()


def test_a_single_cluster_has_no_width_at_all(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    """The width is a comparison, and there is nothing to compare against."""
    distances, labels, _ = sil_case
    pooled = np.where(labels > NOISE_LABEL, 1, NOISE_LABEL)

    assert np.all(np.isnan(silhouette(distances, pooled)))
    assert np.all(np.isnan(silhouette(distances, np.full(labels.size, NOISE_LABEL))))
    assert np.all(np.isnan(silhouette(distances, np.ones(labels.size, dtype=int))))


def test_coincident_points_are_a_tie_and_not_a_division(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    """Two points on top of each other in a cluster of two give ``a == b == 0``."""
    coincident = np.zeros((4, 4))
    coincident[0, 1] = coincident[1, 0] = 0.0
    coincident[2, 3] = coincident[3, 2] = 0.0
    coincident[np.ix_([0, 1], [2, 3])] = 0.0
    coincident[np.ix_([2, 3], [0, 1])] = 0.0

    widths = silhouette(coincident, np.array([1, 1, 2, 2]))
    assert list(widths) == [0.0, 0.0, 0.0, 0.0]

    # The frozen input has a coincident pair inside a larger cluster, where the
    # scale is positive and the width is an ordinary number.
    distances, labels, _ = sil_case
    duplicated = np.flatnonzero(np.count_nonzero(distances == 0, axis=1) > 1)
    assert duplicated.size == 2
    assert np.all(np.isfinite(silhouette(distances, labels)[duplicated]))


def test_a_width_reads_between_minus_one_and_one(
    sil_case: tuple[np.ndarray, np.ndarray, dict],
) -> None:
    distances, labels, _ = sil_case
    widths = silhouette(distances, labels)
    assigned = labels > NOISE_LABEL

    assert np.all(widths[assigned] >= -1)
    assert np.all(widths[assigned] <= 1)
    # These clusters are far apart, so the ones with company sit near the top.
    company = assigned & (np.bincount(labels)[labels] > 1)
    assert np.all(widths[company] > 0.8)


def test_a_point_in_the_wrong_cluster_scores_below_zero() -> None:
    """The sign is the finding, so a mislabelled point has to carry it."""
    points = np.array([[0.0], [0.2], [0.4], [5.0], [5.2], [5.4]])
    distances = squareform(pdist(points))
    honest = np.array([1, 1, 1, 2, 2, 2])
    swapped = np.array([1, 1, 2, 1, 2, 2])

    assert np.all(silhouette(distances, honest) > 0.9)
    widths = silhouette(distances, swapped)
    assert widths[2] < 0
    assert widths[3] < 0


def test_the_definition_agrees_with_scikit_learn_where_the_two_overlap() -> None:
    """The counterpart the R original names as the thing to check against.

    ``silhouette_samples`` has no notion of noise and refuses fewer than two
    labels, so it can only be asked about the case where none of this kernel's
    three conventions applies - which is exactly the case it settles.
    """
    metrics = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(4)
    points = np.vstack(
        [rng.normal(0, 1, size=(12, 3)), rng.normal(4, 1, size=(9, 3)), rng.normal(-3, 1, (7, 3))]
    )
    labels = np.repeat([1, 2, 3], [12, 9, 7])
    distances = squareform(pdist(points))

    assert_close(
        silhouette(distances, labels).tolist(),
        metrics.silhouette_samples(distances, labels, metric="precomputed").tolist(),
        rtol=1e-12,
    )


class TestClusterDist:
    """``cluster_dist()`` against :func:`scipy.spatial.distance.pdist`.

    Complete data is what the two have in common, so that is where the formula is
    settled. The gaps are where they part company, and where R's rule - measure
    the pair on what it shares and scale the sum up to the full width - is what is
    checked instead.
    """

    def test_complete_data_agrees_with_pdist(self) -> None:
        points = np.random.default_rng(11).normal(size=(7, 5))
        for method, metric in (("euclidean", "euclidean"), ("manhattan", "cityblock")):
            assert_close(
                cluster_dist(points, method).tolist(),
                pdist(points, metric=metric).tolist(),
                rtol=1e-12,
            )

    def test_correlation_is_one_minus_the_correlation(self) -> None:
        points = np.random.default_rng(12).normal(size=(6, 8))
        expected = 1 - np.corrcoef(points)
        assert_close(
            cluster_dist(points, "correlation").tolist(),
            squareform(expected, checks=False).tolist(),
            rtol=1e-12,
        )

    def test_a_gap_is_measured_on_what_the_pair_shares_and_scaled_up(self) -> None:
        """R's rule: the sum over the shared columns, times p over how many."""
        left = np.array([1.0, 2.0, np.nan, 4.0])
        right = np.array([1.0, 4.0, 7.0, 8.0])
        shared = np.array([0.0, 2.0, 4.0])
        expected = np.sqrt((shared**2).sum() * 4 / 3)
        assert_close(cluster_dist(np.vstack([left, right]), "euclidean").tolist(), [expected])

    def test_a_pair_that_shares_nothing_has_no_distance(self) -> None:
        left = np.array([1.0, 2.0, np.nan, np.nan])
        right = np.array([np.nan, np.nan, 3.0, 4.0])
        assert np.isnan(cluster_dist(np.vstack([left, right]), "euclidean")).all()

    def test_a_row_with_no_variance_has_no_correlation(self) -> None:
        flat = np.array([2.0, 2.0, 2.0])
        varying = np.array([1.0, 3.0, 7.0])
        assert np.isnan(cluster_dist(np.vstack([flat, varying]), "correlation")).all()

    @pytest.mark.parametrize(
        ("x", "method", "match"),
        [
            (np.zeros((3, 2)), "cosine", "`dist_method` must be one of"),
            (np.zeros((1, 2)), "euclidean", "at least two rows"),
            (np.zeros(3), "euclidean", "at least two rows"),
        ],
    )
    def test_a_bad_argument_is_named_in_the_message(self, x, method, match) -> None:
        with pytest.raises(SaValueError, match=match):
            cluster_dist(x, method)
