"""Grouping points without being told what the groups are.

Four functions, and the disagreement between them is the information.
:func:`cluster_hclust` and :func:`cluster_kmeans` are told how many groups to find
and will always find that many, so they partition: every point lands somewhere and
a point in the middle of nowhere lands somewhere anyway. :func:`cluster_dbscan` and
:func:`cluster_snn` are told how dense a group has to be and derive the count from
that, so they can return two clusters, or nine, or none at all, and they can refuse
to place a point. A structure all four agree on is a different fact from one that
only k-means, having been told to find two things, found two of.

All four read their input through the same helpers the reductions use, so a
clustering and an embedding of the same frame are about the same rows and an
assignment can be painted straight onto a set of coordinates.
"""

from __future__ import annotations

from .dbscan import cluster_dbscan
from .hclust import HCLUST_METHODS, cluster_hclust
from .kmeans import KMEANS_N_START, cluster_kmeans
from .snn import SnnGraph, cluster_snn

__all__ = [
    "HCLUST_METHODS",
    "KMEANS_N_START",
    "SnnGraph",
    "cluster_dbscan",
    "cluster_hclust",
    "cluster_kmeans",
    "cluster_snn",
]
