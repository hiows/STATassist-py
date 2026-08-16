"""Result contracts (sa_* objects)."""

from statassist.contracts.cluster import sa_new_cluster
from statassist.contracts.model import sa_new_model
from statassist.contracts.performance import sa_new_performance
from statassist.contracts.reduction import sa_new_reduction
from statassist.contracts.selection import sa_new_selection
from statassist.contracts.split import sa_new_split

__all__ = [
    "sa_new_model",
    "sa_new_selection",
    "sa_new_performance",
    "sa_new_reduction",
    "sa_new_cluster",
    "sa_new_split",
]
