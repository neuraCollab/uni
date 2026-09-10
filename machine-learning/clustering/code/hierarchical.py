"""
Agglomerative (bottom-up) hierarchical clustering with a choice of linkage.

Ported from `algos/4 sem/5 lab/code/clustering_project/src/clustering/hierarchical.py`.
Logic unchanged; docstrings/type hints and the import path were touched.

Consolidation note: the source project also had a `single_linkage.py` file
implementing single-linkage clustering separately, by building a distance
matrix and delegating to `sklearn.cluster.AgglomerativeClustering(linkage=
'single', metric='precomputed')`. That's functionally the same case already
covered here by `HierarchicalClusterer(linkage="single")` (this file's
from-scratch naive agglomerative loop), just implemented via sklearn instead
of from scratch -- so it was not ported separately; use `linkage="single"`
below to get that behavior.

See ../kmeans-hierarchical-cure-forel-isodata.md for linkage-choice tradeoffs.
"""

from typing import List, Tuple

import numpy as np

from base import BaseClusterer
from distances import pairwise_distance_matrix


class HierarchicalClusterer(BaseClusterer):
    """
    Starts with every point as its own cluster and repeatedly merges the two
    closest clusters until `n_clusters` remain. "Closest" is defined by the
    `linkage` rule:

    - single:   min distance between any pair of points across the two
                clusters -- can chain together elongated/non-convex clusters,
                but is prone to "chaining" through noise.
    - complete: max distance between any pair -- favors compact, similarly
                sized clusters; sensitive to outliers.
    - average:  mean pairwise distance -- a middle ground between the two.

    This is a naive O(n^3)-ish implementation (recomputes closest-pair search
    over all remaining cluster pairs on every merge) suitable for small demo
    datasets; scipy's `linkage`/`fcluster` use a faster algorithm for real use.
    """

    def __init__(self, n_clusters: int = 3, linkage: str = "single", metric: str = "euclidean", p: float = 2):
        """
        :param n_clusters: number of clusters to stop merging at.
        :param linkage: "single" | "complete" | "average".
        """
        super().__init__(metric=metric, p=p)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X: np.ndarray) -> None:
        distances = pairwise_distance_matrix(X, metric=self.distance_func)

        clusters: List[List[int]] = [[i] for i in range(len(X))]
        labels = np.zeros(len(X), dtype=int)

        while len(clusters) > self.n_clusters:
            i, j = self._find_closest_clusters(clusters, distances)
            clusters[i] = clusters[i] + clusters[j]
            clusters.pop(j)

        for label, cluster in enumerate(clusters):
            labels[cluster] = label

        self.labels_ = labels

    def _find_closest_clusters(self, clusters: List[List[int]], distance_matrix: np.ndarray) -> Tuple[int, int]:
        min_dist = np.inf
        closest = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self._cluster_distance(clusters[i], clusters[j], distance_matrix)
                if dist < min_dist:
                    min_dist = dist
                    closest = (i, j)
        return closest

    def _cluster_distance(self, cluster_i: List[int], cluster_j: List[int], distance_matrix: np.ndarray) -> float:
        block = distance_matrix[np.ix_(cluster_i, cluster_j)]
        if self.linkage == "single":
            return float(np.min(block))
        elif self.linkage == "complete":
            return float(np.max(block))
        elif self.linkage == "average":
            return float(np.mean(block))
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.labels_
