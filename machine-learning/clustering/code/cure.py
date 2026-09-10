"""
CURE (Clustering Using REpresentatives).

Ported from `algos/4 sem/5 lab/code/clustering_project/src/clustering/cure.py`.
One real bug was fixed during porting: `_compute_representative_points`
referenced a nonexistent `self.p` attribute instead of the current candidate
point, which raised `AttributeError` once a cluster grew past `num_reps`
points. See the comment in that method for details.

See ../kmeans-hierarchical-cure-forel-isodata.md for how this differs from k-means.
"""

from typing import List, Tuple

import numpy as np

from base import BaseClusterer


class CureClusterer(BaseClusterer):
    """
    Agglomerative clustering where each cluster is summarized by a handful of
    "representative points" (spread-out points pulled slightly toward the
    cluster centroid) instead of a single centroid. Representatives let CURE
    approximate non-spherical cluster shapes that centroid-based methods
    (k-means) can't, while staying cheaper than comparing every point to
    every other point.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        num_reps: int = 5,
        compression_rate: float = 0.2,
        metric: str = "euclidean",
        p: float = 2,
    ):
        """
        :param n_clusters: target number of clusters to stop merging at.
        :param num_reps: max representative points kept per cluster.
        :param compression_rate: how far representatives are shrunk toward the
            cluster centroid (0 = stay at the extremities, 1 = collapse onto
            the centroid, i.e. behaves like a centroid-only method).
        """
        super().__init__(metric=metric, p=p)
        self.n_clusters = n_clusters
        self.num_reps = num_reps
        self.compression_rate = compression_rate
        self.clusters_: List["_Cluster"] = []

    def fit(self, X: np.ndarray) -> None:
        X = np.array(X)
        points = X.tolist()
        # Start with every point as its own singleton cluster.
        self.clusters_ = [
            _Cluster([point], self.num_reps, self.compression_rate, self.distance_func)
            for point in points
        ]

        # Repeatedly merge the two closest clusters (by representative-point
        # distance) until only n_clusters remain -- classic agglomerative loop.
        while len(self.clusters_) > self.n_clusters:
            i, j = self._find_closest_clusters()
            self.clusters_[i].merge(self.clusters_[j], self.distance_func)
            self.clusters_.pop(j)

        self.labels_ = np.zeros(len(X), dtype=int)
        for label, cluster in enumerate(self.clusters_):
            for point in cluster.points:
                idx = np.where((X == point).all(axis=1))[0][0]
                self.labels_[idx] = label

    def _find_closest_clusters(self) -> Tuple[int, int]:
        min_dist = np.inf
        closest = (0, 1)
        for i in range(len(self.clusters_)):
            for j in range(i + 1, len(self.clusters_)):
                dist = self.clusters_[i].distance_to(self.clusters_[j], self.distance_func)
                if dist < min_dist:
                    min_dist = dist
                    closest = (i, j)
        return closest

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.labels_


class _Cluster:
    """A CURE cluster: its member points plus a shrunk set of representatives."""

    def __init__(self, points, num_reps, compression_rate, distance_func):
        self.points = points
        self.num_reps = num_reps
        self.compression_rate = compression_rate
        self.distance_func = distance_func
        self.representative_points = self._compute_representative_points(distance_func)

    def _compute_representative_points(self, distance_func) -> list:
        """Pick well-scattered points, then shrink them toward the centroid."""
        if len(self.points) <= self.num_reps:
            return self.points.copy()

        reps = self.points[:1]  # seed with the first point
        for _ in range(self.num_reps - 1):
            # Classic farthest-first traversal: among points not yet chosen as
            # a representative, pick the one whose distance to its NEAREST
            # already-chosen representative is largest. This is what
            # textbook CURE does to keep representatives well spread out.
            #
            # (The source project this was ported from had a bug here --
            # `distance_func(self.p, rep)` passed the Minkowski exponent
            # `self.p`, a scalar the class never even set, instead of a
            # candidate point. That raises AttributeError the first time a
            # cluster grows past `num_reps` points, i.e. after a few merges
            # -- it does not "run with slightly wrong output" as it might
            # look at a glance. Fixed here; see model-evaluation/data-leakage.md
            # and this file's history for the general lesson: read what code
            # actually does, don't assume a plausible-looking line is correct.)
            best_point, best_min_dist = None, -1.0
            for candidate in self.points:
                if any(np.array_equal(candidate, r) for r in reps):
                    continue
                min_dist_to_reps = min(distance_func(candidate, r) for r in reps)
                if min_dist_to_reps > best_min_dist:
                    best_point, best_min_dist = candidate, min_dist_to_reps
            if best_point is None:
                break
            reps.append(best_point)

        center = np.mean(np.array(self.points), axis=0)
        reps = np.array(reps)
        compressed_reps = center + self.compression_rate * (reps - center)
        return compressed_reps.tolist()

    def distance_to(self, other: "_Cluster", distance_func) -> float:
        """Minimum distance between any pair of representative points -- single-linkage-style."""
        min_distance = np.inf
        for p1 in self.representative_points:
            for p2 in other.representative_points:
                d = distance_func(p1, p2)
                if d < min_distance:
                    min_distance = d
        return min_distance

    def merge(self, other: "_Cluster", distance_func) -> None:
        self.points += other.points
        self.representative_points = self._compute_representative_points(distance_func)
