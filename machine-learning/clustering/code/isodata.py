"""
ISODATA (Iterative Self-Organizing Data Analysis Technique).

Ported from `algos/4 sem/5 lab/code/clustering_project/src/clustering/isodata.py`.
Logic unchanged; docstrings/type hints and the import path were touched.

See ../kmeans-hierarchical-cure-forel-isodata.md for how this differs from
plain k-means.
"""

import numpy as np

from base import BaseClusterer


class ISODATAClusterer(BaseClusterer):
    """
    k-means with dynamic cluster count: after each assignment/update pass,
    clusters that end up too small are discarded, and centers that end up too
    close together are merged. (The classic ISODATA also *splits* clusters
    whose internal spread exceeds a threshold -- this implementation performs
    the discard-small-clusters and merge-close-clusters halves of that loop;
    `sigma_threshold` is accepted for API compatibility with the full
    algorithm but the split step itself is not implemented here.)
    """

    def __init__(
        self,
        k_initial: int = 3,
        max_clusters: int = 10,
        min_points_per_cluster: int = 5,
        sigma_threshold: float = 1.0,
        merge_threshold: float = 1.5,
        max_iterations: int = 10,
        metric: str = "euclidean",
        p: float = 2,
    ):
        super().__init__(metric=metric, p=p)
        self.k = k_initial
        self.max_clusters = max_clusters
        self.min_points_per_cluster = min_points_per_cluster
        self.sigma_threshold = sigma_threshold
        self.merge_threshold = merge_threshold
        self.max_iterations = max_iterations
        self.centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray) -> None:
        X = np.array(X)
        n_samples, _ = X.shape

        indices = np.random.choice(n_samples, size=self.k, replace=False)
        centers = X[indices]
        iteration = 0
        while iteration < self.max_iterations and len(centers) < self.max_clusters:
            # Assignment step (same as k-means): nearest center wins.
            distances = np.array([[self.distance_func(x, c) for c in centers] for x in X])
            labels = np.argmin(distances, axis=1)

            # Update step, with a cluster-size floor: drop clusters that
            # collapsed to too few points instead of keeping a degenerate center.
            unique_labels = np.unique(labels)
            new_centers = []
            for label in unique_labels:
                points = X[labels == label]
                if len(points) >= self.min_points_per_cluster:
                    new_centers.append(np.mean(points, axis=0))
                else:
                    print(f"Cluster {label} dropped (too few points)")

            # Merge centers that ended up too close together.
            merged_centers = self._merge_close_clusters(new_centers)
            centers = merged_centers
            iteration += 1

        self.centers_ = np.array(centers)
        self.labels_ = np.argmin([[self.distance_func(x, c) for c in centers] for x in X], axis=1)

    def _merge_close_clusters(self, centers: list) -> list:
        """Average together any pair of centers within `merge_threshold`."""
        merged_centers = list(centers)
        used = set()
        for i in range(len(merged_centers)):
            for j in range(i + 1, len(merged_centers)):
                if i in used or j in used:
                    continue
                dist = self.distance_func(merged_centers[i], merged_centers[j])
                if dist < self.merge_threshold:
                    merged_centers[i] = (np.array(merged_centers[i]) + np.array(merged_centers[j])) / 2
                    used.add(j)
        return [c for i, c in enumerate(merged_centers) if i not in used]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.labels_
