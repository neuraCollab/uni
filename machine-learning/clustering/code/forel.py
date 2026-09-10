"""
FOREL (FORmal ELement) clustering.

Ported from `algos/4 sem/5 lab/code/clustering_project/src/clustering/forel.py`.
Logic unchanged; docstrings/type hints and the import path were touched.

See ../kmeans-hierarchical-cure-forel-isodata.md for algorithmic detail.
"""

import numpy as np

from base import BaseClusterer


class ForelClusterer(BaseClusterer):
    """
    A density-based, radius-driven method from the Russian pattern-recognition
    literature ("FOREL" = "formal'nyi element", roughly "formal element"):

    1. Pick an unclustered point as a tentative cluster center.
    2. Grab every unclustered point within `radius` of it.
    3. Recompute the center as the mean of that ball's points (the ball then
       tends to drift/"roll" toward denser regions -- unlike k-means, which
       fixes k up front, here the *radius* is the hyperparameter and the
       *number of clusters* falls out of the data).
    4. Remove the settled cluster's points from the pool and repeat from a new
       unclustered point until none remain.

    NOTE: this ported version does one growth step per cluster (grab, take the
    mean, done) rather than looping "recompute center -> re-collect points
    within radius" to convergence before finalizing each cluster, which is
    what many FOREL descriptions do. It is a legitimate one-pass simplification
    of the classic method, not a bug -- flagged here so you don't assume it
    iterates to a fixed point internally.
    """

    def __init__(self, radius: float = 1.0, metric: str = "euclidean", p: float = 2):
        """
        :param radius: neighborhood radius used to grow each cluster.
        :param metric: 'euclidean', 'manhattan', 'minkowski', 'chebyshev', ...
        """
        super().__init__(metric=metric, p=p)
        self.radius = radius
        self.centers_ = []

    def fit(self, X: np.ndarray) -> None:
        X = np.array(X)
        remaining = np.arange(len(X))
        labels = np.full(len(X), -1)

        cluster_id = 0
        while len(remaining) > 0:
            center_idx = remaining[0]
            center_point = X[center_idx]

            distances = np.array([self.distance_func(x, center_point) for x in X[remaining]])
            close_indices = remaining[distances <= self.radius]
            labels[close_indices] = cluster_id

            self.centers_.append(np.mean(X[close_indices], axis=0))

            remaining = np.setdiff1d(remaining, close_indices)
            cluster_id += 1

        self.labels_ = labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "labels_"):
            raise RuntimeError("Model has not been fitted yet.")
        return self.labels_
