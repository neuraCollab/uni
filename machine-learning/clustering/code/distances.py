"""
Distance/similarity metrics used by the from-scratch clusterers in this folder.

Consolidated from `algos/4 sem/5 lab/code/clustering_project/src/distances/*.py`
(one file per metric there) into a single module, plus the `pairwise_distance_matrix`
helper that used to live in that project's `src/utils.py`. Behavior is unchanged.

See ../distance-metrics.md for when to reach for which one.
"""

from typing import Callable

import numpy as np

ArrayLike = np.ndarray


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """L2 distance: straight-line distance between two points."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def squared_euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """L2 distance without the sqrt -- cheaper, same ordering as euclidean."""
    a, b = np.array(a), np.array(b)
    return float(np.sum((a - b) ** 2))


def minkowski_distance(a: ArrayLike, b: ArrayLike, p: float = 2) -> float:
    """
    Generalized L_p distance. p=1 -> Manhattan, p=2 -> Euclidean, p->inf -> Chebyshev.
    """
    a, b = np.array(a), np.array(b)
    return float(np.power(np.sum(np.abs(a - b) ** p), 1 / p))


def chebyshev_distance(a: ArrayLike, b: ArrayLike) -> float:
    """L-infinity distance: the single largest per-coordinate difference."""
    return float(np.max(np.abs(np.array(a) - np.array(b))))


def pearson_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation coefficient between two equal-length vectors."""
    a, b = np.array(a), np.array(b)
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length.")
    return float(np.corrcoef(a, b)[0, 1])


def pearson_distance(a: ArrayLike, b: ArrayLike) -> float:
    """1 - Pearson correlation: 0 for perfectly correlated shapes, 2 for perfectly anti-correlated."""
    return 1 - pearson_correlation(a, b)


def manhattan_distance(a: ArrayLike, b: ArrayLike) -> float:
    """L1 distance, i.e. Minkowski with p=1."""
    return minkowski_distance(a, b, p=1)


def get_distance(metric: str = "euclidean", p: float = 2) -> Callable[[ArrayLike, ArrayLike], float]:
    """Metric-name -> distance-function factory used by BaseClusterer."""
    mapping = {
        "euclidean": euclidean_distance,
        "squared_euclidean": squared_euclidean_distance,
        "pearson": pearson_distance,
        "chebyshev": chebyshev_distance,
        "minkowski": lambda a, b: minkowski_distance(a, b, p=p),
        "manhattan": lambda a, b: minkowski_distance(a, b, p=1),
    }
    if metric not in mapping:
        raise ValueError(f"Unknown metric: {metric}")
    return mapping[metric]


def pairwise_distance_matrix(X: ArrayLike, metric: Callable[[ArrayLike, ArrayLike], float]) -> np.ndarray:
    """
    Dense symmetric matrix of pairwise distances under `metric`. O(n^2) calls
    to `metric` -- fine for the small demo datasets these clusterers target,
    not for large n (use `scipy.spatial.distance.pdist`/`cdist` there instead).
    """
    X = np.array(X)
    n_samples = len(X)
    matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = metric(X[i], X[j])
            matrix[i, j] = matrix[j, i] = d
    return matrix
