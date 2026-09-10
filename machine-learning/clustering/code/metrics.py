"""
Clustering evaluation metrics: external (need ground-truth labels) and
internal (don't).

Consolidated from `algos/4 sem/5 lab/code/clustering_project/src/metrics/
external_indices.py` and `internal_indices.py`. Logic unchanged; docstrings
and type hints were added, and `compactness`/`separation` were adapted to use
this project's own `pairwise_distance_matrix` instead of
`sklearn.metrics.pairwise_distances` (avoids adding a second distance-matrix
code path; same numeric result for the default 'euclidean' metric).

See ../clustering-evaluation-metrics.md for when to use which.
"""

import numpy as np

from distances import euclidean_distance, pairwise_distance_matrix

# ---------------------------------------------------------------------------
# External indices: compare predicted labels against ground-truth labels.
# All are built on counting pairs of points that agree/disagree between the
# two labelings (a pair-counting confusion matrix: TP/FP/FN/TN over pairs,
# not over individual points).
# ---------------------------------------------------------------------------


def rand_index(y_true, y_pred) -> float:
    """
    Rand Index = (TP + TN) / (TP + FP + FN + TN), i.e. the fraction of point
    PAIRS on which the two labelings agree (both put them together, or both
    put them apart). Ranges 0..1; not corrected for chance (see sklearn's
    `adjusted_rand_score` for that).
    """
    A = np.c_[y_true, y_pred]
    n_samples = len(A)

    tp_plus_fp = sum(len(group) * (len(group) - 1) / 2 for _, group in _group_by(A, axis=1))
    tp_plus_fn = sum(len(group) * (len(group) - 1) / 2 for _, group in _group_by(A, axis=0))

    tp = 0
    for i in np.unique(A[:, 0]):
        mask = A[:, 0] == i
        pred_labels = A[mask, 1]
        _, counts = np.unique(pred_labels, return_counts=True)
        tp += sum(c * (c - 1) / 2 for c in counts)

    total_pairs = n_samples * (n_samples - 1) / 2
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = total_pairs - tp - fp - fn

    return (tp + tn) / (tp + fp + fn + tn)


def _group_by(A: np.ndarray, axis: int = 0):
    """Yield (value, rows) groups of A partitioned by column `axis`."""
    unique_values = np.unique(A[:, axis])
    for val in unique_values:
        yield val, A[A[:, axis] == val]


def jaccard_index(y_true, y_pred) -> float:
    """
    Jaccard Index = TP / (TP + FP + FN). Like Rand Index but ignores true
    negatives (pairs correctly kept apart) -- harsher when most pairs are
    "different cluster" by chance, which is the common case for many small
    clusters among many points.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = 0
    for label in np.unique(y_true):
        pred_group = y_pred[y_true == label]
        _, counts = np.unique(pred_group, return_counts=True)
        tp += sum(c * (c - 1) // 2 for c in counts)

    fp_fn_tp = 0
    for label in np.unique(y_pred):
        count = np.sum(y_pred == label)
        fp_fn_tp += count * (count - 1) // 2

    if fp_fn_tp == 0:
        return 0.0
    return tp / fp_fn_tp


def fowlkes_mallows_index(y_true, y_pred) -> float:
    """
    FMI = TP / sqrt((TP + FP)(TP + FN)) -- the geometric mean of pairwise
    precision and recall. Less sensitive to cluster-size imbalance than Rand.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = 0
    for label in np.unique(y_true):
        pred_group = y_pred[y_true == label]
        _, counts = np.unique(pred_group, return_counts=True)
        tp += sum(c * (c - 1) // 2 for c in counts)

    tp_fp = sum(np.sum(y_pred == label) * (np.sum(y_pred == label) - 1) // 2 for label in np.unique(y_pred))
    tp_fn = sum(np.sum(y_true == label) * (np.sum(y_true == label) - 1) // 2 for label in np.unique(y_true))

    denominator = np.sqrt(tp_fp * tp_fn)
    if denominator == 0:
        return 0.0
    return tp / denominator


def phi_index(y_true, y_pred) -> float:
    """
    Phi index / pairwise Matthews correlation coefficient. Ranges -1..1;
    unlike Rand/Jaccard/FMI it accounts for all four pair-confusion cells
    (TP, FP, FN, TN) in a chance-corrected-ish way, and can go negative for
    labelings that actively disagree more than random chance would.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = 0
    for label in np.unique(y_true):
        pred_group = y_pred[y_true == label]
        _, counts = np.unique(pred_group, return_counts=True)
        tp += sum(c * (c - 1) // 2 for c in counts)

    fp = sum(np.sum((y_pred == p) & (y_true != p)) for p in np.unique(y_pred))
    fn = sum(np.sum((y_true == t) & (y_pred != t)) for t in np.unique(y_true))
    tn = len(y_true) - tp - fp - fn

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0.0


# ---------------------------------------------------------------------------
# Internal indices: no ground truth needed, judge the clustering by geometry
# alone.
# ---------------------------------------------------------------------------


def compactness(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Average intra-cluster pairwise distance, summed over clusters. Lower is
    better (tighter clusters). Scales with cluster count/size, so only
    compare runs on the same dataset/k -- it is not normalized like silhouette.
    """
    X = np.asarray(X)
    unique_labels = np.unique(labels)
    total_distance = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) <= 1:
            continue
        distances = pairwise_distance_matrix(cluster_points, metric=euclidean_distance)
        total_distance += np.sum(distances) / (len(cluster_points) ** 2)
    return total_distance


def separation(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Minimum distance between any two cluster centroids. Higher is better
    (well-separated clusters). Pair this with `compactness` -- a good
    clustering is compact AND separated; either alone can be gamed (e.g. one
    giant cluster is maximally "compact" in relative terms but has no
    separation to measure).
    """
    X = np.asarray(X)
    unique_labels = np.unique(labels)
    centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

    min_sep = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d < min_sep:
                min_sep = d
    return min_sep
