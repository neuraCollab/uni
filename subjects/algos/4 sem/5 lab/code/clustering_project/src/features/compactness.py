import numpy as np
from src.clustering import get_clusterer
from src.metrics.internal_indices import compactness


def select_by_compactness(X, n_features=2):
    """
    Выбираем признаки на основе компактности
    """
    scores = []
    for i in range(X.shape[1]):
        feature_col = X[:, i].reshape(-1, 1)
        model = get_clusterer("cure", n_clusters=min(3, len(np.unique(feature_col))))
        labels = model.fit_predict(feature_col)
        score = compactness(X=feature_col, labels=labels)
        scores.append((score, i))

    scores.sort(key=lambda x: x[0])
    return [i for _, i in scores[:n_features]], [f"feature_{i}" for _, i in scores[:n_features]]