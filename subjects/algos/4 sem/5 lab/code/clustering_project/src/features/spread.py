import numpy as np
from src.metrics.internal_indices import separation
from sklearn.cluster import KMeans

def select_by_spread(X, n_features=2):
    """
    Выбираем признаки на основе отделимости
    """
    scores = []
    for i in range(X.shape[1]):
        feature_col = X[:, i].reshape(-1, 1)
        
        labels = KMeans(n_clusters=3).fit_predict(feature_col)
        score = separation(X=feature_col, labels=labels)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [i for _, i in scores[:n_features]], [f"feature_{i}" for _, i in scores[:n_features]]