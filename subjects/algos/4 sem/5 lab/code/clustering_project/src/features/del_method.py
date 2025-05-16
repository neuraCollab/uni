import numpy as np
import pandas as pd
from src.metrics import get_metric
from sklearn.cluster import KMeans

def select_by_del_method(X, y_true, n_features=2, metric_name='rand_index', clusterer=None):
    from src.metrics import get_metric
    from sklearn.cluster import KMeans

    if metric_name not in ["compactness", "separation"] and y_true is None:
        raise ValueError("❌ Для этой метрики нужны истинные метки")

    if clusterer is None:
        clusterer = KMeans(n_clusters=min(3, len(np.unique(y_true))))

    metric_func = get_metric(metric_name)

    selected = list(range(X.shape[1]))
    while len(selected) > n_features:
        worst_score = np.inf
        worst_idx = -1

        for i in range(len(selected)):
            subset = [selected[j] for j in range(len(selected)) if j != i]
            X_subset = X[:, subset]

            labels = clusterer.fit_predict(X_subset)

            if metric_name in ["compactness", "separation"]:
                score = metric_func(X_subset, labels)
            else:
                score = metric_func(y_true, labels)

            if score < worst_score:
                worst_score = score
                worst_idx = i

        # --- Проверка ---
        if worst_idx == -1 or worst_idx >= len(selected):
            raise ValueError("❌ Не удалось удалить признак — возможно, достигнут минимум")
        selected.pop(worst_idx)

    return selected, [f"feature_{i}" for i in selected]