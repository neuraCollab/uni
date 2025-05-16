import numpy as np
import pandas as pd
from src.metrics import get_metric
from sklearn.cluster import KMeans

def select_by_add_method(X, y_true, n_features=2, metric_name='rand_index', clusterer=None):
    from src.metrics import get_metric
    from sklearn.cluster import KMeans

    metric_func = get_metric(metric_name)

    # --- Проверка на наличие y_true ---
    if metric_name not in ["compactness", "separation"] and y_true is None:
        raise ValueError("❌ Внешняя метрика требует истинных меток")

    if clusterer is None:
        clusterer = KMeans(n_clusters=min(3, len(np.unique(y_true)))) if y_true is not None else KMeans(n_clusters=3)

    metric_func = get_metric(metric_name)

    selected = []
    remaining = list(range(X.shape[1]))

    while len(selected) < n_features and remaining:
        best_score = -np.inf
        best_idx = -1

        for idx in remaining:
            candidate = selected + [idx]
            X_candidate = X[:, candidate]

            labels = clusterer.fit_predict(X_candidate)

            # --- ПРАВИЛЬНЫЙ вызов ---
            if metric_name in ["compactness", "separation"]:
                score = metric_func(X_candidate, labels)  # позиционные аргументы
            else:
                if y_true is None:
                    raise ValueError("❌ Невозможно использовать внешнюю метрику без истинных меток")
                score = metric_func(y_true, labels)  # позиционные аргументы

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1 or best_idx not in remaining:
            raise ValueError("❌ Не удалось выбрать признак")

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected, [f"feature_{i}" for i in selected]