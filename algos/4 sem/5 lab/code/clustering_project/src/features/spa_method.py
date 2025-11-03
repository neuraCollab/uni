import numpy as np
import pandas as pd
import random
from src.metrics import get_metric
from sklearn.cluster import KMeans


def select_by_spa_method(X, y_true, n_features=2, metric_name='rand_index', clusterer=None, n_iterations=20):
    """
    Метод случайного поиска с адаптацией (SPA).
    В каждой итерации случайно генерируется набор признаков и оценивается его качество.

    :param X: Матрица признаков [n_samples, n_features]
    :param y_true: Истинные метки (для внешних метрик)
    :param n_features: Число итоговых признаков
    :param metric_name: Имя метрики качества кластеризации
    :param clusterer: Объект кластеризатора (должен иметь fit_predict)
    :param n_iterations: Число итераций случайного поиска
    :return: Индексы лучших признаков
    """

    if metric_name not in ["compactness", "separation"] and y_true is None:
        raise ValueError("❌ Для этой метрики нужны истинные метки")
    
    if clusterer is None:
        # Используем KMeans как дефолтный кластеризатор
        clusterer = KMeans(n_clusters=min(3, len(np.unique(y_true))))
    
    if isinstance(X, pd.DataFrame):
        columns = X.columns.tolist()
        X = X.values
    else:
        columns = list(range(X.shape[1]))

    n_total = X.shape[1]
    feature_sets = [random.sample(range(n_total), n_features) for _ in range(n_iterations)]
    metric_func = get_metric(metric_name)

    best_score = -np.inf
    best_set = None

    for feat_set in feature_sets:
        X_subset = X[:, feat_set]
        labels = clusterer.fit_predict(X_subset)
        score = metric_func(y_true, labels)

        if score > best_score:
            best_score = score
            best_set = feat_set

    return best_set, [columns[i] for i in best_set]