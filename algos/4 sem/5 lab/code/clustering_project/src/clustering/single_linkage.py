import numpy as np
from sklearn.cluster import AgglomerativeClustering
from .base import BaseClusterer


class SingleLinkageClusterer(BaseClusterer):
    def __init__(self, n_clusters=3, metric='euclidean', p=2):
        super().__init__(metric=metric, p=p)
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        from src.utils import pairwise_distance_matrix

        # Создаем матрицу попарных расстояний
        distance_matrix = pairwise_distance_matrix(X, metric=self.distance_func)

        # Убедимся, что мы используем верную форму данных
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='single',
            metric='precomputed'
        )

        # Преобразуем матрицу в формат, который понимает AgglomerativeClustering с precomputed
        # Нужно сделать её треугольной и преобразовать в condensed form
        from scipy.spatial.distance import squareform

        # Сначала убираем дублированные значения (берём только нижний треугольник)
        distances_condensed = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

        # Теперь кластеризуем
        labels = model.fit_predict(squareform(distances_condensed))
        self.labels_ = labels

    def predict(self, X):
        if not hasattr(self, "labels_"):
            raise RuntimeError("Модель не обучена")
        return self.labels_