import numpy as np
from .base import BaseClusterer


class ForelClusterer(BaseClusterer):
    def __init__(self, radius=1.0, metric='euclidean', p=2):
        """
        :param radius: радиус для объединения точек
        :param metric: тип метрики ('euclidean', 'manhattan', 'minkowski', 'chebyshev')
        :param p: степень для Minkowski
        """
        super().__init__(metric=metric, p=p)
        self.radius = radius
        self.centers_ = []

    def fit(self, X):
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

    def predict(self, X):
        if not hasattr(self, "labels_"):
            raise RuntimeError("Модель не обучена")
        return self.labels_