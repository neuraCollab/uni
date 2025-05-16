import numpy as np
from .base import BaseClusterer


class MaxMinDistanceClusterer(BaseClusterer):
    def __init__(self, k=3, metric='euclidean', threshold=0.5, p=2):
        super().__init__(metric=metric, p=p)
        self.k = k
        self.threshold = threshold
        self.centers_ = None
        self.labels_ = None  # <-- Добавлено

    def fit(self, X):
        X = np.array(X)
        n_samples = len(X)

        centers = [X[np.random.choice(n_samples)]]

        while len(centers) < self.k:
            distances = np.array([np.min([self.distance_func(x, c) for c in centers]) for x in X])
            new_center_idx = np.argmax(distances)

            if distances[new_center_idx] > self.threshold:
                centers.append(X[new_center_idx])
            else:
                break

        self.centers_ = np.array(centers)

        # Теперь вычисляем labels_ на основе найденных центров
        dist_to_centers = np.linalg.norm(X[:, np.newaxis] - self.centers_, axis=2)
        self.labels_ = np.argmin(dist_to_centers, axis=1)

    def predict(self, X):
        if self.labels_ is None:
            raise RuntimeError("Модель не обучена. Вызовите метод fit() сначала.")
        return self.labels_