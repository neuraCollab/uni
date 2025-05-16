import numpy as np
from .base import BaseClusterer


class ISODATAClusterer(BaseClusterer):
    def __init__(
        self,
        k_initial=3,
        max_clusters=10,
        min_points_per_cluster=5,
        sigma_threshold=1.0,
        merge_threshold=1.5,
        max_iterations=10,
        metric='euclidean',
        p=2
    ):
        super().__init__(metric=metric, p=p)
        self.k = k_initial
        self.max_clusters = max_clusters
        self.min_points_per_cluster = min_points_per_cluster
        self.sigma_threshold = sigma_threshold
        self.merge_threshold = merge_threshold
        self.max_iterations = max_iterations
        self.centers_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.array(X)
        n_samples, _ = X.shape

        # Инициализируем случайные центры
        indices = np.random.choice(n_samples, size=self.k, replace=False)
        centers = X[indices]
        iteration = 0
        while iteration < self.max_iterations and len(centers) < self.max_clusters:
            # Присвоение кластеров
            distances = np.array([[self.distance_func(x, c) for c in centers] for x in X])
            labels = np.argmin(distances, axis=1)

            # Пересчёт центров
            unique_labels = np.unique(labels)
            new_centers = []
            for label in unique_labels:
                points = X[labels == label]
                if len(points) >= self.min_points_per_cluster:
                    new_centers.append(np.mean(points, axis=0))
                else:
                    print(f"Кластер {label} удален (слишком мало точек)")

            # Проверка на слияние кластеров
            merged_centers = self._merge_close_clusters(new_centers)
            centers = merged_centers
            iteration += 1

        self.centers_ = np.array(centers)
        self.labels_ = np.argmin([[self.distance_func(x, c) for c in centers] for x in X], axis=1)

    def _merge_close_clusters(self, centers):
        """Слияние близких кластеров"""
        merged_centers = list(centers)
        used = set()
        for i in range(len(merged_centers)):
            for j in range(i + 1, len(merged_centers)):
                if i in used or j in used:
                    continue
                dist = self.distance_func(merged_centers[i], merged_centers[j])
                if dist < self.merge_threshold:
                    merged_centers[i] = (np.array(merged_centers[i]) + np.array(merged_centers[j])) / 2
                    used.add(j)

        return [c for i, c in enumerate(merged_centers) if i not in used]

    def predict(self, X):
        return self.labels_