import numpy as np
from .base import BaseClusterer


class CureClusterer(BaseClusterer):
    def __init__(self, n_clusters=3, num_reps=5, compression_rate=0.2, metric='euclidean', p=2):
        super().__init__(metric=metric, p=p)

        self.n_clusters = n_clusters
        self.num_reps = num_reps
        self.compression_rate = compression_rate
        self.clusters_ = []

    def fit(self, X):
        X = np.array(X)
        points = X.tolist()
        self.clusters_ = [Cluster([point], self.num_reps, self.compression_rate, self.distance_func) for point in points]

        while len(self.clusters_) > self.n_clusters:
            i, j = self._find_closest_clusters(self.distance_func)
            self.clusters_[i].merge(self.clusters_[j], self.distance_func)
            self.clusters_.pop(j)

        self.labels_ = np.zeros(len(X), dtype=int)
        for label, cluster in enumerate(self.clusters_):
            for point in cluster.points:
                idx = np.where((X == point).all(axis=1))[0][0]
                self.labels_[idx] = label

    def _find_closest_clusters(self, distance_func):
        min_dist = np.inf
        closest = (0, 1)
        for i in range(len(self.clusters_)):
            for j in range(i + 1, len(self.clusters_)):
                dist = self.clusters_[i].distance_to(self.clusters_[j], distance_func)
                if dist < min_dist:
                    min_dist = dist
                    closest = (i, j)
        return closest

    def predict(self, X):
        return self.labels_


class Cluster:
    def __init__(self, points, num_reps, compression_rate, distance_func, metric='euclidean', p=2):
        self.points = points
        self.num_reps = num_reps
        self.compression_rate = compression_rate
        self.metric = metric
        self.p = p 
        self.distance_func = distance_func
        self.representative_points = self._compute_representative_points(distance_func)

    def _compute_representative_points(self, distance_func):
        """Выбирает несколько наиболее удалённых точек как репрезентативные"""
        if len(self.points) <= self.num_reps:
            return self.points.copy()

        reps = self.points[:1]  # начальная точка
        points_array = np.array(self.points)

        for _ in range(self.num_reps - 1):
            distances = [distance_func(self.p, rep) for rep in reps]
            farthest_idx = np.argmax(distances)
            reps.append(self.points[farthest_idx])

        # Сжимаем точки к центру
        center = np.mean(points_array, axis=0)
        reps = np.array(reps)
        compressed_reps = center + self.compression_rate * (reps - center)
        return compressed_reps.tolist()

    def distance_to(self, other_cluster, distance_func):
        min_distance = np.inf
        for p1 in self.representative_points:
            for p2 in other_cluster.representative_points:
                d = distance_func(p1, p2)
                if d < min_distance:
                    min_distance = d
        return min_distance

    def merge(self, other, distance_func):
        self.points += other.points
        self.representative_points = self._compute_representative_points(distance_func)