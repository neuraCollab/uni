import numpy as np
from .base import BaseClusterer


class HierarchicalClusterer(BaseClusterer):
    def __init__(self, n_clusters=3, linkage="single", metric='euclidean', p=2):
        """
        :param n_clusters: число кластеров
        :param linkage: тип связи (single, complete, average)
        :param metric: метрика расстояния
        :param p: степень для Minkowski
        """
        super().__init__(metric=metric, p=p)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        from src.utils import pairwise_distance_matrix

        # Создаём матрицу попарных расстояний
        distances = pairwise_distance_matrix(X, metric=self.distance_func)

        # Инициализируем кластеры как отдельные группы
        clusters = [[i] for i in range(len(X))]
        labels = np.zeros(len(X), dtype=int)

        while len(clusters) > self.n_clusters:
            # Находим ближайшие кластеры
            i, j = self._find_closest_clusters(X, clusters, distances)
            new_cluster = clusters[i] + clusters[j]
            clusters[i] = new_cluster
            clusters.pop(j)

        # Назначаем финальные метки
        for label, cluster in enumerate(clusters):
            labels[cluster] = label

        self.labels_ = labels

    def _find_closest_clusters(self, X, clusters, distance_matrix):
        min_dist = np.inf
        closest = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self._cluster_distance(X, clusters[i], clusters[j], distance_matrix)
                if dist < min_dist:
                    min_dist = dist
                    closest = (i, j)
        return closest

    def _cluster_distance(self, X, cluster_i, cluster_j, distance_matrix):
        """Рассчитывает расстояние между двумя кластерами по методу linkage"""
        if self.linkage == "single":
            return np.min(distance_matrix[np.ix_(cluster_i, cluster_j)])
        elif self.linkage == "complete":
            return np.max(distance_matrix[np.ix_(cluster_i, cluster_j)])
        elif self.linkage == "average":
            return np.mean(distance_matrix[np.ix_(cluster_i, cluster_j)])
        else:
            raise ValueError(f"Неизвестный linkage: {self.linkage}")