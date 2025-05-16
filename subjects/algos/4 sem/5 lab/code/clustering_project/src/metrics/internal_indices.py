# TODO: Implement this module
import numpy as np
from sklearn.metrics import pairwise_distances


def compactness(X, labels):
    """
    Вычисляет компактность кластеров.
    Чем меньше значение, тем лучше.

    :param X: Матрица признаков [n_samples, n_features]
    :param labels: Предсказанные метки кластеров [n_samples]
    :return: Среднее внутрикластерное расстояние
    """
    unique_labels = np.unique(labels)
    total_distance = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) <= 1:
            continue
        distances = pairwise_distances(cluster_points)
        total_distance += np.sum(distances) / (len(cluster_points) ** 2)
    return total_distance


def separation(X, labels):
    """
    Вычисляет отделимость кластеров.
    Чем больше значение, тем лучше.

    :param X: Матрица признаков [n_samples, n_features]
    :param labels: Предсказанные метки кластеров [n_samples]
    :return: Минимальное расстояние между центрами кластеров
    """
    unique_labels = np.unique(labels)
    centers = []
    for label in unique_labels:
        center = np.mean(X[labels == label], axis=0)
        centers.append(center)
    centers = np.array(centers)

    min_sep = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_sep:
                min_sep = d
    return min_sep