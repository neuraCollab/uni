from abc import ABC, abstractmethod
from src.distances import get_distance


class BaseClusterer(ABC):
    def __init__(self, metric='euclidean', p=2):
        self.metric = metric
        self.p = p
        self.distance_func = self._get_distance_func()

    def _get_distance_func(self):
        """Получает функцию расстояния на основе метрики"""
        return get_distance(self.metric, p=self.p)

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)