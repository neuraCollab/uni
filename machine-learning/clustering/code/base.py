"""
Shared abstract base class for the from-scratch clusterers in this folder
(cure.py, forel.py, isodata.py, hierarchical.py).

Ported from `algos/4 sem/5 lab/code/clustering_project/src/clustering/base.py`,
which every clusterer in that Streamlit project subclassed. Logic unchanged;
only the import path (local `distances.py` instead of the original `src.distances`
package) and docstrings/type hints were added.
"""

from abc import ABC, abstractmethod

import numpy as np

from distances import get_distance


class BaseClusterer(ABC):
    """
    Common scikit-learn-flavored interface (`fit` / `predict` / `fit_predict`)
    plus a metric factory, so every subclass gets `metric='euclidean' | 'manhattan'
    | 'chebyshev' | 'minkowski' | 'pearson' | 'squared_euclidean'` for free
    without reimplementing distance handling.
    """

    def __init__(self, metric: str = "euclidean", p: float = 2):
        self.metric = metric
        self.p = p
        self.distance_func = self._get_distance_func()

    def _get_distance_func(self):
        return get_distance(self.metric, p=self.p)

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)
