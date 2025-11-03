# src/clustering/__init__.py

from .base import BaseClusterer
from .cure import CureClusterer
from .forel import ForelClusterer
from .hierarchical import HierarchicalClusterer
from .isodata import ISODATAClusterer
from .maxmin_distance import MaxMinDistanceClusterer
from .single_linkage import SingleLinkageClusterer


# Автоматический сбор всех доступных кластеризаторов
CLUSTERER_REGISTRY = {
    "cure": CureClusterer,
    "forel": ForelClusterer,
    "hierarchical": HierarchicalClusterer,
    "isodata": ISODATAClusterer,
    "maxmin_distance": MaxMinDistanceClusterer,
    "single_linkage": SingleLinkageClusterer,
}


def get_clusterer(name: str, **kwargs):
    """
    Функция-фабрика для получения кластеризатора по имени
    """
    if name not in CLUSTERER_REGISTRY:
        raise ValueError(f"Кластеризатор '{name}' не найден в registry")
    return CLUSTERER_REGISTRY[name](**kwargs)