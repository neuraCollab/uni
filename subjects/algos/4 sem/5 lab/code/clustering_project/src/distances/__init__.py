# Импортируем все функции
from .euclidean import euclidean_distance
from .squared_euclidean import squared_euclidean_distance
from .pearson import pearson_correlation
from .chebyshev import chebyshev_distance
from .minkowski import minkowski_distance

# Фабрика метрик
def get_distance(metric='euclidean', p=2):
    mapping = {
        'euclidean': euclidean_distance,
        'squared_euclidean': squared_euclidean_distance,
        'pearson': lambda a, b: 1 - pearson_correlation(a, b),
        'chebyshev': chebyshev_distance,
        'minkowski': minkowski_distance,
        "manhattan": lambda a, b: minkowski_distance(a, b, p=1),
    }
    if metric not in mapping:
        raise ValueError(f"Неизвестная метрика: {metric}")
    return mapping[metric]