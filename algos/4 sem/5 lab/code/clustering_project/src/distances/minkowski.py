import numpy as np

def minkowski_distance(a, b, p=2):
    """Обобщённое расстояние Минковского между двумя точками"""
    a, b = np.array(a), np.array(b)
    return np.power(np.sum(np.abs(a - b) ** p), 1/p)

# При p=1 → манхэттенское расстояние
# При p=2 → евклидово
# При p→inf → Чебышево 