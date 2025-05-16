import numpy as np

def squared_euclidean_distance(a, b):
    """Квадрат евклидова расстояния между двумя точками"""
    a, b = np.array(a), np.array(b)
    return np.sum((a - b) ** 2)