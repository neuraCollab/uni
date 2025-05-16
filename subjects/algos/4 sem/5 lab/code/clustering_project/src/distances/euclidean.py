import numpy as np

def euclidean_distance(a, b):
    """Евклидово расстояние между двумя точками"""
    return np.linalg.norm(np.array(a) - np.array(b))