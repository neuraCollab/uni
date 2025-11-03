import numpy as np

def chebyshev_distance(a, b):
    """Расстояние Чебышева между двумя точками"""
    return np.max(np.abs(np.array(a) - np.array(b)))