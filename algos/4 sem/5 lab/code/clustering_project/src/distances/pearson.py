import numpy as np

def pearson_correlation(a, b):
    """Коэффициент корреляции Пирсона между двумя векторами"""
    a, b = np.array(a), np.array(b)
    if len(a) != len(b):
        raise ValueError("Векторы должны быть одинаковой длины.")
    return np.corrcoef(a, b)[0, 1]

def pearson_distanse(a,b):
    return 1 - pearson_correlation(a, b)