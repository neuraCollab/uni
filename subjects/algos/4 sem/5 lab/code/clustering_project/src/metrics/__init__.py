from .external_indices import rand_index, jaccard_index, fowlkes_mallows_index, phi_index
from .internal_indices import compactness, separation


def get_metric(name):
    mapping = {
        "rand_index": lambda y_true, labels: rand_index(y_true, labels),
        "jaccard_index": lambda y_true, labels: jaccard_index(y_true, labels),
        "fowlkes_mallows_index": lambda y_true, labels: fowlkes_mallows_index(y_true, labels),
        "phi_index": lambda y_true, labels: phi_index(y_true, labels),

        "compactness": lambda X_data, labels: compactness(X_data, labels),
        "separation": lambda X_data, labels: separation(X_data, labels)
    }

    if name not in mapping:
        raise ValueError(f"Метрика '{name}' не найдена")
    return mapping[name]