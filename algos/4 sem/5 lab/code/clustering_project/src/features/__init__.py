# Подключаем все методы
from .compactness import select_by_compactness
from .spread import select_by_spread
from .del_method import select_by_del_method
from .add_method import select_by_add_method
from .spa_method import select_by_spa_method

# Фабрика методов отбора признаков
def get_feature_selector(method="compactness"):
    mapping = {
        "compactness": select_by_compactness,
        "spread": select_by_spread,
        "del_method": select_by_del_method,
        "add_method": select_by_add_method,
        "spa_method": select_by_spa_method,
    }
    if method not in mapping:
        raise ValueError(f"Неизвестный метод отбора признаков: {method}")
    return mapping[method]