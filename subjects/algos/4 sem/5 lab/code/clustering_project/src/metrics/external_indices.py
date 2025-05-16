import numpy as np
from collections import Counter


def rand_index(y_true, y_pred):
    """
    Вычисляет индекс Rand.
    Rand Index = (TP + TN) / (TP + FP + FN + TN)

    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: Значение Rand Index от 0 до 1
    """
    A = np.c_[y_true, y_pred]
    n_samples = len(A)
    tp_plus_fp = sum(len(group) * (len(group) - 1) / 2 for _, group in _group_by(A, axis=1))
    tp_plus_fn = sum(len(group) * (len(group) - 1) / 2 for _, group in _group_by(A, axis=0))

    # Подсчёт совпадающих пар
    tp = 0
    for i in np.unique(A[:, 0]):
        mask = (A[:, 0] == i)
        pred_labels = A[mask, 1]
        values, counts = np.unique(pred_labels, return_counts=True)
        tp += sum(c * (c - 1) / 2 for c in counts)

    total_pairs = n_samples * (n_samples - 1) / 2
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = total_pairs - tp - fp - fn

    return (tp + tn) / (tp + fp + fn + tn)


def _group_by(A, axis=0):
    """Вспомогательная функция для группировки по оси"""
    unique_values = np.unique(A[:, axis])
    for val in unique_values:
        yield val, A[A[:, axis] == val]


def jaccard_index(y_true, y_pred):
    """
    Вычисляет индекс Жаккара.
    Jaccard Index = TP / (TP + FP + FN)

    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: Значение Jaccard Index от 0 до 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Подсчёт общего числа совпадающих пар
    tp = 0
    for label in np.unique(y_true):
        mask = (y_true == label)
        pred_group = y_pred[mask]
        values, counts = np.unique(pred_group, return_counts=True)
        tp += sum(c * (c - 1) // 2 for c in counts)

    # Общее число пар в предсказаниях
    fp_fn_tp = 0
    for label in np.unique(y_pred):
        mask = (y_pred == label)
        count = np.sum(mask)
        fp_fn_tp += count * (count - 1) // 2

    # Всего возможных пар
    total_pairs = len(y_true) * (len(y_true) - 1) // 2

    # TN + TP + FP + FN = total_pairs
    # Jaccard = TP / (TP + FP + FN)
    denominator = fp_fn_tp  # FP + FN + TP
    if denominator == 0:
        return 0.0

    return tp / denominator


def fowlkes_mallows_index(y_true, y_pred):
    """
    Вычисляет индекс Фоулкса – Мэллова (FMI).
    FMI = TP / sqrt((TP + FP)(TP + FN))

    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: Значение Fowlkes-Mallows Index от 0 до 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = 0
    for label in np.unique(y_true):
        mask = (y_true == label)
        pred_group = y_pred[mask]
        values, counts = np.unique(pred_group, return_counts=True)
        tp += sum(c * (c - 1) // 2 for c in counts)

    tp_fp = 0
    for label in np.unique(y_pred):
        mask = (y_pred == label)
        count = np.sum(mask)
        tp_fp += count * (count - 1) // 2

    tp_fn = 0
    for label in np.unique(y_true):
        mask = (y_true == label)
        count = np.sum(mask)
        tp_fn += count * (count - 1) // 2

    denominator = np.sqrt(tp_fp * tp_fn)
    if denominator == 0:
        return 0.0

    return tp / denominator


def phi_index(y_true, y_pred):
    """
    Вычисляет Phi-индекс (коэффициент корреляции Мэтьюса).
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: Значение Phi Index от -1 до 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = 0
    for label in np.unique(y_true):
        mask = (y_true == label)
        pred_group = y_pred[mask]
        values, counts = np.unique(pred_group, return_counts=True)
        tp += sum(c * (c - 1) // 2 for c in counts)

    fp = sum(np.sum((y_pred == p) & (y_true != p)) for p in np.unique(y_pred))
    fn = sum(np.sum((y_true == t) & (y_pred != t)) for t in np.unique(y_true))
    tn = len(y_true) - tp - fp - fn

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator if denominator != 0 else 0.0