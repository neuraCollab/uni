import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Конфигурация и логирование
# -----------------------------

def setup_logging(log_file="logs/project.log"):
    """Настройка логгирования"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path="config.yaml"):
    """Загружает параметры из YAML-конфига"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Загрузка и обработка данных
# -----------------------------

def load_data(path):
    df = pd.read_csv(path)
    data = df.values
    return data, df.columns.tolist()

# def load_data(file_path, delimiter=",", header=0):
#     df = pd.read_csv(file_path, delimiter=delimiter, header=header)
#     return df.values, df.columns.tolist()


def save_data(data, file_path, header=None):
    df = pd.DataFrame(data)
    if header:
        df.columns = header
    df.to_csv(file_path, index=False)


def normalize_data(X, method="z-score"):
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "z-score":
        scaler = StandardScaler()
    else:
        raise ValueError("Метод нормализации должен быть 'minmax' или 'z-score'")
    return scaler.fit_transform(X)


# -----------------------------
# Визуализация кластеров
# -----------------------------

def visualize_clusters_2d(X, labels, title="Кластеры (2D)", xlabel="X", ylabel="Y", save_path=None):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label=f"Cluster {label}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_clusters_3d(X, labels, title="Кластеры (3D)", save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    for label in unique_labels:
        ax.scatter(X[labels == label, 0], X[labels == label, 1], X[labels == label, 2], label=f"Cluster {label}")
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def reduce_dimensions(X, method="pca", n_components=2):
    """Снижает размерность до 2D/3D с помощью PCA или t-SNE"""
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Неподдерживаемый метод снижения размерности")
    return reducer.fit_transform(X)

def pairwise_distance_matrix(X, metric='euclidean'):
    """
    Вычисляет матрицу попарных расстояний
    """
    X = np.array(X)
    n_samples = len(X)
    matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = metric(X[i], X[j])
            matrix[i, j] = matrix[j, i] = d

    return matrix
# -----------------------------
# Статистика по кластерам
# -----------------------------

def cluster_statistics(X, labels):
    unique_labels = np.unique(labels)
    stats = {}
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        stats[label] = {
            "count": int(np.sum(mask)),
            "mean": np.mean(cluster_points, axis=0).tolist(),
            "std": np.std(cluster_points, axis=0).tolist(),
            "min": np.min(cluster_points, axis=0).tolist(),
            "max": np.max(cluster_points, axis=0).tolist()
        }
    return stats


def print_cluster_stats(stats):
    for label, data in stats.items():
        logging.info(f"\nКластер {label}:")
        logging.info(f"  Количество точек: {data['count']}")
        logging.info(f"  Среднее значение: {data['mean']}")
        logging.info(f"  Стандартное отклонение: {data['std']}")
        logging.info(f"  Минимум: {data['min']}")
        logging.info(f"  Максимум: {data['max']}")


def save_cluster_stats(stats, file_path):
    with open(file_path, 'w') as f:
        json.dump(stats, f, indent=4)


# -----------------------------
# Генерация отчётов
# -----------------------------
def generate_report(results, file_path, cluster_stats=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Сравнение кластеризаторов</title></head><body>")
        f.write("<h1>📊 Сравнение алгоритмов кластеризации</h1>")

        # Таблица метрик
        f.write("<table border='1'><tr><th>Модель</th>")
        for key in results[0].keys():
            if key != "labels":
                f.write(f"<th>{key}</th>")
        f.write("</tr>")

        for result in results:
            f.write(f"<tr><td>{result['model']}</td>")
            for key, value in result.items():
                if key == "labels":
                    continue
                f.write(f"<td>{value}</td>")
            f.write("</tr>")
        f.write("</table>")

        # Статистика по кластерам
        if cluster_stats:
            f.write("<h2>🧮 Статистика по кластерам</h2>")
            for model_name, stats in cluster_stats.items():
                f.write(f"<h3>{model_name}</h3><ul>")
                for label, data in stats.items():
                    f.write(f"<li>Кластер {label}:<ul>")
                    for k, v in data.items():
                        f.write(f"<li>{k}: {v}</li>")
                    f.write("</ul></li>")
                f.write("</ul>")

        f.write("</body></html>")

    print(f"📄 Отчёт сохранён в {file_path}")