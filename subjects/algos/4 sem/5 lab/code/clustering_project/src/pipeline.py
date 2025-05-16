# src/pipeline.py

import os
import numpy as np
from src.clustering import get_clusterer
from src.metrics import get_metric
from src.features import get_feature_selector
from src.utils import load_data, normalize_data, reduce_dimensions, visualize_clusters_2d, generate_report
from src.utils import cluster_statistics
from sklearn.cluster import KMeans


class ClusteringPipeline:
    def __init__(self, config):
        self.config = config
        self.X = None
        self.y_true = None
        self.results = []  # список словарей: {model, labels, metric1, metric2...}

        # Загружаем модели кластеризации
        self.models = {
            name: get_clusterer(model_config["method"], **model_config.get("params", {}))
            for name, model_config in config.get("clustering", {}).items()
        }

        # Загружаем метрики
        metric_names = config.get("metrics", [])
        self.metric_funcs = {
            name: get_metric(name) for name in metric_names if get_metric(name) is not None
        }

        # Отбор признаков
        feature_config = config.get("feature_selection", {})
        self.feature_selector_name = feature_config.get("method", "add_method")
        self.n_features = feature_config.get("n_features", 2)

        # Получаем функцию отбора
        self.feature_selector = get_feature_selector(self.feature_selector_name)

    def load_data(self):
        """Загрузка и предобработка данных"""
        X, headers = load_data(self.config["data_path"])

        if self.config.get("normalize", False):
            X = normalize_data(X)

        # Проверяем, есть ли истинные метки
        if self.config.get("has_true_labels", True) and X.shape[1] > 1:
            self.y_true = X[:, -1].flatten()
            if len(np.unique(self.y_true)) == 1:
                st.warning("⚠️ Все истинные метки одинаковые. Невозможно использовать внешние метрики.")
                self.y_true = None
        else:
            self.y_true = None

        # Отбор признаков — разный для разных методов
        if self.feature_selector_name in ["add_method", "del_method", "spa_method"]:
            selected_indices, _ = self.feature_selector(
                X=X,
                y_true=self.y_true,
                n_features=self.n_features,
                metric_name="rand_index",
                clusterer=KMeans(n_clusters=self.config["n_clusters"])
            )
        elif self.feature_selector_name in ["compactness", "spread"]:
            selected_indices, _ = self.feature_selector(
                X=X,
                n_features=self.n_features
            )
        else:
            raise ValueError(f"Метод отбора '{self.feature_selector_name}' не найден")

        self.X = X[:, selected_indices]

    def run_comparison(self):
        """Обучение и сравнение всех моделей"""
        print("🚀 Начинаем обучение моделей...")
        for name, model in self.models.items():
            print(f"🧠 Обучение: {name}")
            labels = model.fit_predict(self.X)
            result = {"model": name, "labels": labels}

            for metric_name, func in self.metric_funcs.items():
                try:
                    if metric_name in ["compactness", "separation"]:
                        score = func(self.X, labels)
                    else:
                        if self.y_true is not None:
                            score = func(self.y_true, labels)
                        else:
                            score = "N/A"
                except Exception as e:
                    score = f"Ошибка: {str(e)}"

                result[metric_name] = score

            self.results.append(result)

    def generate_reports(self):
        """Генерация отчётов — графики + HTML"""
        output_dir = self.config.get("output_dir", "../reports")
        os.makedirs(output_dir, exist_ok=True)

        # Визуализация кластеров (PCA)
        X_reduced = reduce_dimensions(self.X)

        for result in self.results:
            model_name = result["model"]
            labels = result["labels"]

            plot_path = os.path.join(output_dir, f"{model_name}_clusters.png")
            visualize_clusters_2d(X_reduced, labels, save_path=plot_path, title=model_name)

        # Статистика по кластерам
        cluster_stats = {}
        for result in self.results:
            model_name = result["model"]
            labels = result["labels"]
            stats = cluster_statistics(self.X, labels)
            cluster_stats[model_name] = stats

        # Генерация HTML-отчёта
        report_path = os.path.join(output_dir, "comparison_report.html")
        generate_report(self.results, report_path, cluster_stats)
        print(f"📄 Отчёт создан: {report_path}")

    def run(self):
        self.load_data()
        self.run_comparison()
        self.generate_reports()
        print("✅ Пайплайн успешно выполнен.")