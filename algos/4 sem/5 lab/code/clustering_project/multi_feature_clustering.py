import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Импорт из src
from src.pipeline import ClusteringPipeline
from src.utils import load_data, normalize_data, reduce_dimensions, visualize_clusters_2d
from src.metrics import (
    rand_index,
    jaccard_index,
    fowlkes_mallows_index,
    phi_index,
    compactness,
    separation
)
from src.features import get_feature_selector


st.set_page_config(page_title="Кластеризация (15+ признаков)", layout="wide")
st.title("🧬 Кластеризация на датасетах с 15+ признаками")

with st.sidebar:
    st.header("⚙️ Настройки")
    data_path = st.text_input("Путь к данным", value="./data/iris.csv")
    n_clusters = st.number_input("Число кластеров", min_value=2, max_value=20, value=3)
    normalize = st.checkbox("Нормализовать данные", value=True)
    has_true_labels = st.checkbox("Данные содержат истинные метки", value=True)

    st.markdown("---")
    st.subheader("🔍 Отбор признаков")
    feature_methods = ["compactness", "spread", "add_method", "del_method", "spa_method"]
    selected_feature_method = st.selectbox("Метод отбора признаков", options=feature_methods, index=2)
    n_features = st.slider("Число признаков после отбора", 2, 10, 5)

    st.markdown("---")
    st.subheader("📊 Метрики оценки качества")
    metric_options = {
        "rand_index": rand_index,
        "jaccard_index": jaccard_index,
        "fowlkes_mallows_index": fowlkes_mallows_index,
        "phi_index": phi_index,
        "compactness": lambda X_data, labels: compactness(X_data, labels),
        "separation": lambda X_data, labels: separation(X_data, labels),
    }

    available_metrics = list(metric_options.keys())
    selected_metric_names = st.multiselect(
        "Выберите метрики",
        options=available_metrics,
        default=["rand_index", "compactness", "separation"]
    )
    metrics = {name: metric_options[name] for name in selected_metric_names}

    run_button = st.button("🚀 Запустить полный сценарий")


if run_button:
    # Шаг 1: Загрузка данных
    X_full, headers = load_data(data_path)
    if normalize:
        X_full = normalize_data(X_full)

    if has_true_labels and X_full.shape[1] > 1:
        y_true = X_full[:, -1].flatten()
        X_full = X_full[:, :-1]
    else:
        y_true = None

    st.success(f"📥 Данные загружены: {X_full.shape[0]} образцов, {X_full.shape[1]} признаков")

    # --- Шаг 2–3: Кластеризация на всех признаках ---
    config_base = {
        "data_path": data_path,
        "n_clusters": n_clusters,
        "normalize": normalize,
        "has_true_labels": has_true_labels,
        "output_dir": "../reports/base_run",
        "metrics": list(metrics.keys()),
        "feature_selection": {"method": "compactness", "n_features": X_full.shape[1]},
        "clustering": {
            "cure": {
                "method": "cure",
                "params": {"n_clusters": n_clusters, "num_reps": 5, "compression_rate": 0.2, "metric": "euclidean", "p": 2},
            },
            "single_linkage": {
                "method": "single_linkage",
                "params": {"n_clusters": n_clusters, "metric": "manhattan", "p": 1}
            },
            "maxmin_distance": {
                "method": "maxmin_distance",
                "params": {"k": n_clusters, "threshold": 0.5, "metric": "chebyshev", "p": 1}
            }
        }
    }

    st.subheader("🧪 Шаг 2–3: Кластеризация на всех признаках")
    pipeline_base = ClusteringPipeline(config_base)
    pipeline_base.X = X_full
    pipeline_base.y_true = y_true
    pipeline_base.run_comparison()

    results_base = pipeline_base.results
    df_base = pd.DataFrame(results_base).drop(columns=["labels"], errors='ignore')
    st.dataframe(df_base)

    # --- Шаг 4–6: Отбор признаков и новая кластеризация ---
    selector = get_feature_selector(selected_feature_method)

    if selected_feature_method in ["add_method", "del_method", "spa_method"]:
        selected_indices, _ = selector(
            X=X_full,
            y_true=y_true,
            n_features=n_features,
            metric_name="rand_index",
            clusterer=KMeans(n_clusters=n_clusters)
        )
    else:
        selected_indices, _ = selector(X=X_full, n_features=n_features)

    X_selected = X_full[:, selected_indices]

    st.write(f"✅ Выбранные индексы признаков: {selected_indices}")

    config_selected = {
        "data_path": data_path,
        "n_clusters": n_clusters,
        "normalize": normalize,
        "has_true_labels": has_true_labels,
        "output_dir": "./reports/after_selection",
        "metrics": list(metrics.keys()),
        "feature_selection": {"method": selected_feature_method, "n_features": n_features},
        "clustering": config_base["clustering"].copy()
    }

    st.subheader("🧪 Шаг 4–6: Кластеризация после отбора признаков")
    pipeline_selected = ClusteringPipeline(config_selected)
    pipeline_selected.X = X_selected
    pipeline_selected.y_true = y_true
    pipeline_selected.run_comparison()

    results_selected = pipeline_selected.results
    df_selected = pd.DataFrame(results_selected).drop(columns=["labels"], errors='ignore')
    st.dataframe(df_selected)

    # --- Шаг 8–10: Обезличивание и кластеризация ---
    anon_metrics = [m for m in selected_metric_names if m in ["compactness", "separation"]]

    config_anon = {
        "data_path": data_path,
        "n_clusters": n_clusters,
        "normalize": normalize,
        "has_true_labels": False,
        "output_dir": "../reports/anonymized",
        "metrics": anon_metrics,
        "feature_selection": {"method": selected_feature_method, "n_features": n_features},
        "clustering": config_base["clustering"].copy()
    }

    st.subheader("🧽 Шаг 8–10: Кластеризация обезличенных данных")
    pipeline_anon = ClusteringPipeline(config_anon)
    pipeline_anon.X = X_selected
    pipeline_anon.y_true = None
    pipeline_anon.run_comparison()

    results_anon = pipeline_anon.results
    df_anon = pd.DataFrame(results_anon).drop(columns=["labels"], errors='ignore')
    st.dataframe(df_anon)

    # --- Шаг 7 и 11: Сравнение результатов ---
    st.subheader("📊 Сравнение результатов")
    comparison_table = []

    all_models = {r["model"] for r in results_base + results_selected + results_anon}
    for model_name in all_models:
        row = {"Модель": model_name}
        for r in results_base:
            if r["model"] == model_name:
                for metric in selected_metric_names:
                    row[f"Базовая ({len(X_full[0])}) признаков"] = r.get(metric, "N/A")

        for r in results_selected:
            if r["model"] == model_name:
                for metric in selected_metric_names:
                    row[f"С отбором ({n_features}) признаков"] = r.get(metric, "N/A")

        for r in results_anon:
            if r["model"] == model_name:
                for metric in anon_metrics:
                    row[f"Обезличенные ({n_features}) признаков"] = r.get(metric, "N/A")

        comparison_table.append(row)

    comp_df = pd.DataFrame(comparison_table)
    st.dataframe(comp_df)

    # --- Визуализация ---
    st.subheader("📈 Визуализация кластеров (PCA)")
    X_reduced = reduce_dimensions(X_selected)

    for result in results_selected:
        model_name = result["model"]
        labels_pred = result["labels"]

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_pred, cmap='Set1')
            ax.set_title(f"{model_name} — Предсказанные кластеры")
            st.pyplot(fig)

        with col2:
            if y_true is not None:
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true, cmap='Set1')
                ax.set_title("Истинные метки")
                st.pyplot(fig)