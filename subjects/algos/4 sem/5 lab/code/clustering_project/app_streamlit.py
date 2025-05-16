import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import yaml
from sklearn.cluster import KMeans

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

st.set_page_config(page_title="Кластеризация", layout="wide")
st.title("🧠 Система сравнения алгоритмов кластеризации")

# --- Боковая панель: общие настройки ---
with st.sidebar:
    st.header("⚙️ Настройки проекта")
    data_path = st.text_input("Путь к данным", value="./data/iris.csv")
    n_clusters = st.number_input("Число кластеров", min_value=2, max_value=10, value=3)
    normalize = st.checkbox("Нормализовать данные", value=True)
    has_true_labels = st.checkbox("Данные содержат истинные метки", value=True)

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
    available_metrics = (
        ["rand_index", "jaccard_index", "fowlkes_mallows_index", "phi_index", "compactness", "separation"]
        if has_true_labels else ["compactness", "separation"]
    )
    selected_metric_names = st.multiselect("Выберите метрики", options=available_metrics, default=available_metrics[:])

    st.markdown("---")
    st.subheader("🔍 Отбор признаков")
    feature_methods = ["compactness", "spread", "add_method", "del_method", "spa_method"]
    selected_feature_methods = st.multiselect("Методы отбора признаков", options=feature_methods, default=["compactness"])
    n_features = st.slider("Число признаков после отбора", 1, 10, 2)

    st.markdown("---")
    run_button = st.button("🚀 Выполнить полный сценарий")

if run_button:
    # --- Шаг 1: Загрузка и подготовка данных ---
    X_full, headers = load_data(data_path)
    if normalize:
        X_full = normalize_data(X_full)
    if has_true_labels:
        y_true = X_full[:, -1].flatten()
        X_full = X_full[:, :-1]
        if len(np.unique(y_true)) < 2:
            st.warning("⚠️ Все истинные метки одинаковые — внешние метрики будут недоступны")
            y_true = None
    else:
        y_true = None
    st.success(f"📥 Данные загружены: {X_full.shape[0]} образцов, {X_full.shape[1]} признаков")

    # --- Шаг 2: Кластеризация на всех признаках ---
    base_config = load_config("configs/base_config.yaml")
    base_config.update({
        "data_path": data_path,
        "n_clusters": n_clusters,
        "normalize": normalize,
        "has_true_labels": has_true_labels,
        "metrics": selected_metric_names
    })

    st.subheader("🧪 Шаг 1: Настройка и кластеризация без отбора признаков")
    clustering_methods = ["cure", "forel", "isodata", "single_linkage", "maxmin_distance"]

    with st.expander("🔧 Настройка кластеризаторов"):
        tabs = st.tabs(clustering_methods)
        for i, method in enumerate(clustering_methods):
            with tabs[i]:
                st.markdown(f"#### {method.capitalize()}")
                if method == "cure":
                    params = {
                        "n_clusters": st.slider("Число кластеров", 2, 10, n_clusters, key=f"cure_k_base"),
                        "num_reps": st.slider("Число репрезентативных точек", 1, 10, 5, key="cure_reps_base"),
                        "compression_rate": st.slider("Compression rate", 0.0, 1.0, 0.2, key="cure_comp_base"),
                        "metric": st.selectbox("Метрика расстояния", ["euclidean", "manhattan", "chebyshev", "minkowski"], key="cure_metric_base"),
                        "p": st.slider("p для Minkowski", 1, 10, 2, key="cure_p_base") if st.session_state.cure_metric_base == "minkowski" else 2
                    }
                elif method == "forel":
                    params = {
                        "radius": st.slider("Радиус", 0.1, 5.0, 1.2, key="forel_radius_base"),
                        "metric": st.selectbox("Метрика расстояния", ["euclidean", "manhattan", "chebyshev", "minkowski"], key="forel_metric_base"),
                        "p": st.slider("p для Minkowski", 1, 10, 2, key="forel_p_base") if st.session_state.forel_metric_base == "minkowski" else 2
                    }
                elif method == "isodata":
                    params = {
                        "k_initial": st.slider("Начальное число кластеров", 2, 10, 4, key="isodata_k_base"),
                        "max_clusters": st.slider("Максимум кластеров", 2, 10, 6, key="isodata_max_base"),
                        "min_points_per_cluster": st.slider("Мин. точек в кластере", 1, 10, 5, key="isodata_min_base"),
                        "sigma_threshold": st.slider("Sigma Threshold", 0.1, 5.0, 1.0, key="isodata_sigma_base"),
                        "merge_threshold": st.slider("Merge Threshold", 0.1, 5.0, 1.5, key="isodata_merge_base"),
                        "max_iterations": st.slider("Макс. итераций", 1, 50, 10, key="isodata_iters_base"),
                        "metric": st.selectbox("Метрика расстояния", ["euclidean", "manhattan", "chebyshev", "minkowski"], key="isodata_metric_base"),
                        "p": st.slider("p для Minkowski", 1, 10, 2, key="isodata_p_base") if st.session_state.isodata_metric_base == "minkowski" else 2
                    }
                elif method == "single_linkage":
                    params = {
                        "n_clusters": n_clusters,
                        "metric": st.selectbox("Метрика расстояния", ["euclidean", "manhattan", "chebyshev", "minkowski"], key="sl_metric_base"),
                        "p": st.slider("p для Minkowski", 1, 10, 2, key="sl_p_base") if st.session_state.sl_metric_base == "minkowski" else 2
                    }
                elif method == "maxmin_distance":
                    params = {
                        "k": st.slider("Число кластеров", 2, 10, 3, key="mm_k_base"),
                        "threshold": st.slider("Threshold", 0.1, 5.0, 0.5, key="mm_threshold_base"),
                        "metric": st.selectbox("Метрика расстояния", ["euclidean", "manhattan", "chebyshev", "minkowski"], key="mm_metric_base"),
                        "p": st.slider("p для Minkowski", 1, 10, 2, key="mm_p_base") if st.session_state.mm_metric_base == "minkowski" else 2
                    }

                base_config["clustering"][method] = {"method": method, "params": params}

    pipeline_base = ClusteringPipeline(base_config)
    pipeline_base.run()
    df_base = pd.DataFrame(pipeline_base.results).drop(columns=["labels"], errors='ignore')
    st.dataframe(df_base)

    # --- Шаг 3–5: Отбор признаков и кластеризация ---
    results_by_method = {}

    for method in selected_feature_methods:
        selector = get_feature_selector(method)
        try:
            if method in ["add_method", "del_method", "spa_method"]:
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

            selected_config = load_config("configs/after_selection.yaml")
            selected_config.update({
                "data_path": data_path,
                "n_clusters": n_clusters,
                "feature_selection": {"method": method, "n_features": n_features},
                "clustering": base_config["clustering"],
                "metrics": selected_metric_names
            })
            pipeline_selected = ClusteringPipeline(selected_config)
            pipeline_selected.X = X_selected
            pipeline_selected.y_true = y_true
            pipeline_selected.run()
            results_by_method[method] = pipeline_selected.results

        except Exception as e:
            st.error(f"❌ Ошибка при использовании метода '{method}': {e}")

    # --- Сравнение методов отбора признаков ---
    if len(results_by_method) > 0:
        comparison_table = []
        for model_idx in range(len(pipeline_base.results)):
            model_name = pipeline_base.results[model_idx]["model"]
            row = {"Модель": model_name}
            for method in results_by_method:
                result = results_by_method[method][model_idx]
                for metric in selected_metric_names:
                    row[f"{method}_{metric}"] = result.get(metric, "N/A")
            comparison_table.append(row)

        st.subheader("📊 Сравнение методов отбора признаков")
        df_comparison = pd.DataFrame(comparison_table)
        st.dataframe(df_comparison)

        # Визуализация (например, bar plot)
        st.subheader("📈 График сравнения метрик по методам отбора признаков")
        for metric in selected_metric_names:
            metric_data = df_comparison[[col for col in df_comparison.columns if col.endswith(f"_{metric}")]]
            metric_data.index = df_comparison['Модель']
            metric_data.plot(kind='bar', title=f"Сравнение по метрике: {metric}")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())