#  %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
# %%
paths = ["sm_dataset", "m_dataset", "lg_dataset"]

def read_csv_by_pd(paths: list[str], info: bool, head: bool):
    
    csv_by_pd = []
    
    for path in paths:
        dataset = pd.read_csv("data/" + path + ".csv")
        dataset = dataset.drop(columns=["Unnamed: 0"])
        dataset = dataset.dropna()
        dataset = dataset.drop_duplicates()
        dataset = dataset.reset_index(drop=True)
        if info:
            print(dataset.info())
        if head:
           print(dataset.head()) 
        csv_by_pd.append(dataset)
    
    return csv_by_pd

datasets = read_csv_by_pd(paths, True, True)
# %%
for i, df in enumerate(datasets):
    df['pasport_number'] = df['pasport_number'].astype(str).str.replace(' ', '', regex=False).str.strip()
    
    df['wagon_seat'] = df['wagon_seat'].astype(str).str.replace('1-', '', regex=False).str.strip()
    
    df['price'] = df['price'].astype(str).str.replace('руб', '', regex=False).str.strip()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # переводим в число, ошибки -> NaN
# %%
for col in datasets[0].columns:
    
    plt.figure(figsize=(8, 4))
    title = f"{col}"
    
    if pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}T').any():
        try:
            df[col] = pd.to_datetime(df[col])
            sns.lineplot(x=df[col], y=range(len(df)))
            plt.xlabel(col)
            plt.ylabel('Index')
            plt.title(title)
        except Exception as e:
            print(f"Ошибка преобразования даты в '{col}': {e}")
            continue

    elif pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=True)

        # Расчёт среднего и медианы
        mean_val = df[col].mean()
        median_val = df[col].median()

        # Отображение линий
        plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.2f}")
        plt.axvline(median_val, color='green', linestyle='-', label=f"Median = {median_val:.2f}")

        plt.xlabel(col)
        plt.title(title)
        plt.legend()

    else:
        if col == "full_name" or col == "pasport_number":
            continue
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'count']
        sns.barplot(data=counts, x=col, y='count')
        plt.title(title)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline

def fill_missing(df, method: str, **kwargs):
    df = df.copy()

    if method == "drop_rows":
        return df.dropna()

    elif method == "pairwise_deletion":
        return df.corr(method='pearson', min_periods=1)

    elif method == "hot_deck":
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        return df

    elif method == "group_mean":
        group_col = kwargs.get("group_col")
        if not group_col or group_col not in df.columns:
            raise ValueError("Не указана колонка для группировки (group_col)")
        return df.apply(lambda col: col.fillna(df.groupby(group_col)[col.name].transform("mean")) if col.name != group_col else col)

    elif method == "mean":
        return df.fillna(df.mean(numeric_only=True))

    elif method == "median":
        return df.fillna(df.median(numeric_only=True))

    elif method == "mode":
        return df.fillna(df.mode().iloc[0])

    elif method == "ffill":
        return df.fillna(method="ffill")

    elif method == "linear_regression":
        target_col = kwargs.get("target_col")
        feature_cols = kwargs.get("feature_cols")
        if not target_col or not feature_cols:
            raise ValueError("Укажите target_col и feature_cols")
        
        known = df[df[target_col].notna()]
        unknown = df[df[target_col].isna()]
        model = LinearRegression()
        model.fit(known[feature_cols], known[target_col])
        predicted = model.predict(unknown[feature_cols])
        df.loc[unknown.index, target_col] = predicted
        return df

    elif method == "stochastic_regression":
        target_col = kwargs.get("target_col")
        feature_cols = kwargs.get("feature_cols")
        if not target_col or not feature_cols:
            raise ValueError("Укажите target_col и feature_cols")

        known = df[df[target_col].notna()]
        unknown = df[df[target_col].isna()]
        model = LinearRegression()
        model.fit(known[feature_cols], known[target_col])
        predicted = model.predict(unknown[feature_cols])
        noise = np.random.normal(0, np.std(known[target_col] - model.predict(known[feature_cols])), size=len(predicted))
        df.loc[unknown.index, target_col] = predicted + noise
        return df

    elif method == "spline":
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                x = df.index[df[col].notna()]
                y = df.loc[x, col]
                spline = UnivariateSpline(x, y, k=2, s=0)
                df[col] = df[col].combine_first(pd.Series(spline(df.index), index=df.index))
        return df

    else:
        raise ValueError(f"Метод '{method}' не поддерживается.")

# %%

import pandas as pd
import numpy as np

def generate_datasets_with_removed_outliers(df: pd.DataFrame, percentages: list[int]) -> dict:
    datasets = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for perc in percentages:
        df_copy = df.copy()

        rows_to_remove = set()

        for col in numeric_cols:
            q1 = df_copy[col].quantile(0.25)
            q3 = df_copy[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_indices = df_copy[(df_copy[col] < lower) | (df_copy[col] > upper)].index.tolist()

            # Сколько удалить
            n_remove = int(len(df_copy) * (perc / 100))
            selected = outlier_indices[:n_remove]

            rows_to_remove.update(selected)

        # Удаление строк по индексам
        df_result = df_copy.drop(index=rows_to_remove).reset_index(drop=True)
        datasets[f"{perc}%"] = df_result

    return datasets


# %%

import pandas as pd
import numpy as np
from typing import List, Dict

def evaluate_imputation_methods(df: pd.DataFrame, 
                              methods: List[str], 
                              missing_percentages: List[int] = [3, 5, 10, 20, 30],
                              n_runs: int = 5) -> pd.DataFrame:
    """
    Оценивает методы заполнения пропусков на разных уровнях пропущенных значений
    
    Параметры:
        df - исходный DataFrame
        methods - список методов заполнения
        missing_percentages - проценты пропусков для тестирования
        n_runs - количество запусков для каждого процента пропусков
        
    Возвращает:
        DataFrame с результатами оценки методов
    """
    print("Оценка методов заполнения пропусков...")
    
    # Шаг 1: Создаем датасет только с полными наблюдениями
    complete_df = df.select_dtypes(include=[np.number]).dropna().copy()
    
    results = []
    
    for pct in missing_percentages:
        print(f"\nТестирование с {pct}% пропусков:")
        
        for run in range(n_runs):
            print(f"Попытка {run + 1}/{n_runs}")
            
            # Шаг 2: Создаем копию и вносим случайные пропуски
            df_masked = complete_df.copy()
            n_missing = int(len(df_masked) * pct / 100)
            
            # Выбираем случайные ячейки для маскировки
            rows = np.random.choice(df_masked.index, size=n_missing, replace=False)
            cols = np.random.choice(df_masked.columns, size=n_missing)
            
            # Сохраняем истинные значения
            true_values = [df_masked.at[row, col] for row, col in zip(rows, cols)]
            
            # Маскируем значения
            for row, col in zip(rows, cols):
                df_masked.at[row, col] = np.nan
            
            # Шаг 3: Тестируем каждый метод
            for method in methods:
                try:
                    # Заполняем пропуски
                    if method in ["linear_regression", "stochastic_regression"]:
                        # Для регрессионных методов нужно указать целевую колонку
                        filled = df_masked.copy()
                        for col in df_masked.columns:
                            if df_masked[col].isna().any():
                                feature_cols = [c for c in df_masked.columns if c != col]
                                temp_filled = fill_missing(filled, method, 
                                                         target_col=col, 
                                                         feature_cols=feature_cols)
                                filled[col] = temp_filled[col]
                    else:
                        filled = fill_missing(df_masked.copy(), method)
                    
                    # Шаг 4: Рассчитываем ошибки
                    errors = []
                    for row, col, true_val in zip(rows, cols, true_values):
                        pred_val = filled.at[row, col]
                        if pd.notna(pred_val):
                            rel_error = abs(true_val - pred_val) / abs(true_val) * 100
                            errors.append(rel_error)
                    
                    if errors:
                        mean_error = np.mean(errors)
                        results.append({
                            'Method': method,
                            'Missing%': pct,
                            'Run': run + 1,
                            'MeanRelativeError%': mean_error,
                            'NumEvaluated': len(errors)
                        })
                        
                except Exception as e:
                    print(f"Ошибка при методе {method} с {pct}% пропусков: {e}")
    
    # Шаг 5: Агрегируем результаты
    result_df = pd.DataFrame(results)
    
    # Группируем по методу и проценту пропусков, вычисляя среднюю ошибку
    final_results = result_df.groupby(['Method', 'Missing%'])['MeanRelativeError%'].mean().reset_index()
    final_results = final_results.sort_values(by=['Missing%', 'MeanRelativeError%'])
    
    return final_results


# Пример использования
methods = [
    "mean", "median", "mode", "ffill",
    "linear_regression", "stochastic_regression", "spline"
]

for dataset in datasets:
    # Генерация датасетов с удаленными выбросами
    result_df = evaluate_imputation_methods(dataset, methods)
    print(result_df)

# %%
