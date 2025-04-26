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

def normalize_columns(df):
    return (df - df.mean()) / df.std()

def zet_impute(df, p=5, q=5, alpha_r=1.0, alpha_c=1.0):
    df = df.copy()
    df_norm = normalize_columns(df.select_dtypes(include=[np.number]))

    for (row_idx, col_idx), _ in df_norm.isna().stack().items():
        # Пропущенное значение
        y, x = row_idx, col_idx

        # 1. Компетентные строки
        row_similarities = []
        for i in df_norm.index:
            if i == y or pd.isna(df_norm.loc[i, x]):
                continue
            common = df_norm.loc[[y, i]].dropna(axis=1)
            if common.shape[1] == 0:
                continue
            r = np.linalg.norm(common.loc[y] - common.loc[i])
            t = common.shape[1]
            if r == 0:  # избежание деления на 0
                L = float('inf')
            else:
                L = t / r
            row_similarities.append((i, L))

        row_similarities.sort(key=lambda x: -x[1])
        competent_rows = row_similarities[:p]

        # 2. Компетентные столбцы
        col_similarities = []
        for j in df_norm.columns:
            if j == x or pd.isna(df_norm.loc[y, j]):
                continue
            common = df_norm[[x, j]].dropna()
            if common.shape[0] == 0 or j == x:
                continue
            t = common.shape[0]
            k = common.corr().iloc[0, 1]
            L = t * k
            col_similarities.append((j, L))

        col_similarities.sort(key=lambda x: -x[1])
        competent_cols = col_similarities[:q]

        # 3. Прогноз по строкам
        row_numerators, row_denominators = 0.0, 0.0
        for i, L in competent_rows:
            known_cols = df_norm.loc[[y, i]].dropna(axis=1).columns
            if x not in known_cols:
                continue
            xi = df_norm.loc[i, x]
            row_numerators += (L ** alpha_r) * xi
            row_denominators += (L ** alpha_r)
        b_row = row_numerators / row_denominators if row_denominators != 0 else np.nan

        # 4. Прогноз по столбцам
        col_numerators, col_denominators = 0.0, 0.0
        for j, L in competent_cols:
            known_rows = df_norm[[x, j]].dropna().index
            if y not in known_rows:
                continue
            xj = df_norm.loc[y, j]
            col_numerators += (L ** alpha_c) * xj
            col_denominators += (L ** alpha_c)
        b_col = col_numerators / col_denominators if col_denominators != 0 else np.nan

        # 5. Итоговое значение
        b_final = np.nanmean([b_row, b_col])
        df.loc[y, x] = b_final * df[x].std() + df[x].mean()  # обратная денормализация

    return df

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

    elif method == "zet":
        return zet_impute(df)

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

def evaluate_imputation_methods(df: pd.DataFrame, methods: list[str], n_values: int = 100) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
    results = {method: 0 for method in methods}
    total = 0

    # Выбираем n случайных значений, которые будем "маскировать"
    for _ in range(n_values):
        row = np.random.randint(0, len(numeric_df))
        col = np.random.choice(numeric_df.columns)

        true_value = numeric_df.at[row, col]

        df_masked = numeric_df.copy()
        df_masked.at[row, col] = np.nan

        for method in methods:
            try:
                if method in ["linear_regression", "stochastic_regression"]:
                    feature_cols = [c for c in numeric_df.columns if c != col]
                    filled = fill_missing(df_masked, method, target_col=col, feature_cols=feature_cols)
                else:
                    filled = fill_missing(df_masked, method)

                predicted_value = filled.at[row, col]
                if pd.notna(predicted_value):
                    rel_error = abs(true_value - predicted_value) / abs(true_value)
                    results[method] += rel_error
                    total += 1
            except Exception as e:
                print(f"Ошибка при методе {method}: {e}")

    # Средняя относительная ошибка для каждого метода
    return pd.DataFrame({
        'Method': list(results.keys()),
        'Mean Relative Error': [results[m] / total if total else np.nan for m in results]
    }).sort_values(by='Mean Relative Error')


# %%

methods = [
    "mean", "median", "mode", "ffill",
    "linear_regression", "stochastic_regression", "spline", "zet"
]

result_df = evaluate_imputation_methods(datasets[0], methods, n_values=100)
print(result_df)

# %%
best_methods = result_df.loc[result_df.groupby("Missing Level")["Avg Mean Shift"].idxmin()]
print(best_methods)
result_df.to_json("result_metrics.json", orient="records", lines=True)