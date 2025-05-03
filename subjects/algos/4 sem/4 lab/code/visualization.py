import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_columns(datasets: list[pd.DataFrame]):
    """Создает визуализации для каждого столбца в датасетах"""
    for df in datasets:
        for col in df.columns:
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
                mean_val = df[col].mean()
                median_val = df[col].median()
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