#  %%

import pandas as pd
from data_loading import read_csv_by_pd
from data_preprocessing import preprocess_data
from evaluation import evaluate_imputation_methods

# def main():
    # Загрузка данных
# %%
paths = ["sm_dataset", "m_dataset", "lg_dataset"]
datasets = read_csv_by_pd(paths, info=True, head=True)

# Предварительная обработка
datasets = preprocess_data(datasets)

# for i in datasets:
#     i.head()

# Визуализация
# %%
# visualize_columns([datasets[0]])

# %%
# Оценка методов заполнения пропусков
methods = [
    "mean", "hot_deck", "linear_regression"
]

for dataset in datasets:
    result_df = evaluate_imputation_methods(dataset, methods)
    print(result_df)

# if __name__ == "__main__":
#     main()
# %%
