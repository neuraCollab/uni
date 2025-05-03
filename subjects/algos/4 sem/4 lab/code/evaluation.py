import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from imputation_methods import fill_missing

def evaluate_imputation_methods(df: pd.DataFrame, 
                                methods: List[str], 
                                missing_percentages: List[int] = [3, 5, 10, 20, 30], 
                                n_runs: int = 5) -> pd.DataFrame:
    """Оценивает методы заполнения пропусков на разных уровнях пропущенных значений и сравнивает с эталонными результатами"""
    print("Оценка методов заполнения пропусков...")
    
    # 1. Формируем датасет, состоящий только из полных наблюдений
    complete_df = df.select_dtypes(include=[np.number]).dropna().copy()
    complete_df = complete_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Оцениваем параметры распределений переменных на полном датасете
    true_distributions = complete_df.describe().T
    
    results = []
    
    for pct in missing_percentages:
        print(f"\nТестирование с {pct}% пропусков:")
        
        for run in range(n_runs):
            print(f"Попытка {run + 1}/{n_runs}")
            
            # 3. Вносим случайные пропуски в датасет
            df_masked = complete_df.copy()
            n_missing = int(len(df_masked) * pct / 100)
            rows = np.random.choice(df_masked.index, size=n_missing, replace=False)
            cols = np.random.choice(df_masked.columns, size=n_missing)
            true_values = [df_masked.at[row, col] for row, col in zip(rows, cols)]
            
            # Внесение пропусков
            for row, col in zip(rows, cols):
                df_masked.at[row, col] = np.nan
            
            # 4. Оценка смещений
            for method in methods:
                try:
                    if method in ["linear_regression", "stochastic_regression"]:
                        # Заполнение пропусков с использованием регрессионных методов
                        filled = df_masked.copy()
                        for col in df_masked.columns:
                            if df_masked[col].isna().any():
                                feature_cols = [c for c in df_masked.columns if c != col]
                                temp_filled = fill_missing(filled, method, 
                                                           target_col=col, 
                                                           feature_cols=feature_cols)
                                filled[col] = temp_filled[col]
                    else:
                        # Для других методов
                        filled = fill_missing(df_masked.copy(), method)
                    # 5. Заполнение пропусков и оценка ошибок
                    errors = []
                    distribution_errors = []


                    for row, col, true_val in zip(rows, cols, true_values):
                        pred_val = filled.at[row, col]
                        if pd.notna(pred_val):
                            if true_val != 0:
                                rel_error = abs(true_val - pred_val) / abs(true_val) * 100
                            else:
                                rel_error = 0  # или np.nan, если хочешь пропустить такие случаи
                            errors.append(rel_error)
                   
                    # Сравнение распределений: отклонение от истинного среднего и стандартного отклонения
                    metrics = true_distributions.columns
                    for col in complete_df.columns:
                        if col in filled.columns:
                            for metric in metrics:
                                true_val = true_distributions.loc[col, metric]
                                filled_val = getattr(filled[col], metric)() if callable(getattr(filled[col], metric, None)) else np.percentile(filled[col], float(metric.strip('%'))) if '%' in metric else np.nan
                                if not pd.isna(filled_val) and true_val != 0:
                                    err = abs(true_val - filled_val) / abs(true_val) * 100
                                    distribution_errors.append({'Column': col, f'{metric}Error%': err})

                    # 6. Сравнение с эталонными результатами
                    if errors:
                        mean_error = np.mean(errors)
                        for dist_error in distribution_errors:
                            row = {
                                'Method': method,   
                                'Missing%': pct,
                                'Run': run + 1,
                                'MeanRelativeError%': mean_error,
                                'NumEvaluated': len(errors),
                                'Column': dist_error['Column'],
                            }
                            row.update({k: v for k, v in dist_error.items() if k != 'Column'})
                            results.append(row)

                    
                except Exception as e:
                    print(f"Ошибка при методе {method} с {pct}% пропусков: {e}")
    
    # 7. Подведение итогов
    result_df = pd.DataFrame(results)
    metric_cols = [col for col in result_df.columns if col not in ['Method', 'Missing%', 'Run', 'Column']]
    final_results = result_df.groupby(['Method', 'Missing%', 'Column'])[metric_cols].mean().reset_index()
    final_results = final_results.sort_values(by=['Missing%', 'MeanRelativeError%'])
    
    # 8. Определение лучшего метода для каждого уровня пропусков
    best_methods = final_results.loc[final_results.groupby('Missing%')['MeanRelativeError%'].idxmin()]
    
    # 9. Резюмирующая таблица с лучшими методами
    metric_cols = [col for col in best_methods.columns if col not in ['Missing%', 'Method', 'Column']]
    best_methods_summary = best_methods[['Missing%', 'Method'] + metric_cols]

    print("\nРезюмирующая таблица с лучшими методами:")
    print(best_methods_summary)
    
    # 10. Построение графика
    plt.figure(figsize=(10, 6))
    for method in best_methods['Method'].unique():
        method_data = best_methods[best_methods['Method'] == method]
        plt.plot(method_data['Missing%'], method_data['MeanRelativeError%'], label=method, marker='o')
    
    plt.title("Лучшие методы заполнения пропусков по уровням пропусков")
    plt.xlabel("Процент пропусков")
    plt.ylabel("Средняя относительная ошибка (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_methods_summary
