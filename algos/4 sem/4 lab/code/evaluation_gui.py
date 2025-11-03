import pandas as pd
import numpy as np
from typing import List, Tuple
from imputation_methods import fill_missing
from visualization import plot_all_methods_comparison, plot_best_methods
import time


def run_evaluation(
    df: pd.DataFrame,
    methods: List[str],
    missing_percentages: List[int] = [3, 5, 10, 20, 30],
    n_runs: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete_df = df.select_dtypes(include=[np.number]).dropna().copy()
    complete_df = complete_df.sample(frac=1, random_state=42).reset_index(drop=True)
    true_distributions = complete_df.describe().T

    results = []

    for pct in missing_percentages:
        for run in range(n_runs):
            df_masked = complete_df.copy()
            n_missing = int(len(df_masked) * pct / 100)
            rows = np.random.choice(df_masked.index, size=n_missing, replace=False)
            cols = np.random.choice(df_masked.columns, size=n_missing)
            true_values = [df_masked.at[row, col] for row, col in zip(rows, cols)]

            for row, col in zip(rows, cols):
                df_masked.at[row, col] = np.nan

            for method in methods:
                try:
                    start = time.time()
                    if method in ["linear_regression", "stochastic_regression"]:
                        filled = df_masked.copy()
                        for col in df_masked.columns:
                            if df_masked[col].isna().any():
                                feature_cols = [c for c in df_masked.columns if c != col]
                                temp_filled = fill_missing(filled, method, target_col=col, feature_cols=feature_cols)
                                filled[col] = temp_filled[col]
                    else:
                        filled = fill_missing(df_masked.copy(), method)

                    errors = []
                    distribution_errors = []

                    for row, col, true_val in zip(rows, cols, true_values):
                        pred_val = filled.at[row, col]
                        if pd.notna(pred_val):
                            rel_error = abs(true_val - pred_val) / abs(true_val) * 100 if true_val != 0 else 0
                            errors.append(rel_error)

                    metrics = true_distributions.columns
                    for col in complete_df.columns:
                        if col in filled.columns:
                            for metric in metrics:
                                true_val = true_distributions.loc[col, metric]
                                if callable(getattr(filled[col], metric, None)):
                                    filled_val = getattr(filled[col], metric)()
                                elif '%' in metric:
                                    filled_val = np.percentile(filled[col], float(metric.strip('%')))
                                else:
                                    continue

                                if not pd.isna(filled_val) and true_val != 0:
                                    err = abs(true_val - filled_val) / abs(true_val) * 100
                                    distribution_errors.append({'Column': col, f'{metric}Error%': err})

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
                                "TimeSeconds": round(time.time() - start, 3),
                            }
                            row.update({k: v for k, v in dist_error.items() if k != 'Column'})
                            results.append(row)

                except Exception as e:
                    print(f"Ошибка в методе {method}: {e}")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return pd.DataFrame(), None, None

    metric_cols = [col for col in result_df.columns if col not in ['Method', 'Missing%', 'Run', 'Column']]
    final_results = result_df.groupby(['Method', 'Missing%', 'Column'])[metric_cols].mean().reset_index()
    best_methods = final_results.loc[final_results.groupby('Missing%')['MeanRelativeError%'].idxmin()]

    return best_methods, final_results, result_df
def generate_plots(
    final_results: pd.DataFrame,
    best_methods: pd.DataFrame,
    parent_widget=None,
    return_fig: bool = False
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if final_results is None or final_results.empty:
        return

    if parent_widget is not None:
        for widget in parent_widget.winfo_children():
            widget.destroy()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=final_results,
        x="Missing%",
        y="MeanRelativeError%",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax
    )

    ax.set_title("Средняя относительная ошибка восстановления пропущенных значений")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent_widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    if return_fig:
        return [fig]
