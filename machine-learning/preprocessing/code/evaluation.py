"""
Evaluate imputation strategies by injecting artificial missingness into a
complete dataset, imputing it, and measuring the error against the (known)
true values -- the standard way to score an imputation method when you don't
have a labeled "correct" answer for real-world missing data.

Ported from `algos/4 sem/4 lab/code/evaluation.py`. Logic unchanged; the
dependency on that project's `visualization.py` (not part of this KB's scope)
was replaced with a small self-contained matplotlib helper so this file has
no ports outside imputation_methods.py. Docstrings/type hints and English
comments were added.

See ../missing-values-imputation.md for the write-up.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imputation_methods import fill_missing

os.makedirs("data", exist_ok=True)


def evaluate_imputation_methods(
    df: pd.DataFrame,
    methods: List[str],
    missing_percentages: List[int] = [3, 5, 10, 20, 30],
    n_runs: int = 5,
) -> pd.DataFrame:
    """
    For each missingness level in `missing_percentages`, repeat `n_runs` times:
    1. Start from a fully-observed subset of `df` (numeric columns only).
    2. Randomly null out `pct`% of (row, col) cells and remember their true values.
    3. Impute with each method in `methods`.
    4. Score: mean relative error on the recovered cells, plus how far each
       column's summary statistics (mean/std/min/25%/50%/75%/max, whatever
       `describe()` reports) drift from the true, un-masked distribution.

    Returns a summary table of the best-scoring method at each missingness
    level (also written to data/full_results.csv and data/best_methods_summary.csv).
    """
    print("Evaluating imputation methods...")

    complete_df = df.select_dtypes(include=[np.number]).dropna().copy()
    complete_df = complete_df.sample(frac=1, random_state=42).reset_index(drop=True)
    true_distributions = complete_df.describe().T

    results = []

    for pct in missing_percentages:
        print(f"Testing at {pct}% missing:")

        for run in range(n_runs):
            print(f"  run {run + 1}/{n_runs}")

            df_masked = complete_df.copy()
            n_missing = int(len(df_masked) * pct / 100)
            rows = np.random.choice(df_masked.index, size=n_missing, replace=False)
            cols = np.random.choice(df_masked.columns, size=n_missing)
            true_values = [df_masked.at[row, col] for row, col in zip(rows, cols)]

            for row, col in zip(rows, cols):
                df_masked.at[row, col] = np.nan

            for method in methods:
                try:
                    if method in ("linear_regression", "stochastic_regression"):
                        # Regression imputation needs an explicit target/feature
                        # split, so impute one column at a time using the rest
                        # of the columns as predictors.
                        filled = df_masked.copy()
                        for col in df_masked.columns:
                            if df_masked[col].isna().any():
                                feature_cols = [c for c in df_masked.columns if c != col]
                                temp_filled = fill_missing(
                                    filled, method, target_col=col, feature_cols=feature_cols
                                )
                                filled[col] = temp_filled[col]
                    else:
                        filled = fill_missing(df_masked.copy(), method)

                    errors = []
                    for row, col, true_val in zip(rows, cols, true_values):
                        pred_val = filled.at[row, col]
                        if pd.notna(pred_val):
                            rel_error = abs(true_val - pred_val) / abs(true_val) * 100 if true_val != 0 else 0
                            errors.append(rel_error)

                    distribution_errors = []
                    metrics = true_distributions.columns
                    for col in complete_df.columns:
                        if col not in filled.columns:
                            continue
                        for metric in metrics:
                            true_val = true_distributions.loc[col, metric]
                            stat = getattr(filled[col], metric, None)
                            if callable(stat):
                                filled_val = stat()
                            elif "%" in metric:
                                filled_val = np.percentile(filled[col], float(metric.strip("%")))
                            else:
                                filled_val = np.nan
                            if not pd.isna(filled_val) and true_val != 0:
                                err = abs(true_val - filled_val) / abs(true_val) * 100
                                distribution_errors.append({"Column": col, f"{metric}Error%": err})

                    if errors:
                        mean_error = np.mean(errors)
                        for dist_error in distribution_errors:
                            row_result = {
                                "Method": method,
                                "Missing%": pct,
                                "Run": run + 1,
                                "MeanRelativeError%": mean_error,
                                "NumEvaluated": len(errors),
                                "Column": dist_error["Column"],
                            }
                            row_result.update({k: v for k, v in dist_error.items() if k != "Column"})
                            results.append(row_result)

                except Exception as e:
                    print(f"  {method} failed at {pct}% missing: {e}")

    result_df = pd.DataFrame(results)
    metric_cols = [c for c in result_df.columns if c not in ("Method", "Missing%", "Run", "Column")]
    final_results = (
        result_df.groupby(["Method", "Missing%", "Column"])[metric_cols].mean().reset_index()
    )
    final_results = final_results.sort_values(by=["Missing%", "MeanRelativeError%"])

    best_methods = final_results.loc[final_results.groupby("Missing%")["MeanRelativeError%"].idxmin()]
    metric_cols = [c for c in best_methods.columns if c not in ("Missing%", "Method", "Column")]
    best_methods_summary = best_methods[["Missing%", "Method"] + metric_cols]

    print("Best method per missingness level:")
    print(best_methods_summary)

    result_df.to_csv("data/full_results.csv", index=False)
    best_methods_summary.to_csv("data/best_methods_summary.csv", index=False)

    plot_all_methods_comparison(final_results)
    plot_best_methods(best_methods)

    return best_methods_summary


def plot_all_methods_comparison(final_results: pd.DataFrame) -> None:
    """Mean relative error vs. missingness %, one line per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in final_results.groupby("Method"):
        by_pct = group.groupby("Missing%")["MeanRelativeError%"].mean()
        ax.plot(by_pct.index, by_pct.values, marker="o", label=method)
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Mean relative error %")
    ax.set_title("Imputation methods compared")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_best_methods(best_methods: pd.DataFrame) -> None:
    """Bar chart of the winning method's error at each missingness level."""
    fig, ax = plt.subplots(figsize=(8, 5))
    by_pct = best_methods.groupby("Missing%")["MeanRelativeError%"].mean()
    ax.bar(by_pct.index.astype(str), by_pct.values)
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Mean relative error % (best method)")
    ax.set_title("Best imputation method's error by missingness level")
    fig.tight_layout()
    plt.show()
