"""
Ten missing-value imputation strategies behind one dispatch function.

Ported from `algos/4 sem/4 lab/code/imputation_methods.py`. Logic unchanged;
docstrings/type hints and English comments were added.

See ../missing-values-imputation.md for when each strategy applies (MCAR /
MAR / MNAR) and for the injected-missingness evaluation approach in
evaluation.py.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 -- required to unlock IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


def fill_missing(df: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """
    Dispatch to one of ten imputation strategies. `df` is never mutated in
    place (a copy is made up front).

    method:
        "drop_rows"              -- listwise deletion (drop any row with a NaN).
        "pairwise_deletion"      -- NOT an imputation; returns the pairwise-
                                     complete correlation matrix instead (useful
                                     when you only need correlations/covariance,
                                     not a filled dataset).
        "hot_deck"                -- forward-fill then back-fill (carries the
                                     nearest observed value along row order).
        "group_mean"              -- fill with the mean of the same group
                                     (kwargs: group_col).
        "mean" / "median" / "mode" -- fill numeric columns with a single global
                                     statistic.
        "ffill"                   -- forward-fill only.
        "linear_regression"       -- predict the missing column from other
                                     columns via OLS (kwargs: target_col, feature_cols).
        "stochastic_regression"   -- same, plus injected residual noise so the
                                     imputed values don't artificially shrink variance.
        "spline"                  -- univariate spline fit over row order, per
                                     numeric column.
        "iterative"               -- MICE-style `IterativeImputer` (models each
                                     column as a function of the others, iterated
                                     to convergence).
    """
    df = df.copy()

    if method == "drop_rows":
        return df.dropna()

    elif method == "pairwise_deletion":
        return df.corr(method="pearson", min_periods=1)

    elif method == "hot_deck":
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        return df

    elif method == "group_mean":
        group_col = kwargs.get("group_col")
        if not group_col or group_col not in df.columns:
            raise ValueError("group_col is required for method='group_mean'")
        return df.apply(
            lambda col: col.fillna(df.groupby(group_col)[col.name].transform("mean"))
            if col.name != group_col
            else col
        )

    elif method == "mean":
        return df.fillna(df.mean(numeric_only=True))

    elif method == "median":
        return df.fillna(df.median(numeric_only=True))

    elif method == "mode":
        return df.fillna(df.mode().iloc[0])

    elif method == "ffill":
        return df.ffill()

    elif method in ("linear_regression", "stochastic_regression"):
        target_col = kwargs.get("target_col")
        feature_cols = kwargs.get("feature_cols")
        if not target_col or not feature_cols:
            raise ValueError("target_col and feature_cols are required for regression imputation")

        # Simple-impute the predictor columns first so the regression itself
        # has no missing inputs to choke on.
        df_filled = df.copy()
        for col in feature_cols:
            if df_filled[col].isna().any():
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())

        known = df_filled[df_filled[target_col].notna()]
        unknown = df_filled[df_filled[target_col].isna()]
        if len(known) == 0 or len(unknown) == 0:
            return df_filled

        model = LinearRegression()
        model.fit(known[feature_cols], known[target_col])
        predicted = model.predict(unknown[feature_cols])

        if method == "stochastic_regression":
            # Add back residual noise (drawn from the training residuals'
            # spread) so imputed values reproduce the target's original
            # variance instead of all landing exactly on the regression line.
            residuals = known[target_col] - model.predict(known[feature_cols])
            noise = np.random.normal(0, np.std(residuals), size=len(predicted))
            predicted += noise

        df_filled.loc[unknown.index, target_col] = predicted
        return df_filled

    elif method == "spline":
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                known = df[col].notna()
                if known.sum() > 1:  # need at least 2 points to fit a spline
                    x = np.where(known)[0]
                    y = df.loc[known, col]
                    spline = UnivariateSpline(x, y, k=min(2, len(x) - 1), s=0)
                    df[col] = df[col].combine_first(pd.Series(spline(np.arange(len(df))), index=df.index))
        return df

    elif method == "iterative":
        imputer = IterativeImputer(random_state=0)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    else:
        raise ValueError(f"Unsupported method: '{method}'")
