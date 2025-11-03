import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def fill_missing(df, method: str, **kwargs):
    df = df.copy()

    if method == "drop_rows":
        return df.dropna()

    elif method == "pairwise_deletion":
        return df.corr(method='pearson', min_periods=1)

    elif method == "hot_deck":
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
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
        return df.ffill()

    elif method in ["linear_regression", "stochastic_regression"]:
        target_col = kwargs.get("target_col")
        feature_cols = kwargs.get("feature_cols")
        if not target_col or not feature_cols:
            raise ValueError("Укажите target_col и feature_cols")
        
        # Сначала заполним пропуски в признаках простым методом (median)
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
            residuals = known[target_col] - model.predict(known[feature_cols])
            noise = np.random.normal(0, np.std(residuals), size=len(predicted))
            predicted += noise
        
        df_filled.loc[unknown.index, target_col] = predicted
        return df_filled

    elif method == "spline":
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                known = df[col].notna()
                if known.sum() > 1:  # Нужно хотя бы 2 точки для сплайна
                    x = np.where(known)[0]
                    y = df.loc[known, col]
                    spline = UnivariateSpline(x, y, k=min(2, len(x)-1), s=0)
                    df[col] = df[col].combine_first(pd.Series(spline(np.arange(len(df))), index=df.index))
        return df

    elif method == "iterative":
        imputer = IterativeImputer(random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df_imputed

    else:
        raise ValueError(f"Метод '{method}' не поддерживается.")