"""
LightGBM regression (log-target house-price prediction) with RandomizedSearchCV
hyperparameter tuning.

Ported from `AI/1 lab/h2.py` in the source coursework repo, with two real bugs
fixed during migration -- both are good "what not to do" teaching examples:

1. DATA LEAKAGE: the original fit the `ColumnTransformer` preprocessor
   (median-impute + StandardScaler for numeric, constant-impute + OneHotEncoder
   for categorical) on the *full* training set before the train/validation
   split and before cross-validation:

       preprocessor.fit(train.drop(columns=["Id", "SalePrice"]))
       train_set, val_set = split_data(train)
       ...

   That means the scaler's mean/std and the imputer's median were computed
   using rows that later ended up in the validation fold -- information from
   "unseen" data leaked into the transform applied to it. Fixed here by
   wrapping the preprocessor and the model in a single `sklearn.Pipeline`, so
   `cross_val_score` / `RandomizedSearchCV` refit the preprocessor from
   scratch on the training portion of *each* fold and only transform the held
   -out portion. This is the standard fix any time you see
   `preprocessor.fit(...)` called before a CV loop or split.

2. INVALID STRATIFICATION: the original used `StratifiedKFold` to cross-validate
   a regression target (`SalePrice`, continuous after log1p):

       kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   `StratifiedKFold` preserves class-label proportions across folds, which
   only makes sense for a categorical target -- passing it a continuous
   target either errors out or silently treats each unique float as its own
   "class", producing near-meaningless folds. Fixed here by using plain
   `KFold`, which is the correct choice for regression. (If you need
   stratified CV for a skewed regression target, bucket it into quantile bins
   first and stratify on the bin, not on the raw scores.)

See ../gradient-boosting-catboost-lgbm.md for the write-up.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train_hw.csv")
    test = pd.read_csv(data_dir / "test_hw.csv")
    return train, test


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """
    numeric  -> median impute + StandardScaler
    categorical -> constant impute ("None") + OneHotEncoder
    Columns with >70% missing are dropped outright (imputing them would mostly
    be inventing data).
    """
    df = train_df.drop(columns=[c for c in ["Id", "SalePrice"] if c in train_df.columns])

    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.7].index.tolist()
    df = df.drop(columns=high_missing_cols)
    if high_missing_cols:
        print(f"Dropped columns with >70% missing: {high_missing_cols}")

    num_feats = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = df.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ])


def find_best_params(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """RandomizedSearchCV over the LightGBM step of `pipeline`.

    Parameter names are prefixed with the pipeline step name ("model__") --
    required so RandomizedSearchCV can route them past the preprocessor step.
    """
    param_dist = {
        "model__n_estimators": [200, 500, 1500, 2000, 2500, 3000],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [31, 63],
        "model__max_depth": [3, 5],
        "model__min_child_samples": [10, 20],
        "model__min_child_weight": [0.001, 0.01],
        "model__colsample_bytree": [0.6, 0.8],
        "model__subsample": [0.6, 0.8],
        "model__reg_alpha": [0.01, 0.1],
        "model__reg_lambda": [0.01, 0.1],
    }

    # Plain KFold: SalePrice (after log1p) is a continuous regression target,
    # so there are no class labels to stratify on (see module docstring, bug 2).
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=40,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    print("Searching hyperparameters...")
    search.fit(X_train, y_train)

    print("Best params:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best RMSE: {-search.best_score_:.4f}")
    return search.best_params_


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    """Build a preprocessor+model Pipeline, tune it, fit it, and score on X_val.

    The preprocessor is NOT fit here directly -- it's fit implicitly by the
    Pipeline every time `.fit(X_train, ...)` runs inside cross-validation, on
    that fold's training rows only (bug 1 fix, see module docstring).
    """
    preprocessor = build_preprocessor(X_train)
    base_model = lgb.LGBMRegressor(
        objective="regression",
        metric="msle",
        random_state=42,
        min_gain_to_split=0.01,
        min_data_in_leaf=20,
        min_sum_hessian_in_leaf=1e-3,
        feature_pre_filter=False,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", base_model),
    ])

    best_params = find_best_params(pipeline, X_train, y_train)
    pipeline.set_params(**best_params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    print(f"CV RMSE (log1p target): {-scores.mean():.4f}")

    pipeline.fit(X_train, y_train)

    preds_log = pipeline.predict(X_val)
    preds = np.expm1(preds_log)
    return pipeline, preds


def main():
    data_dir = Path("./data")  # expects train_hw.csv / test_hw.csv (Kaggle House Prices schema)
    train, test = load_data(data_dir)

    train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)

    y_train = np.log1p(train_set["SalePrice"])
    y_val = np.log1p(val_set["SalePrice"])
    X_train = train_set.drop(columns=["SalePrice", "Id"])
    X_val = val_set.drop(columns=["SalePrice", "Id"])

    pipeline, val_preds = train_and_evaluate(X_train, y_train, X_val, y_val)

    X_test = test.drop(columns=["Id"])
    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)

    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
