"""
Binary classification with LightGBM + CatBoost, hyperparameters tuned via Optuna,
finished off with a simple 2-model averaging ensemble.

Ported from `AI/2 lab/hp.py` in the source coursework repo. The original file's
markdown header said "House Prices" (copy-paste leftover from a regression lab) --
that was a cosmetic bug only: the actual task is binary classification of a
mental-health screening target (`Depression`, binarized at the median). Fixed
here; no logic changed.

Interview-relevant points this script demonstrates:
- Optuna's `suggest_*` API builds the search space *inside* the objective
  function, so you can make parameters conditional on each other (see
  `optimize_catboost_params`: `bagging_temperature` is only sampled when
  `bootstrap_type == 'Bayesian'` -- Bernoulli/MVS bootstrap don't use it).
  A grid/random search can't express that without wasting trials.
- Early stopping (`lgb.early_stopping`, CatBoost's `od_type='Iter'`) picks the
  boosting round count automatically instead of tuning `n_estimators` by hand.
- A naive (unweighted) ensemble of two uncorrelated-ish models is a cheap way
  to shave variance off a single model's predictions.

Note: the original also cast the averaged *probability* straight to `int`
(`(test_preds_ensemble).astype(int)`), which truncates almost everything to 0
since the average rarely reaches 1.0. Fixed here to threshold at 0.5 first
(`(test_preds_ensemble > 0.5).astype(int)`) -- a good example of why you
always sanity-check the value range right before a type cast.

See ../gradient-boosting-catboost-lgbm.md for the write-up.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def prepare_binary_target(df: pd.DataFrame, target_column: str, threshold: float) -> pd.DataFrame:
    """Binarize a numeric target column around `threshold` (e.g. its median)."""
    df[f"{target_column}_binary"] = (df[target_column] > threshold).astype(int)
    return df


def build_preprocessor(train_df: pd.DataFrame, is_train: bool):
    """
    numeric  -> median impute + StandardScaler
    categorical -> constant impute ("None") + OneHotEncoder
    """
    df = train_df.copy()
    columns_to_drop = ["Id"]
    if is_train:
        columns_to_drop.append("Depression")
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    num_feats = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = df.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ])
    return preprocessor, num_feats, cat_feats


def optimize_lightgbm_params(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    """Optuna search over LightGBM hyperparameters, minimizing validation logloss."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000)
        return model.best_score["valid_0"]["binary_logloss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_catboost_params(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    """
    Optuna search over CatBoost hyperparameters. `bagging_temperature` is a
    CONDITIONAL parameter: it only controls Bayesian bootstrap, so it's only
    sampled (and only meaningful) when `bootstrap_type == 'Bayesian'`.
    """

    def objective(trial: optuna.Trial) -> float:
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0),
            "bootstrap_type": bootstrap_type,
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0),
            "od_type": "Iter",
            "od_wait": 50,
            "verbose": False,
            "eval_metric": "Logloss",
        }
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return model.get_best_score()["validation"]["Logloss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train_lightgbm_binary(X_train, y_train, X_val, y_val):
    best_params = optimize_lightgbm_params(X_train, y_train, X_val, y_val)
    print(f"Best LightGBM params: {best_params}")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    return lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)],
    )


def train_catboost_binary(X_train, y_train, X_val, y_val):
    best_params = optimize_catboost_params(X_train, y_train, X_val, y_val)
    print(f"Best CatBoost params: {best_params}")
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
    return model


def evaluate_binary_model(model, X_test, y_test, model_type: str = "lightgbm"):
    if model_type == "lightgbm":
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:  # catboost
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(report)
    return accuracy, auc, report


def main():
    data_dir = Path("./data")  # expects train.csv / test.csv with a "Depression" target
    train, test = load_data(data_dir)

    target_column = "Depression"
    threshold = train[target_column].median()
    train = prepare_binary_target(train, target_column, threshold=threshold)
    target_column = f"{target_column}_binary"

    preprocessor, num_feats, cat_feats = build_preprocessor(train, is_train=True)
    X = preprocessor.fit_transform(train)
    y = train[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training LightGBM...")
    lgb_model = train_lightgbm_binary(X_train, y_train, X_test, y_test)
    evaluate_binary_model(lgb_model, X_test, y_test, "lightgbm")

    print("Training CatBoost...")
    cat_model = train_catboost_binary(X_train, y_train, X_test, y_test)
    evaluate_binary_model(cat_model, X_test, y_test, "catboost")

    test_processed = preprocessor.transform(test)
    test_preds_lgb = lgb_model.predict(test_processed)
    test_preds_cat = cat_model.predict_proba(test_processed)[:, 1]

    # Simple averaging ensemble -- cheap variance reduction across two
    # differently-biased models (leaf-wise histogram GBDT vs ordered boosting).
    test_preds_ensemble = (test_preds_lgb + test_preds_cat) / 2
    test_preds_final = (test_preds_ensemble > 0.5).astype(int)

    submission = pd.DataFrame({"id": test["id"], "Depression": test_preds_final})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
