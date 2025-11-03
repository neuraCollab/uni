# %%
# %% [markdown]
# # Автоматизированный пайплайн для House Prices в Jupyter Notebook

# %%
# 1. Импорты
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import optuna
from catboost import CatBoostClassifier


# %%

# %%
# 2. Функции пайплайна

def unzip_data(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test  = pd.read_csv(data_dir / "test.csv")
    return train, test

def prepare_binary_target(df: pd.DataFrame, target_column: str, threshold: float = 0.5):
    """Преобразует числовую целевую переменную в бинарную"""
    df[f'{target_column}_binary'] = (df[target_column] > threshold).astype(int)
    return df

def optimize_lightgbm_params(X_train, y_train, X_val, y_val, n_trials=100):
    """Оптимизация гиперпараметров LightGBM с помощью Optuna"""
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
        )
        
        return model.best_score['valid_0']['binary_logloss']
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def optimize_catboost_params(X_train, y_train, X_val, y_val, n_trials=100):
    """Оптимизация гиперпараметров CatBoost с помощью Optuna"""
    def objective(trial):
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0),
            'bootstrap_type': bootstrap_type,
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0),
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': False,
            'eval_metric': 'Logloss'
        }
        
        # Добавляем bagging_temperature только для Bayesian bootstrap
        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        return model.get_best_score()['validation']['Logloss']
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def train_lightgbm_binary(X_train, y_train, X_val, y_val):
    """Обучение LightGBM для бинарной классификации с оптимизацией параметров"""
    print("Оптимизация гиперпараметров LightGBM...")
    best_params = optimize_lightgbm_params(X_train, y_train, X_val, y_val)
    print(f"Лучшие параметры LightGBM: {best_params}")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)],
    )
    
    return model

def train_catboost_binary(X_train, y_train, X_val, y_val):
    """Обучение CatBoost для бинарной классификации с оптимизацией параметров"""
    print("Оптимизация гиперпараметров CatBoost...")
    best_params = optimize_catboost_params(X_train, y_train, X_val, y_val)
    print(f"Лучшие параметры CatBoost: {best_params}")
    
    model = CatBoostClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    
    return model

def evaluate_binary_model(model, X_test, y_test, model_type='lightgbm'):
    """Оценка модели бинарной классификации"""
    if model_type == 'lightgbm':
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
    print("\nClassification Report:")
    print(report)
    
    return accuracy, auc, report

# %%
# 3. Распаковка и загрузка данных
zip_path = Path("./mental-health-prediction.zip")
data_dir = Path("./data")
unzip_data(zip_path, data_dir)
train, test = load_data(data_dir)

# %%
# Анализ данных
print("Информация о данных:")
print(train.info())
print("\nСтатистика числовых признаков:")
print(train.describe())

# %%
# Проверка целевой переменной
target_column = 'Depression'
print(f"\nРаспределение значений в {target_column}:")
print(train[target_column].value_counts(normalize=True))
print("\nПроцент пропущенных значений:")
print(train[target_column].isnull().mean() * 100)

# Визуализация распределения целевой переменной
plt.figure(figsize=(10, 6))
sns.histplot(data=train, x=target_column, bins=30)
plt.title(f'Распределение {target_column}')
plt.show()

# %%
# Подготовка данных для бинарной классификации
if target_column not in train.columns:
    raise ValueError(f"Колонка {target_column} не найдена в данных. Доступные колонки: {train.columns.tolist()}")

# Определяем порог для бинаризации на основе медианы
threshold = train[target_column].median()
print(f"\nПорог для бинаризации (медиана): {threshold:.2f}")

train = prepare_binary_target(train, target_column, threshold=threshold)
target_column = f'{target_column}_binary'

print(f"\nРаспределение после бинаризации:")
print(train[target_column].value_counts(normalize=True))

# %%
def build_preprocessor(train_df: pd.DataFrame, is_train: bool):
    """
    Возвращает ColumnTransformer и списки признаков:
    - числовые: SimpleImputer(median) + StandardScaler
    - категориальные: SimpleImputer(constant) + OneHotEncoder(sparse_output=False)
    """
    # Создаем копию датафрейма
    df = train_df.copy()
    
    # Удаляем только существующие колонки
    columns_to_drop = ['Id']
    if is_train:
        columns_to_drop.append('Depression')
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Разделяем на числовые и категориальные признаки
    num_feats = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = df.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats)
    ])
    return preprocessor, num_feats, cat_feats

# %%
# Подготовка данных и обучение моделей
preprocessor, num_feats, cat_feats = build_preprocessor(train, True)
X = preprocessor.fit_transform(train)
y = train[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nРазмеры обучающей и тестовой выборок:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Обучение LightGBM
print("\nОбучение LightGBM...")
lgb_model = train_lightgbm_binary(X_train, y_train, X_test, y_test)
print("\nОценка LightGBM:")
lgb_metrics = evaluate_binary_model(lgb_model, X_test, y_test, 'lightgbm')

# Обучение CatBoost
print("\nОбучение CatBoost...")
cat_model = train_catboost_binary(X_train, y_train, X_test, y_test)
print("\nОценка CatBoost:")
cat_metrics = evaluate_binary_model(cat_model, X_test, y_test, 'catboost')

# Подготовка тестовых данных
test_processed = preprocessor.transform(test)
test_preds_lgb = lgb_model.predict(test_processed)
test_preds_cat = cat_model.predict_proba(test_processed)[:, 1]

# Ансамбль предсказаний (среднее)
test_preds_ensemble = (test_preds_lgb + test_preds_cat) / 2
test_preds_final = (test_preds_ensemble).astype(int)

# Сохранение результатов
submission = pd.DataFrame({
    'id': test['id'],
    'Depression': test_preds_final
})
submission.to_csv("submission.csv", index=False)
print("\nРезультаты сохранены в submission.csv")



