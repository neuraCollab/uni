# %% [markdown]
# # Автоматизированный пайплайн для House Prices в Jupyter Notebook

# %%
# 1. Импорты
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_linear_models(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    """Обучение и сравнение линейных моделей"""
    models = {
        'L1 (Lasso)': Lasso(alpha=1.0, max_iter=1000),
        'L2 (Ridge)': Ridge(alpha=1.0, max_iter=1000),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nОбучение {name}...")
        model.fit(X_train, y_train)
        
        # Оценка на валидационном наборе
        y_pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"{name} результаты:")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2
        }
    
    return results

def plot_model_comparison(results):
    """Визуализация сравнения моделей"""
    models = list(results.keys())
    mses = [results[m]['mse'] for m in models]
    r2s = [results[m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График MSE
    rects1 = ax1.bar(x, mses, width)
    ax1.set_ylabel('MSE')
    ax1.set_title('Сравнение MSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # График R2
    rects2 = ax2.bar(x, r2s, width)
    ax2.set_ylabel('R2')
    ax2.set_title('Сравнение R2')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Обучение и оценка модели
    
    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная для обучения
        X_test: Тестовые признаки
        y_test: Целевая переменная для тестирования
    """
    # Построение препроцессора
    preprocessor, num_feats, cat_feats = build_preprocessor(X_train, is_train=True)
    preprocessor.fit(X_train)
    
    # Преобразование данных
    X_train_proc = transform_data(preprocessor, X_train, num_feats, cat_feats)
    
    # Проверка и очистка данных
    # Замена бесконечных значений
    X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan)
    # Заполнение пропусков медианными значениями
    X_train_proc = X_train_proc.fillna(X_train_proc.median())
    # Проверка на отрицательные значения
    X_train_proc = X_train_proc.clip(lower=0)
    
    # Обучение и оценка линейных моделей
    # print("Обучение линейных моделей...")
    # linear_results = train_linear_models(X_train_proc, y_train, X_test, y_test)
    # plot_model_comparison(linear_results)
    
    # Поиск лучших параметров для LightGBM
    best_params = find_best_params(X_train_proc, y_train)
    
    # Создание и обучение модели с лучшими параметрами
    lgb_model = lgb.LGBMRegressor(
        **best_params,
        objective="regression",
        random_state=42,
        min_gain_to_split=0.01,
        min_data_in_leaf=20,
        min_sum_hessian_in_leaf=1e-3,
        feature_pre_filter=False
    )
    
    # Кросс-валидация
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        lgb_model,
        X_train_proc,
        y_train,
        cv=kf
    )
    print("CV RMSE (log1p):", -scores.mean())
    
    # Обучение финальной модели
    lgb_model.fit(
        X_train_proc, 
        y_train,
        eval_metric='msle',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ],
        eval_set=[(X_train_proc, y_train)]
    )
    
    # Подготовка тестового набора
    X_test_proc = transform_data(preprocessor, X_test, num_feats, cat_feats)
    
    # Проверка тестовых данных
    X_test_proc = X_test_proc.replace([np.inf, -np.inf], np.nan)
    X_test_proc = X_test_proc.fillna(X_test_proc.median())
    X_test_proc = X_test_proc.clip(lower=0)
    
    # Предсказания
    preds_log = lgb_model.predict(X_test_proc)
    preds = np.expm1(preds_log)
    
    return lgb_model, preds, X_train_proc

# %%
# 2. Функции пайплайна

def unzip_data(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train_hw.csv")
    test  = pd.read_csv(data_dir / "test_hw.csv")
    return train, test

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def build_preprocessor(train_df: pd.DataFrame, is_train: bool = True):
    """
    Возвращает ColumnTransformer и списки признаков:
    - числовые: SimpleImputer(median) + StandardScaler
    - категориальные: SimpleImputer(constant) + OneHotEncoder(sparse_output=False)
    
    Args:
        train_df: DataFrame для обучения
        is_train: Флаг, указывающий является ли набор тренировочным
    """
    # Создаем копию датафрейма
    df = train_df.copy()
    
    # Удаляем колонки в зависимости от типа набора
    columns_to_drop = ['Id']
    if is_train:
        columns_to_drop.append('SalePrice')
    
    # Удаляем только существующие колонки
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Находим колонки с большим количеством пропусков (>70%)
    missing_threshold = 0.7
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    
    # Удаляем колонки с большим количеством пропусков
    df = df.drop(columns=high_missing_cols)
    print(f"Удалены колонки с >70% пропусков: {high_missing_cols}")
    
    # Разделяем на числовые и категориальные признаки
    num_feats = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Создаем пайплайны для числовых и категориальных признаков
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)) 
    ])
    
    # Создаем ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats)
    ])
    
    return preprocessor, num_feats, cat_feats

def transform_data(preprocessor, df: pd.DataFrame, num_feats: list, cat_feats: list):
    """
    Преобразует данные с помощью предобученного препроцессора
    
    Args:
        preprocessor: Обученный ColumnTransformer
        df: DataFrame для преобразования
        num_feats: Список числовых признаков
        cat_feats: Список категориальных признаков
    
    Returns:
        DataFrame с преобразованными данными
    """
    # Получаем имена признаков после OneHotEncoder
    cat_feature_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_feats)
    
    # Создаем список всех имен признаков
    feature_names = list(num_feats) + list(cat_feature_names)
    
    # Преобразуем данные
    transformed_data = pd.DataFrame(
        preprocessor.transform(df),
        columns=feature_names,
        index=df.index
    )
    
    return transformed_data

def find_best_params(X_train: pd.DataFrame, y_train: pd.Series):
    base_params = {
        'objective': 'regression',
        'metric': 'msle',
        'random_state': 42,
        'min_gain_to_split': 0.01,  # Минимальный выигрыш для разбиения
        'min_data_in_leaf': 20,     # Минимальное количество данных в листе
        'min_sum_hessian_in_leaf': 1e-3,  # Минимальная сумма гессиана в листе
        'feature_pre_filter': False  # Отключаем предварительную фильтрацию признаков
    }

    param_dist = {
        'n_estimators': [200, 500, 1500, 2000, 2500, 3000],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [3, 5],
        'min_child_samples': [10, 20],
        'min_child_weight': [0.001, 0.01],
        'colsample_bytree': [0.6, 0.8],
        'subsample': [0.6, 0.8],
        'reg_alpha': [0.01, 0.1],
        'reg_lambda': [0.01, 0.1]
    }

    lgb_model = lgb.LGBMRegressor(**base_params)

    search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_dist,
        n_iter=40,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("Начинаем быстрый поиск параметров...")
    search.fit(X_train, y_train)

    print("\nЛучшие параметры:")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nЛучший RMSE: {-search.best_score_:.4f}")

    return search.best_params_

# %%
# 3. Распаковка и загрузка данных
zip_path = Path("./house-prices-hw.zip")
data_dir = Path("./data")
unzip_data(zip_path, data_dir)
train, test = load_data(data_dir)
# %% 
# 4. Быстрый обзор и разбиение
#

# Создаем препроцессор на всем обучающем наборе
preprocessor, num_feats, cat_feats = build_preprocessor(train)
preprocessor.fit(train.drop(columns=["Id", "SalePrice"]))

# Разбиваем данные после создания препроцессора
train_set, val_set = split_data(train)
print(f"Train set:      {train_set.shape}")
print(f"Validation set: {val_set.shape}")

# %% 
# 5. Лог-трансформация целевой переменной
y_train = np.log1p(train_set["SalePrice"])
y_val   = np.log1p(val_set["SalePrice"])
X_train = train_set.drop(columns=["SalePrice", "Id"])
X_val   = val_set.drop(columns=["SalePrice", "Id"])

# %% 
# 6. Преобразование данных
X_train_proc = transform_data(preprocessor, X_train, num_feats, cat_feats)
X_val_proc = transform_data(preprocessor, X_val, num_feats, cat_feats)

# %% 
# 7. Кросс‑валидация с LightGBM
lgb_model, preds, X_train_proc = train_and_evaluate_model(X_train_proc, y_train, X_val_proc, y_val)

# %% 
# 8. Обучение финальной модели и предсказания
X_test_proc = transform_data(preprocessor, test.drop(columns=["Id"]), num_feats, cat_feats)
preds_log = lgb_model.predict(X_test_proc)
preds = np.expm1(preds_log)

# Сохраняем результат
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": preds
})
submission.to_csv("submission.csv", index=False)
print("Done: submission.csv generated")

# %%
