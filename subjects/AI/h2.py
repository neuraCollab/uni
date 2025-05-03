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

def build_preprocessor(train_df: pd.DataFrame):
    """
    Возвращает ColumnTransformer и списки признаков:
    - числовые: SimpleImputer(median) + StandardScaler
    - категориальные: SimpleImputer(constant) + OneHotEncoder(sparse_output=False)
    """
    df = train_df.drop(columns=["Id", "SalePrice"])
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
# 3. Распаковка и загрузка данных
zip_path = Path("./house-prices-hw.zip")
data_dir = Path("./data")
unzip_data(zip_path, data_dir)
train, test = load_data(data_dir)

# %% 
# 4. Быстрый обзор и разбиение
display(train.head())
train.info()

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
# 6. Построение и применение препроцессора
preprocessor, num_feats, cat_feats = build_preprocessor(train_set)
preprocessor.fit(train_set.drop(columns=["Id", "SalePrice"]))

X_train_proc = pd.DataFrame(
    preprocessor.transform(X_train),
    columns=list(num_feats) +
            list(preprocessor.named_transformers_["cat"]["ohe"]
                 .get_feature_names_out(cat_feats)),
    index=X_train.index
)
X_val_proc = pd.DataFrame(
    preprocessor.transform(X_val),
    columns=X_train_proc.columns,
    index=X_val.index
)

# %% 
# 7. Кросс‑валидация с LightGBM
lgb_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    lgb_model,
    pd.concat([X_train_proc, X_val_proc]),
    np.log1p(pd.concat([train_set["SalePrice"], val_set["SalePrice"]])),
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
print("CV RMSE (log1p):", -scores.mean())

# %% 
# 8. Обучение финальной модели и предсказания
lgb_model.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
)

X_test_proc = pd.DataFrame(
    preprocessor.transform(test.drop(columns=["Id"])),
    columns=X_train_proc.columns,
    index=test.index
)
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
