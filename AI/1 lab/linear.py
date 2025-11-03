import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

# Загрузка данных
train = pd.read_csv(Path("./data/train_hw.csv"))
test = pd.read_csv(Path("./data/test_hw.csv"))

# Логарифм целевой переменной
train["SalePrice"] = np.log1p(train["SalePrice"])

# Удаление столбцов с >80% пропусков
thresh = 0.8
na_cols = train.isnull().mean()
drop_cols = na_cols[na_cols > thresh].index.tolist()
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)

# Целевая переменная
y = train["SalePrice"]
X = train.drop(columns=["SalePrice", "Id"])
X_test = test.drop(columns=["Id"])

# Обработка категориальных/числовых
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Препроцессор
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# Разделение на train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Модели
alphas = np.logspace(-4, 4, 100)
ridge = Pipeline([
    ("prep", preprocessor),
    ("model", RidgeCV(alphas=alphas, cv=5))
])

lasso = Pipeline([
    ("prep", preprocessor),
    ("model", LassoCV(alphas=alphas, cv=5, max_iter=5000))
])

elastic = Pipeline([
    ("prep", preprocessor),
    ("model", ElasticNetCV(alphas=alphas, l1_ratio=[.1, .5, .9, 1], cv=5, max_iter=5000))
])

# Обучение и оценка
for name, model in [("Ridge", ridge), ("Lasso", lasso), ("ElasticNet", elastic)]:
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"{name} RMSE: {rmse:.4f}")

# Прогноз и экспорт сабмита (по Ridge, например)
best_model = ridge
best_model.fit(X, y)
final_preds = np.expm1(best_model.predict(X_test))
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": final_preds})
submission.to_csv("submission.csv", index=False)
print("Готово!")
