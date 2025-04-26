# %%
# 1. Импорты
import zipfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import ydf

# %%
# 2. Распаковка архива
zip_path = Path("./house-prices-hw.zip")
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(data_dir)

# %%
# 3. Загрузка данных
train = pd.read_csv(data_dir / "train_hw.csv")
test  = pd.read_csv(data_dir / "test_hw.csv")

# %%
# 4. Быстрый обзор
print(train.head())
train.info()

# %%
# 5. Разбиение на train/validation
train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)
print(f"Train set shape:      {train_set.shape}")
print(f"Validation set shape: {val_set.shape}")

# %%
# 6. Описание целевой переменной и её распределение
print(train["SalePrice"].describe())

plt.figure(figsize=(9, 8))
sns.histplot(data=train, x="SalePrice", bins=100, kde=True, alpha=0.4)
plt.title("Распределение SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Количество")
plt.tight_layout()
plt.show()

# %%
# 7. Числовые признаки и удаление идентификатора
df_num = train.select_dtypes(include=["float64", "int64"]).drop(columns=["Id"])
print(df_num.head())

# %%
# 8. Гистограммы всех числовых признаков
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()

# %%
# 9. Обучение модели Random Forest через YDF
learner = ydf.RandomForestLearner(label="SalePrice")
model   = learner.train(train_set )

# Печать краткого описания модели
print(model.describe())

# %%
# 10. Оценка модели и предсказания
print("Evaluation:", model.evaluate(val_set))

# %%

preds = model.predict(val_set)
print("Predictions:", preds[:10], "...")  # показываем первые 10 предсказаний

# %%
# 11. Анализ модели и бенчмарк
print("Analysis:",   model.analyze(val_set).to_file("analysis.html"))
print("Benchmark:",  model.benchmark(val_set))

# %%
