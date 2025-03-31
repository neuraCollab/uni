# %% [code]
# Многозадачная регрессия с использованием Lasso-регрессии
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error

# Генерация случайных данных для многозадачной регрессии
X, y = make_regression(n_samples=100, n_features=10, n_targets=3, noise=0.5, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели MultiTaskLassoCV
model = MultiTaskLassoCV(cv=5, n_jobs=-1)
model.fit(X_train, y_train)

# Оценка модели
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Вычисление ошибки
train_error = mean_squared_error(y_train, train_predictions)
test_error = mean_squared_error(y_test, test_predictions)

print(f"Обучающая ошибка: {train_error:.4f}")
print(f"Тестовая ошибка: {test_error:.4f}")

# Коэффициенты модели
print("Коэффициенты модели:", model.coef_)

# Визуализация коэффициентов для каждой задачи
plt.figure(figsize=(10, 6))
for i in range(y.shape[1]):
    plt.plot(model.coef_[:, i], label=f'Задача {i+1}')
plt.title('Коэффициенты для каждой целевой переменной (задачи)')
plt.xlabel('Признаки')
plt.ylabel('Коэффициенты')
plt.legend()
plt.show()

# %%
model.score(X_test, y_test)
# %%
