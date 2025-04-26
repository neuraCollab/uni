# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars, lars_path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 1. Генерация синтетических данных
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Обучение модели LARS
model = Lars(n_nonzero_coefs=10)
model.fit(X_train, y_train)

# 3. Оценка
r2 = model.score(X_test, y_test)
print(f"R² на тесте: {r2:.3f}")

# 4. Построение пути коэффициентов
alphas, active, coefs = lars_path(X, y, method='lar')

# 5. Визуализация пути коэффициентов
plt.figure(figsize=(8, 5))
for i in range(coefs.shape[0]):
    plt.plot(alphas, coefs[i], label=f'Feature {i}')
plt.xlabel('Alpha')
plt.ylabel('Коэффициенты')
plt.title('LARS — путь коэффициентов')
plt.legend(loc='best')
plt.gca().invert_xaxis()  # alpha убывает слева направо
plt.grid()
plt.tight_layout()
plt.show()

# %%
