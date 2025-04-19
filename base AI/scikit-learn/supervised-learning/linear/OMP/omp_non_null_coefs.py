import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Генерация данных
X, y, coef = make_regression(n_samples=100, n_features=100, n_informative=10,
                             coef=True, noise=5, random_state=42)

# 2. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучение OMP (с ограничением на число ненулевых коэффициентов)
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
omp.fit(X_train, y_train)

# 4. Предсказание и метрика
y_pred = omp.predict(X_test)
print(f"R² на тесте: {r2_score(y_test, y_pred):.3f}")
print(f"Число ненулевых коэффициентов: {np.sum(omp.coef_ != 0)}")

# 5. Визуализация: истинные vs предсказанные
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("OMP: True vs Predicted")
plt.grid()
plt.show()
