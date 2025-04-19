import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Генерация синтетических данных
X, y, coef = make_regression(n_samples=100, n_features=100, n_informative=10,
                             coef=True, noise=5, random_state=42)

# 2. Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучение OMP с ограничением по ошибке
tol_value = 1e-4  # допустимый уровень ошибки
omp_tol = OrthogonalMatchingPursuit(tol=tol_value)
omp_tol.fit(X_train, y_train)

# 4. Предсказания и метрики
y_pred = omp_tol.predict(X_test)
r2 = r2_score(y_test, y_pred)
nnz = np.sum(omp_tol.coef_ != 0)

print(f"R² на тесте: {r2:.3f}")
print(f"Число ненулевых коэффициентов: {nnz}")
print(f"Заданное значение tol: {tol_value}")

# 5. Визуализация
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("OMP с ограничением по ошибке (tol)")
plt.grid()
plt.show()
