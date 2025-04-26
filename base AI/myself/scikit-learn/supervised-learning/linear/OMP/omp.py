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

# === Модель 1: Ограничение по числу признаков ===
omp_fixed = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
omp_fixed.fit(X_train, y_train)
y_pred_fixed = omp_fixed.predict(X_test)
r2_fixed = r2_score(y_test, y_pred_fixed)
nnz_fixed = np.sum(omp_fixed.coef_ != 0)

# === Модель 2: Ограничение по ошибке ===
tol_value = 1e-4
omp_tol = OrthogonalMatchingPursuit(tol=tol_value)
omp_tol.fit(X_train, y_train)
y_pred_tol = omp_tol.predict(X_test)
r2_tol = r2_score(y_test, y_pred_tol)
nnz_tol = np.sum(omp_tol.coef_ != 0)

# === Сравнение результатов ===
print("📊 Сравнение моделей:")
print(f"[n_nonzero_coefs=10]     R²: {r2_fixed:.3f}, Ненулевых коэффициентов: {nnz_fixed}")
print(f"[tol={tol_value}]        R²: {r2_tol:.3f}, Ненулевых коэффициентов: {nnz_tol}")

# === Визуализация ===
plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred_fixed, label='n_nonzero_coefs=10', color='blue', alpha=0.6)
plt.scatter(y_test, y_pred_tol, label=f'tol={tol_value}', color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Идеал')

plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("OMP: сравнение стратегий отбора")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
