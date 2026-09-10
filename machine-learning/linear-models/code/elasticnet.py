"""
Elastic Net: single-task regression with a mixed L1/L2 penalty, plus the
multi-task variant that shares a sparsity pattern across several targets.

See ../regularization.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# --- Single-task Elastic Net ---
X, y = make_regression(n_samples=200, n_features=30, n_informative=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# l1_ratio=1 -> pure Lasso, l1_ratio=0 -> pure Ridge; CV searches both alpha and l1_ratio.
model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, random_state=42)
model.fit(X_train, y_train)
print("Best alpha:", model.alpha_, " Best l1_ratio:", model.l1_ratio_)
print("R^2 on test:", model.score(X_test, y_test))
print("Nonzero coefficients:", np.sum(model.coef_ != 0), "/", X.shape[1])

# --- Multi-task Elastic Net: several correlated targets, shared feature selection ---
X_mt, Y_mt = make_regression(n_samples=100, n_features=100, n_informative=85,
                              n_targets=2, shuffle=True, random_state=42)
Xm_train, Xm_test, Ym_train, Ym_test = train_test_split(X_mt, Y_mt, test_size=0.2, random_state=42)

mt_model = MultiTaskElasticNetCV(eps=1e-4, n_jobs=-1, cv=5)
mt_model.fit(Xm_train, Ym_train)
print("\nMultiTaskElasticNetCV R^2:", mt_model.score(Xm_test, Ym_test))

plt.figure(figsize=(6, 4))
for i in range(Ym_train.shape[1]):
    plt.plot(mt_model.coef_[i, :], label=f"Target {i}", alpha=0.7)
plt.title("Multi-task Elastic Net: shared sparsity pattern across targets")
plt.xlabel("Feature index")
plt.ylabel("Coefficient")
plt.legend()
plt.tight_layout()
plt.show()
