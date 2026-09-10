"""
Side-by-side comparison of OLS / Ridge / Lasso / Elastic Net on the same
synthetic regression problem. Mirrors the kind of "try several linear models
and compare" step you'd do early in a modeling pipeline.

See ../regularization.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Correlated, slightly noisy synthetic data -> a good stress test for
# regularization (OLS should be visibly less stable here).
X, y = make_regression(
    n_samples=200, n_features=20, n_informative=8, noise=15.0, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling matters: the penalty term is applied to raw coefficient magnitudes.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "OLS": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.5),
    "Elastic Net": ElasticNet(alpha=0.5, l1_ratio=0.5),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-8)
    results[name] = {"mse": mse, "r2": r2, "nonzero_coefs": n_nonzero}
    print(f"{name:12s}  MSE={mse:8.2f}  R2={r2:6.3f}  nonzero_coefs={n_nonzero}/{X.shape[1]}")

# Coefficient magnitudes side by side -> visualizes shrinkage/sparsity differences.
plt.figure(figsize=(10, 5))
for name, model in models.items():
    plt.plot(model.coef_, marker="o", label=name, alpha=0.7)
plt.axhline(0, color="k", linewidth=0.5)
plt.xlabel("Feature index")
plt.ylabel("Coefficient value")
plt.title("OLS vs Ridge vs Lasso vs Elastic Net: coefficient shrinkage/sparsity")
plt.legend()
plt.tight_layout()
plt.show()
