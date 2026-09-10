"""
Ridge regression: a plain fit, residual diagnostics, and RidgeCV for
automatic alpha selection.

See ../regularization.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# Correlated features by construction (both driven by the same underlying signal)
X = rng.normal(0, 0.2, size=(300, 2))
y = 4 * np.mean(X, axis=1) + 1 + rng.normal(0, 0.1, 300)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Plain Ridge with a fixed alpha ---
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
print("Ridge R^2:", ridge.score(X_test, y_test))
print("Coefficients:", ridge.coef_, "Intercept:", ridge.intercept_)

residuals = y_test - ridge.predict(X_test)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=20)
plt.title("Residual distribution")
plt.xlabel("y_true - y_pred")

plt.subplot(1, 2, 2)
plt.scatter(ridge.predict(X_test), residuals)
plt.axhline(0, color="r", linestyle="--")
plt.title("Residuals vs predicted")
plt.xlabel("Predicted value")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# --- RidgeCV: search alpha via efficient built-in leave-one-out CV ---
ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13))
ridge_cv.fit(X_train, y_train)
print("Best alpha found by RidgeCV:", ridge_cv.alpha_)
print("RidgeCV R^2 on test:", ridge_cv.score(X_test, y_test))
