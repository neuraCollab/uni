"""
Lasso regression: fitting, the train/test error gap as an overfitting
diagnostic, LassoCV's alpha path, and LassoLarsIC (AIC/BIC model selection
as a cheaper alternative to cross-validation).

See ../regularization.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=15, n_informative=6, noise=10, random_state=44)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Plain Lasso fit ---
model = Lasso(alpha=0.5)
model.fit(X_train, y_train)
print("Lasso R^2:", model.score(X_test, y_test))
print("Nonzero coefficients:", np.sum(model.coef_ != 0), "/", X.shape[1])

# --- Overfitting diagnostic: train vs test error as training set grows ---
train_errors, test_errors = [], []
for m in range(2, len(X_train)):
    model.fit(X_train[:m], y_train[:m])
    train_errors.append(mean_squared_error(y_train[:m], model.predict(X_train[:m])))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.plot(np.sqrt(train_errors), label="Train RMSE")
plt.plot(np.sqrt(test_errors), label="Test RMSE")
plt.legend()
plt.title("Learning curve: gap between train/test error signals over/underfitting")
plt.xlabel("Number of training examples")
plt.ylabel("RMSE")
plt.show()

# --- LassoCV: cross-validated alpha search ---
cv_model = LassoCV(cv=5, random_state=42)
cv_model.fit(X_train, y_train)
plt.plot(cv_model.alphas_, cv_model.mse_path_, ":")
plt.axvline(cv_model.alpha_, linestyle="--", color="k", label=f"chosen alpha={cv_model.alpha_:.3f}")
plt.xlabel("alpha")
plt.ylabel("MSE (per fold)")
plt.title("LassoCV: MSE path across folds")
plt.legend()
plt.show()

# --- LassoLarsIC: pick alpha via AIC/BIC instead of CV (needs n_samples > n_features) ---
aic_model = LassoLarsIC(criterion="aic").fit(X_train, y_train)
bic_model = LassoLarsIC(criterion="bic").fit(X_train, y_train)
print("AIC-selected alpha:", aic_model.alpha_, " R^2:", aic_model.score(X_test, y_test))
print("BIC-selected alpha:", bic_model.alpha_, " R^2:", bic_model.score(X_test, y_test))
