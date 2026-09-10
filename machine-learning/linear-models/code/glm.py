"""
A real Generalized Linear Model example: Poisson regression on count data
(the source material only had a 4-line TweedieRegressor stub with no
worked example, so this synthesizes one).

See ../generalized-linear-models.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance, mean_squared_error

rng = np.random.default_rng(0)

# Simulate count data, e.g. "number of customer support tickets per day"
# driven by two features, with a log-link (counts are strictly non-negative
# and their variance grows with the mean -> a Gaussian/OLS assumption breaks).
n_samples = 2000
X = rng.normal(size=(n_samples, 2))
true_coef = np.array([0.6, -0.3])
log_rate = 1.0 + X @ true_coef
rate = np.exp(log_rate)
y = rng.poisson(rate)  # counts: 0, 1, 2, 3, ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Poisson GLM: minimizes Poisson unit deviance, not squared error ---
poisson_reg = PoissonRegressor(alpha=1e-4)
poisson_reg.fit(X_train, y_train)
poisson_pred = poisson_reg.predict(X_test)

# --- Naive OLS baseline for comparison (assumes symmetric Gaussian noise) ---
ols = LinearRegression()
ols.fit(X_train, y_train)
ols_pred = np.clip(ols.predict(X_test), a_min=0, a_max=None)  # counts can't be negative

print("Poisson deviance (lower is better for count data):")
print("  PoissonRegressor:", mean_poisson_deviance(y_test, np.clip(poisson_pred, 1e-6, None)))
print("  OLS (clipped):   ", mean_poisson_deviance(y_test, np.clip(ols_pred, 1e-6, None)))
print("\nMSE (for reference):")
print("  PoissonRegressor:", mean_squared_error(y_test, poisson_pred))
print("  OLS:             ", mean_squared_error(y_test, ols_pred))

# TweedieRegressor generalizes Gaussian/Poisson/Gamma via a single `power`
# parameter (0=Gaussian, 1=Poisson, 2=Gamma, ...). power=1, link='log'
# reproduces Poisson regression:
tweedie = TweedieRegressor(power=1, alpha=1e-4, link="log")
tweedie.fit(X_train, y_train)
print("\nTweedieRegressor(power=1) matches PoissonRegressor:",
      np.allclose(tweedie.coef_, poisson_reg.coef_, atol=1e-2))

plt.scatter(y_test, poisson_pred, alpha=0.4, label="Poisson prediction")
plt.plot([0, y_test.max()], [0, y_test.max()], "k--", label="ideal")
plt.xlabel("True count")
plt.ylabel("Predicted rate (mean)")
plt.title("Poisson regression: predicted rate vs true count")
plt.legend()
plt.show()
