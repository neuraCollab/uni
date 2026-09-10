"""
Bayesian Ridge regression: curve fitting with a predictive mean AND a
predictive uncertainty band, plus a small ARDRegression comparison to show
its sparser weights.

See ../bayesian-regression.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, ARDRegression


def true_function(x):
    return np.sin(2 * np.pi * x)


rng = np.random.RandomState(1234)
size = 25
x_train = rng.uniform(0.0, 1.0, size)
y_train = true_function(x_train) + rng.normal(scale=0.1, size=size)
x_test = np.linspace(0.0, 1.0, 100)

# Polynomial basis expansion (degree 3) fed into a linear Bayesian model
n_order = 3
X_train = np.vander(x_train, n_order + 1, increasing=True)
X_test = np.vander(x_test, n_order + 1, increasing=True)

bayes_ridge = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
bayes_ridge.fit(X_train, y_train)
y_mean, y_std = bayes_ridge.predict(X_test, return_std=True)

ard = ARDRegression(fit_intercept=False)
ard.fit(X_train, y_train)
y_mean_ard, y_std_ard = ard.predict(X_test, return_std=True)

print("BayesianRidge: alpha_ (noise precision) =", round(bayes_ridge.alpha_, 3),
      " lambda_ (weight precision) =", round(bayes_ridge.lambda_, 3))
print("BayesianRidge coefficients:", np.round(bayes_ridge.coef_, 3))
print("ARDRegression coefficients:", np.round(ard.coef_, 3))
print("-> ARD typically drives more coefficients close to zero: each weight")
print("   gets its OWN precision (lambda_i), so irrelevant basis terms can be")
print("   pruned individually, unlike BayesianRidge's single shared lambda.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, mean, std) in zip(
    axes, [("BayesianRidge", y_mean, y_std), ("ARDRegression", y_mean_ard, y_std_ard)]
):
    ax.plot(x_test, true_function(x_test), color="blue", label="true sin(2*pi*x)")
    ax.scatter(x_train, y_train, s=30, alpha=0.5, label="observations")
    ax.plot(x_test, mean, color="red", label="predicted mean")
    ax.fill_between(x_test, mean - std, mean + std, color="pink", alpha=0.5, label="predicted std")
    ax.set_title(name)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
