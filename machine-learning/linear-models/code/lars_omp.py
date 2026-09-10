"""
LARS (path visualization) and OMP (fixed-k vs tolerance-based sparsity
control), consolidated into one script since they're both greedy sparse
linear methods.

See ../regularization.md for the theory.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars, lars_path, OrthogonalMatchingPursuit
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# LARS: full piecewise-linear regularization path
# ============================================================
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lars = Lars(n_nonzero_coefs=10)
lars.fit(X_train, y_train)
print(f"LARS R^2 on test: {lars.score(X_test, y_test):.3f}")

alphas, active, coefs = lars_path(X, y, method="lar")

plt.figure(figsize=(8, 5))
for i in range(coefs.shape[0]):
    plt.plot(alphas, coefs[i], label=f"Feature {i}")
plt.xlabel("alpha")
plt.ylabel("Coefficient")
plt.title("LARS path: each feature joins the active set one at a time")
plt.gca().invert_xaxis()  # alpha decreases left to right along the path
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

# ============================================================
# OMP: fixed number of nonzero coefficients vs. error-tolerance stopping
# ============================================================
X2, y2, true_coef = make_regression(
    n_samples=100, n_features=100, n_informative=10, coef=True, noise=5, random_state=42
)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Formulation 1: fix k, minimize error
omp_fixed_k = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
omp_fixed_k.fit(X2_train, y2_train)
r2_fixed = r2_score(y2_test, omp_fixed_k.predict(X2_test))
nnz_fixed = np.sum(omp_fixed_k.coef_ != 0)

# Formulation 2: fix an error tolerance, minimize number of features used
omp_tol = OrthogonalMatchingPursuit(tol=1e-4)
omp_tol.fit(X2_train, y2_train)
r2_tol = r2_score(y2_test, omp_tol.predict(X2_test))
nnz_tol = np.sum(omp_tol.coef_ != 0)

print(f"\n[n_nonzero_coefs=10]  R2={r2_fixed:.3f}  nonzero_coefs={nnz_fixed}")
print(f"[tol=1e-4]            R2={r2_tol:.3f}  nonzero_coefs={nnz_tol}")

plt.figure(figsize=(6, 6))
plt.scatter(y2_test, omp_fixed_k.predict(X2_test), label="n_nonzero_coefs=10", alpha=0.6)
plt.scatter(y2_test, omp_tol.predict(X2_test), label="tol=1e-4", alpha=0.6)
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], "k--", lw=2, label="Ideal")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("OMP: fixed-k vs tolerance-based sparsity control")
plt.legend()
plt.tight_layout()
plt.show()
