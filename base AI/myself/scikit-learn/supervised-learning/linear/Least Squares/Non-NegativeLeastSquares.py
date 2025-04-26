# %% [markdown]
# # Non-Negative Least Squares
# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures
# %%
np.random.seed(100)
# %%
X = np.random.randint(0, 200, size=(100, 2)) + np.random.pareto(1, size=(100, 2))
Y = np.sqrt(np.mean(X, axis=1) ) + 1 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
# %%
NNLR = lm.LinearRegression(positive=True)
NNLR.fit(X_train, y_train)

NNLR.coef_, NNLR.intercept_

NNLR.score(X_test, y_test)
# %%
plt.plot(y_test, NNLR.predict(X_test), 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
# %%
residuals = y_test - NNLR.predict(X_test)

plt.hist(residuals, bins=20)
plt.title("Гистограмма остатков")
plt.xlabel("Ошибка (y_true - y_pred)")
plt.ylabel("Частота")
plt.show()
# %%
plt.scatter(y_test, residuals)
plt.title("Остатки vs Предсказанные значения")
plt.xlabel("Предсказанное значение")
plt.ylabel("Остаток")
plt.show()
# %%
# model = make_pipeline(PolynomialFeatures(degree=2), lm.LinearRegression(positive=True))
# model.fit(X_train, y_train)

# model.score(X_test, y_test)
# %%
