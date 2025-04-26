# %% [markdown]
# # Ridge

# %%
import numpy as np 
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn
# %%
np.random.seed(100)
# %%
X = np.random.random_sample(size=(1000, 2)) * np.random.beta(.2, .4, size=(1000, 2))
Y = 2 * np.mean(X, 1) + 1 + np.random.normal(0, .1, 1000)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
# %%
ridge = lm.Ridge(alpha=0.01)
# ridge = lm.Ridge(alpha=np.logspace(-6, 6, 13))
ridge.fit(X_train, y_train)

ridge.score(X=X_test, y=y_test), ridge.coef_, ridge.intercept_
# %%
plt.plot(y_test, ridge.predict(X=X_test), 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
# %%
residuals = y_test - ridge.predict(X_test)
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
