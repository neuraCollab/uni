# %% [markdown]
# # Lasso Regression
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# %% 
np.random.seed(44)
# %%
X, y = make_regression(n_samples=1000, n_features=1, 
                       noise=10, n_informative=1 )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
model = Lasso(alpha=0.1)
model.fit(X=X_train, y=y_train)

y_pred = model.predict(X_test)
model.score(X_test, y_test)
MSE = mean_squared_error(y_test, y_pred)
# %%
plt.plot(X_test, y_test, 'o', label='True')
plt.plot(X_test, y_pred, 'o', label='Predicted')
plt.legend()
plt.show()
# %%
plt.plot(y_test, model.predict(X=X_test), 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# %%
train_errors, test_errors = [], []
for m in range(1, len(X_train)):
    model.fit(X_train[:m], y_train[:m])
    train_errors.append(mean_squared_error(y_train[:m], model.predict(X_train[:m])))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.plot(np.sqrt(train_errors), label="Обучающая ошибка")
plt.plot(np.sqrt(test_errors), label="Тестовая ошибка")
plt.legend()
plt.title('Learning Curve')
plt.xlabel('Количество обучающих примеров')
plt.ylabel('RMSE')
plt.show()
# %%
from sklearn.linear_model import LassoCV
model = LassoCV(cv=5)
model.fit(X_train, y_train)

plt.plot(model.alphas_, model.mse_path_, ':')
plt.axvline(model.alpha_, linestyle='--', color='k')
plt.xlabel('Значения alpha')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.title('Кривые кросс-валидации для Lasso')
plt.show()