# Документация: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html
# %% [markdown] 
# # LassoLarsCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

# Генерация случайных данных для регрессии
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.random.seed(44)
# %%
model = LassoLarsCV(cv=5, n_jobs=-1, verbose=1)
model.fit(X=X_train, y=y_train)

# model.alpha_
model.score(X_test, y_test)
# %%
plt.scatter(y_test, model.predict(X_test))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # линия идеальных предсказаний
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Истинные vs. Предсказанные значения')
plt.show()

# %%
residuals = y_test - model.predict(X_test)
plt.scatter(model.predict(X_test), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков')
plt.show()

# %%
# Грвфик важности признаков
plt.bar(range(len(model.coef_)), model.coef_)
plt.xlabel('Номер признака')
plt.ylabel('Важность признака')
plt.title('Важность признаков')
plt.show()
# %%
# Learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1, 10))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning curve')
plt.legend()
plt.show()
# %%
