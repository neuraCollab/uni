# %% [markdown] 
# # LassoLarsIC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoLarsIC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

# Генерация случайных данных для регрессии
X, y = make_regression(n_samples=100, n_features=20, noise=0.5, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.random.seed(44)
# %%
model = LassoLarsIC(criterion='aic')
model.fit(X=X_train, y=y_train)

model.score(X_test, y_test)
