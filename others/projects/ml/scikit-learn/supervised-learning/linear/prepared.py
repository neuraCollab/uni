# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn
# %%
np.random.seed(100)
# %%
X = np.random.randint(0, 100, size=(100, 2))
Y = 2 * np.mean(X, 1) + 1 + np.random.normal(0, 10, 100)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
