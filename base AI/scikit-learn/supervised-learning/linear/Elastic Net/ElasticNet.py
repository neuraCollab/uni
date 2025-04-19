# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
# %%

X, Y = make_regression(n_samples=100,  n_features=100, n_informative=85,
                       shuffle=True, n_targets=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
# %%
model = MultiTaskElasticNetCV(eps=1e-4, n_jobs=-1, cv=5)
model.fit(X_train, Y_train)

model.score(X_test, Y_test)

# model.alphas_
# %%
for i in range(Y_test.shape[1]):
    plt.figure(figsize=(5, 5))
    plt.plot(Y_test[:, i], model.predict(X_test)[:, i], 'o', label='Predicted vs True')
    plt.plot([Y_test[:, i].min(), Y_test[:, i].max()],
             [Y_test[:, i].min(), Y_test[:, i].max()], 'k--', lw=2, label='Ideal')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(f"Target {i}")
    plt.legend()
    plt.grid()
    plt.show()
# %%
