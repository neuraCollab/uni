# %% [markdown]
# Ridge regrassion cross validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model as lm
import sklearn.model_selection
# %%
np.random.seed(42)
# %%
X = np.random.normal(0, .2, size=(100, 2)) 
Y = 4 * np.mean(X, axis=1) + 1 + np.random.normal(0, .1, 100)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# %%
model = lm.RidgeCV(alphas=np.logspace(-6, 6, 13))

model.fit(X=X_train, y=y_train)

# model.coef_, model.intercept_

model.score(X_test, y_test)
# %%
plt.plot(y_test, model.predict(X_test), 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# %%
