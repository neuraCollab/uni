# %% [markdown]
# # RidgeClassifierCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# %%
np.random.seed(42)
# %%
X, y = make_classification(n_samples=1000, n_features=2000, 
                           n_classes=2, n_informative=100, 
                           n_clusters_per_class=1, n_redundant=1)

pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=.2, 
                                                    random_state=42)

# %%
print(pca.components_) # Вектора главных компонент (направления вроль которыз 
    # данные имеют наибольшую дисперсию)
print(pca.explained_variance_ratio_) # Доля объясненной дисперсии каждой компоненты
print(pca.explained_variance_) # Объясненная дисперсия каждой компоненты
print(x_pca)    # Преобразованные данные
# %%
model = RidgeClassifierCV(alphas=np.logspace(-6,6,13), cv=5)

model.fit(X_train, y_train)
model.score(X_test, y_test)
model.alpha_
# %%
print(classification_report(y_test, model.predict(X_test)))
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
# %%
x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                     np.arange(y_min, y_max, .01))

z= model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# %%
plt.contourf(xx, yy, z, alpha=.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("RidgeClassifierCV")
plt.show()
# %%
