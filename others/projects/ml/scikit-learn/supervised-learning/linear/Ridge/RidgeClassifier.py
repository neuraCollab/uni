# %% [markdown]
# # RidgeClassifier
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
# %%
X, Y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=10, n_clusters_per_class=2, 
                           n_redundant=0, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# %%
clf = RidgeClassifier(alpha=0.1, random_state=42)
clf.fit(X_train, Y_train)

print(f"Accuracy: {clf.score(X_test, Y_test)}")
print(classification_report(Y_test, clf.predict(X=X_test)))
# %%
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, Y_test)
disp.ax_.set_title("Confusion Matrix of RidgeClassifier")
disp.plot()
# %%
# %%
# 1) Проекция X_test в 2D через PCA
pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(X_test)

# 2) Предсказания модели
y_pred = clf.predict(X_test)

# 3) Рисуем scatter — реальные классы
plt.scatter(X2d[:,0], X2d[:,1], c=Y_test, marker='o', label='True', alpha=0.6)
# 4) Отметим неверно классифицированные
mis = Y_test != y_pred
plt.scatter(X2d[mis,0], X2d[mis,1], facecolors='none', edgecolors='red', label='Misclassified')

plt.title("PCA projection of X_test")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
# %%
