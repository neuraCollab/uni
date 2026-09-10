"""
PassiveAggressiveClassifier: an online margin-based learner. "Passive" when
the current example is already classified with sufficient margin (no
update), "aggressive" when it's misclassified or inside the margin (large
corrective update). See ../online-learning-sgd-pa.md for the theory.
"""
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = (y.astype(int) % 2) * 2 - 1  # binarize to {-1, +1}: odd vs even digit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = PassiveAggressiveClassifier(
    max_iter=1000,
    C=1.0,        # aggressiveness / regularization tradeoff (PA-II); C -> inf gives PA-I behavior
    tol=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
