"""
SGDClassifier: online/mini-batch-style linear classification via stochastic
gradient descent, with feature scaling (important! see ../online-learning-sgd-pa.md).
"""
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# MNIST digits, multiclass (0-9)
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)

# SGD is extremely sensitive to feature scale: gradient step size is shared
# across all features, so unscaled pixel values (0-255) would make the
# effective per-feature learning rate wildly inconsistent.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SGDClassifier(
    loss="log_loss",           # log_loss -> equivalent to logistic regression
    penalty="l2",
    learning_rate="optimal",   # sklearn picks a decaying schedule automatically
    eta0=0.01,                 # only used by 'constant'/'invscaling' schedules
    max_iter=500,
    tol=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
