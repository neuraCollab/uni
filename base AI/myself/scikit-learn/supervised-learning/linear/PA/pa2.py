from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Загрузка MNIST (или другой выборки)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = (y.astype(int) % 2) * 2 - 1  # бинаризуем метки в {-1,+1}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = PassiveAggressiveClassifier(
    max_iter=1000,
    C=1.0,           # параметр PA-II; C → ∞ даст PA-I
    tol=1e-3,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
