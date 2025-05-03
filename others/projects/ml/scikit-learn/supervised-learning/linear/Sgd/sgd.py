from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Загрузка и предобработка данных
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# приводим метки к целочисленному виду
y = y.astype(int)

# делим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=42
)

# стандартизация признаков (очень важна для SGD)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2. Настройка SGDClassifier
# loss='log' — логистическая регрессия, penalty='l2' — L2-регуляризация
clf = SGDClassifier(
    loss='log_loss',      # логистическая регрессия
    learning_rate='optimal',  # автоматический подбор шага
    eta0=0.01,       # начальный learning rate (играет роль при 'constant')
    max_iter=500,
    tol=1e-3,
    random_state=42,
    # verbose=1        # чтобы видеть прогресс
)

# 3. Обучение
clf.fit(X_train, y_train)

# 4. Оценка
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc*100:.2f}%")
