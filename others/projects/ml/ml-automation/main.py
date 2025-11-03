import pandas as pd
from sklearn.model_selection import train_test_split
from src.automl_hub import AutoMLHub

# Загрузка данных
df = pd.read_csv("data/sample.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение
automl = AutoMLHub(
    backend="flaml",
    task_type="classification",
    metric="f1",
    time_budget=10
)

automl.fit(X_train, y_train)

# Оценка
score = automl.score(X_test, y_test)
print(f"🏁 Итоговый скор: {score:.4f}")
print(f"🥇 Лучшая модель: {automl.get_best_model_name()}")

# Сохранение
automl.save_model("models/best_automl.pkl")

# Построение ROC и importance (сохранение в папку models)
try:
    automl.plot_roc_auc(X_test, y_test, savepath='models/roc.png')
except Exception as e:
    print('Не удалось построить ROC:', e)

try:
    automl.plot_feature_importance(top_n=15, savepath='models/feature_importance.png')
except Exception as e:
    print('Не удалось построить важности признаков:', e)