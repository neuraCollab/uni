## AI Labs — ML/DS 

Набор учебных лабораторных работ по машинному обучению: регрессия цен на дома (Kaggle House Prices), бинарная классификация ментального здоровья, и классификация текстов с TF‑IDF. Репозиторий демонстрирует полный цикл ML: подготовка данных, пайплайны, валидация, тюнинг гиперпараметров, обучение и экспорт предсказаний.

### Структура
- `1 lab/`
  - `linear.py` — линейные модели (Ridge/Lasso/ElasticNet) на House Prices с полноценным `ColumnTransformer` пайплайном и CV.
  - `h2.py` — продвинутый пайплайн: препроцессинг, лог‑трансформация, подбор гиперпараметров и обучение `LightGBM`, генерация `submission.csv`.
  - `hp.ipynb` — ноутбук с экспериментами по той же задаче.
- `2 lab/`
  - `hp.py` — бинарная классификация с `LightGBM` и `CatBoost`, тюнинг с `Optuna`, метрики (`Accuracy`, `AUC`), энсамбль, экспорт `submission.csv`.
  - `mh.ipynb` — ноутбук с анализом данных и экспериментами.
- `3 lab/`
  - `tf-idf.py` — текстовая классификация: `TF-IDF` + `LogisticRegression`, сохранение артефактов (`joblib`) и `submission.csv`.

### Навыки и технологии
- **Пайплайны и препроцессинг**: `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`.
- **Модели**: линейные модели (Ridge/Lasso/ElasticNet), `LightGBM`, `CatBoost`, `LogisticRegression`.
- **Текстовый ML**: `TfidfVectorizer`, объединение полей (`title` + `text`).
- **Валидация и метрики**: `train_test_split`, `KFold/StratifiedKFold`, `cross_val_score`, `RMSE`, `Accuracy`, `AUC-ROC`, `classification_report`.
- **Тюнинг гиперпараметров**: `RandomizedSearchCV` (sklearn), `Optuna` (Bayesian/TPESampler).
- **Практики воспроизводимости**: `requirements.txt`, фиксированный `random_state`, экспорт сабмитов и моделей (`joblib`).
- **Визуализация**: `matplotlib`, `seaborn`.

### Требования
- Python 3.10+
- Windows PowerShell (команды ниже даны под Windows; на macOS/Linux — эквиваленты с `python3`/`source`)

Установите зависимости:
```bash
py -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Данные
- `1 lab` (House Prices): положите `house-prices-hw.zip` в корень проекта. Скрипт сам распакует в `./data` и ожидает файлы `train_hw.csv`, `test_hw.csv` внутри архива.
- `2 lab` (Mental Health): положите `mental-health-prediction.zip` в корень проекта. Скрипт распакует в `./data` и ожидает `train.csv`, `test.csv`.
- `3 lab` (Text): поместите `./data/train.csv` и `./data/test.csv` со столбцами: `id`, `title`, `text`, `label` (для train).

### Быстрый запуск
- Запуск линейных моделей (House Prices):
```bash
py "1 lab/linear.py"
```

- Продвинутый пайплайн c LightGBM (House Prices):
```bash
py "1 lab/h2.py"
```

- Классификация ментального здоровья (LightGBM + CatBoost + Optuna):
```bash
py "2 lab/hp.py"
```

- TF‑IDF классификация текстов:
```bash
py "3 lab/tf-idf.py"
```

Артефакты и результаты:
- `submission.csv` в соответствующей рабочей директории скрипта
- `3 lab/results/` — сохранённые `logistic_regression_model.pkl` и `vectorizer.pkl`

### Запуск ноутбуков (опционально)
```bash
jupyter notebook
# Откройте: "1 lab/hp.ipynb" или "2 lab/mh.ipynb"
```

### Примечания
- В скриптах предусмотрена лог‑трансформация целевой переменной для устойчивости к выбросам (House Prices).
- Для воспроизводимости зафиксированы `random_state`, используется кросс‑валидация и ранняя остановка (`LightGBM`).
- Тюнинг с `Optuna` может занимать существенное время; при необходимости уменьшите `n_trials`.


