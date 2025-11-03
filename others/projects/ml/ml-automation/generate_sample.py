# generate_sample.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)

n = 1000  # количество строк

df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.randint(30000, 200000, n),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
    'city': np.random.choice(['New York', 'LA', 'Chicago', 'Houston', 'Phoenix'], n),
    'score': np.random.randint(50, 100, n),
    'text': np.random.choice([
        'I love this product',
        'This is terrible',
        'Average experience',
        'Will buy again',
        'Not as expected',
    ], n)
})

# Генерируем целевую переменную с зависимостью от признаков
df['target'] = (
    (df['age'] > 35) |
    (df['income'] > 80000) |
    (df['education'].isin(['Master', 'PhD'])) |
    (df['score'] > 80)
).astype(int)

# Добавим немного шума
flip = np.random.rand(n) < 0.1  # 10% шума
df.loc[flip, 'target'] = 1 - df.loc[flip, 'target']

df.to_csv("data/sample.csv", index=False)
print(f"✅ Сгенерировано {n} строк в data/sample.csv")
# Файл просто генерирует примерные данные (включая колонку `text` для NLP)