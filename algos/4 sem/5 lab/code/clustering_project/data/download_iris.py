from sklearn.datasets import load_iris
import pandas as pd

# Загрузить Iris датасет
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Сохранить как iris.csv
df.to_csv('iris.csv', index=False)
print("Файл iris.csv успешно сохранён!")