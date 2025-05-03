# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
train_data, test_data_with_survived = train_test_split(pd.read_csv("./train.csv"), test_size=0.3, random_state=42) 
# test_data = test_data_with_survived.drop("Survived", axis=1)
test_data = pd.read_csv("./test.csv")


# %%
# Analyze
(train_data.info())
(train_data.isna().sum())
(train_data.describe())
(train_data.head())

# %% 
# тут должно быть масштабирование данных (хотя для этого датасета оно и не нужно)
# потом можно использовать Метод главных компонент ( совокупная объясненная дисперсия от смотрим график зависимости количества компонентов) 
#  далее птимизация гиперпараметров. RandomizedSearchCV. Состовляем всевозможные гиперпараметры и выбираем наилучшие. Пример см. image.png
#  потмле RSCV (ОБЯЗАТЕЛЬНО ПОСЛЕ) используем Оптимизация гиперпараметров GridSearchCV с подобранными диапозонами
ss = StandardScaler()

# %%

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)

print('% survived women:', rate_women)
# %%
man = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_man = sum(man) / len(man)

print('% survived man:', rate_man)

# %%
Y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
# %%
model = RandomForestClassifier(n_estimators=700, max_depth=5, random_state=1, min_samples_leaf=7, bootstrap=False, max_features="sqrt", min_samples_split=23)
model.fit(X, Y)
predictions = model.predict(X_test)

# %%
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.head()
output.to_csv('submission.csv', index=False)
# %%

# print(sum(predictions == test_data_with_survived["Survived"])/len(test_data_with_survived['Survived']))
# %%
