import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import joblib

output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv("./data/train.csv")
df = df.dropna(subset=["text", "label"])

df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["content"])
print(vectorizer.vocabulary_)
X_val = vectorizer.transform(val_df["content"])

y_train = train_df["label"]
y_val = val_df["label"]

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

joblib.dump(model, os.path.join(output_dir, 'logistic_regression_model.pkl'))
joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))

test_df = pd.read_csv("./data/test.csv")
test_df["content"] = test_df["title"] + " " + test_df["text"]

X_test = vectorizer.transform(test_df["content"])

predictions = model.predict(X_test)

submission_df = pd.DataFrame({
    "id": test_df["id"],
    "label": predictions
})

submission_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

print("Submission file created successfully.")
