# %%
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from src.pipeline import ClusteringPipeline

# %%
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Загруженная конфигурация:")
for key, value in config.items():
    print(f"{key}: {value}")

# %%
pipeline = ClusteringPipeline(config)
pipeline.run()

# %%
df = pd.read_csv(config["data_path"])
X = df.iloc[:, :-1].values
y_true = df.iloc[:, -1].values

from src.utils import reduce_dimensions
X_reduced = reduce_dimensions(X, method="pca")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y_true, palette='Set1', s=100)
plt.title("Исходные данные (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="True Labels")
plt.show()

# %%
from sklearn.metrics.cluster import contingency_matrix

for result in pipeline.results:
    model_name = result["model"]
    labels_pred = result["labels"]

    matrix = contingency_matrix(y_true, labels_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(labels_pred),
                yticklabels=np.unique(y_true))
    plt.title(f"Матрица схожести — {model_name}")
    plt.xlabel("Предсказанные кластеры")
    plt.ylabel("Истинные классы")
    plt.show()

# %%
import pandas as pd

results_df = pd.DataFrame(pipeline.results)
results_df = results_df.drop(columns=["labels"])
print(results_df.to_string(index=False))

# %%
from IPython.display import display, HTML
html_report_path = os.path.join(config["output_dir"], "comparison_report.html")
display(HTML(filename=html_report_path))


