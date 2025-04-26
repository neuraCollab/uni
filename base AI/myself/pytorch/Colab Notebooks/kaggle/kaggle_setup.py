import os
import subprocess
from google.colab import files

# Загрузите ваш kaggle.json файл
uploaded = files.upload()

# Создайте kaggle директорию
os.makedirs('/root/.kaggle', exist_ok=True)

# Переместите kaggle.json в правильное место
subprocess.run(["mv", "kaggle.json", "/root/.kaggle/"])

# Установите права доступа
subprocess.run(["chmod", "600", "/root/.kaggle/kaggle.json"])

# Установите kaggle API
subprocess.run(["pip", "install", "kaggle"])
