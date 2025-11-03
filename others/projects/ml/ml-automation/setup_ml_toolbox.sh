#!/bin/bash

echo "🚀 Создание toolbox..."
toolbox create ml-olympiad 2>/dev/null || echo "Toolbox уже существует"

echo "📥 Вход в toolbox..."
toolbox run ml-olympiad bash <<'EOF'

    echo "🔧 Установка системных зависимостей..."
    sudo dnf update -y
    sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ redhat-rpm-config libjpeg-devel zlib-devel openssl-devel libffi-devel git make curl which htop nano vim

    echo "🌸 Установка Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

    echo "📁 Создание проекта..."
    mkdir -p ~/ml-project && cd ~/ml-project
    poetry init -n
    poetry env use python3

    echo "🧩 Установка Python-зависимостей..."
    poetry add numpy pandas scikit-learn matplotlib seaborn jupyter notebook ipykernel
    poetry add torch torchvision torchaudio --source pytorch -E cpu
    poetry add tensorflow
    poetry add transformers datasets accelerate sentence-transformers
    poetry add pycaret h2o optuna flaml autofeat
    poetry add xgboost lightgbm catboost shap eli5 scikit-optimize
    poetry add --group dev jupyterlab black flake8 pytest

    echo "🔧 Установка fallback-пакетов через pip..."
    poetry run pip install autogluon tabular tpot ydata-profiling

    echo "🧪 Установка зависимостей..."
    poetry install

    echo "🧠 Настройка Jupyter ядра..."
    poetry run python -m ipykernel install --user --name=ml-olympiad-poetry

    echo "✅ ВСЁ ГОТОВО! Активируй: cd ~/ml-project && poetry shell"
EOF
