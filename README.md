## «Большие данные и распределённая цифровая платформа» (2024–2027)

 Фокус на практических решениях: ML пайплайны, алгоритмы и визуализация, системное программирование под Linux, функциональные и сетевые приложения на Python.



---

## Структура репозитория
- `AI/` — ML‑лабы: конкурсные задачи, текстовая классификация, тюнинг (LightGBM, CatBoost)
- `algos/` — алгоритмы и проекты: C++ классика + Python GUI/Streamlit (TSP, кластеризация)
- `functional_prog/` — функциональное и параллельное программирование, асинхронный чат, рекомендательная система
- `linux/` — системное программирование: демон бэкапа (systemd), мониторинг сети/процессов


---

## Что смотреть в первую очередь (для hr/developers)
- `algos/4 sem/5 lab/code/clustering_project/` — законченное Streamlit‑приложение с пайплайном и метриками
- `algos/4 sem/3 lab/code/` — TSP GUI с эвристиками и структурой модулей
- `AI/` — полный цикл ML: от препроцессинга до сабмитов и моделей
- `linux/1 lab` и `linux/2 lab` — работа с `systemd`, `scapy`, `iptables`, GUI

Если нужна быстрая демонстрация — напишите, пришлю скринкасты/демо. (контакты внизу страницы)

---

## Выделенные проекты (быстрый просмотр)

### 1) AI — учебные ML‑проекты
Расположение: `AI/`
- Линейные модели для House Prices, LightGBM с пайплайном и лог‑трансформацией
- Бинарная классификация (CatBoost/LightGBM) с тюнингом в Optuna
- Текстовая классификация (TF‑IDF + LogisticRegression), сохранение артефактов `joblib`

Быстрый старт (Windows PowerShell):
```powershell
cd AI
py -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Примеры запусков
py "1 lab/linear.py"
py "1 lab/h2.py"
py "2 lab/hp.py"
py "3 lab/tf-idf.py"
```

### 2) Algos — TSP GUI и Streamlit‑кластерайзер
Расположение: `algos/4 sem/`
- `3 lab/code/` — настольный TSP на Tkinter с набором эвристик
- `5 lab/code/clustering_project/` — Streamlit‑приложение для сравнения алгоритмов кластеризации (CURE, FOREL, ISODATA и др.), метрики Rand/Jaccard/F‑M, YAML‑конфиги

Запуск Streamlit‑проекта:
```powershell
cd "algos/4 sem/5 lab/code/clustering_project"
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
streamlit run app_streamlit.py
```

### 3) Linux — демон бэкапа и мониторинг
Расположение: `linux/`
- `1 lab/code/backup_daemon.py` + `backup_daemon.service` — пример systemd‑сервиса, конфиг через INI
- `2 lab/code/checker.py` — мониторинг сети (Scapy + Tkinter) с блокировкой IP через `iptables`
- `3 lab/code/main.py` — дашборд процессов (psutil + Matplotlib)

Запуск демона (локально):
```bash
cd "linux/1 lab/code"
python backup_config_control.py display       # посмотреть конфиг
python backup_daemon.py                       # запустить демон
```

Установка как systemd‑сервис (Linux):
```bash
sudo cp backup_daemon.service /etc/systemd/system/backup_daemon.service
sudo systemctl daemon-reload
sudo systemctl enable --now backup_daemon.service
sudo systemctl status backup_daemon.service
```

Запуск сетевого чекера (требует root):
```bash
cd "linux/2 lab/code"
sudo -E python checker.py
```

---

## Технологии
- Python 3.10+: NumPy, pandas, scikit‑learn, LightGBM, CatBoost, Optuna, matplotlib, seaborn, psutil, scapy, Tkinter
- C++17 (STL)
- Streamlit, Tkinter GUI, YAML‑конфиги

---

## Как работать с репозиторием
Клонирование полностью:
```bash
git clone https://github.com/neuraCollab/uni.git
```

Выборочное клонирование папки (sparse‑checkout), пример для `algos`:
```bash
git init uni && cd uni
git remote add -f origin https://github.com/neuraCollab/uni.git
git sparse-checkout init --cone
git sparse-checkout set algos
git pull origin main
```

Рекомендации по окружению:
- Для каждого Python‑подпроекта создавайте отдельный `venv`
- В Windows используйте PowerShell; в Linux/macOS заменяйте на `python3`/`source`

---

## Контакты: 
- telegram `@vbjgfc`
- email mihialpersonalemai@gmail.com (редко читаю)
