# Алгоритмы и проекты  

Короткое портфолио по учебным работам: классические алгоритмы на C++, проекты по анализу данных и визуализации на Python (Tkinter, Streamlit), эксперименты с кластеризацией, отбором признаков и метриками качества. Репозиторий структурирован по семестрам; в каждой папке — самостоятельные мини‑проекты.

## Ключевые навыки
- Алгоритмическое мышление: сортировки, слияние, инверсии, радикс, Shell/Pratt, стек/очереди
- Языки: C++ (STL), Python (NumPy, pandas, scikit‑learn, matplotlib)
- Кластеризация: CURE, FOREL, ISODATA, Single Linkage, Max‑Min Distance; метрики Rand/Jaccard/Fowlkes‑Mallows/Φ, compactness, separation
- Отбор признаков: compactness, spread, add/del, SPA
- GUI и визуализация: Tkinter (настольное приложение TSP), Streamlit (интерактивная аналитика)
- Организация кода: модульная архитектура, конфигурации YAML, воспроизводимые пайплайны

## Структура
- `1_sem/` — классические алгоритмы на C++ (сортировки, инверсии, radix, т.д.)
- `2_sem/` — ноутбуки Jupyter по алгоритмам/аналитике
- `3 sem/` — криптография/хеширование (скрипты), ноутбуки, отчёты
- `4 sem/1 lab` — шаблон для задачи TSP и графов (Python)
- `4 sem/3 lab` — настольное приложение TSP на Tkinter: `code/main.py` (вход)
- `4 sem/4 lab` — обработка пропусков/оценка моделей/GUI (Python)
- `4 sem/5 lab` — проект кластеризации с пайплайном и Streamlit‑приложением: `code/clustering_project/`

## Быстрый старт
Ниже — быстрые команды для запуска наиболее показательных проектов. Команды даны для Windows PowerShell. Для Python проектов рекомендуется отдельное виртуальное окружение.

### 1) Streamlit‑приложение «Сравнение алгоритмов кластеризации»
Расположение: `4 sem/5 lab/code/clustering_project/`

Зависимости указаны в `requirements.txt`:
```
cd "4 sem/5 lab/code/clustering_project"
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
Запуск Streamlit:
```
streamlit run app_streamlit.py
```
По умолчанию приложение открывается в браузере. В боковой панели можно выбрать данные (`./data/iris.csv`), число кластеров, метрики качества, методы отбора признаков и параметры алгоритмов (CURE/FOREL/ISODATA/SingleLinkage/Max‑Min).

### 2) Настольное приложение TSP (Tkinter)
Расположение: `4 sem/3 lab/code/`
```
cd "4 sem/3 lab/code"
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r ..\..\requirements.txt  # если есть общий список; иначе без этой строки
python main.py
```
Откроется окно Tkinter. Внутри используются алгоритмы из `code/algorithms/` (ближайший сосед, имитация отжига, муравьиный алгоритм и др.).

### 3) Классические алгоритмы на C++
Расположение: `1_sem/`

Пример компиляции и запуска (необходим MinGW или LLVM):
```
# Counting Sort
cd "1_sem/1"
g++ -std=c++17 counting_sort.cpp -O2 -o counting_sort.exe
./counting_sort.exe

# Merge Sort
cd "..\3\merge-sort"
g++ -std=c++17 merge_sort.cpp -O2 -o merge_sort.exe
./merge_sort.exe

# Radix Sort, Shell/Pratt, инверсии и т.п. по аналогии
```
Если есть входные файлы (например, `sum`, `line`, `radix sort`), запускайте бинарник рядом с ожидаемыми входами.

### 4) Ноутбуки (Jupyter)
Расположение: `2_sem/`, `3 sem/`
```
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install jupyter numpy pandas matplotlib scikit-learn
jupyter lab
```
Далее открывайте соответствующие `*.ipynb`.

## Что посмотреть работодателю
- `4 sem/5 lab/code/clustering_project/` — законченная архитектура пайплайна кластеризации
  - `src/clustering/*.py` — реализации алгоритмов (CURE, FOREL, ISODATA, и др.)
  - `src/metrics/*.py` — внутренние/внешние метрики качества
  - `src/features/*.py` — методы отбора признаков
  - `src/pipeline.py` — воспроизводимый сценарий с конфигами YAML
  - `app_streamlit.py` — интерактивный UI для экспериментов
- `4 sem/3 lab/code/` — TSP GUI на Tkinter + набор эвристик
- `1_sem/` — реализация базовых алгоритмов и структур данных на C++

## Технологический стек
- C++17 (STL)
- Python 3.10+ (NumPy, pandas, scikit‑learn, matplotlib, PyYAML)
- Streamlit (интерактивные панели)
- Tkinter (настольные GUI)
- Jupyter (исследования), YAML‑конфиги

## Требования
- Windows 10/11, PowerShell
- Python 3.10+; для C++ — компилятор `g++` (MinGW) или `clang++`
- Для Streamlit‑аппа: зависимости из `4 sem/5 lab/code/clustering_project/requirements.txt`

