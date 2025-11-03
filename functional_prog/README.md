## Functional Programming Labs (Python)

### Если вы работодатель, можете скипнуть, тут ничего полезного.

Набор учебных лабораторных работ по функциональному и параллельному программированию на Python. Проекты демонстрируют практические навыки: функциональные трансформации `map/filter/reduce`, параллельную обработку изображений (multiprocessing + OpenCV), асинхронный сетевой чат (asyncio + Tkinter), а также простую рекомендательную систему с GUI.

### Быстрый старт (Windows PowerShell)

1) Клонирование и окружение
```powershell
cd C:\Users\morro\prog\uni\functional_prog
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Установка зависимостей по лабам
- Лаба 2 (CV и Excel):
```powershell
pip install opencv-python numpy openpyxl
```

- Лаба 3 (чат):
```powershell
pip install async-tkinter-loop
```

3) Запуск по лабораторным

- Лаба 1 — функциональные примеры:
```powershell
python "1 lab\code\1.py"
python "1 lab\code\2.py"
python "1 lab\code\3.py"
```

- Лаба 2 — параллельная обработка изображений и экспорт в Excel:
```powershell
cd "2 lab\code"
python parallel_processing_space.py
```
В GUI выберите:
- Source Folder — папку с изображениями (есть пример `photo`),
- Destination — папку для сохранения `results.xlsx`.
Скрипт разрежет изображения, распараллелит обработку и сохранит аннотированные фрагменты в `processed_parts`.

- Лаба 3 — асинхронный чат (сначала сервер, затем клиент):
```powershell
cd "3 lab\code"
python server.py
```
В новом окне терминала:
```powershell
cd "3 lab\code"
python client.py
```
Параметры по умолчанию: адрес `127.0.0.2`, порт `8888`. Введите логин и комнату (например, `main`). Для теста можно запустить несколько клиентов.

- Лаба 4 — рекомендатель книг (GUI):
```powershell
cd "4 lab\code"
python main.py
```
В окне приложения введите авторов, жанры, ключевые слова и выберите сортировку — найденные книги появятся справа со сводной строкой.

### Технические детали по ключевым файлам
- `2 lab/code/parallel_processing_space.py`:
  - Разбиение изображений на блоки, параллельная обработка в `multiprocessing.Pool`, поиск контуров, классификация областей по площади/яркости, отрисовка окружностей, экспорт результатов в Excel.
- `3 lab/code/server.py` / `client.py`:
  - Сервер: очередь сообщений `asyncio.Queue`, рассылка по группам, ответы на команды `@help` и `@list`, поддержка личных адресатов вида `[User]`.
  - Клиент: интеграция `asyncio` с Tkinter через `async-tkinter-loop`, неблокирующий ввод/вывод, переключение экранов входа/чата.
- `4 lab/code/main.py`:
  - Скоринг книг по совпадению авторов/жанров/ключевых слов и сортировка по выбранному критерию, сбор данных из `books.json`.
