## Linux Labs — демонстрации: бэкап-демон, мониторинг сети и процессов

### Если вы работодатель, можете скипнуть, тут ничего полезного.

Сборник из трёх лабораторных работ по Linux, демонстрирующих системное программирование, работу с сервисами, сетевой анализ и мониторинг ОС на Python. Проекты показывают навыки работы с системными инструментами Linux, `systemd`, обработкой пакетов, процессами, графическими интерфейсами и конфигурациями.

### Что я умею (ключевые навыки)
- **Демонизация и автоматизация**: периодическое резервное копирование с логированием, конфигурацией через INI и примером юнита `systemd`.
- **Управление конфигурацией**: CLI‑утилита для просмотра/изменения параметров `backup_config.ini`.
- **Сеть и безопасность**: живой сниффинг и простые эвристики выявления активности на Scapy; блокировка/разблокировка IP через `iptables` из Tkinter‑GUI.
- **Интроспекция ОС**: мониторинг процессов в реальном времени с `psutil`, логированием и визуализацией временных рядов в Matplotlib (Tkinter‑дашборд).
- **Параллелизм и UX**: фоновые потоки для «реактивных» GUI без подвисаний.

### Техстек
- Python 3.10+
- Scapy (захват пакетов), `iptables` (межсетевой экран Linux)
- Tkinter (настольный GUI), Matplotlib (графики)
- psutil (процессы), logging, configparser
- Рекомендуется Linux/WSL2 (используются `/proc`, сигналы, `iptables`, `systemd`)

### Структура репозитория
- `1 lab/`
  - `code/backup_daemon.py` — демон резервного копирования: читает `backup_config.ini`, пишет логи, корректно обрабатывает сигналы.
  - `code/backup_config_control.py` — CLI для просмотра/обновления INI‑конфигурации.
  - `code/backup_daemon.service` — пример юнита `systemd`.
  - `code/backup_config.ini` — пример конфигурации.
  - `code/source_dir/` — образцы файлов для бэкапа.
- `2 lab/`
  - `code/checker.py` — Tkinter‑GUI + Scapy: сниффер, выявление подозрительных IP и управление `iptables`.
  - `code/cmds.txt` — пример последовательности команд (checker/sender/block).
  - `code/README.md` — краткое описание модуля.
- `3 lab/`
  - `code/main.py` — Tkinter‑дашборд: логирование метрик процессов и графики (running/sleeping/zombie).
  - `code/man.py`, `code/test.py`, `code/tests.py` — связанные примеры/эксперименты.

## Быстрый старт
Лабы используют Linux‑специфику. Запуск на Linux или Windows 11 с WSL2 Ubuntu. Для захвата пакетов и изменения фаервола требуются права root.

### 0) Предварительные требования
- Python 3.10+
- Linux/WSL2 с `pip`, `venv`, `iptables`, `libc` (glibc) и поддержкой GUI (X‑server при необходимости в WSL)

### 1) Виртуальное окружение
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Зависимости
```bash
pip install scapy psutil matplotlib
```
Tkinter обычно идёт с системным Python. Если отсутствует — установите через дистрибутив (например, `sudo apt-get install python3-tk`).

---

## Лаба 1 — Демон резервного копирования
- Периодически копирует `source_dir` в подкаталоги с таймстампом внутри `backup_dir` и пишет лог.
- Настраивается через `backup_config.ini`. Пути и интервал задаются параметрами.

Важно: В образце `backup_config.ini` указаны абсолютные пути из другой среды. Обновите пути под вашу систему перед запуском.

### Настройка
```bash
cd "1 lab/code"
# Показать текущую конфигурацию
python backup_config_control.py display

# Установить параметры (примеры)
python backup_config_control.py set source_dir /path/to/source
python backup_config_control.py set backup_dir /path/to/backup
python backup_config_control.py set backup_interval 60
python backup_config_control.py set log_file /path/to/backup_daemon.log
```

### Запуск (в терминале)
```bash
python backup_daemon.py
```
При необходимости отредактируйте путь к `config_file` в конце `backup_daemon.py` (сейчас там жёстко прописан путь). Альтернатива — добавить чтение пути из переменной окружения/аргумента.

### Запуск как сервис systemd (Linux)
```bash
# Подправьте пути внутри backup_daemon.service под вашу систему
sudo cp backup_daemon.service /etc/systemd/system/backup_daemon.service
sudo systemctl daemon-reload
sudo systemctl enable --now backup_daemon.service
sudo systemctl status backup_daemon.service
```

---

## Лаба 2 — Мониторинг сетевого трафика (Scapy + Tkinter)
- Захватывает пакеты и помечает подозрительные шаблоны (напр., SYN‑сканы, запрещённые порты).
- Отображает три списка: все пакеты, обнаруженные IP и заблокированные IP.
- Позволяет блокировать/разблокировать IP через `iptables` прямо из GUI.

### Запуск
```bash
cd "2 lab/code"
sudo -E python checker.py
```
Примечания:
- Требуются права root для Scapy и изменений `iptables`.
- В WSL2 возможности захвата пакетов/`iptables` ограничены — лучше использовать полноценный Linux.

---

## Лаба 3 — Дашборд мониторинга процессов
- Показывает процессы с временем CPU, памятью и пользователем; пишет логи в `system_audit.log`.
- Считает число процессов в состояниях running/sleeping/zombie и строит графики по времени.
- Использует потоки, чтобы интерфейс оставался отзывчивым при сборе данных.

### Запуск
```bash
cd "3 lab/code"
python main.py
```
Если не установлен `python3-tk`:
```bash
sudo apt-get update && sudo apt-get install -y python3-tk
```

---

### Компетенции, которые демонстрируются
- Системное программирование Linux: сигналы, `/proc`, `ptrace` (см. `tests.py`), `iptables`, `systemd`.
- Практичный Python: `configparser`, `logging`, `threading`, GUI на Tkinter.
- Сетевой анализ и основы безопасности: Scapy, простые эвристики подозрительного трафика, активная блокировка.
- Наблюдаемость: структурированные логи, живые дашборды, визуализация временных рядов.

### Заметки и возможные улучшения
- Убрать жёсткий путь к конфигу в `backup_daemon.py` (переменная окружения/CLI‑флаг).
- Добавить тесты и тайпинги; оформить каждую лабу как пакет с entry‑point.
- Где возможно — контейнеризировать для упрощения развёртывания.
- Вместо прямых вызовов `iptables` рассмотреть Netfilter Queue для более тонкого контроля.
