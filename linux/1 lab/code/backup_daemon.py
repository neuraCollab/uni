import os
import time
import shutil
import logging
import configparser
from datetime import datetime
import signal
import sys

# Загрузка конфигурации из файла
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return {
        'source_dir': config.get('Settings', 'source_dir'),
        'backup_dir': config.get('Settings', 'backup_dir'),
        'backup_interval': config.getint('Settings', 'backup_interval'),
        'log_file': config.get('Settings', 'log_file'),
    }

# Настройка логирования
def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Функция создания резервной копии
def backup_data(source_dir, backup_dir):
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    timestamp = datetime.now().strftime('%H.%M.%S %d.%m.%Y')
    backup_subdir = os.path.join(backup_dir, f"{timestamp}")
    shutil.copytree(source_dir, backup_subdir)
    logging.info(f"Backup created: {backup_subdir}")

# Функция обработки сигналов для остановки демона
def handle_signal(signum, frame):
    logging.info("Daemon stopping...")
    sys.exit(0)

# Главная функция демона
def run_daemon(config_file):
    config = load_config(config_file)
    setup_logging(config['log_file'])

    logging.info("Daemon started...")

    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while True:
        backup_data(config['source_dir'], config['backup_dir'])
        time.sleep(config['backup_interval'])

if __name__ == "__main__":
    config_file = '/mnt/c/Users/Lenovo/OneDrive/Рабочий стол/jas/linux/Laboratory_1/backup_config.ini'
    run_daemon(config_file)

