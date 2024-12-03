import os
import time
import json
import logging
import shutil
import daemon
from datetime import datetime

# Настройка логирования
def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Чтение конфигурации
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# Создание резервной копии
def create_backup(source, backup):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup, f"backup_{timestamp}")
    shutil.copytree(source, backup_path)
    logging.info(f"Backup created at: {backup_path}")

# Основной процесс демона
def backup_process(config):
    setup_logging(config['log_file'])
    
    while True:
        create_backup(config['source_directory'], config['backup_directory'])
        time.sleep(config['backup_interval'])

if __name__ == "__main__":
    config = load_config('/home/bubu/Documents/uni/functional_prog/lab2/backup_config.json')

    with daemon.DaemonContext():
        backup_process(config)
