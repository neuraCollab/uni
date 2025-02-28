import configparser
import sys

# Имя конфигурационного файла
CONFIG_FILE = 'backup_config.ini'

# Функция для чтения и отображения текущей конфигурации
def display_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    print("Текущие параметры конфигурации:")
    for section in config.sections():
        print(f"[{section}]")
        for key, value in config.items(section):
            print(f"{key} = {value}")
        print()

# Функция для обновления конфигурационного параметра
def update_config(param, value):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    # Проверка на существование необходимых секций и параметров
    if not config.has_section('Settings'):
        print("Секция 'Settings' отсутствует в конфигурационном файле.")
        return

    # Обновляем параметр
    if param in config['Settings']:
        config['Settings'][param] = value
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        print(f"Параметр {param} успешно обновлён на {value}.")
    else:
        print(f"Параметр {param} не найден в конфигурационном файле.")

# Основная функция для обработки команд
def main():
    if len(sys.argv) < 2:
        print("Недостаточно аргументов. Используйте команды: display, set [param] [value].")
        return

    command = sys.argv[1]

    if command == 'display':
        display_config()
    elif command == 'set' and len(sys.argv) == 4:
        param = sys.argv[2]
        value = sys.argv[3]
        update_config(param, value)
    else:
        print("Некорректная команда. Используйте: display, set [param] [value].")

if __name__ == '__main__':
    main()

