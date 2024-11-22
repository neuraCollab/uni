import os
import subprocess
import time
import hashlib
import pandas as pd

# Конфигурация алгоритмов, типов соли и масок
config = [
    {"algorithm": "MD5", "salt_type": "Численный", "salt": "12345", "mask": "?d?d?d?d"},
    {"algorithm": "SHA1", "salt_type": "Численный", "salt": "12345", "mask": "?d?d?d?d?d"},
    {"algorithm": "SHA256", "salt_type": "Численный", "salt": "12345", "mask": "?d?d?d?d?d?d"},
    {"algorithm": "SHA512", "salt_type": "Численный", "salt": "12345", "mask": "?d?d?d?d?d?d?d"},
    {"algorithm": "SHA1", "salt_type": "Буквенный (1 символ)", "salt": "a", "mask": "?l?l?u?u"},  # Только буквы
    {"algorithm": "SHA512", "salt_type": "Буквенный (1 символ)", "salt": "a", "mask": "?l?u?l?l?u"},  # Только буквы
    {"algorithm": "SHA256", "salt_type": "Буквенный (1 символ)", "salt": "a", "mask": "?u?u?l?l"},  # Только буквы
    {"algorithm": "SHA1", "salt_type": "Комбинированная", "salt": "1a2b3c", "mask": "?a?a?a?a?a?a"},  # Любые символы
]

# Карта алгоритмов для hashcat
hashcat_algorithms = {
    "MD5": "0",
    "SHA1": "100",
    "SHA256": "1400",
    "SHA512": "1700",
}

# Функция для создания хэша с солью
def create_hash(algorithm, phone_number, salt):
    combined = phone_number + salt
    if algorithm == "MD5":
        return hashlib.md5(combined.encode()).hexdigest()
    elif algorithm == "SHA1":
        return hashlib.sha1(combined.encode()).hexdigest()
    elif algorithm == "SHA256":
        return hashlib.sha256(combined.encode()).hexdigest()
    elif algorithm == "SHA512":
        return hashlib.sha512(combined.encode()).hexdigest()
    else:
        raise ValueError(f"Алгоритм {algorithm} не поддерживается")

# Генерация данных и выполнение hashcat
def generate_table(input_file):
    data = []

    # Чтение номеров телефонов из файла
    with open(input_file, "r") as file:
        phone_numbers = [line.strip() for line in file.readlines()]

    for config_item in config:
        algorithm = config_item["algorithm"]
        salt_type = config_item["salt_type"]
        salt = config_item["salt"]
        mask = config_item["mask"]

        for phone_number in phone_numbers:
            # Генерация хэша
            hash_value = create_hash(algorithm, phone_number, salt)

            # Сохраняем хэш в файл
            hash_file = f"hash_{algorithm}_{phone_number}.txt"
            with open(hash_file, "w") as f:
                f.write(hash_value + "\n")

            # Команда hashcat для теста
            hashcat_mode = hashcat_algorithms[algorithm]
            command = [
                "hashcat",
                "-a", "3",  # Указание на использование маски
                "-m", hashcat_mode,
                hash_file,
                mask,
                "--quiet",
                "--potfile-disable",  # Отключение использования potfile для чистоты эксперимента
            ]

            # Измерение времени выполнения
            try:
                start_time = time.time()
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                end_time = time.time()

                if result.returncode == 0:
                    time_to_crack = f"{end_time - start_time:.2f} сек"
                else:
                    time_to_crack = "Не удалось взломать"
            except subprocess.TimeoutExpired:
                time_to_crack = "> 300 сек (таймаут)"

            # Удаляем временный файл
            os.remove(hash_file)

            # Добавляем данные в таблицу
            data.append({
                "Алгоритм шифрования": algorithm,
                "Вид Соли": salt_type,
                "Телефон": phone_number,
                "Маска": mask,
                "Время дешифровки": time_to_crack,
            })
    
    # Создание таблицы
    df = pd.DataFrame(data)
    print("\nТаблица результатов дешифровки:")
    print(df)
    df.to_csv("hashcat_results_with_mask.csv", index=False)

# Укажите путь к файлу с номерами телефонов
input_file = "output_with_salt_removed.txt"

# Запуск генерации таблицы
generate_table(input_file)
