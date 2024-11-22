import hashlib
import bcrypt

def generate_hashes(input_file, salt):
    # Преобразуем соль в целое число
    salt_value = int(salt)

    # Чтение номеров телефонов из файла
    with open(input_file, 'r') as file:
        phone_numbers = [line.strip() for line in file.readlines()]

    # Генерация хэшей SHA-256 и SHA-1
    with open("hashed_sha256.txt", 'w') as sha256_file, open("hashed_sha1.txt", 'w') as sha1_file:
        for phone in phone_numbers:
            # Добавляем соль к номеру телефона алгебраически
            phone_with_salt = str(int(phone) + salt_value)

            # Генерация SHA-256
            sha256_hash = hashlib.sha3_256(phone_with_salt.encode()).hexdigest()
            sha256_file.write(f"{sha256_hash}\n")  # Запись SHA-256 хэша

            # Генерация SHA-1
            sha1_hash = hashlib.sha1(phone_with_salt.encode()).hexdigest()
            sha1_file.write(f"{sha1_hash}\n")  # Запись SHA-1 хэша

    # Генерация хэшей PBKDF2
    with open("hashed_ripemd160.txt", 'w') as ripemd160_file:
        for phone in phone_numbers:
            phone_with_salt = str(int(phone) + int(salt)).encode()
            ripemd160_hash = hashlib.new('ripemd160', phone_with_salt).hexdigest()
            ripemd160_file.write(f"{ripemd160_hash}\n")  # Запись RIPEMD-160 хэша

# Задаем входной файл и соль
input_file = "output_with_salt_removed.txt"
salt = "32941766"

# Запуск функции генерации хэшей
generate_hashes(input_file, salt)

print("Хэши успешно сгенерированы.")
