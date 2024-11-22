import hashlib

def load_decrypted_hashes(filename):
    decrypted_hashes = {}
    with open(filename, 'r') as file:
        for line in file:
            hash_val, decrypted_phone = line.strip().split(":")
            decrypted_hashes[hash_val] = decrypted_phone
    return decrypted_hashes

def load_known_numbers(filename):
    known_numbers = []
    with open(filename, 'r') as file:
        for line in file:
            known_numbers.append(line.strip())
    return known_numbers

def hash_with_salt(phone, salt):
    phone_with_salt = str(int(phone) + salt)
    return hashlib.md5(phone_with_salt.encode()).hexdigest()

def verify_salt(salt, known_numbers, decrypted_hashes):
    for phone in known_numbers:
        hashed_phone = hash_with_salt(phone, salt)
        
        # Проверяем, есть ли хэш в расшифрованных данных
        if hashed_phone in decrypted_hashes:
            print(f"Совпадение найдено для номера {phone} с хэшем {hashed_phone}")
        else:
            print(f"Нет совпадения для номера {phone} с хэшем {hashed_phone}")

# Загрузка данных
decrypted_hashes = load_decrypted_hashes("output.txt")
known_numbers = load_known_numbers("known_numbers.txt")

# Вставьте здесь найденную соль
found_salt = 32941766  # Пример значения соли

# Проверка
verify_salt(found_salt, known_numbers, decrypted_hashes)
