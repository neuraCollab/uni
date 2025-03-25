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

def find_common_salt(decrypted_hashes, known_phones):
    salt_counts = {}  # Словарь для хранения количества вхождений каждой возможной соли

    for known_phone in known_phones:
        for decrypted_phone in decrypted_hashes.values():
            try:
                # Вычисляем разницу между расшифрованным номером и известным телефоном
                salt = int(decrypted_phone) - int(known_phone)
                
                # Увеличиваем счётчик вхождений для этой соли
                if salt in salt_counts:
                    salt_counts[salt] += 1
                else:
                    salt_counts[salt] = 1
            except ValueError:
                continue
    # Ищем соль с наибольшим количеством совпадений
    most_common_salt = max(salt_counts, key=salt_counts.get)
    return most_common_salt, salt_counts[most_common_salt]

# Загрузка данных
decrypted_hashes = load_decrypted_hashes("output.txt")
known_phones = load_known_numbers("knows_numbers.txt")

# Поиск и вывод наиболее часто встречающейся соли
common_salt, occurrences = find_common_salt(decrypted_hashes, known_phones)
print(f"Наиболее часто встречающаяся соль: {common_salt} (встречается {occurrences} раз)")
