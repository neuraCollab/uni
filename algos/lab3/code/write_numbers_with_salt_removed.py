def load_numbers_from_hashed_file(filename):
    numbers = []
    with open(filename, 'r') as file:
        for line in file:
            _, number = line.strip().split(":")
            numbers.append(number)
    return numbers

def write_numbers_with_salt_removed(numbers, salt, output_file):
    with open(output_file, 'w') as file:
        for number in numbers:
            number_with_salt_removed = str(int(number) - salt)
            file.write(f"{number_with_salt_removed}\n")

# Загрузка номеров из файла с хэшами и номерами
hashed_numbers = load_numbers_from_hashed_file("output.txt")

# Указание соли
salt = 32941766

# Запись номеров с учетом вычитания соли
write_numbers_with_salt_removed(hashed_numbers, salt, "output_with_salt_removed.txt")
