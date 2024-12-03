
from functools import reduce


users = [
    {"name": "Alice", "expenses": [100, 50, 75, 200]},
    {"name": "Bob", "expenses": [50, 75, 80, 100]},
    {"name": "Charlie", "expenses": [200, 300, 50, 150]},
    {"name": "David", "expenses": [100, 200, 300, 400]},
    {"name": "Eve", "expenses": [120, 80, 60, 140]},
    {"name": "Frank", "expenses": [90, 130, 120, 150]},
    {"name": "Grace", "expenses": [110, 70, 95, 85]},
    {"name": "Hannah", "expenses": [200, 150, 180, 130]},
    {"name": "Ian", "expenses": [60, 90, 70, 110]},
    {"name": "Jack", "expenses": [140, 160, 180, 200]},
    {"name": "Karen", "expenses": [70, 80, 100, 90]},
    {"name": "Leo", "expenses": [95, 110, 85, 100]},
    {"name": "Megan", "expenses": [140, 120, 160, 180]},
    {"name": "Nathan", "expenses": [80, 100, 120, 150]},
    {"name": "Olivia", "expenses": [300, 250, 200, 150]},
    {"name": "Paul", "expenses": [85, 95, 90, 100]},
    {"name": "Quincy", "expenses": [150, 170, 130, 120]},
    {"name": "Rachel", "expenses": [120, 110, 105, 115]},
    {"name": "Sam", "expenses": [100, 110, 90, 130]},
    {"name": "Tina", "expenses": [160, 140, 150, 130]},
    {"name": "Umar", "expenses": [100, 90, 80, 110]},
    {"name": "Vera", "expenses": [110, 130, 120, 150]},
    {"name": "Wendy", "expenses": [85, 75, 80, 95]},
    {"name": "Xavier", "expenses": [150, 160, 170, 140]},
    {"name": "Yara", "expenses": [180, 200, 220, 210]},
    {"name": "Zach", "expenses": [90, 95, 85, 100]},
    {"name": "Aiden", "expenses": [130, 120, 110, 150]},
    {"name": "Bella", "expenses": [140, 130, 120, 110]},
    {"name": "Carter", "expenses": [160, 170, 150, 180]},
    {"name": "Daisy", "expenses": [170, 160, 180, 190]}
]

# Функция для вычисления общей суммы расходов пользователя с использованием map
def calculate_total_expenses(user):
    return sum(user["expenses"])

# Фильтрация пользователей по минимальной и максимальной сумме расходов с использованием filter
def filter_users(users, min_expenses=None, max_expenses=None):
    return list(filter(
        lambda user: (min_expenses is None or calculate_total_expenses(user) >= min_expenses) and
                     (max_expenses is None or calculate_total_expenses(user) <= max_expenses),
        users
    ))

# Вычисление общей суммы расходов всех отфильтрованных пользователей с использованием reduce
def total_expenses_of_users(filtered_users):
    return reduce(lambda acc, user: acc + calculate_total_expenses(user), filtered_users, 0)

# Запрос минимальной и максимальной суммы расходов у пользователя
min_exp_input = input("Введите минимальную сумму расходов для фильтрации (или нажмите Enter для пропуска): ")
max_exp_input = input("Введите максимальную сумму расходов для фильтрации (или нажмите Enter для пропуска): ")

# Преобразование введенных данных в числа, если они введены
min_expenses = int(min_exp_input) if min_exp_input.isdigit() else None
max_expenses = int(max_exp_input) if max_exp_input.isdigit() else None

# Фильтрация пользователей по введенным критериям
filtered_users = filter_users(users, min_expenses, max_expenses)

# Рассчет общей суммы расходов для отфильтрованных пользователей
overall_total_expenses = total_expenses_of_users(filtered_users)

# Вывод результатов
print("\nОтфильтрованные пользователи:")
for user in filtered_users:
    total_expenses = calculate_total_expenses(user)
    print(f"Имя: {user['name']}, Общая сумма расходов: {total_expenses}")

print(f"\nОбщая сумма расходов всех отфильтрованных пользователей: {overall_total_expenses}")
