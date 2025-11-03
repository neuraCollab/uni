from functools import reduce

users = [
    {"name": "Alice", "expenses": [100, 50, 75, 200]},
    {"name": "Bob", "expenses": [50, 75, 80, 100]},
    {"name": "Charlie", "expenses": [200, 300, 50, 150]},
    {"name": "David", "expenses": [100, 200, 300, 400]},
    {"name": "Eve", "expenses": [150, 60, 90, 120]},
    {"name": "Frank", "expenses": [80, 200, 150, 100]},
    {"name": "Grace", "expenses": [300, 400, 250, 500]},
    {"name": "Hannah", "expenses": [120, 60, 70, 90]},
    {"name": "Ivy", "expenses": [500, 300, 250, 150]},
    {"name": "Jack", "expenses": [75, 100, 50, 150]},
    {"name": "Kim", "expenses": [100, 90, 80, 60]},
    {"name": "Liam", "expenses": [120, 200, 180, 160]},
    {"name": "Mia", "expenses": [400, 500, 450, 300]},
    {"name": "Noah", "expenses": [90, 80, 100, 150]},
    {"name": "Olivia", "expenses": [60, 90, 110, 140]},
    {"name": "Paul", "expenses": [200, 300, 150, 100]},
    {"name": "Quincy", "expenses": [100, 120, 130, 140]},
    {"name": "Rachel", "expenses": [90, 60, 50, 80]},
    {"name": "Sam", "expenses": [500, 400, 350, 600]},
    {"name": "Tina", "expenses": [150, 200, 300, 250]},
]

# Используем map для добавления поля total_expenses каждому пользователю
users_with_total_expenses = list(map(lambda user: {'name': user['name'], 'expenses': user['expenses'], 'total_expenses': sum(user['expenses']) }, users))

# Фильтрация пользователей с суммарными расходами >= 750 через filter
filtered_users = list(filter(lambda user: user['total_expenses'] >= 750, users_with_total_expenses))

total_expenses_filtered = reduce(lambda acc, user: acc + user['total_expenses'], filtered_users, 0)

print("Фильтрованные пользователи:")
for user in filtered_users:
    print(user)

print("\nОбщая сумма расходов каждого пользователя:")
for user in users_with_total_expenses:
    print((user['name'], user['total_expenses']))

print("\nОбщая сумма расходов всех отфильтрованных пользователей:", total_expenses_filtered)
