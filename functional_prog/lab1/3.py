from functools import reduce

orders = [
    {"order_id": 1, "customer_id": 101, "amount": 150.0},
    {"order_id": 2, "customer_id": 102, "amount": 200.0},
    {"order_id": 3, "customer_id": 101, "amount": 75.0},
    {"order_id": 4, "customer_id": 103, "amount": 100.0},
    {"order_id": 5, "customer_id": 101, "amount": 50.0},
    {"order_id": 6, "customer_id": 104, "amount": 120.0},
    {"order_id": 7, "customer_id": 105, "amount": 90.0},
    {"order_id": 8, "customer_id": 106, "amount": 300.0},
    {"order_id": 9, "customer_id": 107, "amount": 250.0},
    {"order_id": 10, "customer_id": 104, "amount": 110.0},
    {"order_id": 11, "customer_id": 108, "amount": 130.0},
    {"order_id": 12, "customer_id": 109, "amount": 75.0},
    {"order_id": 13, "customer_id": 110, "amount": 220.0},
    {"order_id": 14, "customer_id": 105, "amount": 85.0},
    {"order_id": 15, "customer_id": 102, "amount": 90.0},
    {"order_id": 16, "customer_id": 101, "amount": 130.0},
    {"order_id": 17, "customer_id": 111, "amount": 95.0},
    {"order_id": 18, "customer_id": 103, "amount": 80.0},
    {"order_id": 19, "customer_id": 106, "amount": 150.0},
    {"order_id": 20, "customer_id": 110, "amount": 175.0},
    {"order_id": 21, "customer_id": 101, "amount": 50.0},
    {"order_id": 22, "customer_id": 112, "amount": 250.0},
    {"order_id": 23, "customer_id": 113, "amount": 180.0},
    {"order_id": 24, "customer_id": 114, "amount": 70.0},
    {"order_id": 25, "customer_id": 115, "amount": 90.0},
    {"order_id": 26, "customer_id": 116, "amount": 130.0},
    {"order_id": 27, "customer_id": 112, "amount": 120.0},
    {"order_id": 28, "customer_id": 117, "amount": 95.0},
    {"order_id": 29, "customer_id": 105, "amount": 160.0},
    {"order_id": 30, "customer_id": 107, "amount": 220.0},
    {"order_id": 31, "customer_id": 108, "amount": 70.0},
    {"order_id": 32, "customer_id": 103, "amount": 110.0},
    {"order_id": 33, "customer_id": 106, "amount": 80.0},
    {"order_id": 34, "customer_id": 109, "amount": 200.0},
    {"order_id": 35, "customer_id": 101, "amount": 90.0},
    {"order_id": 36, "customer_id": 113, "amount": 130.0},
    {"order_id": 37, "customer_id": 115, "amount": 50.0},
    {"order_id": 38, "customer_id": 118, "amount": 140.0},
    {"order_id": 39, "customer_id": 104, "amount": 110.0},
    {"order_id": 40, "customer_id": 102, "amount": 60.0},
    {"order_id": 41, "customer_id": 101, "amount": 120.0},
    {"order_id": 42, "customer_id": 110, "amount": 90.0},
    {"order_id": 43, "customer_id": 119, "amount": 180.0},
    {"order_id": 44, "customer_id": 105, "amount": 70.0},
    {"order_id": 45, "customer_id": 101, "amount": 130.0},
    {"order_id": 46, "customer_id": 112, "amount": 160.0},
    {"order_id": 47, "customer_id": 114, "amount": 90.0},
    {"order_id": 48, "customer_id": 116, "amount": 85.0},
    {"order_id": 49, "customer_id": 120, "amount": 300.0},
    {"order_id": 50, "customer_id": 121, "amount": 230.0}
]

# Функция для фильтрации заказов по customer_id с использованием filter
def filter_orders_by_customer(orders, customer_id):
    return list(filter(lambda order: order["customer_id"] == customer_id, orders))

# Функция для подсчета общей суммы заказов с использованием reduce
def calculate_total_orders(orders):
    return reduce(lambda acc, order: acc + order["amount"], orders, 0)

# Функция для подсчета средней стоимости заказов
def calculate_average_order(orders):
    total_amount = calculate_total_orders(orders)
    return total_amount / len(orders) if orders else 0

# Запрос customer_id от пользователя
customer_id_input = input("Введите идентификатор клиента (customer_id): ")

# Преобразование customer_id в целое число
customer_id = int(customer_id_input)

# Фильтрация заказов по клиенту
filtered_orders = filter_orders_by_customer(orders, customer_id)

# Подсчет общей суммы заказов для клиента
total_amount = calculate_total_orders(filtered_orders)

# Подсчет средней стоимости заказов для клиента
average_order_amount = calculate_average_order(filtered_orders)

# Вывод результатов
print(f"\nЗаказы клиента с ID {customer_id}:")
for order in filtered_orders:
    print(f"ID заказа: {order['order_id']}, Сумма заказа: {order['amount']}")

print(f"\nОбщая сумма заказов: {total_amount}")
print(f"Средняя стоимость заказов: {average_order_amount}")
