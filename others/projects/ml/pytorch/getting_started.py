import torch
import time

# Проверяем доступность CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Размеры матриц
matrix_size = 10000  # Достаточно большой размер для загрузки видеокарты
num_iterations = 500  # Количество повторов

# Создаем тензоры
A = torch.randn((matrix_size, matrix_size), device=device)
B = torch.randn((matrix_size, matrix_size), device=device)

# Функция для выполнения матричного умножения
def matrix_multiplication(A, B):
    return torch.mm(A, B)

# Прогрев (для стабильности измерений)
for _ in range(5):
    matrix_multiplication(A, B)

# Основной тест
start_time = time.time()

for i in range(num_iterations):
    result = matrix_multiplication(A, B)
    torch.cuda.synchronize()  # Синхронизируем устройство для точных замеров
    print(f"Iteration {i + 1}/{num_iterations} completed")

end_time = time.time()

# Результаты
total_time = end_time - start_time
average_time = total_time / num_iterations

print(f"\nTotal execution time: {total_time:.2f} seconds")
print(f"Average time per iteration: {average_time:.2f} seconds")
