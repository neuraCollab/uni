import random
import math
import time

def simulated_annealing(distances, initial_temp=100.0, cooling_rate=0.995, min_temp=0.01):
    """
    Решение задачи коммивояжера методом имитации отжига
    """
    start_time = time.time()
    n = len(distances)
    
    # Создаем начальное решение (случайная перестановка городов)
    current_path = list(range(n))
    random.shuffle(current_path)
    current_cost = calculate_path_cost(current_path, distances)
    
    # Лучшее найденное решение
    best_path = current_path[:]
    best_cost = current_cost
    
    # Текущая температура
    temp = initial_temp
    
    while temp > min_temp:
        # Генерируем нового соседа путем обмена двух случайных городов
        i, j = random.sample(range(n), 2)
        new_path = current_path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        new_cost = calculate_path_cost(new_path, distances)
        
        # Вычисляем разницу в стоимости
        delta = new_cost - current_cost
        
        # Принимаем новое решение с вероятностью e^(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_path = new_path
            current_cost = new_cost
            
            # Обновляем лучшее решение, если нашли лучше
            if current_cost < best_cost:
                best_path = current_path[:]
                best_cost = current_cost
        
        # Охлаждаем систему
        temp *= cooling_rate
    
    end_time = time.time()
    execution_time = end_time - start_time
    return best_path, best_cost, execution_time

def calculate_path_cost(path, distances):
    """
    Вычисляет общую стоимость маршрута
    """
    cost = 0
    n = len(path)
    for i in range(n):
        cost += distances[path[i]][path[(i + 1) % n]]
    return cost 