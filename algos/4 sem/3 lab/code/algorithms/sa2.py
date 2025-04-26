import random
import math
import time
from typing import List, Tuple

def solve_tsp_simulated_annealing(distances: List[List[float]], 
                                initial_temp: float = 100.0,
                                cooling_rate: float = 0.995,
                                min_temp: float = 0.01,
                                iterations_per_temp: int = 100) -> Tuple[List[int], float]:
    """
    Решает задачу коммивояжера методом имитации отжига.
    
    Args:
        distances: Матрица расстояний между городами
        initial_temp: Начальная температура
        cooling_rate: Коэффициент охлаждения (0 < cooling_rate < 1)
        min_temp: Минимальная температура
        iterations_per_temp: Количество итераций на каждой температуре
    
    Returns:
        Tuple[List[int], float]: Кортеж (путь, стоимость)
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
        for _ in range(iterations_per_temp):
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

def calculate_path_cost(path: List[int], distances: List[List[float]]) -> float:
    """
    Вычисляет общую стоимость пути.
    
    Args:
        path: Список городов в порядке посещения
        distances: Матрица расстояний между городами
    
    Returns:
        float: Общая стоимость пути
    """
    cost = 0
    n = len(path)
    for i in range(n):
        cost += distances[path[i]][path[(i + 1) % n]]
    return cost 