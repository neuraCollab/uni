import time
import random

def nearest_neighbor(distances, start_city=None):
    """
    Решение задачи коммивояжера методом ближайшего соседа
    с возможностью случайного выбора начального города
    """
    n = len(distances)
    unvisited = set(range(n))  # все города изначально не посещены
    
    # Выбираем начальный город (если не задан, выбираем случайный)
    if start_city is None:
        current = random.choice(range(n))
    else:
        current = start_city
    
    path = [current]
    unvisited.remove(current)
    total_cost = 0
    
    start_time = time.perf_counter()

    while unvisited:
        next_city = min(unvisited, key=lambda x: distances[current][x])
        total_cost += distances[current][next_city]
        current = next_city
        path.append(current)
        unvisited.remove(current)
    
    # Возвращаемся в начальный город
    total_cost += distances[current][path[0]]
    path.append(path[0])

    execution_time = time.perf_counter() - start_time
    return path, total_cost, execution_time