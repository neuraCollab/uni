import time

def nearest_neighbor(distances):
    """
    Решение задачи коммивояжера методом ближайшего соседа
    """
    start_time = time.time()
    n = len(distances)
    unvisited = set(range(1, n))  # все города кроме начального
    current = 0  # начинаем с первого города
    path = [current]
    total_cost = 0
    
    while unvisited:
        # Находим ближайший непосещенный город
        next_city = min(unvisited, key=lambda x: distances[current][x])
        total_cost += distances[current][next_city]
        current = next_city
        path.append(current)
        unvisited.remove(current)
    
    # Возвращаемся в начальный город
    total_cost += distances[current][0]
    
    end_time = time.time()
    execution_time = end_time - start_time
    return path, total_cost, execution_time 