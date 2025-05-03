import random
import numpy as np
import time

def ant_colony_optimization(distances, n_ants=10, n_iterations=50, decay=0.1, alpha=1, beta=2):
    """
    Решение задачи коммивояжера методом муравьиной колонии с поиском оптимальной начальной точки
    """
    start_time = time.time()
    n = len(distances)
    pheromone = np.ones((n, n)) / n
    best_path = None
    best_cost = float('inf')
    best_start = 0
    
    # Перебираем все возможные начальные точки
    for start_point in range(n):
        current_pheromone = pheromone.copy()
        
        for iteration in range(n_iterations):
            paths = []
            costs = []
            
            # Каждый муравей строит свой путь
            for ant in range(n_ants):
                path = construct_solution(distances, current_pheromone, alpha, beta, start_point)
                cost = calculate_path_cost(path, distances)
                paths.append(path)
                costs.append(cost)
                
                # Обновляем лучшее решение
                if cost < best_cost:
                    best_cost = cost
                    best_path = path[:]
                    best_start = start_point
            
            # Обновляем феромоны
            current_pheromone *= (1 - decay)  # Испарение
            for path, cost in zip(paths, costs):
                for i in range(len(path)):
                    j = (i + 1) % len(path)
                    current_pheromone[path[i]][path[j]] += 1.0 / cost
    
    end_time = time.time()
    execution_time = end_time - start_time
    return best_path, best_cost, execution_time

def wandering_colony_optimization(distances, n_ants=10, n_iterations=50, decay=0.1, alpha=1, beta=2):
    """
    Решение задачи коммивояжера методом блуждающей муравьиной колонии,
    где начальная точка муравьев меняется на каждой итерации.
    """
    start_time = time.time()
    n = len(distances)
    pheromone = np.ones((n, n)) / n
    best_path = None
    best_cost = float('inf')
    
    for iteration in range(n_iterations):
        # Меняем стартовую точку для всей колонии на каждой итерации
        current_start = random.randint(0, n-1)
        
        paths = []
        costs = []
        
        # Каждый муравей строит свой путь из текущей стартовой точки
        for ant in range(n_ants):
            path = construct_solution(distances, pheromone, alpha, beta, current_start)
            cost = calculate_path_cost(path, distances)
            paths.append(path)
            costs.append(cost)
            
            # Обновляем лучшее решение
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]
        
        # Обновляем феромоны
        pheromone *= (1 - decay)  # Испарение
        for path, cost in zip(paths, costs):
            for i in range(len(path)):
                j = (i + 1) % len(path)
                pheromone[path[i]][path[j]] += 1.0 / cost
    
    end_time = time.time()
    execution_time = end_time - start_time
    return best_path, best_cost, execution_time

def construct_solution(distances, pheromone, alpha, beta, start_point):
    """
    Построение решения одним муравьем с заданной начальной точкой
    """
    n = len(distances)
    unvisited = set(range(n))
    unvisited.remove(start_point)
    path = [start_point]
    current = start_point
    
    while unvisited:
        # Вычисляем вероятности перехода
        probabilities = []
        for j in unvisited:
            # Привлекательность = феромоны ^ alpha * (1/расстояние) ^ beta
            p = (pheromone[current][j] ** alpha) * ((1.0 / distances[current][j]) ** beta)
            probabilities.append((j, p))
        
        # Выбираем следующий город
        total = sum(p for j, p in probabilities)
        if total == 0:
            # Если все вероятности нулевые, выбираем случайный город
            next_city = random.choice(list(unvisited))
        else:
            r = random.random() * total
            cum_prob = 0
            for j, p in probabilities:
                cum_prob += p
                if cum_prob >= r:
                    next_city = j
                    break
            else:
                next_city = probabilities[-1][0]
        
        path.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    
    return path

def calculate_path_cost(path, distances):
    """
    Вычисляет общую стоимость маршрута
    """
    cost = 0
    n = len(path)
    for i in range(n):
        cost += distances[path[i]][path[(i + 1) % n]]
    return cost 