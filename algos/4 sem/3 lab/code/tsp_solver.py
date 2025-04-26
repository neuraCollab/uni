import math
import random
import time

def solve_tsp_nn(dist_matrix):
    n = len(dist_matrix)
    visited = [False]*n
    visited[0] = True
    path = [0]
    total = 0.0
    current = 0
    for _ in range(n-1):
        nxt, md = min(
            ((j, dist_matrix[current][j]) for j in range(n) if not visited[j]),
            key=lambda x: x[1], default=(None, None)
        )
        if nxt is None: break
        visited[nxt] = True
        path.append(nxt)
        total += md
        current = nxt
    if dist_matrix[current][0] < math.inf:
        total += dist_matrix[current][0]
        path.append(0)
    return path, total

def solve_tsp_simulated_annealing(dist_matrix, initial_temp=1000, cooling_rate=0.995, cauchy=False):
    n = len(dist_matrix)
    if n < 2:
        return None, float('inf'), 0

    # initial solution: random permutation
    current_solution = list(range(n))
    random.shuffle(current_solution)
    best_solution = current_solution[:]

    def tour_length(sol):
        length = 0
        for k in range(n):
            i = sol[k]
            j = sol[(k+1)%n]
            length += dist_matrix[i][j]
        return length

    current_length = tour_length(current_solution)
    best_length = current_length
    temp = initial_temp
    start_time = time.time()

    iteration = 1
    while temp > 1e-3:
        # generate neighbor: swap two
        i, j = random.sample(range(n), 2)
        neighbor = current_solution[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_length = tour_length(neighbor)
        delta = neighbor_length - current_length
        # acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_solution = neighbor
            current_length = neighbor_length
            if current_length < best_length:
                best_solution = current_solution[:]
                best_length = current_length
        # update temperature
        if cauchy:
            temp = initial_temp / (1 + cooling_rate * iteration)
        else:
            temp *= cooling_rate
        iteration += 1

    elapsed = time.time() - start_time
    # close cycle
    best_solution.append(best_solution[0])
    return best_solution, best_length, elapsed

def solve_tsp_aco(dist_matrix, num_ants=20, num_iters=100, alpha=1.0, beta=5.0, rho=0.5, wandering=True):
    n = len(dist_matrix)
    if n < 2:
        return None, float('inf'), 0

    # 2) Инициализация феромона
    tau0 = 1.0 / (n * sum(min(row) for row in dist_matrix if min(row) < math.inf))
    tau = [[tau0]*n for _ in range(n)]

    # 3) Эвристика η = 1/d
    eta = [[0 if dist_matrix[i][j]==math.inf else 1.0/dist_matrix[i][j] for j in range(n)] for i in range(n)]

    best_path, best_len = None, float('inf')
    start_time = time.time()

    for _ in range(num_iters):
        all_paths = []
        for _ in range(num_ants):
            # стартовая вершина
            current = random.randrange(n) if wandering else 0
            visited = {current}
            path = [current]

            while len(path) < n:
                i = path[-1]
                # вероятности перехода
                probs = []
                for j in range(n):
                    if j not in visited and dist_matrix[i][j] < math.inf:
                        probs.append(((i,j), (tau[i][j]**alpha)*(eta[i][j]**beta)))
                if not probs:
                    break
                total = sum(p for (_,p) in probs)
                r = random.random() * total
                s = 0
                for (edge,p) in probs:
                    s += p
                    if s >= r:
                        nxt = edge[1]
                        break
                visited.add(nxt)
                path.append(nxt)

            # замкнём цикл
            if len(path)==n and dist_matrix[path[-1]][path[0]]<math.inf:
                path.append(path[0])
                length = sum(dist_matrix[path[k]][path[k+1]] for k in range(n))
                all_paths.append((path, length))

                if length < best_len:
                    best_len = length
                    best_path = path[:]

        # испарение
        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1 - rho)
        # обновление феромона
        for path, length in all_paths:
            delta = 1.0 / length
            for k in range(len(path)-1):
                i,j = path[k], path[k+1]
                tau[i][j] += delta

    elapsed = time.time() - start_time
    return best_path, best_len, elapsed 