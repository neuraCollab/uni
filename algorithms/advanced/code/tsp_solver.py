"""
Traveling Salesman Problem (TSP) heuristics — nearest neighbor, simulated
annealing (two cooling schedules), and ant colony optimization.

Ported from algos/4 sem/3 lab/code/tsp_solver.py (chosen over the two other
duplicate/simpler copies in `4 sem/1 lab/template/` and `4 sem/2 lab/code.py`
because this is the most complete, GUI-decoupled version). Comments
translated from Russian; logic unchanged.

All three functions operate on a dense distance matrix (`dist_matrix[i][j]`
= distance from city i to city j; use `math.inf` for missing edges on a
non-complete graph).
"""

from __future__ import annotations

import math
import random
import time


def solve_tsp_nearest_neighbor(dist_matrix: list[list[float]]) -> tuple[list[int], float]:
    """Greedy construction heuristic: start at city 0, repeatedly go to the
    nearest unvisited city, then return to the start.

    Fast (O(n^2)) and simple, but has no quality guarantee and is a common
    "obviously greedy, obviously suboptimal" baseline to compare against.
    """
    n = len(dist_matrix)
    visited = [False] * n
    visited[0] = True
    path = [0]
    total = 0.0
    current = 0

    for _ in range(n - 1):
        nxt, dist = min(
            ((j, dist_matrix[current][j]) for j in range(n) if not visited[j]),
            key=lambda pair: pair[1],
            default=(None, None),
        )
        if nxt is None:
            break
        visited[nxt] = True
        path.append(nxt)
        total += dist
        current = nxt

    if dist_matrix[current][0] < math.inf:
        total += dist_matrix[current][0]
        path.append(0)

    return path, total


def solve_tsp_simulated_annealing(
    dist_matrix: list[list[float]],
    initial_temp: float = 1000,
    cooling_rate: float = 0.995,
    cauchy: bool = False,
) -> tuple[list[int] | None, float, float]:
    """Simulated annealing over random-swap neighbors.

    Two cooling schedules, selected by `cauchy`:
      - exponential (default): temp *= cooling_rate each step -> fast,
        aggressive cooling.
      - Cauchy/fast annealing: temp = initial_temp / (1 + cooling_rate * t)
        -> slower, heavier-tailed cooling that keeps accepting worse moves
        for longer, which can escape local optima that exponential cooling
        would have already frozen out of.

    Acceptance rule is the standard Metropolis criterion: always accept an
    improving move; accept a worsening move with probability
    exp(-delta / temp), so early (hot) iterations explore more freely.

    Returns (best_tour, best_length, elapsed_seconds). best_tour is a cycle
    (first city repeated at the end).
    """
    n = len(dist_matrix)
    if n < 2:
        return None, float("inf"), 0

    current_solution = list(range(n))
    random.shuffle(current_solution)
    best_solution = current_solution[:]

    def tour_length(sol: list[int]) -> float:
        length = 0.0
        for k in range(n):
            i, j = sol[k], sol[(k + 1) % n]
            length += dist_matrix[i][j]
        return length

    current_length = tour_length(current_solution)
    best_length = current_length
    temp = initial_temp
    start_time = time.time()

    iteration = 1
    while temp > 1e-3:
        i, j = random.sample(range(n), 2)
        neighbor = current_solution[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_length = tour_length(neighbor)
        delta = neighbor_length - current_length

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_solution = neighbor
            current_length = neighbor_length
            if current_length < best_length:
                best_solution = current_solution[:]
                best_length = current_length

        if cauchy:
            temp = initial_temp / (1 + cooling_rate * iteration)
        else:
            temp *= cooling_rate
        iteration += 1

    elapsed = time.time() - start_time
    best_solution.append(best_solution[0])  # close the cycle
    return best_solution, best_length, elapsed


def solve_tsp_ant_colony(
    dist_matrix: list[list[float]],
    num_ants: int = 20,
    num_iterations: int = 100,
    alpha: float = 1.0,   # pheromone influence
    beta: float = 5.0,    # heuristic (1/distance) influence
    rho: float = 0.5,     # pheromone evaporation rate
    random_start: bool = True,
) -> tuple[list[int] | None, float, float]:
    """Ant Colony Optimization (ACO).

    Each ant builds a tour by probabilistically choosing the next city,
    weighted by pheromone^alpha * (1/distance)^beta. After all ants finish
    a round, pheromone evaporates (scaled by 1 - rho) and each ant deposits
    pheromone proportional to 1/tour_length on the edges it used, so
    shorter tours reinforce their edges more strongly over time.

    Returns (best_tour, best_length, elapsed_seconds).
    """
    n = len(dist_matrix)
    if n < 2:
        return None, float("inf"), 0

    # Initial pheromone level, seeded from the average of each row's
    # shortest edge so it's roughly scale-appropriate for this instance.
    tau0 = 1.0 / (n * sum(min(row) for row in dist_matrix if min(row) < math.inf))
    tau = [[tau0] * n for _ in range(n)]

    # Heuristic desirability: closer cities are more attractive (eta = 1/d).
    eta = [
        [0 if dist_matrix[i][j] == math.inf else 1.0 / dist_matrix[i][j] for j in range(n)]
        for i in range(n)
    ]

    best_path: list[int] | None = None
    best_length = float("inf")
    start_time = time.time()

    for _ in range(num_iterations):
        all_paths: list[tuple[list[int], float]] = []

        for _ in range(num_ants):
            current = random.randrange(n) if random_start else 0
            visited = {current}
            path = [current]

            while len(path) < n:
                i = path[-1]
                candidates = [
                    ((i, j), (tau[i][j] ** alpha) * (eta[i][j] ** beta))
                    for j in range(n)
                    if j not in visited and dist_matrix[i][j] < math.inf
                ]
                if not candidates:
                    break  # dead end (can happen on incomplete graphs)

                total_weight = sum(weight for _, weight in candidates)
                r = random.random() * total_weight
                cumulative = 0.0
                nxt = None
                for (_, j), weight in candidates:
                    cumulative += weight
                    if cumulative >= r:
                        nxt = j
                        break

                visited.add(nxt)
                path.append(nxt)

            if len(path) == n and dist_matrix[path[-1]][path[0]] < math.inf:
                path.append(path[0])  # close the cycle
                length = sum(dist_matrix[path[k]][path[k + 1]] for k in range(n))
                all_paths.append((path, length))
                if length < best_length:
                    best_length = length
                    best_path = path[:]

        # Evaporation.
        for i in range(n):
            for j in range(n):
                tau[i][j] *= 1 - rho

        # Deposit: shorter tours contribute more pheromone.
        for path, length in all_paths:
            deposit = 1.0 / length
            for k in range(len(path) - 1):
                i, j = path[k], path[k + 1]
                tau[i][j] += deposit

    elapsed = time.time() - start_time
    return best_path, best_length, elapsed


if __name__ == "__main__":
    # Small symmetric example instance.
    D = [
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0],
    ]

    print("nearest neighbor:", solve_tsp_nearest_neighbor(D))
    print("simulated annealing (exponential):", solve_tsp_simulated_annealing(D))
    print("simulated annealing (cauchy):", solve_tsp_simulated_annealing(D, cauchy=True))
    print("ant colony:", solve_tsp_ant_colony(D))
