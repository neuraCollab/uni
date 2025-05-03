import math
import time
import random

def solve_nearest_neighbor(vertices, edges, start_index=0):
    n = len(vertices)
    dist_matrix = [[math.inf] * n for _ in range(n)]
    for i, j, dist in edges:
        dist_matrix[i][j] = dist

    visited = [False] * n
    visited[start_index] = True
    path = [start_index]
    total_dist = 0.0
    current = start_index

    start_time = time.time()
    for _ in range(n - 1):
        next_v = None
        min_dist = math.inf
        for j in range(n):
            if not visited[j] and dist_matrix[current][j] < min_dist:
                min_dist = dist_matrix[current][j]
                next_v = j
        if next_v is None:
            break
        visited[next_v] = True
        path.append(next_v)
        total_dist += min_dist
        current = next_v

    if dist_matrix[current][start_index] < math.inf:
        total_dist += dist_matrix[current][start_index]
        path.append(start_index)
    end_time = time.time()

    return path, total_dist, end_time - start_time
