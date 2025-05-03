from collections import deque

def line_up_children(n, dislikes):
    graph = {i: set() for i in range(n)}
    in_degree = {i: 0 for i in range(n)}
    
    for i, j in dislikes:
        graph[i].add(j)
        in_degree[j] += 1
    
    # Тут просто создаем очередь
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    if len(result) == n:
        return result
    else:
        return "Детей не получится поставить в ряд (((((((((((((((((("

# n = 5
n = 7
dislikes = [(0, 2), (1, 3), (2, 5), (3, 6), (4, 1)]

# dislikes = [(0, 1), (1, 2), (3, 4), (4, 0)]
print(line_up_children(n, dislikes))
