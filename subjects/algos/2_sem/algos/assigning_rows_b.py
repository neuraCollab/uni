from collections import deque

def assign_rows_to_children(n, dislikes):
    graph = {i: set() for i in range(n)}
    in_degree = {i: 0 for i in range(n)}
    
    for i, j in dislikes:
        graph[i].add(j)
        in_degree[j] += 1
    
    # Level assignment for each child
    levels = [-1] * n
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    
    # Assign levels
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            levels[neighbor] = max(levels[neighbor], levels[node] + 1)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all nodes are processed
    if all(level >= 0 for level in levels):
        max_level = max(levels)
        # Return number of rows and the row assignments
        return max_level + 1, {child: row for child, row in enumerate(levels)}
    else:
        return "Impossible to assign rows to children"

# Example usage
n = 5
# dislikes = [(0, 1), (1, 2), (3, 4), (4, 0)]
# dislikes = [(0, 2), (1, 3), (2, 5), (3, 6), (4, 1)]
dislikes = [(0, 2), (1, 3)]
# dislikes = [(0, 1), (1, 2), (3, 4)]
print(assign_rows_to_children(n, dislikes))
