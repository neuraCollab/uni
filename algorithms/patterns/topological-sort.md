# Topological Sort

## Recognition
### Key clues
- **Directed** graph with a notion of "must happen before" (dependencies, prerequisites, build order).
- "Course Schedule" / task-ordering-with-prerequisites phrasing.
- Asked to detect whether a valid ordering exists at all (equivalent to: is the graph a DAG, i.e. cycle-free?).
- Compilation order, package dependency resolution.

## Pattern
Produce a linear ordering of nodes in a directed acyclic graph (DAG) such that for every edge `u -> v`, `u` comes before `v`. Two standard approaches: **Kahn's algorithm** (BFS, repeatedly remove nodes with in-degree 0) or **DFS with post-order** (push each node after all its descendants are processed, then reverse).

## Why this works
A node with in-degree 0 has no unresolved prerequisites, so it's always safe to place it next; removing it can only reduce other nodes' in-degrees, eventually exposing the next safe node(s) — this greedily builds a valid order (Kahn's). Equivalently, in DFS post-order, a node is only finished after everything it depends on is finished, so reversing post-order gives a valid "dependencies first" sequence.

## Template
```python
from collections import deque


def topological_sort_kahn(num_nodes: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """edges are (u, v) meaning u must come before v. Returns None if a
    cycle exists (no valid ordering)."""
    graph: list[list[int]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque(node for node in range(num_nodes) if in_degree[node] == 0)
    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_nodes else None  # None => cycle detected
```

## Example Problems
- Course Schedule / Course Schedule II
- Alien Dictionary
- Sequence Reconstruction
- Minimum Height Trees
- Build Order (dependency resolution)

## Common Mistakes
- Not detecting cycles — if `len(order) != num_nodes` at the end, the graph has a cycle and there is no valid topological order; forgetting this check silently returns a partial/wrong order.
- Mixing up edge direction (`u -> v` meaning "u before v" vs. accidentally building the reverse graph).
- Using plain DFS/BFS (for connectivity) when the problem actually needs edge *direction* respected — topological sort only makes sense on directed graphs.

## Complexity
Time O(V + E). Space O(V + E) for the graph and in-degree/visited tracking.

## Related Patterns
- [BFS/DFS](bfs-dfs.md) — both algorithms underlying topological sort are BFS/DFS variants.
- [Union-Find](union-find.md) — different tool, but also commonly paired with graph-construction problems (e.g. Kruskal's MST vs. topological sort for scheduling).
