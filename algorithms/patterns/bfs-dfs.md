# BFS / DFS (Graph & Tree Traversal)

## Recognition
### Key clues
- Any tree or graph traversal: "visit every node", "is X reachable from Y", "flood fill".
- **BFS** clues: shortest path / minimum steps in an **unweighted** graph, "level order", "minimum number of moves".
- **DFS** clues: explore all paths, detect cycles, connected components, "count islands", backtracking-flavored graph problems, topological ordering.
- Grid problems (2D array treated as an implicit graph, 4- or 8-directionally connected).

## Pattern
Systematically visit every reachable node exactly once, tracking visited state to avoid infinite loops. **BFS** explores level by level using a queue (FIFO) — the first time you reach a node is via a shortest path in hop count. **DFS** explores as deep as possible before backtracking, using a stack (explicit or the call stack via recursion) — natural for exhaustive exploration, cycle detection, and problems structured as decision trees.

## Why this works
Both are exhaustive-but-non-repeating traversals: marking a node visited the moment it's discovered guarantees $O(V + E)$ total work (each node visited once, each edge examined once/twice). BFS's queue ordering specifically guarantees the *first* visit to any node is via the fewest edges, which is exactly what "shortest path in an unweighted graph" needs.

## Template
```python
from collections import deque


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    visited = {start}
    order = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs_recursive(graph: dict[int, list[int]], start: int) -> list[int]:
    visited = set()
    order = []

    def visit(node: int) -> None:
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visit(neighbor)

    visit(start)
    return order


def dfs_iterative(graph: dict[int, list[int]], start: int) -> list[int]:
    """Explicit-stack DFS — avoids recursion depth limits on large/deep graphs."""
    visited = {start}
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return order


def num_islands(grid: list[list[str]]) -> int:
    """Classic grid-as-graph DFS: count connected components of '1's."""
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])

    def sink(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"  # mark visited in place
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sink(r + dr, c + dc)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                sink(r, c)
    return count
```

## Example Problems
- Number of Islands
- Binary Tree Level Order Traversal (BFS)
- Word Ladder (BFS, shortest transformation sequence)
- Clone Graph
- Course Schedule (DFS cycle detection — see also [Topological Sort](topological-sort.md))
- Rotting Oranges (multi-source BFS)
- Surrounded Regions

## Common Mistakes
- Forgetting to mark a node visited *when enqueuing* (BFS) rather than when dequeuing — without this, the same node can be added to the queue multiple times, wasting work or breaking shortest-path guarantees.
- Using DFS to find shortest paths in an unweighted graph — it doesn't guarantee shortest path unless you specifically track and compare path lengths; BFS gets this for free.
- Recursive DFS stack overflow on very deep/large graphs — switch to the iterative, explicit-stack version.
- Not handling disconnected graphs — a single traversal from one start node won't visit everything; loop over all nodes and traverse from each unvisited one when asked about the *whole* graph (e.g. counting all connected components).

## Complexity
Time $O(V + E)$ for both. Space $O(V)$ for the visited set/queue/stack (plus $O(V)$ recursion stack for recursive DFS in the worst case).

## Related Patterns
- [Shortest Paths](shortest-paths.md) — BFS is the unweighted special case.
- [Topological Sort](topological-sort.md) — built on BFS (Kahn's) or DFS (post-order).
- [Union-Find](union-find.md) — alternative for connectivity queries, especially incremental ones.
- [Backtracking](backtracking.md) — DFS over an implicit decision-tree "graph".
