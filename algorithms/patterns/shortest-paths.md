# Shortest Paths

## Recognition
### Key clues
- "Minimum cost/distance/time to get from A to B" on a **weighted** graph.
- Weights can be negative -> think Bellman-Ford, not Dijkstra.
- All edge weights equal (or "minimum number of steps/hops") -> plain BFS is enough, no need for a weighted algorithm at all.
- "Cheapest flights within K stops", network delay time, grid with varying costs per cell.
- All-pairs distances needed (not just from one source) -> Floyd-Warshall.

## Pattern
Relax edges: repeatedly try to improve the best known distance to each node by going through another node, until no improvement is possible. The specific algorithm is chosen by graph properties (weighted vs. unweighted, negative weights or not, single-source vs. all-pairs).

## Why this works
Shortest-path problems have optimal substructure: the shortest path to node `v` through `u` is `dist[u] + weight(u, v)`, so once you know the true shortest distance to every node that could precede `v` on an optimal path, you can compute `v`'s. Dijkstra's greedy "always finalize the closest unfinalized node next" is correct specifically because non-negative weights guarantee no future relaxation could ever beat an already-finalized shortest distance.

## Template
```python
import heapq
from collections import deque


def bfs_shortest_path(graph: dict[int, list[int]], source: int) -> dict[int, int]:
    """Unweighted graph (or all edges cost 1): plain BFS gives shortest
    hop-count from source to every reachable node."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def dijkstra(graph: dict[int, list[tuple[int, int]]], source: int) -> dict[int, float]:
    """Weighted graph, NON-NEGATIVE weights. graph[u] = list of (v, weight).
    Greedy: always expand the closest unfinalized node next, via a min-heap.
    """
    dist = {source: 0}
    visited = set()
    heap = [(0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph.get(node, []):
            nd = d + weight
            if nd < dist.get(neighbor, float("inf")):
                dist[neighbor] = nd
                heapq.heappush(heap, (nd, neighbor))
    return dist


def bellman_ford(num_nodes: int, edges: list[tuple[int, int, int]], source: int) -> list[float] | None:
    """Weighted graph, weights MAY be negative. edges = (u, v, weight).
    Relax all edges V-1 times; one more pass detects a negative cycle.
    """
    dist = [float("inf")] * num_nodes
    dist[source] = 0
    for _ in range(num_nodes - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # negative cycle reachable from source
    return dist
```

## Example Problems
- Network Delay Time (Dijkstra)
- Cheapest Flights Within K Stops (Bellman-Ford-style relaxation, bounded)
- Path With Minimum Effort (Dijkstra variant / binary search + BFS)
- Word Ladder (unweighted BFS)
- Floyd-Warshall for all-pairs (e.g. Find the City With the Smallest Number of Neighbors at a Threshold Distance)

## Common Mistakes
- Reaching for Dijkstra on a graph with negative edge weights — it silently produces wrong answers instead of erroring; check the weight range before choosing an algorithm.
- Using a weighted-shortest-path algorithm on an unweighted graph — plain BFS is simpler and already optimal (O(V+E) vs. O(E log V)).
- Not checking `if node in visited: continue` in Dijkstra — the heap can contain stale/outdated entries for the same node, and processing them again wastes time (or, without a visited check at all, can be relied upon incorrectly since distances only get pushed when improved — but the standard safe pattern is to skip already-finalized nodes).
- Forgetting the extra Bellman-Ford pass to detect negative cycles when they're possible in the input.

## Complexity
- BFS: O(V + E).
- Dijkstra (binary heap): O((V + E) log V).
- Bellman-Ford: O(V * E).
- Floyd-Warshall (all-pairs): O(V^3).

## Related Patterns
- [BFS/DFS](bfs-dfs.md) — BFS is the unweighted special case of shortest paths.
- [Greedy](greedy.md) — Dijkstra is a greedy algorithm; its correctness proof relies on non-negative weights.
