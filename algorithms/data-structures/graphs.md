# Graphs — Representation & Terminology

This file covers graph **representation, terminology, and complexity**. For the algorithms themselves, see the dedicated pattern notes: [BFS/DFS](../patterns/bfs-dfs.md), [Shortest Paths](../patterns/shortest-paths.md), [Topological Sort](../patterns/topological-sort.md), [Union-Find](../patterns/union-find.md).

## Terminology
- **Vertex/node** and **edge** — the basic building blocks; `V` = number of vertices, `E` = number of edges.
- **Directed** — edges have a direction (`u -> v` doesn't imply `v -> u`); **undirected** — edges are symmetric.
- **Weighted** — edges carry a cost/distance; **unweighted** — all edges are equivalent (or implicitly cost 1).
- **DAG** (directed acyclic graph) — directed, no cycles; required for [topological sort](../patterns/topological-sort.md) to produce a valid ordering.
- **Connected component** — a maximal set of vertices reachable from each other (undirected); **strongly connected component** — the directed analogue (mutually reachable both ways).
- **Degree** — number of edges touching a vertex (in undirected graphs); **in-degree**/**out-degree** for directed graphs.
- **Dense** graph: `E` close to `V^2`. **Sparse** graph: `E` close to `V`.

## Representations

### Adjacency list
`dict[node, list[neighbor]]` (or `list[list[int]]` for integer-labeled nodes), optionally storing `(neighbor, weight)` pairs for weighted graphs.
```python
graph: dict[int, list[int]] = {0: [1, 2], 1: [2], 2: [0]}
weighted_graph: dict[int, list[tuple[int, int]]] = {0: [(1, 4), (2, 1)]}
```
- Space: O(V + E) — proportional to what's actually in the graph.
- "List neighbors of node u": O(degree(u)) — fast, only touches actual edges.
- "Is there an edge u -> v?": O(degree(u)) worst case (must scan u's neighbor list), unless neighbors are stored in a set (then O(1) average).
- **Preferred for sparse graphs** (most interview graphs, and most real-world graphs) — avoids wasting space on the many non-edges a matrix would store.

### Adjacency matrix
`matrix[i][j]` = weight of edge i -> j (or `1`/`0`, or `math.inf` for "no edge").
```python
matrix = [[0, 4, 1], [0, 0, 0], [0, 0, 0]]  # matrix[i][j] = weight, 0/inf = no edge
```
- Space: O(V^2) regardless of how many edges actually exist.
- "Is there an edge u -> v?": O(1) — direct index.
- "List neighbors of node u": O(V) — must scan the whole row even if u has few neighbors.
- **Preferred for dense graphs**, or when O(1) edge-existence lookups matter more than space (e.g. Floyd-Warshall all-pairs shortest paths is naturally matrix-based — see [`../patterns/shortest-paths.md`](../patterns/shortest-paths.md)).

### Implicit graphs
Many interview "graphs" are never built explicitly: a 2D grid (cells are nodes, 4/8-directional adjacency is implicit), or a state-space search (each state is a node, transitions are computed on the fly) — see the grid examples in [`../patterns/bfs-dfs.md`](../patterns/bfs-dfs.md). Recognizing an implicit graph is often the harder part of the problem; representation choice barely matters once you see it.

## Complexity summary

| | Adjacency list | Adjacency matrix |
|---|---|---|
| Space | O(V + E) | O(V^2) |
| Add edge | O(1) | O(1) |
| Check edge u->v exists | O(degree(u)) (O(1) if neighbor set) | O(1) |
| Iterate all neighbors of u | O(degree(u)) | O(V) |
| Best for | sparse graphs (most cases) | dense graphs, frequent edge-existence checks |

## Related Patterns
- [BFS/DFS](../patterns/bfs-dfs.md) — traversal.
- [Shortest Paths](../patterns/shortest-paths.md) — Dijkstra, Bellman-Ford, Floyd-Warshall.
- [Topological Sort](../patterns/topological-sort.md) — ordering on a DAG.
- [Union-Find](../patterns/union-find.md) — connectivity without full traversal, especially for incremental edges.
