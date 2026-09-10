# Union-Find (Disjoint Set Union)

## Recognition
### Key clues
- "Are these two nodes connected?" queries, possibly interleaved with edges being added over time.
- Counting **connected components** in a graph, especially when edges arrive incrementally.
- Detecting a **cycle** while building a graph edge by edge (e.g. Kruskal's MST).
- "Number of provinces/islands/friend circles" style grouping problems.
- Grid or graph problems where you repeatedly need to merge groups.

## Pattern
Maintain a forest where each set (connected component) is represented by a tree, and each node points to a parent. `find(x)` walks up to the root (with path compression to flatten the tree). `union(x, y)` links the roots of x's and y's trees (using union by rank/size to keep trees shallow).

## Why this works
Path compression + union by rank together give near-constant amortized time per operation (inverse Ackermann, effectively $O(1)$ in practice) — far better than re-scanning the whole structure with BFS/DFS on every query, which matters when there are many interleaved union/find operations.

## Template
```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # already connected -> this edge creates a cycle
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        self.components -= 1
        return True
```

## Example Problems
- Number of Provinces / Number of Connected Components in an Undirected Graph
- Graph Valid Tree (cycle detection)
- Redundant Connection
- Accounts Merge
- Kruskal's Minimum Spanning Tree
- Number of Islands II (online connectivity)

## Common Mistakes
- Forgetting path compression and/or union by rank — without them, `find` degrades to $O(n)$ per call on adversarial input (a long chain).
- Off-by-one when mapping 2D grid coordinates to a flat index for the DSU array.
- Using `union`'s return value (or lack of a cycle check) incorrectly — `union` returning `False` (roots already equal) is exactly the cycle-detection signal in Kruskal's algorithm.

## Complexity
Time: $O(\alpha(n))$ amortized per `find`/`union` ($\alpha$ = inverse Ackermann, effectively constant). Space $O(n)$.

## Related Patterns
- [BFS/DFS](bfs-dfs.md) — alternative way to find connected components when there's no need for incremental/online queries.
- [Topological Sort](topological-sort.md)
