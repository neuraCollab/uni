# Algorithms — Interview Prep Index

Cram notes for coding interviews: recognize a pattern from the problem statement, recall the template, know the complexity and the common mistakes. Not a textbook — each note assumes you already know what an array or a graph is and focuses on the interview-relevant judgment calls.

## Structure
- [`patterns/`](patterns/) — the 13 core algorithmic patterns (technique-first: given a problem shape, which approach applies).
- [`data-structures/`](data-structures/) — the underlying ADTs (structure-first: what each data structure gives you and when to reach for it).
- [`sorting/`](sorting/notes.md) — sorting algorithm complexity, stability, and "which would you implement from memory" guidance.
- [`advanced/metaheuristics.md`](advanced/metaheuristics.md) — GA, PSO, simulated annealing, ACO for combinatorial/non-convex optimization (less common, but shows up in optimizer/ML-adjacent system design).
- [`leetcode/`](leetcode/README.md) — solved problems, one file per problem, using [`leetcode/template.md`](leetcode/template.md).

## Pattern lookup table

Match what the problem statement is telegraphing to the pattern that handles it.

| If the problem says... | Think... |
|---|---|
| Sorted array/list + find a pair/triplet summing to target | [Two Pointers](patterns/two-pointers.md) |
| Palindrome check, or "in-place" / O(1) extra space on an array | [Two Pointers](patterns/two-pointers.md) |
| Contiguous subarray/substring + longest/shortest/max/min | [Sliding Window](patterns/sliding-window.md) |
| "At most K distinct", fixed-size window aggregate | [Sliding Window](patterns/sliding-window.md) |
| Sorted array/monotonic predicate, O(log n) lookup | [Binary Search](patterns/binary-search.md) |
| "Find the min/max X such that condition holds" | [Binary Search](patterns/binary-search.md) (binary search on the answer) |
| Tree/graph traversal, "is X reachable from Y", flood fill | [BFS/DFS](patterns/bfs-dfs.md) |
| Shortest path / fewest steps in an **unweighted** graph | [BFS/DFS](patterns/bfs-dfs.md) (BFS) |
| Count all paths, detect cycle, connected components, islands | [BFS/DFS](patterns/bfs-dfs.md) (DFS) |
| Min cost/distance on a **weighted** graph | [Shortest Paths](patterns/shortest-paths.md) (Dijkstra / Bellman-Ford) |
| Negative edge weights | [Shortest Paths](patterns/shortest-paths.md) (Bellman-Ford, not Dijkstra) |
| All-pairs shortest distances needed | [Shortest Paths](patterns/shortest-paths.md) (Floyd-Warshall) |
| "Number of ways", min cost/steps to reach a state, overlapping subproblems | [Dynamic Programming](patterns/dynamic-programming.md) |
| Longest/shortest subsequence, knapsack-shaped include/exclude choice | [Dynamic Programming](patterns/dynamic-programming.md) |
| Locally-best choice provably leads to a globally-best result | [Greedy](patterns/greedy.md) |
| Interval scheduling, "maximum non-overlapping meetings" | [Greedy](patterns/greedy.md) / [Intervals](patterns/intervals.md) |
| List of `[start, end]` ranges to merge, overlap, or count | [Intervals](patterns/intervals.md) |
| "Minimum rooms/resources needed" (max concurrent overlap) | [Intervals](patterns/intervals.md) |
| Generate all permutations/combinations/subsets, or every valid arrangement | [Backtracking](patterns/backtracking.md) |
| Constraint satisfaction: N-Queens, Sudoku, word search | [Backtracking](patterns/backtracking.md) |
| "Next greater/smaller element", histogram/rectangle problems | [Monotonic Stack](patterns/monotonic-stack.md) |
| Many range-sum/range-count queries on a static array | [Prefix Sums](patterns/prefix-sums.md) |
| "Subarray sum equals K" | [Prefix Sums](patterns/prefix-sums.md) (+ hash map, see [`data-structures/hash-tables.md`](data-structures/hash-tables.md)) |
| Directed graph, "must happen before", prerequisites/build order | [Topological Sort](patterns/topological-sort.md) |
| "Are these two nodes connected?", incremental edges, cycle detection while building a graph | [Union-Find](patterns/union-find.md) |
| Top/smallest K elements, kth largest, merge K sorted lists, running median | [Heaps](data-structures/heaps.md) |
| Fast key -> value lookup, frequency counting, "have I seen this before" | [Hash Tables](data-structures/hash-tables.md) |
| Cycle detection or middle-node finding on a linked structure | [Linked Lists](data-structures/linked-lists.md) (fast/slow pointers) |
| Balanced brackets, "next"/"most recent" ordering, undo | [Stacks, Queues & Deque](data-structures/stacks-queues-deque.md) |
| BST property, validate/serialize a tree, LCA | [Trees & BST](data-structures/trees-bst.md) |
| Values are integers in a small known range | [Sorting notes](sorting/notes.md) (counting sort) |
| Fixed-width keys / bounded-length strings to sort | [Sorting notes](sorting/notes.md) (radix sort) |
| Huge/non-convex search space, no exact algorithm scales (e.g. TSP-shaped) | [Metaheuristics](advanced/metaheuristics.md) (GA / PSO / SA / ACO) |

## How to use this repo before an interview
1. Skim the lookup table above to refresh pattern recognition.
2. For each pattern you're rusty on, re-read its `Template` and `Common Mistakes` sections in [`patterns/`](patterns/) — those are the highest-signal parts.
3. Drill actual problems into [`leetcode/`](leetcode/README.md) using [`leetcode/template.md`](leetcode/template.md), cross-linking back to the pattern/data-structure note it exercises.
