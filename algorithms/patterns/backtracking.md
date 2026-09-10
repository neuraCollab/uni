# Backtracking

## Recognition
### Key clues
- "Generate all" permutations, combinations, subsets, valid arrangements.
- Constraint satisfaction: N-Queens, Sudoku, word search on a grid.
- Asked for every solution (or to count them), not just one optimal value.
- Problem naturally decomposes into "choose one of several options, then recurse on the rest".

## Pattern
Explore a decision tree depth-first: make a choice, recurse into the consequence of that choice, and undo (backtrack) the choice before trying the next option. Prune branches early whenever a partial state already violates a constraint.

## Why this works
Backtracking is exhaustive search with early termination — pruning invalid branches as soon as possible avoids wasting time exploring subtrees that can never lead to a valid solution, which is usually the difference between exponential-but-tractable and truly intractable.

## Template
```python
def subsets(nums: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        result.append(path.copy())  # every partial state is a valid subset
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()  # undo the choice

    backtrack(0)
    return result


def permutations(nums: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    path: list[int] = []
    used = [False] * len(nums)

    def backtrack() -> None:
        if len(path) == len(nums):
            result.append(path.copy())
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return result
```

## Example Problems
- Subsets / Subsets II
- Permutations / Permutations II
- Combination Sum
- N-Queens
- Word Search
- Generate Parentheses
- Sudoku Solver

## Common Mistakes
- Forgetting to undo (`pop()`/reset) the choice after recursing — leaves stale state polluting sibling branches.
- Appending a *reference* to the mutable `path` list instead of a copy (`path.copy()` / `path[:]`), so all recorded results end up pointing at the same, later-mutated list.
- Not pruning early — checking a constraint only at a leaf instead of as soon as it can be determined wastes exponential time in the worst case.
- Missing the dedup step (`if i > start and nums[i] == nums[i-1]: continue`) when the input has duplicates and the problem wants unique results.

## Complexity
Typically O(2^n) (subsets) to O(n!) (permutations) time in the worst case — inherent to exhaustive search — but effective pruning often keeps real-world runtime far below the bound.

## Related Patterns
- [Dynamic Programming](dynamic-programming.md) — when overlapping subproblems let you memoize instead of re-exploring.
- [Graphs: BFS/DFS](bfs-dfs.md) — backtracking is DFS over an implicit decision tree/graph.
