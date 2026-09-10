# Dynamic Programming

## Recognition
### Key clues
- Asked for an **optimal value** (min/max/count of ways), not the set of all solutions.
- The problem has **overlapping subproblems** — a brute-force recursive solution would recompute the same state many times.
- **Optimal substructure**: the optimal solution to the whole problem is built from optimal solutions to subproblems.
- Keywords: "number of ways", "minimum cost/steps to reach", "longest/shortest ... subsequence/substring", "can you partition into...".
- Choices with constraints that look like knapsack (include/exclude an item under a capacity).

## Pattern
Break the problem into a sequence of overlapping subproblems defined by some state (often an index, or index + remaining capacity/target). Solve each state once, either top-down with memoization (recursion + cache) or bottom-up by filling a table in an order that guarantees dependencies are solved first.

## Why this works
Without memoization, a naive recursive solution can be exponential because the same subproblem is recomputed on every path that reaches it (classic example: naive Fibonacci is O(2^n)). Caching each state's result the first time it's computed turns that into O(number of distinct states * work per state).

## Template
```python
from functools import lru_cache


def climbing_stairs(n: int) -> int:
    """Top-down memoized DP: number of ways to climb n stairs, 1 or 2 steps
    at a time. State = remaining stairs.
    """
    @lru_cache(maxsize=None)
    def ways(remaining: int) -> int:
        if remaining <= 1:
            return 1
        return ways(remaining - 1) + ways(remaining - 2)

    return ways(n)


def coin_change(coins: list[int], amount: int) -> int:
    """Bottom-up tabulation: minimum coins to make `amount`, or -1 if
    impossible. State = amount remaining; dp[a] built from smaller a's.
    """
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1


def longest_common_subsequence(a: str, b: str) -> int:
    """2D DP: classic two-string LCS. dp[i][j] = LCS length of a[:i], b[:j]."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]
```

## Example Problems
- Climbing Stairs / Fibonacci
- Coin Change / Coin Change II
- Longest Common Subsequence
- Longest Increasing Subsequence
- 0/1 Knapsack / Partition Equal Subset Sum
- Edit Distance
- House Robber

## Common Mistakes
- Not identifying the state precisely — a wrong or incomplete state (missing a dimension, e.g. forgetting "remaining capacity" in knapsack) silently produces wrong answers rather than crashing.
- Wrong iteration order in bottom-up tabulation, referencing a `dp` cell that hasn't been computed yet.
- Confusing 0/1 knapsack (each item once — iterate amount in decreasing order, or items on the outer loop) with unbounded knapsack/coin-change (item reusable — amount can be inner loop, items outer, ascending).
- Jumping straight to code without first writing the recurrence relation in words/math — this is the step interviewers actually want to see.
- Using recursion without memoization ("I'll just write the recursive version") on a problem that clearly has overlapping subproblems — always state the complexity of the naive approach to justify why memoization is needed.

## Complexity
Time O(number of distinct states * transition cost) — e.g. O(n) for 1D DP, O(n*m) for 2D DP like LCS/edit distance, O(n*capacity) for knapsack. Space can often be reduced from O(n) or O(n*m) to O(1) or O(m) via rolling arrays when only the previous row/state is needed.

## Related Patterns
- [Backtracking](backtracking.md) — DP is backtracking/recursion + memoization when subproblems overlap.
- [Greedy](greedy.md) — sometimes a greedy choice is provably optimal and replaces a DP entirely; know how to argue which applies.
