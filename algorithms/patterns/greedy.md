# Greedy

## Recognition
### Key clues
- Asked for an optimal value where a **locally best choice at each step** intuitively seems to lead to a globally best result.
- Interval scheduling ("maximum non-overlapping meetings"), activity selection.
- Problems involving sorting first, then a single linear pass making irrevocable choices.
- "Minimum number of..." / "maximum number of..." where a DP would work but is overkill — always check if greedy is provably correct first (it's faster to code and reason about).

## Pattern
At each step, make the choice that looks best *right now*, without reconsidering past choices, and never backtrack. Correctness requires proving the **greedy-choice property** (a locally optimal choice is part of some globally optimal solution) and (usually) **optimal substructure**.

## Why this works
When the greedy-choice property genuinely holds, exchange arguments show that any optimal solution can be transformed into the greedy solution without making it worse — so there's no need to explore alternatives via backtracking or DP. The hard part is *proving* the property holds; greedy that "seems to work" on examples but isn't proven can fail on adversarial inputs.

## Template
```python
def max_non_overlapping_intervals(intervals: list[tuple[int, int]]) -> int:
    """Classic activity-selection greedy: sort by END time, greedily take
    an interval if it starts after the last taken interval ends.
    """
    intervals = sorted(intervals, key=lambda iv: iv[1])
    count = 0
    last_end = float("-inf")
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    return count


def min_coins_greedy(coins: list[int], amount: int) -> int | None:
    """Greedy coin change — ONLY correct for 'canonical' coin systems
    (e.g. US coins). Included as a cautionary template: for arbitrary coin
    sets, this can give a suboptimal or wrong answer — use DP instead
    (see dynamic-programming.md) unless you've verified the coin system is
    canonical.
    """
    coins = sorted(coins, reverse=True)
    count = 0
    for c in coins:
        count += amount // c
        amount %= c
    return count if amount == 0 else None
```

## Example Problems
- Non-overlapping Intervals / Meeting Rooms
- Jump Game / Jump Game II
- Gas Station
- Task Scheduler
- Best Time to Buy and Sell Stock II
- Partition Labels

## Common Mistakes
- Assuming greedy works without proving the greedy-choice property — the most common interview trap is a problem that *looks* greedy but actually requires DP (e.g. general coin change with non-canonical denominations).
- Sorting by the wrong key (e.g. by start time instead of end time for interval scheduling — sorting by end time is what makes the exchange argument work).
- Not handling ties correctly in the sort/comparison, which can silently break the greedy invariant.

## Complexity
Typically O(n log n) (dominated by the initial sort) + O(n) for the greedy pass. Space O(1) to O(n) depending on whether sorting is in place.

## Related Patterns
- [Dynamic Programming](dynamic-programming.md) — the fallback when greedy can't be proven correct.
- [Intervals](intervals.md) — greedy is the dominant technique for interval-scheduling problems specifically.
