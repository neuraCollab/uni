# Sliding Window

## Recognition
### Key clues
- "**Contiguous** subarray/substring" + longest/shortest/maximum/minimum/count.
- Constraint on the window contents: "at most K distinct characters", "sum <= target", "no repeating characters".
- Asked for a fixed-size window aggregate (e.g. "max sum of any subarray of size k").
- Input is a 1D array or string; brute force would be $O(n^2)$ checking every subarray.

## Pattern
Maintain a window `[left, right]` over the array/string with some invariant (sum, character counts, distinct count). Expand `right` to grow the window; when the invariant is violated (or, for fixed-size windows, once size k is reached), shrink from `left` until it's valid again. Each element enters and leaves the window at most once, giving $O(n)$ total.

## Why this works
Because the window's validity is monotonic in a useful sense (growing the window can only make an "at most K" constraint harder, shrinking can only make it easier), you never need to re-examine a `left`/`right` pair once you've moved past it — no nested loop required.

## Template
```python
def longest_substring_no_repeat(s: str) -> int:
    """Variable-size window: shrink while the invariant (no repeats) is violated."""
    last_seen: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best


def max_sum_fixed_window(a: list[int], k: int) -> int:
    """Fixed-size window: slide by adding the new element and removing the
    one that fell out, instead of recomputing the sum from scratch."""
    window_sum = sum(a[:k])
    best = window_sum
    for right in range(k, len(a)):
        window_sum += a[right] - a[right - k]
        best = max(best, window_sum)
    return best
```

## Example Problems
- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Longest Repeating Character Replacement
- Maximum Sum Subarray of Size K
- Permutation in String
- Fruit Into Baskets (at most 2 distinct types)

## Common Mistakes
- Using a fixed-size template for a variable-size problem (or vice versa) — first decide whether the window size is given or must be discovered.
- Forgetting to update/decrement counts when shrinking the window (classic bug: incrementing a character count on expand but never decrementing on shrink).
- Off-by-one in window length (`right - left + 1` vs `right - left`).

## Complexity
Time $O(n)$ — each pointer traverses the array at most once. Space $O(1)$ to $O(k)$ depending on what the window tracks (a fixed alphabet counter vs. a hash map).

## Related Patterns
- [Two Pointers](two-pointers.md) — sliding window is a specialized, same-direction case.
- [Prefix Sums](prefix-sums.md) — alternative for sum-range queries when the window isn't strictly contiguous-and-growing, or when queries are arbitrary (not streaming).
