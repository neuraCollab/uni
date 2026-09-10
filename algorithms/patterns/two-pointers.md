# Two Pointers

## Recognition
### Key clues
- Input is a **sorted** array (or can be sorted without losing needed info).
- Asked to find a **pair/triplet** with a target sum, difference, or product.
- Asked to **remove duplicates in place**, partition, or merge two sorted sequences.
- Palindrome checks (pointers converging from both ends).
- "In-place", $O(1)$ extra space is required or implied.

## Pattern
Use two indices that move through the array (toward each other, or both forward at different speeds) instead of nested loops, exploiting sortedness or a monotonic relationship to eliminate a whole branch of the search space on each step.

## Why this works
On a sorted array, if `a[left] + a[right] > target`, no pair involving the current `right` and anything to its left of `left` can work better than moving `right` down — so we can safely discard one end instead of checking all pairs. This turns an $O(n^2)$ pair search into $O(n)$.

## Template
```python
def two_sum_sorted(a: list[int], target: int) -> tuple[int, int] | None:
    """Anchor example — ported from algos/1_sem/3/sum/sum.cpp
    (findPairsWithSum): given a sorted array, find indices of a pair
    summing to `target`.
    """
    left, right = 0, len(a) - 1
    while left < right:
        current = a[left] + a[right]
        if current == target:
            return left, right
        elif current < target:
            left += 1
        else:
            right -= 1
    return None


# Same-direction variant (fast/slow pointers) — e.g. removing duplicates
# in place from a sorted array.
def remove_duplicates(a: list[int]) -> int:
    if not a:
        return 0
    slow = 0
    for fast in range(1, len(a)):
        if a[fast] != a[slow]:
            slow += 1
            a[slow] = a[fast]
    return slow + 1  # new length
```

## Example Problems
- Two Sum II (sorted array)
- 3Sum / 3Sum Closest
- Container With Most Water
- Remove Duplicates from Sorted Array
- Valid Palindrome
- Trapping Rain Water (two pointers variant)

## Common Mistakes
- Forgetting the array must be sorted (or sorting it first) before applying the converging-pointer technique — sorting costs $O(n \log n)$ and can change the answer if original indices matter (keep an index-value pairing if you need original positions).
- Off-by-one on the `while left < right` vs `left <= right` boundary.
- Not skipping duplicate values in `3Sum`-style problems, causing duplicate triplets in the output.

## Complexity
Time $O(n)$ (or $O(n \log n)$ if a sort is required first). Space $O(1)$ extra (excluding output).

## Related Patterns
- [Sliding Window](sliding-window.md) — same-direction two pointers with a "window" invariant.
- [Binary Search](binary-search.md) — alternative when only one of the two values is unknown per step.
