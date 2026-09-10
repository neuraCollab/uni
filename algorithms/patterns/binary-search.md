# Binary Search

## Recognition
### Key clues
- Input is **sorted** (or has a monotonic/"boolean" structure — a predicate that's `False` then `True` across the range).
- Asked for O(log n) on a search/lookup problem.
- "Find the minimum/maximum X such that condition holds" — **binary search on the answer**, not necessarily on a sorted array.
- Rotated sorted array, or "search in a matrix sorted row-wise and column-wise".

## Pattern
Repeatedly halve the search space by checking the midpoint against a monotonic predicate, discarding the half that can't contain the answer.

## Why this works
If a predicate `P(x)` is `False` for all `x < threshold` and `True` for all `x >= threshold` (monotonic), checking the midpoint tells you which half of the remaining range can possibly contain the threshold — so each check eliminates half the candidates, giving O(log n) checks total.

## Template
```python
def binary_search(a: list[int], target: int) -> int:
    """Classic search for an exact value in a sorted array. Returns index
    or -1."""
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if a[mid] == target:
            return mid
        elif a[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def binary_search_on_answer(lo: int, hi: int, feasible) -> int:
    """'Search the answer space' variant: find the smallest x in [lo, hi]
    for which feasible(x) is True, given feasible is monotonic
    (False...False True...True). Classic for "minimize the maximum" /
    "maximize the minimum" optimization problems.
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def find_left_boundary(a: list[int], target: int) -> int:
    """Leftmost insertion point for target (a.k.a. lower_bound / bisect_left).
    Standard building block for "first/last occurrence" problems.
    """
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

Python's `bisect` module (`bisect_left`, `bisect_right`, `insort`) implements the boundary-search variant directly — reach for it before hand-rolling.

## Example Problems
- Binary Search (basic)
- Search in Rotated Sorted Array
- Find First and Last Position of Element in Sorted Array
- Koko Eating Bananas (binary search on the answer)
- Capacity To Ship Packages Within D Days (binary search on the answer)
- Median of Two Sorted Arrays
- Search a 2D Matrix

## Common Mistakes
- `mid = (lo + hi) // 2` can overflow in languages with fixed-width ints (not a Python concern, but know `lo + (hi - lo) // 2` as the safe idiom — interviewers sometimes ask about it).
- Infinite loops from an inconsistent `lo`/`hi` update rule — always double check that the search space strictly shrinks on every branch.
- Applying binary search to data that *looks* sorted but isn't fully monotonic (e.g. a rotated array needs a modified check, not a plain `a[mid] < target`).
- Confusing `bisect_left` (leftmost insertion point) with `bisect_right` (rightmost) when looking for first vs. last occurrence.

## Complexity
Time O(log n) per search. Space O(1) iterative (O(log n) if recursive, due to call stack).

## Related Patterns
- [Two Pointers](two-pointers.md)
- [Monotonic Stack](monotonic-stack.md) — different tool, but also exploits a monotonic structure.
