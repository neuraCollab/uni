# Prefix Sums

## Recognition
### Key clues
- Many **range-sum** or **range-count** queries over a static (non-mutating) array.
- "Subarray sum equals K" style problems.
- Asked for cumulative totals, running balance, or "number of elements in range [l, r]".
- 2D grid range-sum queries (region sum) — 2D prefix sums.

## Pattern
Precompute a running total `prefix[i] = a[0] + a[1] + ... + a[i-1]` once in $O(n)$. Any range sum `a[l..r]` is then `prefix[r+1] - prefix[l]` in $O(1)$, instead of re-summing the range every query.

## Why this works
Sum is associative and the "total up to index i" only needs to be computed once; every range query is just a subtraction of two precomputed totals. This trades $O(n)$ preprocessing + $O(1)$ space per prefix array for $O(1)$ queries, vs. $O(n)$ per query with no preprocessing.

## Template
```python
def build_prefix_sums(a: list[int]) -> list[int]:
    prefix = [0] * (len(a) + 1)
    for i, x in enumerate(a):
        prefix[i + 1] = prefix[i] + x
    return prefix


def range_sum(prefix: list[int], left: int, right: int) -> int:
    """Sum of a[left..right] inclusive, O(1)."""
    return prefix[right + 1] - prefix[left]


def subarray_sum_equals_k(a: list[int], k: int) -> int:
    """Count subarrays summing to k using a running prefix sum + hash map
    of prefix-sum frequencies (classic combination with hashing)."""
    count = {0: 1}
    running = 0
    total = 0
    for x in a:
        running += x
        total += count.get(running - k, 0)
        count[running] = count.get(running, 0) + 1
    return total
```

A related, non-arithmetic version: a **frequency-array prefix sum** for
bounded-range values, used to answer "how many elements have value in
[low, high]" — see `range_count_query` in
[`sorting/code/counting_sort.py`](../sorting/code/counting_sort.py), ported
from `algos/1_sem/1/delimetr_sort_finder.cpp`. There, instead of prefix-summing
the array itself, you prefix-sum a frequency/bucket array built by counting
sort, then a range-count query is a slice-sum over buckets.

```python
def range_count_query(a: list[int], k: int, low: int, high: int) -> int:
    """Count elements of `a` (values in [0, k)) that fall in [low, high]."""
    counts = [0] * k
    for x in a:
        counts[x] += 1
    # counts is now a frequency table; prefix-sum it for O(1) queries,
    # or just sum the relevant slice for a one-off query.
    return sum(counts[low : high + 1])
```

## Example Problems
- Range Sum Query - Immutable
- Subarray Sum Equals K
- Product of Array Except Self (prefix/suffix products)
- Contiguous Array (equal 0s and 1s)
- Range Sum Query 2D - Immutable
- Continuous Subarray Sum (divisible by K)

## Common Mistakes
- Off-by-one: `prefix` is usually sized `n+1` with `prefix[0] = 0` so `range_sum(l, r) = prefix[r+1] - prefix[l]`; mixing up inclusive/exclusive bounds is the #1 bug here.
- Rebuilding prefix sums from scratch after a mutation — plain prefix sums assume a **static** array; if updates are needed, use a Fenwick tree / BIT or segment tree instead.
- Forgetting the `{0: 1}` seed in the prefix-sum + hashmap pattern (needed so a subarray starting at index 0 that itself sums to k is counted).

## Complexity
Build: $O(n)$ time, $O(n)$ space. Query: $O(1)$ time per query after preprocessing.

## Related Patterns
- [Sliding Window](sliding-window.md) — better when the array *does* change or when you need the actual subarray, not just its sum, and the values are non-negative (monotonic window sum).
- [Two Pointers](two-pointers.md)
