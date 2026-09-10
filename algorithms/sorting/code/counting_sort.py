"""
Counting sort.

Non-comparison sort for integers in a small, known range [0, k).
Stable, O(n + k) time, O(k) extra space.

Ported/translated from algos/1_sem/1/counting_sort.cpp (which fused the sort
with a "how many elements fall in range [l, r]" query — that range-query use
case is split out separately below, and also documented as a prefix-sums
example in algorithms/patterns/prefix-sums.md).
"""

from __future__ import annotations


def counting_sort(a: list[int], k: int | None = None) -> list[int]:
    """Sort a list of non-negative ints in [0, k) using counting sort.

    Args:
        a: input list of non-negative integers.
        k: exclusive upper bound on values. Defaults to max(a) + 1.

    Returns:
        A new sorted list (input is not mutated).
    """
    if not a:
        return []
    if k is None:
        k = max(a) + 1

    counts = [0] * k
    for x in a:
        counts[x] += 1

    result = []
    for value, count in enumerate(counts):
        result.extend([value] * count)
    return result


def counting_sort_stable(a: list[int], key=lambda x: x, k: int | None = None) -> list[int]:
    """Stable counting sort that preserves relative order of equal keys.

    This is the variant radix sort relies on internally (see radix_sort.py):
    it uses a cumulative/prefix-sum count array to place each element at its
    final index in one pass, rather than just rebuilding by frequency.
    """
    if not a:
        return []
    if k is None:
        k = max(key(x) for x in a) + 1

    counts = [0] * k
    for item in a:
        counts[key(item)] += 1

    # Cumulative counts -> counts[v] becomes "index just past the last slot
    # for value v" (prefix sum over the frequency array).
    for v in range(1, k):
        counts[v] += counts[v - 1]

    output = [None] * len(a)
    for item in reversed(a):  # reversed traversal keeps it stable
        v = key(item)
        counts[v] -= 1
        output[counts[v]] = item
    return output


def range_count_query(a: list[int], k: int, low: int, high: int) -> int:
    """How many elements of `a` (values in [0, k)) fall in [low, high]?

    This is the pattern from delimetr_sort_finder.cpp: build the frequency
    array once with counting sort, then answer range-count queries in O(1)
    each by summing (or prefix-summing) the relevant buckets.
    See also: algorithms/patterns/prefix-sums.md
    """
    counts = [0] * k
    for x in a:
        counts[x] += 1
    return sum(counts[low : high + 1])


if __name__ == "__main__":
    data = [4, 2, 2, 8, 3, 3, 1]
    print(counting_sort(data))              # [1, 2, 2, 3, 3, 4, 8]
    print(range_count_query(data, 9, 2, 3))  # elements with value in [2,3] -> 4
