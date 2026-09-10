"""
Merge sort.

Ported from algos/1_sem/3/merge-sort/merge_sort.cpp.

Classic divide-and-conquer: split the array in half, recursively sort each
half, then merge the two sorted halves in O(n). Stable, O(n log n) time,
O(n) extra space (not in-place — this is the standard interview trade-off
vs. heap sort/quicksort).
"""

from __future__ import annotations


def merge_sort(a: list[int]) -> list[int]:
    """Return a new sorted list. Non-mutating, top-down recursive merge sort."""
    if len(a) <= 1:
        return list(a)

    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return _merge(left, right)


def _merge(left: list[int], right: list[int]) -> list[int]:
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def merge_sort_inplace(a: list[int], lo: int = 0, hi: int | None = None) -> None:
    """In-place-ish variant matching the source's index-range signature
    (mergeSort(arr, from, to)). Still allocates O(n) temp space per merge,
    same as the C++ original — true in-place merging is not practical.
    """
    if hi is None:
        hi = len(a) - 1
    if lo < hi:
        mid = lo + (hi - lo) // 2
        merge_sort_inplace(a, lo, mid)
        merge_sort_inplace(a, mid + 1, hi)
        _merge_inplace(a, lo, mid, hi)


def _merge_inplace(a: list[int], lo: int, mid: int, hi: int) -> None:
    left = a[lo : mid + 1]
    right = a[mid + 1 : hi + 1]
    i = j = 0
    k = lo
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            a[k] = left[i]
            i += 1
        else:
            a[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        a[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        a[k] = right[j]
        j += 1
        k += 1


if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10]
    print(merge_sort(data))

    data2 = [38, 27, 43, 3, 9, 82, 10]
    merge_sort_inplace(data2)
    print(data2)
