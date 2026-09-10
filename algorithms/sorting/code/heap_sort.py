"""
Ternary (3-ary) heap sort.

Ported from algos/1_sem/4/triple_pile/triple_pile.cpp — note the source
filename ("triple_pile") is misleading; this is a heapsort variant, not a
pile/stack structure. "Triple" refers to each heap node having up to 3
children instead of the usual 2.

Standard heap sort uses a binary max-heap (child indices 2i+1, 2i+2). This
variant uses a ternary max-heap (child indices 3i+1, 3i+2, 3i+3), which
makes the heap shorter (height ~log_3 n instead of ~log_2 n) at the cost of
comparing 3 children instead of 2 per sift-down step. Asymptotically both
are O(n log n); the base of the log changes but not the complexity class.
This is a good "do you actually understand heapify, or did you memorize
binary heap indices" interview probe.
"""

from __future__ import annotations


def _heapify(a: list[int], heap_size: int, root: int) -> None:
    largest = root
    left = 3 * root + 1
    mid = 3 * root + 2
    right = 3 * root + 3

    for child in (left, mid, right):
        if child < heap_size and a[child] > a[largest]:
            largest = child

    if largest != root:
        a[root], a[largest] = a[largest], a[root]
        _heapify(a, heap_size, largest)


def heap_sort_ternary(a: list[int]) -> list[int]:
    """Sort ascending using a ternary max-heap. Returns a new list."""
    a = list(a)
    n = len(a)

    # Build max-heap: last parent index in a ternary heap is n // 3 - 1.
    for i in range(n // 3 - 1, -1, -1):
        _heapify(a, n, i)

    # Repeatedly move the max to the end and shrink the heap.
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        _heapify(a, end, 0)

    return a


def _heapify_binary(a: list[int], heap_size: int, root: int) -> None:
    """Standard binary heapify, included for direct comparison with the
    ternary version above.
    """
    largest = root
    left = 2 * root + 1
    right = 2 * root + 2

    if left < heap_size and a[left] > a[largest]:
        largest = left
    if right < heap_size and a[right] > a[largest]:
        largest = right

    if largest != root:
        a[root], a[largest] = a[largest], a[root]
        _heapify_binary(a, heap_size, largest)


def heap_sort_binary(a: list[int]) -> list[int]:
    """The textbook binary-heap heap sort, for reference/comparison."""
    a = list(a)
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        _heapify_binary(a, n, i)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        _heapify_binary(a, end, 0)
    return a


if __name__ == "__main__":
    data = [12, 3, 45, 6, 89, 1, 34, 22]
    print(heap_sort_ternary(data))
    print(heap_sort_binary(data))
