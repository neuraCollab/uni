"""
Shell sort with a Pratt gap sequence.

Ported from algos/1_sem/2/shell_pratt.cpp.

Shell sort generalizes insertion sort: instead of comparing adjacent
elements, it compares elements `gap` apart, for a decreasing sequence of
gaps ending in 1 (so the final pass is a plain insertion sort, but by then
the array is "almost sorted" and that pass is cheap).

The Pratt sequence uses gaps of the form 2^i * 3^j (i.e. 3-smooth numbers:
1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 27, ...). It gives O(n log^2 n) worst case,
better than the naive gap/2 sequence's O(n^2).
"""

from __future__ import annotations


def _pratt_sequence(n: int) -> list[int]:
    """All gaps of the form 2^i * 3^j that are <= n // 2, ascending."""
    gaps = {1}
    i = 0
    while (1 << i) <= n // 2 or i == 0:
        p = 1 << i
        j = 0
        while p * (3**j) <= n // 2:
            gaps.add(p * (3**j))
            j += 1
        if (1 << i) > n // 2:
            break
        i += 1
    return sorted(gaps)


def shell_sort(a: list[int]) -> list[int]:
    """Shell sort using Pratt gaps, descending gap order. In-place-style
    (returns a new list built from a copy; the algorithm itself sorts via
    index swaps, matching the original C++ pointer-based implementation).
    """
    a = list(a)
    n = len(a)
    if n < 2:
        return a

    gaps = _pratt_sequence(n)

    for gap in reversed(gaps):
        for i in range(gap, n):
            j = i
            while j >= gap and a[j - gap] > a[j]:
                a[j - gap], a[j] = a[j], a[j - gap]
                j -= gap

    return a


if __name__ == "__main__":
    data = [64, 34, 25, 12, 22, 11, 90, 5]
    print(shell_sort(data))
