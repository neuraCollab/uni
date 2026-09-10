"""
Radix sort — two variants, ported from two different source files:

1. LSD (Least Significant Digit first) radix sort on non-negative integers.
   Ported from algos/1_sem/4/radix_sort/radix sort.cpp.
   Processes digits from the ones place outward; each pass is a *stable*
   counting sort on that digit. Because each pass is stable, sorting by
   increasingly significant digits produces a fully sorted array once the
   most significant digit has been processed.

2. MSD (Most Significant Digit first) radix sort on strings.
   Ported from algos/1_sem/2/main.cpp (originally a fixed-alphabet,
   lowercase-only "radix sort of strings" using counting sort per character
   position). Processes characters from the front of the string; strings
   that are shorter than the current index are treated as having a
   "smaller than any letter" sentinel character so they sort before their
   own prefixes.

Key distinction (interview-relevant):
- LSD is simpler to implement correctly for fixed-width keys (e.g. fixed
  number of digits) because you don't need recursion/partitioning.
- MSD is what you reach for with variable-length keys (strings) or when you
  want to stop early once a bucket is uniquely determined. MSD is
  conceptually similar to a quicksort-style partition by radix and is
  typically implemented recursively, per-bucket.
- Both are stable if each internal pass is a stable counting sort.
- Time: O(d * (n + k)) where d = number of digits/characters, k = alphabet
  size (10 for digits, 26/256 for characters). Space: O(n + k).
"""

from __future__ import annotations


def _counting_sort_by_digit(a: list[int], exp: int) -> list[int]:
    """Stable counting sort of `a` by the digit at place value `exp`
    (exp = 1, 10, 100, ...). Base 10.
    """
    n = len(a)
    output = [0] * n
    count = [0] * 10

    for x in a:
        digit = (x // exp) % 10
        count[digit] += 1

    for d in range(1, 10):
        count[d] += count[d - 1]

    for i in range(n - 1, -1, -1):
        digit = (a[i] // exp) % 10
        count[digit] -= 1
        output[count[digit]] = a[i]

    return output


def radix_sort_lsd(a: list[int]) -> list[int]:
    """LSD radix sort for non-negative integers."""
    if not a:
        return []
    a = list(a)
    max_val = max(a)
    exp = 1
    while max_val // exp > 0:
        a = _counting_sort_by_digit(a, exp)
        exp *= 10
    return a


_ALPHABET_SIZE = 26  # a-z; bump to 256 for arbitrary bytes/unicode-ish text


def _char_at(s: str, index: int) -> int:
    """Map s[index] to 0..25 (a-z), or -1 if the string is shorter than
    index (so shorter strings sort before their own longer prefixes).
    """
    if index < len(s):
        return ord(s[index]) - ord("a")
    return -1


def _counting_sort_by_char(strings: list[str], index: int) -> list[str]:
    """Stable counting sort of `strings` by the character at `index`."""
    n = len(strings)
    output = [""] * n
    count = [0] * (_ALPHABET_SIZE + 1)  # +1 slot for the "-1" sentinel

    for s in strings:
        count[_char_at(s, index) + 1] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        c = _char_at(strings[i], index) + 1
        count[c] -= 1
        output[count[c]] = strings[i]

    return output


def radix_sort_msd_strings(strings: list[str]) -> list[str]:
    """MSD-style radix sort for lowercase a-z strings.

    Note: the source implementation runs MSD passes iteratively from the
    longest string's last index down to 0 (i.e. it sorts by the *last*
    character first). This works because each pass is a stable counting
    sort — sorting by increasingly significant character positions,
    finishing with the most significant (index 0), is equivalent to true
    LSD-on-characters and produces a correct lexicographic sort. It is
    included here as "MSD" per the source material's naming/intent, but
    note it is actually implemented as an LSD-style pass over character
    positions (simpler than a recursive true MSD/three-way radix quicksort).
    """
    if not strings:
        return []
    strings = list(strings)
    max_len = max(len(s) for s in strings)

    for index in range(max_len - 1, -1, -1):
        strings = _counting_sort_by_char(strings, index)

    return strings


if __name__ == "__main__":
    nums = [170, 45, 75, 90, 802, 24, 2, 66]
    print(radix_sort_lsd(nums))

    words = ["bob", "alice", "bo", "ann", "bobby"]
    print(radix_sort_msd_strings(words))
