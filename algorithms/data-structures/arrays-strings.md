# Arrays & Strings

## ADT operations
- Index access/update: $O(1)$.
- Append (amortized $O(1)$ for dynamic arrays like Python `list`), insert/delete at arbitrary position: $O(n)$ (shifts elements).
- Search (unsorted): $O(n)$. Search (sorted): $O(\log n)$ via [binary search](../patterns/binary-search.md).
- Strings: same as arrays of characters, but **immutable** in Python — every "modification" creates a new string object.

## Python notes
- `list` is a dynamic array (over-allocates capacity so `append` is amortized $O(1)$); `list.insert(0, x)` / `list.pop(0)` are $O(n)$ — use `collections.deque` if you need $O(1)$ operations at both ends (see [`stacks-queues-deque.md`](stacks-queues-deque.md)).
- Strings are immutable: repeated concatenation in a loop (`s += char`) is $O(n^2)$ total — build a `list[str]` and `"".join(...)` instead ($O(n)$).
- Slicing (`a[i:j]`) copies — $O(k)$ for a slice of length k, not $O(1)$. Easy to accidentally introduce hidden $O(n)$ costs inside a loop.
- `list.sort()` / `sorted()` use Timsort: $O(n \log n)$ worst case, $O(n)$ best case (already-sorted or nearly-sorted input), stable.
- `str` methods worth knowing cold: `.split()`, `.strip()`, `.join()`, `.find()`/`.index()`, `.count()`, slicing with step (`s[::-1]` for reverse).
- `array` module / `bytearray` for mutable fixed-type buffers if an interview specifically probes memory layout — rarely needed for LeetCode-style problems.

## Common interview questions
- Reverse a string/array in place.
- Rotate an array by k (in place, $O(1)$ extra space — the three-reversals trick).
- Two Sum, Product of Array Except Self, Kadane's algorithm (max subarray sum).
- Valid Anagram, Group Anagrams (sort/count characters as a key).
- In-place partitioning (Dutch National Flag / sort colors).

## Common Mistakes
- Off-by-one errors on slice/range boundaries (`a[i:j]` is half-open — excludes `j`).
- Mutating a list while iterating over it (skips or duplicates elements) — iterate over a copy, or build a new list, when removing elements.
- Assuming string concatenation in a loop is $O(1)$ per operation (it's not — see above).
- Forgetting Python string/list comparisons and equality are $O(n)$, not $O(1)$ — matters when reasoning about complexity of, e.g., putting strings in a set/dict (hashing itself is $O(n)$ on the string, subsequent comparisons on collision also $O(n)$).

## Complexity summary

| Operation | Array/list | String |
|---|---|---|
| Index access | $O(1)$ | $O(1)$ |
| Append (end) | $O(1)$ amortized | $O(n)$ (new object) |
| Insert/delete (middle) | $O(n)$ | $O(n)$ (new object) |
| Search (unsorted) | $O(n)$ | $O(n)$ |
| Search (sorted) | $O(\log n)$ | $O(\log n)$ |
| Slice of length k | $O(k)$ | $O(k)$ |

## Related Patterns
- [Two Pointers](../patterns/two-pointers.md), [Sliding Window](../patterns/sliding-window.md), [Prefix Sums](../patterns/prefix-sums.md) — the three dominant array/string techniques.
