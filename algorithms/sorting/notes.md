# Sorting Algorithms — Interview Notes

Code implementations: [`code/`](code/) — `counting_sort.py`, `radix_sort.py`, `shell_sort.py`, `merge_sort.py`, `heap_sort.py`.

## Complexity table

| Algorithm | Best | Average | Worst | Space | Stable | In-place |
|---|---|---|---|---|---|---|
| Counting sort | $O(n + k)$ | $O(n + k)$ | $O(n + k)$ | $O(n + k)$ | Yes | No |
| Radix sort (LSD) | $O(d(n + k))$ | $O(d(n + k))$ | $O(d(n + k))$ | $O(n + k)$ | Yes | No |
| Shell sort (Pratt gaps) | $O(n \log n)$ | ~$O(n \log n)$ | $O(n \log^2 n)$ | $O(1)$ | No | Yes |
| Merge sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes | No |
| Heap sort (binary or ternary) | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ | No | Yes |
| Quicksort (reference, not ported) | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | No | Yes |
| Python `sorted`/`list.sort` (Timsort) | $O(n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes | No |

`d` = number of digits/characters in the key, `k` = range of values / alphabet size.

## When each is preferred

- **Counting sort** — values are integers in a small known range (e.g. ages 0-120, grades 0-100). Not comparison-based, so it beats the $O(n \log n)$ lower bound. Useless if `k >> n` (wastes memory/time). Also the building block for a bucketed **range-count query**: precompute the frequency array once, answer "how many elements in [l, r]" queries via a prefix sum over buckets — see [`patterns/prefix-sums.md`](../patterns/prefix-sums.md).
- **Radix sort** — fixed-width keys (fixed number of digits, or fixed-length/bounded strings) where `d` is small relative to `log n`. LSD radix sort (process least-significant digit first) is simplest for numbers. MSD (most-significant first) suits variable-length strings and can short-circuit early. Both need a **stable** sort as the per-digit subroutine (we use counting sort).
- **Shell sort** — good "no extra memory, better than insertion sort, and I can code it from memory" answer. The gap sequence matters a lot: naive `n/2, n/4, ..., 1` degrades to $O(n^2)$ worst case; Pratt's 3-smooth sequence (1, 2, 3, 4, 6, 8, 9, 12, ...) gives $O(n \log^2 n)$. Rarely the "right" production answer today but a common systems-programming/embedded choice (no heap allocation, decent cache behavior).
- **Merge sort** — need *guaranteed* $O(n \log n)$ and stability (e.g. sorting objects by one key while preserving original order of ties), or sorting linked lists (merge is naturally $O(1)$ extra space there) or external/out-of-core sorting (merge naturally streams). Costs $O(n)$ extra space for arrays.
- **Heap sort** — need guaranteed $O(n \log n)$ *and* $O(1)$ extra space (in-place), and don't need stability. Building block for "top K" / priority queue problems — see [`data-structures/heaps.md`](../data-structures/heaps.md). A ternary (or d-ary) heap trades more per-node comparisons for a shorter tree; same asymptotic class, occasionally better constants/cache behavior for large fan-out.

## Stability

Stable sort = equal-key elements keep their relative input order. Matters when sorting by one key but the original order encodes another (e.g. stable-sort transactions by amount after having sorted them by date — a second stable sort by amount preserves the date order within equal amounts). Counting sort, radix sort, and merge sort are stable *as implemented above*; heap sort and (naive) shell sort are not.

## Which sort would you implement from memory in an interview?

1. **Merge sort** — the safest "write correct $O(n \log n)$ code under pressure" choice; recursion + merge is easy to reason about and debug.
2. **Quicksort** (not ported here, but know it) — often the expected answer for "in-place $O(n \log n)$ on average"; be ready to discuss worst-case $O(n^2)$ and mitigations (random pivot, median-of-three, introsort fallback to heap sort — this is literally what C++ `std::sort` and many stdlib sorts do).
2. **Heap sort** — when asked specifically for guaranteed $O(n \log n)$ with $O(1)$ space, or when the problem is really "give me the k largest/smallest" (then you don't even need to fully sort — use a heap of size k).
3. **Counting/radix sort** — when the interviewer's problem statement telegraphs bounded integer ranges or fixed-width keys ("ages between 0 and 100", "sort n numbers each with at most d digits"). Recognizing *when* a non-comparison sort applies is often the actual signal being tested.

Common mistake carried over from the original source material: forgetting to guard against arrays of size 0 or 1 before entering the sort (radix sort in particular can loop forever / index out of range on a size-0 or size-1 input if you don't short-circuit).

## Related

- [`patterns/prefix-sums.md`](../patterns/prefix-sums.md) — range-count query built on counting sort's frequency array.
- [`data-structures/heaps.md`](../data-structures/heaps.md) — heaps, `heapq`, top-K pattern.
