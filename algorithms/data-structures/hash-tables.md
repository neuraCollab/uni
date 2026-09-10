# Hash Tables

## ADT
A hash table maps keys to values, supporting average $O(1)$ insert, lookup, and delete by converting a key into an array index via a **hash function**, instead of scanning or maintaining sorted order.

## How Python's `dict`/`set` work under the hood
- A hash function converts the key to an integer (`hash(key)`); the table maps that integer (modulo the table's capacity) to a bucket/slot.
- **Collisions** (two keys hashing to the same slot) are resolved via **open addressing** in CPython — on collision, probe a pseudo-random sequence of alternative slots until an empty one is found, rather than chaining a linked list per slot.
- **Load factor** $= n / m$ (entries / capacity). CPython resizes (grows the table, typically 4x under ~50k entries, 2x above) once load factor crosses roughly 2/3, keeping probe sequences short. Resizing is $O(n)$ but happens rarely enough that insert stays **amortized** $O(1)$, the same argument as dynamic array append (see [`arrays-strings.md`](arrays-strings.md)).
- Worst case is $O(n)$ per operation if many keys collide (e.g. adversarial input crafted against a weak/known hash function, or worst-case hashing of all-equal keys) — average-case $O(1)$ is what interviews assume unless explicitly probing this.
- Keys must be **hashable** (implement `__hash__` and `__eq__`, and be effectively immutable — `list` is unhashable, `tuple`/`str`/`int`/`frozenset` are). Two equal objects must have equal hashes (required invariant); this is why you can't reliably use a mutable object as a dict key.
- `set` is implemented the same way as `dict` internally, just without stored values (a dict with only keys).

## Common interview questions
- **Two Sum** — single pass, store `value -> index` in a dict, check for `target - current` seen so far. $O(n)$ time, $O(n)$ space vs. $O(n^2)$ brute force.
- **Group Anagrams** — key each string by a canonical form (sorted characters, or a 26-length character count tuple) and bucket into `dict[key, list[str]]`.
- **First non-repeating character** — count frequencies in one pass (`dict` or `collections.Counter`), then scan in order for the first count-1 character.
- Also worth knowing cold: `collections.Counter` (multiset/frequency dict with helpers like `.most_common(k)`), `collections.defaultdict` (avoids `if key not in d` boilerplate).

## When a hash table is the wrong tool
- **Ordered data / range queries** — "give me all keys between X and Y", "find the smallest key >= X" — a hash table has no notion of order; use a sorted structure (sorted array + binary search, a balanced BST, or in Python, `sortedcontainers.SortedDict`/SortedList, a third-party structure) instead. See [`trees-bst.md`](trees-bst.md) and [`../patterns/binary-search.md`](../patterns/binary-search.md).
- **Prefix/nearest-neighbor queries** — needs a trie or spatial structure, not a hash table.
- **Preserving insertion or sorted iteration order as a primary requirement** beyond what's incidental — Python dicts do preserve insertion order since 3.7, but that's an implementation guarantee for iteration, not a substitute for actual sorted-order operations (still $O(n)$ to get a sorted view).
- **Memory-constrained settings with many small keys** — hash tables carry overhead (empty slots for load factor headroom, hash storage) vs. a plain sorted array.

## Complexity summary

| Operation | Average | Worst case |
|---|---|---|
| Insert | $O(1)$ amortized | $O(n)$ |
| Lookup | $O(1)$ | $O(n)$ |
| Delete | $O(1)$ | $O(n)$ |
| Iteration | $O(n)$ | $O(n)$ |

## Related Patterns
- [Two Pointers](../patterns/two-pointers.md), [Sliding Window](../patterns/sliding-window.md) — often combined with a hash map/set to track window contents in $O(1)$.
- [Prefix Sums](../patterns/prefix-sums.md) — the `subarray sum equals K` pattern pairs prefix sums with a hash map of seen sums.
- [Arrays & Strings](arrays-strings.md)
