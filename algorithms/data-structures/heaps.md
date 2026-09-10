# Heaps (Priority Queues)

## Heap property
A **binary heap** is a complete binary tree (filled level by level, left to right) satisfying the heap-order invariant:
- **Min-heap**: every parent <= its children -> the minimum element is always at the root.
- **Max-heap**: every parent >= its children -> the maximum element is always at the root.

Unlike a [BST](trees-bst.md), a heap only guarantees the parent/child relationship, not a full ordering between siblings or across subtrees — so it supports fast min/max access but not fast arbitrary search. Typically array-backed (no explicit pointers): for a node at index `i`, children are at `2i+1` and `2i+2`, parent at `(i-1)//2`.

## Python's `heapq`
`heapq` implements a **min-heap only**, operating in place on a plain `list`.
```python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
heapq.heappop(heap)      # 1 (smallest)
heapq.heapify(existing_list)  # convert a list to a heap in place, O(n)
```
**Max-heap trick**: since `heapq` is min-heap only, negate values on push/pop (`heapq.heappush(heap, -x)`, then negate again on pop) to simulate a max-heap. For heaps of tuples/objects, negate just the sort key, or wrap with a custom `__lt__`.

`heapq.nlargest(k, iterable)` / `heapq.nsmallest(k, iterable)` are ready-made $O(n \log k)$ helpers for "top K" — reach for them before hand-rolling unless the interview specifically wants the heap mechanics shown.

## Complexity
- `heappush` / `heappop`: $O(\log n)$ — sift up/down at most the tree's height.
- `heapify` (build a heap from an existing list): $O(n)$, **not** $O(n \log n)$ — a classic surprising result. Pushing n elements one at a time is $O(n \log n)$; bottom-up heapify is $O(n)$ because most nodes are near the bottom and sift down only a short distance.
- Peek (`heap[0]`): $O(1)$.

## Common interview questions
- **Kth largest/smallest element** — maintain a heap of size k (min-heap for kth largest: pop whenever size exceeds k, root ends up being the kth largest) for $O(n \log k)$, beating a full $O(n \log n)$ sort when k is small.
- **Merge k sorted lists** — push the head of each list into a min-heap keyed by value; repeatedly pop the smallest, push its successor. $O(n \log k)$ where n = total elements, k = number of lists. See [`linked-lists.md`](linked-lists.md).
- **Top-K frequent elements** — count frequencies (see [`hash-tables.md`](hash-tables.md)), then use a heap of size k over (frequency, value) pairs, or `heapq.nlargest`.
- **Median from a data stream** — maintain two heaps: a max-heap for the lower half, a min-heap for the upper half, kept balanced in size (differ by at most 1). The median is the top of the larger heap, or the average of both tops when sizes are equal. Each insert is $O(\log n)$; median query is $O(1)$.

## Example Problems
- Kth Largest Element in an Array
- Merge k Sorted Lists
- Top K Frequent Elements
- Find Median from Data Stream
- Task Scheduler (max-heap by remaining count)
- K Closest Points to Origin

## Common Mistakes
- Forgetting `heapq` is min-heap only and comparing/popping as if it were max — always negate explicitly for max-heap behavior.
- Using a full sort ($O(n \log n)$) when a size-bounded heap ($O(n \log k)$) is the whole point of the question — recognize "kth"/"top K" as the heap-of-size-k signal.
- Pushing non-comparable or ambiguous tuples (e.g. `(priority, object)` where `object` isn't comparable) — Python falls back to comparing the second tuple element on ties and can raise a `TypeError`; break ties explicitly with a unique counter, e.g. `(priority, counter, object)`.
- Assuming heap order gives a fully sorted array by iterating the underlying list directly — it doesn't; only repeated `heappop` yields sorted order (see [`heap_sort.py`](../sorting/code/heap_sort.py) for the heap-based sorting version, which repeatedly extracts the root).

## Complexity summary

| Operation | Complexity |
|---|---|
| Push | $O(\log n)$ |
| Pop (min/max) | $O(\log n)$ |
| Peek | $O(1)$ |
| Build from list (heapify) | $O(n)$ |
| Kth largest/smallest via size-k heap | $O(n \log k)$ |

## Related Patterns
- [Sorting notes](../sorting/notes.md) — heap sort uses the same underlying structure.
- [Linked Lists](linked-lists.md) — merge k sorted lists.
- [Hash Tables](hash-tables.md) — frequency counting paired with a heap for top-K problems.
