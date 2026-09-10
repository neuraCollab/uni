# Stacks, Queues & Deque

## ADTs
- **Stack (LIFO)** — last in, first out. Operations: `push`, `pop`, `peek`, all O(1).
- **Queue (FIFO)** — first in, first out. Operations: `enqueue`, `dequeue`, `peek`, all O(1) *with the right underlying structure*.
- **Deque (double-ended queue)** — push/pop from both ends in O(1); a strict generalization of both stack and queue.

## Python implementations
- **Stack**: plain `list` works well — `append()`/`pop()` (from the end) are both O(1) amortized, so a Python list is already a correct, efficient stack.
- **Queue/Deque**: use `collections.deque`, **not** a plain `list`. A `list` is a dynamic array, so `list.pop(0)` / `list.insert(0, x)` have to shift every remaining element — O(n) per operation. `collections.deque` is backed by a doubly linked list of blocks (see [`linked-lists.md`](linked-lists.md)), giving O(1) append/pop at *both* ends (`append`, `appendleft`, `pop`, `popleft`). This is the single most common "wrong data structure" bug in interview code — using `list` as a queue silently turns an O(n) algorithm into O(n^2).

```python
from collections import deque

stack = []
stack.append(1); stack.append(2); stack.pop()   # LIFO, O(1)

queue = deque()
queue.append(1); queue.append(2); queue.popleft()  # FIFO, O(1)
```

`queue.Queue` also exists in the standard library but is a thread-safe, blocking queue meant for producer/consumer concurrency — overkill and the wrong tool for plain algorithmic use; reach for `collections.deque`.

## Classic uses
- **Valid Parentheses / balanced brackets** — push opening brackets, pop and match on closing brackets; stack empty at the end means balanced.
- **BFS uses a queue** — process nodes level by level, matching the FIFO order in which they were discovered. See [`../patterns/bfs-dfs.md`](../patterns/bfs-dfs.md).
- **DFS uses a stack** (explicit, or implicitly via recursion and the call stack) — explore as deep as possible before backtracking. See [`../patterns/bfs-dfs.md`](../patterns/bfs-dfs.md) and [`../patterns/backtracking.md`](../patterns/backtracking.md).
- **Expression evaluation** — infix-to-postfix conversion, evaluating postfix/RPN expressions, both stack-based.
- **Undo functionality, call stack simulation, DFS iterative traversal** — anywhere "most recently seen, handle first" ordering is needed.

## Monotonic stack
A specialized stack kept strictly increasing or decreasing, used to answer "next greater/smaller element" style questions in O(n) instead of O(n^2). Full pattern writeup, template, and example problems: [`../patterns/monotonic-stack.md`](../patterns/monotonic-stack.md).

## Example Problems
- Valid Parentheses
- Implement Queue using Stacks / Implement Stack using Queues (classic ADT-conversion exercise)
- Sliding Window Maximum (monotonic deque — see [`../patterns/monotonic-stack.md`](../patterns/monotonic-stack.md))
- Min Stack (stack that also supports O(1) `get_min`, via an auxiliary stack tracking running minimums)
- Evaluate Reverse Polish Notation
- Design Circular Queue/Deque

## Common Mistakes
- Using `list.pop(0)` for a queue — O(n) per call, see above; use `deque.popleft()`.
- Popping from an empty stack/deque without checking — always guard with `if stack:` before `pop()`.
- Confusing which end represents "front" vs "back" when manually implementing a queue with two stacks.

## Complexity
All core operations (`push`/`pop`/`append`/`appendleft`/`popleft`) are O(1) with `list` (stack use) or `collections.deque` (queue/deque use). Space O(n) for n elements held.

## Related Patterns
- [Monotonic Stack](../patterns/monotonic-stack.md)
- [BFS/DFS](../patterns/bfs-dfs.md)
- [Backtracking](../patterns/backtracking.md) — DFS-driven exploration, implicitly stack-based via recursion.
