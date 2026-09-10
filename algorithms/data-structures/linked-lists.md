# Linked Lists

## Singly vs. doubly linked
- **Singly linked**: each node holds a value and a `next` pointer. $O(1)$ insert/delete at the head; $O(n)$ to reach an arbitrary node or the tail (unless a tail pointer is kept); traversal is one-directional.
- **Doubly linked**: each node also holds a `prev` pointer. $O(1)$ insert/delete given a reference to the node (no need to find its predecessor), and bidirectional traversal — at the cost of extra memory per node and more pointer bookkeeping. `collections.deque` in Python is backed by a doubly linked list of blocks (see [`stacks-queues-deque.md`](stacks-queues-deque.md)).

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

## Common operations
- Traverse: $O(n)$.
- Insert/delete at head: $O(1)$.
- Insert/delete at tail: $O(1)$ with a tail pointer (singly), $O(n)$ without.
- Insert/delete given a node reference (doubly linked): $O(1)$.
- Search by value: $O(n)$ — no random access, unlike arrays (see [`arrays-strings.md`](arrays-strings.md) for the array-side comparison).

## Classic interview tricks

### Fast/slow pointers — cycle detection (Floyd's algorithm)
Advance `slow` by one node and `fast` by two each step. If there's a cycle, `fast` eventually laps `slow` and they meet inside it; if `fast` hits `None`, there's no cycle. To find the cycle's *start* node, after they meet, reset one pointer to the head and advance both one step at a time — they meet again exactly at the cycle's entry (a classic mathematical property of this construction).

```python
def has_cycle(head: ListNode | None) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

### Reversing a linked list
Iterative ($O(1)$ space):
```python
def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev
```
Recursive ($O(n)$ call stack space, but often the cleaner answer to write first):
```python
def reverse_list_recursive(head: ListNode | None) -> ListNode | None:
    if head is None or head.next is None:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

### Finding the middle node
Same fast/slow pointer idea, no cycle needed: when `fast` reaches the end, `slow` is at the middle. One pass, $O(1)$ space — avoids a first pass just to count length.
```python
def middle_node(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### Merging two sorted lists
Same idea as the merge step of [merge sort](../sorting/notes.md): walk both lists with two pointers, always attaching the smaller current node, splice in whatever remains when one list is exhausted.
```python
def merge_two_lists(a: ListNode | None, b: ListNode | None) -> ListNode | None:
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next
```

## Example Problems
- Linked List Cycle / Linked List Cycle II (find the entry point)
- Reverse Linked List / Reverse Linked List II (reverse a sub-range)
- Middle of the Linked List
- Merge Two Sorted Lists / Merge k Sorted Lists (combine with a [heap](heaps.md) for k > 2)
- Remove Nth Node From End of List (fast pointer offset by n, then move both)
- Reorder List, Palindrome Linked List (find middle + reverse second half + merge)

## Common Mistakes
- Losing the reference to the rest of the list before reassigning `.next` during reversal (always save `head.next` before overwriting it, or use tuple assignment as above).
- Off-by-one when using a **dummy head node** — a very common and useful trick for edge cases (removing the actual head, empty list) but easy to forget to `return dummy.next` instead of `dummy`.
- Not handling `None`/empty-list and single-node inputs as edge cases.
- Forgetting to null out `.next` on the final reversed node (leaves a stray forward link, or in cycle-adjacent problems, can accidentally recreate a cycle).

## Complexity
Traversal-based operations are $O(n)$ time; the pointer tricks above are $O(n)$ time, $O(1)$ extra space (except recursive reversal, which is $O(n)$ space on the call stack).

## Related Patterns
- [Two Pointers](../patterns/two-pointers.md) — fast/slow is a same-direction, different-speed variant of two pointers.
- [Heaps](heaps.md) — merging k sorted lists.
