# Trees & Binary Search Trees

## Terminology
- **Root** — the top node (no parent). **Leaf** — a node with no children. **Height** — longest path from a node down to a leaf. **Depth** — distance from the root to a node.
- **Binary tree** — each node has at most 2 children (commonly `left`/`right`).
- **Balanced** — height is $O(\log n)$ relative to the number of nodes (roughly, left and right subtrees don't differ in height by more than a small constant at every node).
- **Degenerate/skewed tree** — every node has only one child, effectively a linked list — height $O(n)$.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## Traversals
- **Inorder** (left, node, right) — visits a BST's values in sorted order (see below).
- **Preorder** (node, left, right) — natural for copying/serializing a tree (parent before children).
- **Postorder** (left, right, node) — natural for deleting a tree or computing values that depend on children first (e.g. subtree size/height).
- **Level-order** (BFS, level by level) — uses a queue; see [`../patterns/bfs-dfs.md`](../patterns/bfs-dfs.md) for the general BFS template.

```python
def inorder(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


def preorder(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)


def postorder(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]


def level_order(root: TreeNode | None) -> list[list[int]]:
    from collections import deque
    if root is None:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```
(The recursive traversals above favor clarity over efficiency — list concatenation is $O(n)$ per call; an interview-quality version accumulates into a shared list passed by reference instead.)

## BST property
For every node, all values in the **left** subtree are smaller and all values in the **right** subtree are larger (assuming no duplicates). Direct consequence: **inorder traversal of a BST visits values in sorted order** — this is the single most useful BST fact for interviews (validate-BST, kth-smallest, and "convert BST to sorted list" all fall out of it directly).

## Insert / search / delete
- **Search/insert**: walk left or right by comparing against the current node, same idea as [binary search](../patterns/binary-search.md) but over a tree instead of an array.
- **Delete**: three cases — leaf (just remove it), one child (splice it up), two children (replace the node's value with its inorder successor — the minimum of the right subtree — then delete that successor, which is now guaranteed to have at most one child).
- **Complexity**: $O(\log n)$ for a **balanced** BST, but $O(n)$ worst case on a degenerate/skewed tree (e.g. inserting already-sorted data one at a time into a naive BST produces a linked list).
- Self-balancing trees (AVL, Red-Black) maintain $O(\log n)$ height automatically via rotations on insert/delete. Know they exist and *why* (guaranteed $O(\log n)$ regardless of insertion order — this is what backs `TreeMap`/`std::map`-style ordered structures in other languages) but they're essentially never implemented from scratch in an interview — focus on recognizing when balance matters, not coding rotations.

## Common interview questions
- **Validate BST** — don't just check `node.left.val < node.val < node.right.val` locally; pass down a valid `(low, high)` range to each recursive call (a node deep in the left subtree must be less than *every* ancestor above it, not just its immediate parent).
- **Lowest Common Ancestor (LCA)** — in a BST, exploit the ordering: walk down from the root, go left if both targets are smaller, right if both are larger, stop at the first node where they diverge. In a general binary tree (no BST property), use a bottom-up recursive search instead.
- **Balanced tree check** — bottom-up recursion returning height (or -1 as a "not balanced" sentinel) so each subtree's balance is checked once, not $O(n)$ times per node.
- **Serialize/deserialize** — preorder traversal with explicit null markers is the standard approach; reconstructing needs the same traversal order on the way back in.
- Kth smallest element in a BST (inorder traversal, stop at the kth), diameter of a binary tree, invert a binary tree.

## Complexity summary

| Operation | Balanced BST | Degenerate BST |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Traversal (any order) | $O(n)$ | $O(n)$ |

## Related Patterns
- [Binary Search](../patterns/binary-search.md) — BST search/insert is binary search generalized to a tree structure.
- [BFS/DFS](../patterns/bfs-dfs.md) — level-order traversal is BFS; the recursive traversals above are DFS.
- [Heaps](heaps.md) — a different, weaker-ordered tree structure (parent/child relation only, not full BST ordering) optimized for repeated min/max extraction rather than arbitrary search.
