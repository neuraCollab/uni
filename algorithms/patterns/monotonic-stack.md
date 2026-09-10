# Monotonic Stack

## Recognition
### Key clues
- "Next greater/smaller element" to the left or right of each position.
- Histogram-style problems: largest rectangle, trapping rain water, daily temperatures.
- Asked for, per element, the nearest element satisfying some comparison — naive approach is O(n^2) with a nested loop.
- Stock span, or any "how far back/forward until X changes" question.

## Pattern
Maintain a stack of indices (or values) that is kept strictly increasing or decreasing as you scan the array. When the current element breaks the monotonic invariant, pop from the stack — each pop resolves the "next greater/smaller" answer for the popped element.

## Why this works
Each element is pushed once and popped at most once, so total work across all pushes/pops is O(n) even though it replaces what looks like an O(n^2) "compare every pair" problem. The stack only ever holds elements that are still "waiting" for their answer.

## Template
```python
def next_greater_element(a: list[int]) -> list[int]:
    """For each index, the value of the next greater element to the right,
    or -1 if none exists. Monotonic decreasing stack of indices.
    """
    n = len(a)
    result = [-1] * n
    stack: list[int] = []  # indices whose "next greater" is still unknown

    for i in range(n):
        while stack and a[stack[-1]] < a[i]:
            j = stack.pop()
            result[j] = a[i]
        stack.append(i)

    return result


def daily_temperatures(temps: list[int]) -> list[int]:
    """Days to wait until a warmer temperature (0 if none). Same pattern,
    storing the *distance* instead of the value."""
    n = len(temps)
    answer = [0] * n
    stack: list[int] = []

    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)

    return answer
```

## Example Problems
- Next Greater Element I / II
- Daily Temperatures
- Largest Rectangle in Histogram
- Trapping Rain Water (monotonic stack variant)
- Remove K Digits
- Sum of Subarray Minimums

## Common Mistakes
- Storing values instead of indices when you need distances or need to write into a result array at the correct position.
- Getting the comparison direction backwards (increasing vs. decreasing stack) — always re-derive from "what invariant do I need broken to know the answer" rather than memorizing.
- Forgetting elements left on the stack at the end never found their answer (correctly left at the default, e.g. -1 or 0) — don't force-pop them incorrectly.

## Complexity
Time O(n) amortized (each element pushed/popped once). Space O(n) for the stack.

## Related Patterns
- [Binary Search](binary-search.md)
- [Sliding Window](sliding-window.md)
