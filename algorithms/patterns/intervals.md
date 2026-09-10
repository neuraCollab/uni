# Intervals

## Recognition
### Key clues
- Input is a list of `[start, end]` ranges: meetings, bookings, time windows.
- Asked to **merge overlapping** intervals, find free time, count overlaps, or insert a new interval into a sorted list.
- "Minimum number of rooms/resources needed" (max concurrent overlap).
- Calendar/scheduling-flavored word problems.

## Pattern
Sort intervals by start (or end, depending on the sub-problem), then do a single linear pass tracking whether the current interval overlaps with the previous one (or with a running "active count"), merging or splitting as needed.

## Why this works
Once sorted by start time, any interval that could overlap the one you're building must be adjacent in the sorted order — you never need to compare against intervals further away, so one pass after the sort suffices.

## Template
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_end = merged[-1][1]
        if start <= last_end:  # overlaps (or touches) the last merged interval
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged


def min_meeting_rooms(intervals: list[list[int]]) -> int:
    """Max number of intervals overlapping at any point in time, via a
    classic 'sweep line' over separately-sorted start/end events.
    """
    starts = sorted(iv[0] for iv in intervals)
    ends = sorted(iv[1] for iv in intervals)
    rooms = max_rooms = 0
    i = j = 0
    while i < len(starts):
        if starts[i] < ends[j]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            i += 1
        else:
            rooms -= 1
            j += 1
    return max_rooms
```

## Example Problems
- Merge Intervals
- Insert Interval
- Meeting Rooms / Meeting Rooms II
- Non-overlapping Intervals
- Employee Free Time
- Interval List Intersections

## Common Mistakes
- Using `<` instead of `<=` when checking overlap — deciding whether touching intervals (`end == next start`) count as overlapping depends on the exact problem statement.
- Sorting by the wrong field (start vs. end) for the specific sub-problem — merging needs sort-by-start, activity-selection/max-count needs sort-by-end (see [Greedy](greedy.md)).
- Mutating the input list of intervals in place when the caller doesn't expect it.

## Complexity
Time O(n log n) for the sort, O(n) for the pass — O(n log n) overall. Space O(n) for the output (O(1) extra if merging in place after sorting).

## Related Patterns
- [Greedy](greedy.md) — interval scheduling (max non-overlapping subset) is a greedy application.
- [Two Pointers](two-pointers.md) — the sweep-line room-counting variant is effectively two pointers over separately sorted start/end arrays.
