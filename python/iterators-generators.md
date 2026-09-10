# Iterators & Generators

## What / why

The single most common Python interview scenario: *"how would you process a
10 GB file without loading it all into memory?"* — the answer is generators.
Understanding the iterator protocol is also what's actually being tested when
someone asks "what does `for` do under the hood?"

## The iterator protocol

`for x in obj` desugars to roughly:

```python
it = iter(obj)          # calls obj.__iter__()
while True:
    try:
        x = next(it)     # calls it.__next__()
    except StopIteration:
        break
```

- **Iterable**: implements `__iter__`, which returns an **iterator**.
- **Iterator**: implements `__next__` (raises `StopIteration` when exhausted)
  *and* `__iter__` (returning `self`) — so an iterator is itself iterable.

```python
class Countdown:
    def __init__(self, start: int) -> None:
        self.current = start

    def __iter__(self) -> "Countdown":
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for n in Countdown(3):
    print(n)   # 3 2 1
```

A `list` is iterable but is **not** its own iterator — `iter([1, 2])` returns
a separate `list_iterator` object, which is why you can have two independent
`for` loops over the same list running at once.

## Generators

A function with `yield` in its body is a **generator function**; calling it
returns a generator object (an iterator, built for you — no manual
`__iter__`/`__next__`). Execution is paused at each `yield` and resumes from
there on the next `next()` call, preserving all local state.

```python
def read_large_file(path: str):
    with open(path) as f:
        for line in f:
            yield line.strip()          # one line in memory at a time

for line in read_large_file("huge.log"):
    process(line)
```

Compare to loading everything: `lines = open(path).readlines()` — a 10 GB
file becomes a 10 GB list. The generator version holds one line.

### `yield from`

Delegates to a sub-iterator, flattening nested generators without a manual
loop:

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

list(flatten([1, [2, 3, [4]], 5]))   # [1, 2, 3, 4, 5]
```

Also used to delegate to a subgenerator and forward `send()`/`throw()`/return
value — relevant for coroutine-style generator chains (mostly legacy since
`async`/`await` replaced this use case).

### Infinite generators

Generators don't need to terminate — useful for streams/counters, always
paired with `itertools.islice` or an explicit break:

```python
import itertools

def naturals():
    n = 0
    while True:
        yield n
        n += 1

first_five = list(itertools.islice(naturals(), 5))   # [0, 1, 2, 3, 4]
```

## Generator expressions vs list comprehensions

Same syntax, different brackets, very different memory profile:

```python
squares_list = [x * x for x in range(1_000_000)]   # builds a 1M-element list now
squares_gen  = (x * x for x in range(1_000_000))    # builds nothing yet — lazy
```

| | List comprehension | Generator expression |
|---|---|---|
| Memory | O(n) — all at once | O(1) — one item at a time |
| Speed to first result | slower (builds everything) | instant |
| Reusable / indexable | yes | no — single-pass, exhausts |
| When to use | need `len()`, indexing, multiple passes, or it's small | large/unbounded data, feeding straight into `sum()`/`for`/another generator |

`sum(x * x for x in range(1_000_000))` never materializes the intermediate
list — the parentheses can even be dropped when it's the sole argument to a
function call.

## Interview questions

- **How do you process a file too big to fit in memory?** Generator over
  lines (`for line in open(path)`), never `.readlines()`.
- **Difference between an iterable and an iterator?** Iterable has
  `__iter__`; iterator has `__iter__` + `__next__` and is stateful/single-use.
- **What happens if you call `next()` past the end?** Raises `StopIteration`
  (a `for` loop catches this silently to end the loop).
- **Can you restart a generator?** No — once exhausted, it's dead; you must
  call the generator function again to get a fresh one.
- **What's `yield` doing under the hood?** Turns the function into a state
  machine; each call to `__next__` resumes execution right after the last
  `yield`, keeping locals alive on the heap (in a frame object) between calls.

## Common mistakes

- Calling `list(gen)` immediately, defeating the whole point of laziness.
- Iterating a generator twice, expecting the second pass to work.
- Building a huge list comprehension when a generator expression would do
  (e.g. as an argument to `sum`, `any`, `all`, `max`).
- Forgetting that dict/set comprehensions exist too — `{k: v for k, v in ...}`
  and `{x for x in ...}` — and are equally preferable to `map`+`dict`/`set`
  constructor combos. See [code/functional_toolkit.py](code/functional_toolkit.py).

## Related

[decorators.md](decorators.md) (generator-based context managers via
`@contextmanager`) · [context-managers.md](context-managers.md) ·
[concurrency-async.md](concurrency-async.md) (`async def` + `yield` = async
generators; `await` is conceptually similar suspension)
