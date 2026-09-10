# Python — Interview Prep

Concise, cram-friendly notes on Python internals commonly probed in ML/DS
and backend interviews. Each file: what/why, how it works, a runnable
example, interview questions, and common mistakes.

## Quick-revision index

- **[oop.md](oop.md)** — classes, `__init__`/`super()`, abstract base
  classes (`abc.ABC`), inheritance vs composition, class/static/instance
  methods. Real example: an ABC-based clustering algorithm hierarchy.
- **[iterators-generators.md](iterators-generators.md)** — the iterator
  protocol (`__iter__`/`__next__`), `yield`, `yield from`, generator vs list
  comprehension memory tradeoffs, infinite generators, "process a huge file
  without loading it into memory."
- **[decorators.md](decorators.md)** — closures, `functools.wraps`,
  parameterized decorators, class decorators, stacking order, timing /
  `lru_cache` / retry examples.
- **[context-managers.md](context-managers.md)** — `with`,
  `__enter__`/`__exit__`, exception suppression, `contextlib.contextmanager`,
  `contextlib.suppress`, file/DB/timing examples.
- **[typing.md](typing.md)** — type hints, `Optional`/`Union`/`|`, `TypeVar`/
  `Generic`, `Protocol` (structural typing), `TypedDict`, why hints aren't
  enforced at runtime.
- **[memory-model-mutability.md](memory-model-mutability.md)** — mutable vs
  immutable types, `id()`/`is` vs `==`, shallow vs deep copy, the mutable
  default argument trap, `__slots__`, refcounting/GC, `__eq__`+`__hash__`
  contract, magic methods overview.
- **[concurrency-async.md](concurrency-async.md)** — the GIL and why
  threading doesn't speed up CPU-bound code, threading vs multiprocessing vs
  asyncio decision table, anchor examples for all three models.
- **[common-interview-traps.md](common-interview-traps.md)** — one-line
  cheat sheet: mutable defaults, late-binding closures, `is` vs `==`
  caching, `/` vs `//`, `*args`/`**kwargs`, string concatenation
  performance, circular imports.

## Code

- **[code/async_chat_server.py](code/async_chat_server.py)** — asyncio TCP
  chat server + client in one file: `asyncio.start_server`/
  `open_connection`, `asyncio.Queue` producer/consumer broadcaster,
  `asyncio.create_task`/`gather`. Anchor example for the asyncio section of
  [concurrency-async.md](concurrency-async.md).
- **[code/multiprocessing_pipeline.py](code/multiprocessing_pipeline.py)** —
  `multiprocessing.Pool` chunk/process/merge pattern for CPU-bound work.
  Anchor example for the multiprocessing section of
  [concurrency-async.md](concurrency-async.md).
- **[code/functional_toolkit.py](code/functional_toolkit.py)** — `map`/
  `filter`/`functools.reduce`/lambdas side by side with their comprehension
  equivalents, with notes on when to prefer each. Referenced from
  [iterators-generators.md](iterators-generators.md) and
  [common-interview-traps.md](common-interview-traps.md).

## Suggested review order

1. [memory-model-mutability.md](memory-model-mutability.md) — foundational
   mental model everything else builds on.
2. [common-interview-traps.md](common-interview-traps.md) — fast pattern
   recognition for the gotchas most likely to come up.
3. [iterators-generators.md](iterators-generators.md) →
   [decorators.md](decorators.md) → [context-managers.md](context-managers.md)
   — these three build on each other (generators underpin
   `@contextmanager`; both are closures under the hood like decorators).
4. [oop.md](oop.md) and [typing.md](typing.md) — class design and how to
   express it statically.
5. [concurrency-async.md](concurrency-async.md) — heaviest topic, save for
   last; skim the decision table first, then work through the three code
   examples.
