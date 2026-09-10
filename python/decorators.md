# Decorators

## What / why

A decorator is a function that takes a function (or class) and returns a
(usually wrapped) replacement — syntactic sugar for `func = decorator(func)`.
Interviewers use decorators to check whether you understand closures, since
that's exactly what a decorator is built from.

## Core mechanics

```python
def shout(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@shout
def greet(name):
    return f"hello {name}"

greet("bob")   # "HELLO BOB"
```

`@shout` above `def greet` is exactly `greet = shout(greet)`. `wrapper`
closes over `func` — that's the closure that makes decorators work.

### `functools.wraps`

Without it, the wrapped function loses its identity (`__name__`,
`__doc__`, `__module__` all become the wrapper's) — breaks introspection,
debugging, and tools like `help()` or documentation generators.

```python
from functools import wraps

def shout(func):
    @wraps(func)                 # copies __name__, __doc__, etc. onto wrapper
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).upper()
    return wrapper
```

### Decorators with arguments

Needs one more level of nesting — a function that *returns* a decorator:

```python
def retry(times: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
            raise last_exc
        return wrapper
    return decorator

@retry(times=3)
def flaky_call():
    ...
```

`@retry(times=3)` first calls `retry(3)`, which returns `decorator`; *that*
is what actually decorates `flaky_call`.

### Class decorators

Same idea applied to a class instead of a function — e.g. `@dataclass`,
or registering a class in a plugin registry:

```python
def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls

@register
class Handler:
    ...
```

A decorator can also *be* a class (implementing `__call__`) instead of a
function — useful when the decorator needs its own state.

### Stacking order

Decorators apply bottom-up, but execute outer-to-inner-to-outer at call time:

```python
@a
@b
def f(): ...
# equivalent to f = a(b(f))  -> b wraps f first, then a wraps that
```

So for `@app.route(...)` + `@login_required` stacks, order matters — the
bottom decorator is the one closest to the original function.

## Real examples

**Timing:**
```python
import time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper
```

**Caching — `functools.lru_cache`** (don't hand-roll this in an interview,
know it exists):
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    return n if n < 2 else fib(n - 1) + fib(n - 2)
```
Memoizes by argument tuple; arguments must be hashable. `maxsize=None` = unbounded
cache; a fixed `maxsize` turns it into an LRU cache that evicts oldest entries.

**Retry logic:** shown above (`@retry(times=3)`).

## Interview questions

- **Why does `functools.wraps` matter?** Preserves metadata for
  introspection/debugging; without it stack traces and `help()` show the
  wrapper's identity, not the original function's.
- **How would you write a decorator that takes arguments?** Three nested
  functions: outer takes the decorator args, middle takes the function,
  inner is the actual wrapper. (Shown above.)
- **What's the difference between a decorator and a higher-order function?**
  A decorator *is* a higher-order function, applied via `@` sugar
  specifically to replace a name at definition time.
- **Can a class be a decorator?** Yes, if it implements `__call__`; the
  instance replaces the function and `instance(*args)` runs the logic.
- **How does `lru_cache` know two calls are "the same"?** It hashes the
  positional+keyword argument tuple, so unhashable arguments (lists, dicts)
  raise `TypeError`.

## Common mistakes

- Forgetting `@wraps(func)` — silently breaks anything relying on
  `__name__`/docstrings (including some test frameworks and `functools.singledispatch`).
- Forgetting `*args, **kwargs` in `wrapper`, breaking on any decorated
  function that isn't zero-argument.
- Mutable default state shared across calls when a decorator uses a
  closure variable as a cache without considering thread-safety
  (see [concurrency-async.md](concurrency-async.md) for why that matters
  with threads but not with `asyncio`'s single-threaded model).
- Applying `@lru_cache` to a method — it caches on `(self, *args)`, keeping
  `self` alive for the cache's lifetime (a subtle memory leak in long-lived
  objects).

## Related

[context-managers.md](context-managers.md) (`@contextmanager` is a decorator
that turns a generator into a context manager) · [typing.md](typing.md)
(`@overload`, `@runtime_checkable` are decorators too) ·
[iterators-generators.md](iterators-generators.md) (closures vs generator
state — both are ways functions "remember" things between calls)
