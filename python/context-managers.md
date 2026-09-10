# Context Managers

## What / why

`with` guarantees cleanup runs even if an exception is raised inside the
block — the Pythonic alternative to `try/finally` for anything with a clear
"acquire/release" shape (files, locks, DB connections, timers). A very common
interview ask: "implement a context manager two ways."

## The protocol

```python
class ManagedResource:
    def __enter__(self):
        print("acquire")
        return self                 # bound to the `as` target

    def __exit__(self, exc_type, exc_value, traceback):
        print("release")
        return False                # False/None = don't suppress the exception

with ManagedResource() as r:
    ...
```

- `__enter__` runs first; its return value is what `as x` binds to.
- `__exit__` **always** runs on the way out, exception or not.
- `__exit__`'s return value controls exception propagation: truthy =
  **suppress** the exception (it vanishes); falsy/`None` = let it propagate
  after `__exit__` finishes.

```python
class SuppressValueError:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return exc_type is ValueError   # swallow only ValueError

with SuppressValueError():
    raise ValueError("ignored")
print("still runs")
```

## `contextlib.contextmanager`

Turns a generator into a context manager without writing a class — the code
before `yield` is `__enter__`, the code after is `__exit__`, and the
yielded value is the `as` target:

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("acquire")
    try:
        yield "resource handle"
    finally:
        print("release")            # finally = still runs on exception

with managed_resource() as handle:
    ...
```

The `try/finally` inside is required if you want cleanup to run on
exceptions — an exception raised in the `with` block is re-raised *at the
`yield` point*, so without `finally` your cleanup code after `yield` would
be skipped.

## `contextlib.suppress`

Shorthand for "ignore this exception type", clearer than an empty
`except: pass`:

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("maybe_missing.txt")
```

## Real examples

**File handling** (the canonical stdlib example — `open()` is a context
manager):
```python
with open("data.txt") as f:
    contents = f.read()
# file is closed here even if .read() raised
```

**DB connection / transaction:**
```python
@contextmanager
def transaction(conn):
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
```

**Timing a block:**
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f}s")

with timer("query"):
    run_query()
```

**Multiple managers** — either stack on one line or nest, both call
`__exit__` in reverse order of `__enter__`:
```python
with open("in.txt") as src, open("out.txt", "w") as dst:
    dst.write(src.read())
```

## Interview questions

- **What does `__exit__` returning `True` do?** Suppresses any exception
  raised in the `with` block — it never propagates further.
- **Class-based vs `@contextmanager`-based — when to pick which?** Class
  form for reusable managers needing extra methods/state or subclassing;
  generator form for a quick, local, single-purpose manager — less
  boilerplate.
- **Is `with` equivalent to `try/finally`?** Roughly, but `with` also
  standardizes exception info passed into cleanup (`exc_type`, `exc_value`,
  `traceback`) and composes cleanly when nesting multiple managers.
- **What happens if `__enter__` itself raises?** `__exit__` is **not**
  called — the manager was never successfully "entered".
- **Async version?** `async with`, using `__aenter__`/`__aexit__`
  (`contextlib.asynccontextmanager` for the generator form) — needed for
  async DB drivers/HTTP clients. See [concurrency-async.md](concurrency-async.md).

## Common mistakes

- Forgetting `try/finally` inside a `@contextmanager` generator, so an
  exception in the `with` block skips the cleanup code after `yield`.
- Returning a truthy value from `__exit__` by accident (e.g. returning the
  result of some check instead of explicitly `return False`), silently
  swallowing real bugs.
- Assuming `__exit__` can "fix" the exception and continue the `with` block
  — it can't; suppressing just means execution resumes *after* the block,
  not back inside it.
- Writing a context manager to acquire a resource but reusing the same
  instance across multiple concurrent `with` blocks without confirming it's
  reentrant/thread-safe.

## Related

[decorators.md](decorators.md) (`@contextmanager` is itself a decorator over
a generator) · [iterators-generators.md](iterators-generators.md) (generator
suspension is exactly the mechanism `@contextmanager` relies on) ·
[concurrency-async.md](concurrency-async.md) (`async with`)
