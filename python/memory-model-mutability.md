# Memory Model & Mutability

## What / why

The mental model behind most "gotcha" Python interview questions: everything
is an object, variables are references (names bound to objects), and
mutability determines whether operations create a new object or modify one
in place. Get this wrong and you'll misdiagnose half the traps in
[common-interview-traps.md](common-interview-traps.md).

## Mutable vs immutable

| Immutable | Mutable |
|---|---|
| `int`, `float`, `bool`, `str`, `tuple`, `frozenset`, `bytes` | `list`, `dict`, `set`, `bytearray`, most custom classes |

Immutable means the object's value can never change after creation — any
"modification" (e.g. `x += 1`, `s = s + "a"`) creates a **new** object and
rebinds the name to it. Mutable objects can be changed in place, and every
reference to them sees the change.

```python
a = [1, 2, 3]
b = a           # b is the SAME list object, not a copy
b.append(4)
print(a)        # [1, 2, 3, 4] — a changed too, because a and b are one object
```

## `id()` / `is` vs `==`

- `id(obj)` — the object's identity (in CPython, its memory address).
- `is` — compares identity (`id(a) == id(b)`).
- `==` — compares value (calls `__eq__`), which can be overridden.

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b        # True  (same value)
a is b        # False (two distinct objects)

x = None
x is None     # correct — always use `is` for None/True/False singletons
```

### Small int / string caching

CPython caches small integers (-5 to 256) and some string literals
(interning), so `is` can accidentally look like it's comparing by value:

```python
a = 100
b = 100
a is b          # True — small ints are cached/interned

x = 100000
y = 100000
x is y          # False (usually) — outside the cached range, separate objects
```

**Never rely on this** — it's a CPython implementation detail, not a
language guarantee. Always use `==` for value comparison; reserve `is` for
identity checks (`is None`, `is True`, sentinel objects).

## Shallow vs deep copy

```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)          # or original[:] or list(original)
shallow[0].append(99)
print(original)   # [[1, 2, 99], [3, 4]] — inner list is SHARED

deep = copy.deepcopy(original)
deep[0].append(100)
print(original)   # unaffected — deepcopy recursively copies nested objects
```

Shallow copy duplicates the outer container only; nested mutable objects are
still shared references. Deep copy recursively duplicates everything.

## The mutable-default-argument trap

Default argument values are evaluated **once**, at function-definition time,
and reused across every call that doesn't override them:

```python
def add_item(item, bucket=[]):     # BUG: bucket is created ONCE, not per call
    bucket.append(item)
    return bucket

add_item(1)   # [1]
add_item(2)   # [1, 2]  <- unexpected! same list object reused
```

Fix: use `None` as the sentinel and create the mutable object inside the
function body:

```python
def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```

See also [common-interview-traps.md](common-interview-traps.md).

## `__slots__`

By default, instances store attributes in a per-instance `__dict__`, which
costs memory and adds a hash-lookup for every attribute access. `__slots__`
declares a fixed set of attributes, replacing `__dict__` with fixed-size
storage:

```python
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

p = Point(1, 2)
p.z = 3          # AttributeError — not in __slots__
```

Tradeoff: less memory (no per-instance `__dict__`), slightly faster
attribute access, but loses dynamic attribute assignment and multiple
inheritance gets restrictive (all bases with slots must not conflict).
Worth it for classes instantiated millions of times (e.g. data records in a
hot loop); not worth it for typical business objects.

## Refcounting / GC mental model

CPython uses **reference counting** as the primary memory management
mechanism: every object tracks how many references point to it; when the
count hits zero, it's freed immediately (deterministic, unlike Java/Go's
GC pause model). A supplementary **cyclic garbage collector**
(`gc` module) periodically finds and collects reference cycles (e.g. two
objects referencing each other) that refcounting alone can't clean up,
since neither's count ever reaches zero on its own.

```python
import sys
a = []
sys.getrefcount(a)   # includes the temporary reference from the call itself
```

This is why `del x` doesn't necessarily free memory immediately — it just
decrements the refcount; the object is freed only if that was the last
reference.

## Magic methods overview

| Method | Purpose |
|---|---|
| `__repr__` | unambiguous, dev-facing string (`repr(obj)`, default in REPL) |
| `__str__` | readable, user-facing string (`str(obj)`, `print(obj)`) |
| `__eq__` | `==` comparison |
| `__hash__` | value used as dict key / set member |
| `__len__` | `len(obj)` |
| `__iter__` / `__next__` | iteration protocol, see [iterators-generators.md](iterators-generators.md) |
| `__enter__` / `__exit__` | context manager protocol, see [context-managers.md](context-managers.md) |
| `__call__` | makes an instance callable like a function |

### The `__eq__` + `__hash__` contract

If you override `__eq__`, Python **sets `__hash__` to `None`** automatically
unless you also define `__hash__` — the object becomes unhashable (can't go
in a `set` or be a `dict` key). This is because the contract requires: if
`a == b`, then `hash(a) == hash(b)` must also hold, or hash-based
collections break silently (an object could be "in" a set but not found by
lookup).

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))   # must be consistent with __eq__
```

If instances are meant to be mutable, it's often safer to leave `__hash__`
as `None` (i.e. don't define `__eq__` without `__hash__`, or explicitly set
`__hash__ = None`) — a mutable object used as a dict key can "disappear"
from the dict if its hash changes after insertion.

## Interview questions

- **`is` vs `==`?** Identity vs value; only ever use `is` for
  `None`/singletons, never for general value comparison.
- **Why did `x is y` return `True` for two `100`s but `False` for two
  `100000`s?** CPython small-int caching (-5..256) — an implementation
  detail, not something to rely on.
- **What's the mutable default argument bug and how do you fix it?** Default
  values are evaluated once at def-time; fix with `None` sentinel + create
  inside the function.
- **Shallow vs deep copy — when do you need which?** Shallow is enough for
  flat structures; deep copy is required whenever the structure has nested
  mutable containers you don't want shared.
- **Why does defining `__eq__` sometimes break `set`/dict usage?** It nulls
  out the inherited `__hash__` unless you redefine it too — must keep both
  consistent.

## Common mistakes

- Assuming `a == b` implies `a is b`, or vice versa.
- Mutable default arguments (`def f(x=[])`).
- Comparing floats with `==` (unrelated to `is`/`==` above, but a nearby
  common trap: use `math.isclose()` for float comparisons).
- Shallow-copying when a deep copy was needed (nested mutation bugs).
- Defining `__eq__` without `__hash__`, silently making instances unusable
  as dict keys/set members.

## Related

[common-interview-traps.md](common-interview-traps.md) ·
[oop.md](oop.md) (magic methods in class design) ·
[typing.md](typing.md) (hints don't affect mutability semantics)
