# Common Interview Traps — Cheat Sheet

Quick-fire gotchas. Each links to the file with the full explanation.

## Mutable default arguments

Default values are evaluated once, at def-time, and reused across calls.

```python
def add(item, bucket=[]):   # BUG: same list reused every call
    bucket.append(item)
    return bucket

add(1)   # [1]
add(2)   # [1, 2]  <- surprise
```
Fix: `bucket=None`, then `if bucket is None: bucket = []` inside the body.
Details: [memory-model-mutability.md](memory-model-mutability.md#the-mutable-default-argument-trap)

## Late-binding closures in loops

Closures capture the **variable**, not its value at closure-creation time —
by the time the closures run, the loop variable holds its final value.

```python
funcs = [lambda: i for i in range(3)]
[f() for f in funcs]   # [2, 2, 2]  <- not [0, 1, 2]
```
Fix: default-argument trick to bind the value at definition time:
```python
funcs = [lambda i=i: i for i in range(3)]
[f() for f in funcs]   # [0, 1, 2]
```

## `is` vs `==` for small ints/strings (caching)

CPython caches small ints (-5..256) and interns some string literals, so
`is` can look like it's comparing by value — it isn't, reliably.

```python
a, b = 100, 100
a is b        # True  — cached small int, don't rely on this

x, y = 100000, 100000
x is y        # False — outside the cache range
```
Always use `==` for values; `is` only for `None`/singletons.
Details: [memory-model-mutability.md](memory-model-mutability.md#id--is-vs-)

## Integer division: `/` vs `//`

```python
7 / 2    # 3.5   — true division, always returns float
7 // 2   # 3     — floor division, returns int if both operands are int
-7 // 2  # -4    — floors toward negative infinity, not toward zero!
```
`//` rounds **down** (toward `-inf`), not toward zero — `-7 // 2` is `-4`,
not `-3`. Use `int(a / b)` or `math.trunc` if you actually want
truncation-toward-zero.

## `*args` / `**kwargs` unpacking

```python
def f(a, b, *args, c, **kwargs):
    print(a, b, args, c, kwargs)

f(1, 2, 3, 4, c=5, d=6)
# a=1, b=2, args=(3, 4), c=5, kwargs={'d': 6}
```
Gotchas: anything after a bare `*args` becomes **keyword-only**
(`c` above must be passed as `c=...`); unpacking a dict into `**kwargs`
requires string keys; `*` alone (no name) in a signature also forces
keyword-only args without collecting them (`def f(a, *, b): ...`).

## String immutability performance

Strings are immutable — repeated concatenation in a loop creates a new
string object every time (potentially $O(n^2)$ total work):

```python
s = ""
for word in words:
    s += word + " "        # O(n) new string allocated each iteration
```
Fix: accumulate in a list and `"".join(...)` once — $O(n)$ total:
```python
s = " ".join(words)
```

## Circular imports

```python
# a.py
import b
def foo(): return b.bar()

# b.py
import a               # ImportError at import time if a.py is still executing
def bar(): return a.foo()
```
Happens when two modules import each other at the top level. Fixes: move the
import inside the function (deferred/local import), restructure so shared
code lives in a third module, or import the module (`import a`) instead of
a name from it (`from a import foo`) and reference `a.foo` lazily.

## Related

[memory-model-mutability.md](memory-model-mutability.md) ·
[oop.md](oop.md) (mutable default arguments in `__init__`) ·
[typing.md](typing.md) (type hints don't prevent any of the above at runtime)
