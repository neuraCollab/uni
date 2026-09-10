# Typing

## What / why

Python type hints (`PEP 484` and later) are optional, gradual, and — this is
the #1 interview trap — **not enforced at runtime**. They exist for static
checkers (mypy, pyright), IDEs, and documentation. Understanding what hints
do and don't do is a very common quick-fire interview question.

## Basics

```python
def greet(name: str, times: int = 1) -> str:
    return (name + " ") * times

age: int = 30
scores: list[int] = [90, 85, 77]
```

Since Python 3.9+, builtin generics (`list[int]`, `dict[str, int]`) work
directly — no need for `typing.List`/`typing.Dict` anymore (still seen in
older code).

## Optional / Union

```python
from typing import Optional, Union

def find_user(id: int) -> Optional[str]:      # same as Union[str, None]
    ...

def parse(value: Union[int, str]) -> int:
    ...

# Python 3.10+ pipe syntax (preferred now)
def find_user_new(id: int) -> str | None: ...
def parse_new(value: int | str) -> int: ...
```

`Optional[X]` is exactly `Union[X, None]` — it does **not** mean "has a
default value", just that `None` is a valid value.

## Generics: `TypeVar` and `Generic`

For a function/class whose type varies but is consistent across usage:

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]
```

`first(list[int])` is known by the checker to return `int`; `first(list[str])`
returns `str` — same function, tracked type per call.

Generic classes:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

int_stack: Stack[int] = Stack()
```

## `Protocol` — structural typing

Unlike ABCs (nominal typing: "must inherit from X"), `Protocol` checks shape
("must have these methods") — Python's version of duck typing made
statically checkable, closer to Go interfaces than Java interfaces:

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render(shape: Drawable) -> None:
    shape.draw()

class Circle:            # note: no inheritance from Drawable at all
    def draw(self) -> None:
        print("○")

render(Circle())          # type-checks fine — Circle "fits" the protocol
```

No explicit `class Circle(Drawable)` needed — any object with a matching
`draw()` method satisfies the protocol. Compare to
[oop.md](oop.md#abstract-base-classes--real-example)'s `BaseClusterer`,
which uses nominal typing (`ABC`) instead — requiring explicit subclassing.

## `TypedDict`

Types a dict's shape (keys + value types) without making it a real class —
useful for JSON-like data:

```python
from typing import TypedDict

class UserRecord(TypedDict):
    name: str
    age: int

def greet(user: UserRecord) -> str:
    return f"Hi {user['name']}"

greet({"name": "Alice", "age": 30})   # checker verifies keys/types
```

Still a plain `dict` at runtime — `TypedDict` is purely a static-checking
construct, same caveat as everything else on this page.

## Runtime vs static checking

**Python does not enforce type hints at runtime.** This is the single most
common typing interview question:

```python
def add(a: int, b: int) -> int:
    return a + b

add("hello", "world")   # runs fine, returns "helloworld" — no error!
```

Hints are metadata, inspectable via `typing.get_type_hints()` or
`__annotations__`, but nothing checks them unless you run a static checker
(`mypy`, `pyright`) as a separate step, or explicitly validate at runtime
(e.g. with `pydantic`, which *does* enforce at runtime by generating
validation code from the hints).

## Interview questions

- **Does Python enforce type hints at runtime?** No — they're purely
  advisory for tools; nothing stops you from passing the wrong type.
- **`Optional[X]` vs a default value?** Unrelated concepts — `Optional[X]`
  is about `None` being a valid value, a default is about what happens when
  the argument is omitted; you often want both: `x: Optional[int] = None`.
- **`Protocol` vs ABC?** Protocol = structural ("has the right methods"),
  no inheritance required; ABC = nominal ("explicitly inherits and
  implements the abstract methods"). Protocol also supports
  `@runtime_checkable` for `isinstance()` checks against structure.
- **Why use `TypedDict` instead of a dataclass?** When you're stuck with
  plain dict-shaped data (e.g. JSON from an API) and want static checking
  without changing the runtime representation.
- **How would you validate types at runtime, since hints don't?** Explicit
  `isinstance` checks, `assert`, or a library like `pydantic`/`attrs` with
  validators.

## Common mistakes

- Assuming a type hint will raise an error if violated — it silently does
  nothing at runtime.
- Confusing `Optional[X]` with "has a default".
- Mutable default arguments typed correctly but still buggy — typing a
  parameter `items: list[int] = []` doesn't fix the mutable-default trap,
  see [common-interview-traps.md](common-interview-traps.md).
- Forgetting `from __future__ import annotations` (or a recent enough Python)
  when using new-style `list[int]`/`X | None` syntax on older versions.

## Related

[oop.md](oop.md) (`Protocol` vs `ABC`) ·
[memory-model-mutability.md](memory-model-mutability.md) (`TypedDict`/generics
say nothing about mutability of the underlying object) ·
[common-interview-traps.md](common-interview-traps.md)
