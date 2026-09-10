# OOP in Python

## What / why

Python OOP is duck-typed and dynamic: classes are runtime objects, methods are
just functions on the class dict, and `self` is passed explicitly. Interviews
usually probe whether you understand *why* a mechanism exists, not just its
syntax — abstract base classes, `super()` and MRO, and composition-vs-inheritance
come up constantly for anyone claiming ML/DS or backend experience.

## Core mechanics

```python
class Animal:
    def __init__(self, name: str) -> None:
        self.name = name          # instance attribute

    def speak(self) -> str:
        raise NotImplementedError

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says Woof"
```

- `__init__` initializes an *already-created* instance (compare `__new__`,
  which actually constructs it — rarely overridden except for singletons/immutables).
- `super()` walks the **MRO** (Method Resolution Order, C3 linearization) —
  matters as soon as you have multiple inheritance.
- Everything is public by convention: `_leading_underscore` = "internal, don't
  touch"; `__leading_dunder` triggers **name mangling** (`_ClassName__attr`),
  used to avoid subclass attribute clashes, not real privacy.

## Abstract base classes — real example

From a clustering-algorithms project in this repo
(`algos/4 sem/5 lab/code/clustering_project/src/clustering/`), a clean
`abc.ABC` hierarchy where every clustering algorithm shares a constructor
and a `fit_predict` convenience method, but must supply its own `fit`/`predict`:

```python
from abc import ABC, abstractmethod

class BaseClusterer(ABC):
    def __init__(self, metric="euclidean", p=2):
        self.metric = metric
        self.p = p
        self.distance_func = self._get_distance_func()

    def _get_distance_func(self):
        return get_distance(self.metric, p=self.p)

    @abstractmethod
    def fit(self, X): ...

    @abstractmethod
    def predict(self, X): ...

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
```

`BaseClusterer(...)` cannot be instantiated directly — Python raises
`TypeError: Can't instantiate abstract class` if `fit`/`predict` aren't
overridden. A concrete subclass (`CureClusterer` in the same project) calls
`super().__init__(metric=metric, p=p)` to reuse the shared setup, then
implements `fit`/`predict` with its own algorithm:

```python
class CureClusterer(BaseClusterer):
    def __init__(self, n_clusters=3, metric="euclidean", p=2):
        super().__init__(metric=metric, p=p)   # reuse base setup
        self.n_clusters = n_clusters
        self.clusters_ = []

    def fit(self, X):
        ...           # CURE-specific clustering logic

    def predict(self, X):
        return self.labels_
```

This is **method overriding**: `fit`/`predict` are redefined per algorithm,
while `fit_predict` and the constructor plumbing are inherited unchanged —
the base class enforces a contract ("every clusterer has fit/predict") without
knowing any algorithm's implementation.

The same project also shows **composition over inheritance**: `CureClusterer`
doesn't inherit from a `Cluster` — it *has* a list of `Cluster` objects and
delegates to their `.merge()` / `.distance_to()` methods. Prefer composition
when the relationship is "uses/has", not "is-a".

## Interview questions

- **ABC vs duck typing?** — ABCs enforce a contract at instantiation time;
  duck typing just calls the method and lets it fail at runtime if missing.
  Python favors duck typing generally, but ABCs are the right tool for a
  plugin/strategy hierarchy like the clusterer example above.
- **`__init__` vs `__new__`?** — `__new__` creates and returns the instance
  (rarely touched); `__init__` configures it. Needed for immutable types
  (`int`, `str`, `tuple` subclasses) or singleton patterns.
- **Composition vs inheritance?** — inheritance for "is-a" + shared interface
  contract; composition for "has-a"/"uses-a" to avoid deep, fragile hierarchies.
- **What does `super()` actually do?** — returns a proxy that resolves
  attribute lookups against the next class in the MRO, not literally "the
  parent class" (important with multiple inheritance / mixins).
- **Classmethod vs staticmethod vs instance method?** — `@classmethod`
  receives `cls` (used for alternate constructors, e.g. `from_dict`);
  `@staticmethod` receives neither `self` nor `cls` (a namespaced plain
  function); instance methods receive `self` and need instance state.

## Common mistakes

- Mutable default arguments in `__init__` (see
  [common-interview-traps.md](common-interview-traps.md)) — e.g.
  `def __init__(self, items=[])` shares one list across all instances.
- Forgetting `super().__init__()` in a subclass, silently skipping base setup.
- Overusing inheritance for code reuse when composition/mixins would be less
  coupled.
- Implementing `__eq__` without `__hash__` (or vice versa) — see
  [memory-model-mutability.md](memory-model-mutability.md#magic-methods-overview).

## Related

[memory-model-mutability.md](memory-model-mutability.md) (magic methods,
`__slots__`) · [typing.md](typing.md) (`Protocol` as a structural alternative
to ABCs) · [decorators.md](decorators.md) (`@abstractmethod`,
`@classmethod`, `@staticmethod` are all decorators)
