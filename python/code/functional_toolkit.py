"""Functional toolkit: map / filter / functools.reduce / lambda vs comprehensions.

Canonical, interview-cram version merging three near-identical university
demos (student grades, user expenses, customer orders - all synthetic,
fictional data) into a single file. Each operation is shown two ways: the
map/filter/reduce style and the comprehension style, so the tradeoff is
visible side by side.

Interview question this file answers: "When would you prefer a list
comprehension over map/filter, and vice versa?"
  - Comprehensions are the modern default: they read left-to-right, don't
    need a throwaway lambda, and are usually faster (no per-element Python
    function call overhead).
  - map/filter still earn their keep when you already have a named function
    (no lambda needed) and want to pass a callable around lazily, or when
    chaining directly into other iterator tools (itertools, zip) without
    materializing an intermediate list.
  - functools.reduce has no comprehension equivalent - reach for it whenever
    you're folding a sequence down into one accumulated value.
"""

from __future__ import annotations

from functools import reduce
from typing import TypedDict


class Student(TypedDict):
    name: str
    age: int
    grades: list[int]


students: list[Student] = [
    {"name": "Alice", "age": 20, "grades": [85, 90, 88, 92]},
    {"name": "Bob", "age": 22, "grades": [78, 89, 76, 85]},
    {"name": "Charlie", "age": 21, "grades": [92, 95, 88, 94]},
    {"name": "David", "age": 23, "grades": [65, 70, 68, 72]},
    {"name": "Eve", "age": 20, "grades": [88, 87, 90, 91]},
]


def average(grades: list[int]) -> float:
    return sum(grades) / len(grades)


# --- filter -------------------------------------------------------------

# map/filter style
students_aged_20 = list(filter(lambda s: s["age"] == 20, students))
# comprehension style (preferred)
students_aged_20_comp = [s for s in students if s["age"] == 20]


# --- map ------------------------------------------------------------------

# map/filter style
names_and_avg = list(map(lambda s: (s["name"], average(s["grades"])), students))
# comprehension style (preferred)
names_and_avg_comp = [(s["name"], average(s["grades"])) for s in students]

# dict comprehension: name -> average grade
avg_by_name: dict[str, float] = {s["name"]: average(s["grades"]) for s in students}

# set comprehension: distinct ages present in the cohort
distinct_ages: set[int] = {s["age"] for s in students}


# --- functools.reduce -------------------------------------------------------

# reduce has no comprehension equivalent - use it to fold into a single value
overall_average = reduce(lambda acc, s: acc + average(s["grades"]), students, 0.0) / len(students)

# same result without reduce, for comparison (often clearer for simple sums)
overall_average_alt = sum(average(s["grades"]) for s in students) / len(students)


# --- combining filter + max --------------------------------------------

best_average = max(average(s["grades"]) for s in students)
top_students = [s["name"] for s in students if average(s["grades"]) == best_average]


if __name__ == "__main__":
    print("Students aged 20:", [s["name"] for s in students_aged_20_comp])
    print("Name -> average:", names_and_avg_comp)
    print("Average by name:", avg_by_name)
    print("Distinct ages:", distinct_ages)
    print("Overall average:", round(overall_average, 2))
    print("Top student(s):", top_students)
