# Joins

## What

Joins combine rows from two or more tables based on a related column. Understanding exactly which rows each join type keeps — and what happens to unmatched rows — is the #1 practical SQL skill tested in interviews.

## Sample tables

**`employees`**

| id | name    | dept_id | manager_id |
|----|---------|---------|------------|
| 1  | Alice   | 10      | NULL       |
| 2  | Bob     | 10      | 1          |
| 3  | Carol   | 20      | 1          |
| 4  | Dave    | NULL    | 2          |

**`departments`**

| id | name        |
|----|-------------|
| 10 | Engineering |
| 20 | Sales       |
| 30 | Marketing   |

Note: `dept_id = 30` (Marketing) has no matching employee, and Dave's `dept_id` is `NULL`.

## INNER JOIN

Returns only rows where the join condition matches in **both** tables.

```sql
SELECT e.name, d.name AS dept
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;
```

| name  | dept        |
|-------|-------------|
| Alice | Engineering |
| Bob   | Engineering |
| Carol | Sales       |

Dave is dropped (NULL dept_id never matches), Marketing is dropped (no employee).

## LEFT (OUTER) JOIN

All rows from the left table, matched rows from the right table where they exist, `NULL`s for the right side otherwise.

```sql
SELECT e.name, d.name AS dept
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;
```

| name  | dept        |
|-------|-------------|
| Alice | Engineering |
| Bob   | Engineering |
| Carol | Sales       |
| Dave  | NULL        |

## RIGHT (OUTER) JOIN

Mirror of `LEFT JOIN` — all rows from the right table, matched left rows where they exist.

```sql
SELECT e.name, d.name AS dept
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;
```

| name  | dept        |
|-------|-------------|
| Alice | Engineering |
| Bob   | Engineering |
| Carol | Sales       |
| NULL  | Marketing   |

(`RIGHT JOIN` is rarely used in practice — you can always rewrite it as a `LEFT JOIN` by swapping table order; some engines like older SQLite don't support `RIGHT JOIN` at all.)

## FULL OUTER JOIN

All rows from both tables; unmatched rows on either side get `NULL`s on the other side.

```sql
SELECT e.name, d.name AS dept
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.id;
```

| name  | dept        |
|-------|-------------|
| Alice | Engineering |
| Bob   | Engineering |
| Carol | Sales       |
| Dave  | NULL        |
| NULL  | Marketing   |

**Dialect note**: MySQL has no `FULL OUTER JOIN`. Emulate with `LEFT JOIN UNION RIGHT JOIN` (or `LEFT JOIN ... UNION ALL SELECT ... FROM b LEFT JOIN a ... WHERE a.id IS NULL` to avoid duplicating matched rows).

## CROSS JOIN

Cartesian product — every row of the left table paired with every row of the right table. No `ON` clause. `N x M` rows.

```sql
SELECT e.name, d.name
FROM employees e
CROSS JOIN departments d;   -- 4 employees x 3 departments = 12 rows
```

Common legitimate uses: generating all date x category combinations, building calendar tables.

## SELF JOIN

A table joined to itself, typically to compare rows within the same table (e.g. employee-to-manager).

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

| employee | manager |
|----------|---------|
| Alice    | NULL    |
| Bob      | Alice   |
| Carol    | Alice   |
| Dave     | Bob     |

## Classic pattern: "find employees with no manager"

```sql
-- Approach 1: LEFT JOIN + IS NULL
SELECT e.name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id
WHERE m.id IS NULL;

-- Approach 2 (simpler here, since manager_id is directly on the row):
SELECT name FROM employees WHERE manager_id IS NULL;
```

The `LEFT JOIN ... WHERE right.key IS NULL` pattern generalizes to any "find rows in A with no matching row in B" problem — e.g. "customers who never placed an order":

```sql
SELECT c.id, c.name
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
WHERE o.id IS NULL;
```

This is functionally equivalent to `WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)` — see [`ctes-subqueries.md`](./ctes-subqueries.md) for when to prefer `NOT EXISTS` over this pattern (NULL-safety, performance).

## Join vs. subquery tradeoffs

| | JOIN | Subquery/EXISTS |
|---|---|---|
| Returns columns from both tables | Yes | No (unless correlated + selected) |
| Risk of row duplication (1-to-many) | Yes, if not careful | No, for existence checks |
| Typical use | Combining/enriching data | Existence/filtering checks |
| Optimizer | Usually very well optimized | `EXISTS` often optimizes to a semi-join internally — comparable |

Rule of thumb: use a `JOIN` when you need columns from both tables in the output; use `EXISTS`/`IN` when you only need to filter one table based on the presence of related rows (avoids duplicate rows from a 1-to-many join).

## Common mistake: NULLs break naive equality joins

Join conditions use `=`, and per three-valued logic, `NULL = NULL` is `UNKNOWN`, not `TRUE` — so rows with `NULL` in the join column **never match**, even against another `NULL`.

```sql
-- Dave (dept_id = NULL) will NEVER match any department, even if departments had a NULL id row.
FROM employees e JOIN departments d ON e.dept_id = d.id
```

This is expected behavior, not a bug — but it trips people up when they expect "no department" to somehow join to something. If you need NULL-aware matching, use `IS NOT DISTINCT FROM` (Postgres/SQL standard) instead of `=`:

```sql
... ON e.dept_id IS NOT DISTINCT FROM d.id   -- treats NULL = NULL as a match
```
(SQL Server equivalent is more verbose: no direct `IS NOT DISTINCT FROM` until recent versions — use `(a = b OR (a IS NULL AND b IS NULL))`.)

## Multi-table joins & duplicate row trap

Joining a "one" table to two separate "many" tables (a fan-out) multiplies rows unexpectedly:

```sql
-- If an employee has 3 orders and 2 reviews, this produces 3 x 2 = 6 rows for that employee!
SELECT e.name, o.id AS order_id, r.id AS review_id
FROM employees e
JOIN orders o ON o.emp_id = e.id
JOIN reviews r ON r.emp_id = e.id;
```
Fix: aggregate each side first (in a CTE/subquery) before joining, or use separate queries.

## Common mistakes summary

- Forgetting `LEFT JOIN` semantics — filtering on a right-table column in `WHERE` (instead of `ON`) silently turns it back into an `INNER JOIN`:
  ```sql
  -- WRONG: this filters out Dave's NULL dept row, defeating the LEFT JOIN
  FROM employees e LEFT JOIN departments d ON e.dept_id = d.id
  WHERE d.name = 'Engineering'   -- d.name IS NULL for Dave → row dropped

  -- RIGHT: move the condition into ON to preserve LEFT JOIN semantics
  FROM employees e LEFT JOIN departments d ON e.dept_id = d.id AND d.name = 'Engineering'
  ```
- Fan-out row multiplication from joining multiple one-to-many relationships at once.
- Forgetting `RIGHT`/`FULL OUTER JOIN` aren't supported (or are limited) in some engines (MySQL: no `FULL OUTER`; older SQLite: no `RIGHT`/`FULL OUTER` before 3.39).

## Interview questions

1. What's the difference between `WHERE` filtering and `ON` filtering in a `LEFT JOIN`?
2. Write a query to find customers with no orders.
3. How would you emulate `FULL OUTER JOIN` in MySQL?
4. Why might a self-join on `manager_id` return fewer rows than expected if some managers have left the company (manager row deleted)?
5. What happens to `NULL` values in join columns during an equality join?

## See also

- [`select-where-groupby.md`](./select-where-groupby.md)
- [`ctes-subqueries.md`](./ctes-subqueries.md) — EXISTS vs IN vs JOIN
- [`window-functions.md`](./window-functions.md)
- [`interview-problems.md`](./interview-problems.md) — employees earning more than their manager (self-join pattern)
