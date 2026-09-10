# CTEs, Subqueries, and Temp Tables

## What

Three ways to structure intermediate/nested query logic:
- **Subquery**: a query nested inside another (in `SELECT`, `FROM`, `WHERE`, etc.).
- **CTE (`WITH ... AS (...)`)**: a named, temporary result set scoped to one query, defined before the main query.
- **Temp table**: an actual materialized table (session/transaction-scoped), created with `CREATE TEMP TABLE` / `CREATE TABLE #tmp` (SQL Server).

## CTE vs subquery vs temp table

```sql
-- Subquery (nested in FROM)
SELECT dept, avg_sal
FROM (
  SELECT department AS dept, AVG(salary) AS avg_sal
  FROM employees
  GROUP BY department
) sub
WHERE avg_sal > 90000;

-- CTE — same logic, more readable, can be referenced multiple times
WITH dept_avg AS (
  SELECT department AS dept, AVG(salary) AS avg_sal
  FROM employees
  GROUP BY department
)
SELECT * FROM dept_avg WHERE avg_sal > 90000;
```

| | CTE | Subquery | Temp table |
|---|---|---|---|
| Readability | High — named, top-down | Lower — nested, reads inside-out | High but verbose (extra DDL) |
| Reusable within same query | Yes, multiple times | No, must repeat | Yes, across multiple statements |
| Materialized? | Depends on engine (see below) | Usually inlined by optimizer | Always materialized (real table + stats) |
| Scope | Single statement | Single statement | Session/transaction |
| Can be indexed | No | No | Yes |
| Good for | Readability, recursion, breaking complex logic into steps | Quick one-off nesting | Large intermediate results reused across several queries, or when you need indexes on the intermediate result |

**Optimizer fence**: historically, Postgres always materialized CTEs (an "optimization fence" — the planner couldn't push predicates into them), which could hurt performance. Since **Postgres 12**, non-recursive CTEs are inlined by default like subqueries, unless referenced multiple times, involves side effects, or you force it with `MATERIALIZED`. SQL Server and MySQL 8+ generally inline non-recursive CTEs too. Recursive CTEs are always materialized (they have to be, structurally).

```sql
WITH x AS MATERIALIZED (...)   -- Postgres 12+: force materialization (e.g. to avoid recomputation)
WITH x AS NOT MATERIALIZED (...)  -- force inlining
```

## Recursive CTEs

Used for hierarchical/graph traversal — org charts, bill-of-materials, category trees.

```sql
-- org_chart: find all reports (direct and indirect) under Alice (id = 1)
WITH RECURSIVE subordinates AS (
  -- anchor member: the starting row
  SELECT id, name, manager_id, 1 AS depth
  FROM employees
  WHERE id = 1

  UNION ALL

  -- recursive member: joins back to the CTE itself
  SELECT e.id, e.name, e.manager_id, s.depth + 1
  FROM employees e
  JOIN subordinates s ON e.manager_id = s.id
)
SELECT * FROM subordinates;
```

Structure:
1. **Anchor member** — the base case (non-recursive `SELECT`).
2. `UNION ALL` (or `UNION` to dedupe, but usually `ALL` for performance and to avoid infinite loops being silently masked).
3. **Recursive member** — references the CTE's own name, joins to get the "next level."
4. Recursion stops automatically when the recursive member returns no new rows.

**Dialect note**: MySQL requires `WITH RECURSIVE` (same as Postgres); SQL Server just uses `WITH cte AS (...)` (no `RECURSIVE` keyword needed, recursion is implicit from the self-reference). SQLite supports `WITH RECURSIVE` since 3.8.3.

**Gotcha**: always guard against infinite loops (e.g. cyclic manager references) with a `depth` counter and a `WHERE depth < N` cap, or `MAXRECURSION` hint in SQL Server.

## Correlated vs. uncorrelated (non-correlated) subqueries

**Uncorrelated**: inner query is independent, runs once.
```sql
SELECT name FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

**Correlated**: inner query references a column from the outer query, conceptually re-runs per outer row.
```sql
SELECT e.name FROM employees e
WHERE e.salary > (
  SELECT AVG(salary) FROM employees e2 WHERE e2.department = e.department
);   -- "employees earning more than their department's average"
```
Modern optimizers often rewrite correlated subqueries into joins/semi-joins internally, but conceptually treat them as "runs per outer row" — this matters for reasoning about correctness and rough performance.

## EXISTS vs IN vs JOIN for existence checks

**Goal**: "customers who have placed at least one order."

```sql
-- JOIN (risk: duplicates a customer row per matching order — need DISTINCT)
SELECT DISTINCT c.* FROM customers c
JOIN orders o ON o.customer_id = c.id;

-- IN
SELECT * FROM customers
WHERE id IN (SELECT customer_id FROM orders);

-- EXISTS
SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

| | Duplicate rows risk | NULL-safety | Typical performance |
|---|---|---|---|
| `JOIN` (+ DISTINCT) | Yes, without DISTINCT | Safe (equality join just won't match NULLs) | Good, optimizer-friendly, but needs DISTINCT for existence semantics |
| `IN` | No | **Unsafe** with NULLs in subquery for `NOT IN` (see below) | Good for small/static lists; engine typically rewrites to semi-join |
| `EXISTS` | No | Safe — NULL-safe by construction | Often best for correlated existence checks; can short-circuit on first match |

### The `NOT IN` + NULL trap — a top interview gotcha

```sql
-- If ANY customer_id in orders is NULL, this returns ZERO rows for ALL customers!
SELECT * FROM customers
WHERE id NOT IN (SELECT customer_id FROM orders WHERE customer_id IS NULL OR TRUE);
```
Why: `NOT IN (1, 2, NULL)` expands to `id <> 1 AND id <> 2 AND id <> NULL`. The last term is `UNKNOWN` (not `TRUE`/`FALSE`), and `AND` with `UNKNOWN` can never yield `TRUE` — so the whole condition becomes `UNKNOWN` or `FALSE`, never `TRUE`, for every row. This silently returns an empty result set with no error — extremely dangerous in production code.

**Fix**: use `NOT EXISTS` instead, which is NULL-safe:
```sql
SELECT * FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```
Or filter NULLs explicitly if you must use `NOT IN`:
```sql
WHERE id NOT IN (SELECT customer_id FROM orders WHERE customer_id IS NOT NULL)
```

**Rule of thumb**: prefer `EXISTS`/`NOT EXISTS` over `IN`/`NOT IN` for subqueries, especially `NOT IN`, unless you're certain the subquery column can never be NULL.

## Subquery in SELECT (scalar subquery)

```sql
SELECT name,
  (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count
FROM customers c;
```
Must return exactly one row/column (or NULL) per outer row, or it errors at runtime. Often better expressed as a `LEFT JOIN ... GROUP BY` for performance at scale.

## Common mistakes

- Using `NOT IN` with a subquery that can return NULLs (silent empty result).
- Forgetting `DISTINCT` when using `JOIN` purely for an existence check, causing row duplication.
- Writing a recursive CTE without a depth/cycle guard, risking infinite loops on cyclic data.
- Assuming CTEs are always materialized (behavior differs by engine and version — see above).
- Using a correlated subquery in `SELECT` for every row when a single `JOIN` + `GROUP BY` would be far cheaper.

## Interview questions

1. What's the practical difference between a CTE and a subquery?
2. Why is `NOT IN` dangerous when the subquery can return NULLs? How do you fix it?
3. Write a recursive CTE to find all descendants of a given category in a category tree.
4. When would you prefer `EXISTS` over `JOIN` for an existence check?
5. What's a correlated subquery, and how does it differ from a regular subquery?

## See also

- [`window-functions.md`](./window-functions.md) — the ROW_NUMBER + CTE top-N pattern
- [`joins.md`](./joins.md) — LEFT JOIN + IS NULL vs NOT EXISTS
- [`query-execution-order.md`](./query-execution-order.md)
- [`interview-problems.md`](./interview-problems.md) — several problems use recursive CTEs / EXISTS
